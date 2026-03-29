# Flow Matching (Rectified Flow) for Action Generation
#
# References:
#   - NVIDIA Isaac-GR00T: Rectified flow with Beta time sampling and Euler ODE solver
#   - thu-ml/RDT-2: Rectified flow with LogisticNormal time sampling and Euler ODE solver
#   - Lipman et al. (2023): Flow Matching for Generative Modeling
#
# Flow matching formulation:
#   - Interpolation: x_t = (1 - t) * noise + t * x_1   (t=0: noise, t=1: data)
#   - Velocity target: v = x_1 - noise
#   - Loss: MSE(v_pred, v)
#   - Inference: Euler ODE from t=0 to t=1

import torch
import torch.nn as nn
import math


class FlowMatching:
    """
    Rectified Flow / Conditional Flow Matching for action generation.

    Supports two time sampling strategies:
    - 'logit_normal': LogisticNormal(loc, scale) — biases toward mid-range t (RDT-2 style)
    - 'beta': Beta(alpha, beta) — can bias toward early steps (GR00T style)
    """

    def __init__(
        self,
        time_sampling='logit_normal',
        logit_normal_loc=0.0,
        logit_normal_scale=1.0,
        beta_alpha=1.5,
        beta_beta=1.0,
        noise_s=0.999,
        num_timestep_buckets=1000,
    ):
        self.time_sampling = time_sampling
        self.logit_normal_loc = logit_normal_loc
        self.logit_normal_scale = logit_normal_scale
        self.beta_alpha = beta_alpha
        self.beta_beta = beta_beta
        self.noise_s = noise_s
        self.num_timestep_buckets = num_timestep_buckets

    def sample_time(self, batch_size, device, dtype=torch.float32):
        """
        Sample continuous timesteps t in (0, 1) for training.

        Returns:
            t: (batch_size,) tensor of timesteps
        """
        if self.time_sampling == 'logit_normal':
            # LogisticNormal: sample z ~ N(loc, scale), then t = sigmoid(z)
            # Concentrates samples near t=0.5 (RDT-2 approach)
            z = torch.randn(batch_size, device=device, dtype=dtype)
            z = self.logit_normal_loc + self.logit_normal_scale * z
            t = torch.sigmoid(z)
        elif self.time_sampling == 'beta':
            # Beta distribution, then transform (GR00T approach)
            # Beta(1.5, 1.0) skewed toward 1, transform (1-s)*noise_s biases toward 0
            dist = torch.distributions.Beta(self.beta_alpha, self.beta_beta)
            s = dist.sample((batch_size,)).to(device=device, dtype=dtype)
            t = (1 - s) * self.noise_s
        elif self.time_sampling == 'uniform':
            t = torch.rand(batch_size, device=device, dtype=dtype)
        else:
            raise ValueError(f"Unknown time sampling: {self.time_sampling}")

        # Clamp to avoid boundary issues
        t = t.clamp(1e-5, 1.0 - 1e-5)
        return t

    def q_sample(self, x_1, t, noise):
        """
        Compute noisy sample x_t via linear interpolation (forward process).

        x_t = (1 - t) * noise + t * x_1

        Args:
            x_1: (B, T, C) clean action data
            t: (B,) continuous timesteps in [0, 1]
            noise: (B, T, C) Gaussian noise

        Returns:
            x_t: (B, T, C) noisy sample
        """
        t = t[:, None, None]  # (B, 1, 1) for broadcasting
        x_t = (1 - t) * noise + t * x_1
        return x_t

    def compute_velocity(self, x_1, noise):
        """
        Compute ground-truth velocity field (target for training).

        v = x_1 - noise

        Args:
            x_1: (B, T, C) clean action data
            noise: (B, T, C) Gaussian noise

        Returns:
            velocity: (B, T, C) target velocity
        """
        return x_1 - noise

    def discretize_timestep(self, t):
        """
        Discretize continuous t to integer bucket indices for timestep embedding.

        Args:
            t: (B,) continuous timesteps in [0, 1]

        Returns:
            t_discrete: (B,) integer timestep indices in [0, num_timestep_buckets)
        """
        return (t * self.num_timestep_buckets).long().clamp(0, self.num_timestep_buckets - 1)

    @torch.no_grad()
    def euler_sample(
        self,
        model_fn,
        shape,
        num_steps=10,
        device='cuda',
        model_kwargs=None,
    ):
        """
        Generate samples via Euler ODE integration from t=0 (noise) to t=1 (data).

        一阶 Euler 方法: x_{t+dt} = x_t + dt * v_theta(x_t, t)
        简单快速，但步数少时误差大，可能导致轨迹抖动。

        Args:
            model_fn: callable(x, t, **kwargs) -> velocity prediction
            shape: (B, T, C) output shape
            num_steps: number of Euler integration steps
            device: torch device
            model_kwargs: dict of extra kwargs for model_fn

        Returns:
            x: (B, T, C) generated samples
        """
        if model_kwargs is None:
            model_kwargs = {}

        # Start from pure noise at t=0
        x = torch.randn(shape, device=device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t_cont = i / float(num_steps)
            t = torch.full((shape[0],), t_cont, device=device, dtype=x.dtype)
            t_discrete = self.discretize_timestep(t)

            v_pred = model_fn(x, t_discrete, **model_kwargs)
            x = x + dt * v_pred

        return x

    @torch.no_grad()
    def midpoint_sample(
        self,
        model_fn,
        shape,
        num_steps=10,
        device='cuda',
        model_kwargs=None,
    ):
        """
        Generate samples via Midpoint (2nd-order Runge-Kutta) ODE integration.

        二阶中点法: 每步调用两次模型，精度远高于 Euler，
        相同步数下轨迹更平滑，减少机械臂抖动。

        算法:
            k1 = v_theta(x_t, t)
            x_mid = x_t + (dt/2) * k1
            k2 = v_theta(x_mid, t + dt/2)
            x_{t+dt} = x_t + dt * k2

        注意: 每步需要 2 次前向传播，10 步 midpoint ≈ 20 步 Euler 的精度。

        Args:
            model_fn: callable(x, t, **kwargs) -> velocity prediction
            shape: (B, T, C) output shape
            num_steps: number of integration steps (each step = 2 model calls)
            device: torch device
            model_kwargs: dict of extra kwargs for model_fn

        Returns:
            x: (B, T, C) generated samples
        """
        if model_kwargs is None:
            model_kwargs = {}

        x = torch.randn(shape, device=device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t_cont = i / float(num_steps)
            t_mid_cont = (i + 0.5) / float(num_steps)

            # k1: velocity at current point
            t = torch.full((shape[0],), t_cont, device=device, dtype=x.dtype)
            t_discrete = self.discretize_timestep(t)
            k1 = model_fn(x, t_discrete, **model_kwargs)

            # Midpoint: x_mid = x + (dt/2) * k1
            x_mid = x + (dt / 2) * k1

            # k2: velocity at midpoint
            t_mid = torch.full((shape[0],), t_mid_cont, device=device, dtype=x.dtype)
            t_mid_discrete = self.discretize_timestep(t_mid)
            k2 = model_fn(x_mid, t_mid_discrete, **model_kwargs)

            # Full step using midpoint velocity
            x = x + dt * k2

        return x
