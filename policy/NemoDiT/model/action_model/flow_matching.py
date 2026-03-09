# Flow Matching for Action Generation
# References:
#   - Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023
#   - NVIDIA Isaac-GR00T: https://github.com/NVIDIA/Isaac-GR00T
#   - thu-ml/RDT2: https://github.com/thu-ml/RDT2

import torch
import torch.nn as nn
import math


class LogisticNormal:
    """
    Logistic-Normal distribution for timestep sampling.

    Samples t ~ LogisticNormal(loc, scale), which concentrates samples
    away from 0 and 1, improving training stability for flow matching.
    """

    def __init__(self, loc=0.0, scale=1.0):
        self.loc = loc
        self.scale = scale

    def sample(self, shape, device='cpu'):
        """Sample from LogisticNormal distribution, returning values in (0, 1)."""
        normal_samples = torch.randn(shape, device=device) * self.scale + self.loc
        return torch.sigmoid(normal_samples)


class FlowMatching(nn.Module):
    """
    Conditional Flow Matching (CFM) with optimal transport (rectified flow) paths.

    Uses linear interpolation paths between noise and data:
        x_t = (1 - t) * x_0 + t * x_1
    where x_0 is noise (t=0) and x_1 is data (t=1).

    The model learns the velocity field v(x_t, t) = x_1 - x_0,
    and sampling is done via Euler integration from t=0 to t=1.

    Args:
        num_inference_steps: Number of Euler integration steps for sampling (default: 10).
        timestep_loc: Location parameter for LogisticNormal timestep sampling (default: 0.0).
        timestep_scale: Scale parameter for LogisticNormal timestep sampling (default: 1.0).
    """

    def __init__(self, num_inference_steps=10, timestep_loc=0.0, timestep_scale=1.0):
        super().__init__()
        self.num_inference_steps = num_inference_steps
        self.timestep_sampler = LogisticNormal(loc=timestep_loc, scale=timestep_scale)

    def sample_timesteps(self, batch_size, device):
        """
        Sample timesteps from LogisticNormal distribution.

        Args:
            batch_size: Number of timesteps to sample.
            device: Torch device.

        Returns:
            t: (batch_size,) timesteps in (0, 1).
        """
        return self.timestep_sampler.sample((batch_size,), device=device)

    def interpolate(self, x_1, noise, t):
        """
        Linear interpolation (optimal transport path) between noise and data.

        x_t = (1 - t) * noise + t * x_1

        Args:
            x_1: (B, T, C) clean data (target actions).
            noise: (B, T, C) Gaussian noise (x_0).
            t: (B,) timesteps in [0, 1].

        Returns:
            x_t: (B, T, C) interpolated samples.
        """
        # Reshape t for broadcasting: (B,) -> (B, 1, 1)
        t = t.view(-1, 1, 1)
        return (1 - t) * noise + t * x_1

    def velocity_target(self, x_1, noise):
        """
        Compute the target velocity for rectified flow.

        v = x_1 - x_0 = data - noise

        Args:
            x_1: (B, T, C) clean data.
            noise: (B, T, C) Gaussian noise.

        Returns:
            v: (B, T, C) target velocity.
        """
        return x_1 - noise

    def compute_loss(self, model, x_1, z=None, state=None):
        """
        Compute flow matching training loss.

        Args:
            model: The neural network that predicts velocity v(x_t, t).
            x_1: (B, T, C) ground truth actions (clean data).
            z: (B, 1, D) vision condition features.
            state: (B, C) robot current state.

        Returns:
            loss: Scalar MSE loss between predicted and target velocity.
        """
        batch_size = x_1.shape[0]
        device = x_1.device

        # Sample noise and timesteps
        noise = torch.randn_like(x_1)
        t = self.sample_timesteps(batch_size, device)

        # Create interpolated sample: x_t = (1-t)*noise + t*data
        x_t = self.interpolate(x_1, noise, t)

        # Target velocity: v = data - noise
        target = self.velocity_target(x_1, noise)

        # Predict velocity from model
        v_pred = model(x_t, t, z, state=state)

        assert v_pred.shape == target.shape == x_1.shape
        # MSE loss
        loss = ((v_pred - target) ** 2).mean()
        return loss

    @torch.no_grad()
    def sample(self, model, shape, z=None, state=None, device=None,
               num_steps=None, cfg_scale=0, progress=False):
        """
        Generate samples via Euler integration of the learned velocity field.

        Integrates from t=0 (pure noise) to t=1 (clean data):
            x_{t+dt} = x_t + v_theta(x_t, t) * dt

        Args:
            model: Velocity prediction model v(x_t, t).
            shape: (B, T, C) shape of samples to generate.
            z: (B, 1, D) vision condition features.
            state: (B, C) robot current state.
            device: Torch device.
            num_steps: Number of Euler steps (overrides self.num_inference_steps).
            cfg_scale: Classifier-free guidance scale (0 = no guidance).
            progress: Whether to show progress bar.

        Returns:
            x: (B, T, C) generated samples.
        """
        if device is None:
            device = next(iter([])) if False else 'cpu'

        num_steps = num_steps if num_steps is not None else self.num_inference_steps
        dt = 1.0 / num_steps

        # Start from pure noise (t=0)
        x = torch.randn(*shape, device=device)

        timesteps = torch.linspace(0, 1 - dt, num_steps, device=device)

        if progress:
            from tqdm.auto import tqdm
            timesteps = tqdm(timesteps, desc="Flow Matching Sampling")

        for t_scalar in timesteps:
            t = torch.full((shape[0],), t_scalar.item(), device=device)

            if cfg_scale > 1.0:
                v_pred = model.forward_with_cfg(x, t, z, cfg_scale, state=state)
            else:
                v_pred = model(x, t, z, state=state)

            # Euler step: x_{t+dt} = x_t + v * dt
            x = x + v_pred * dt

        return x
