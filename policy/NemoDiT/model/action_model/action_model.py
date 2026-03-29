from model.action_model.models import DiT
from model.action_model.flow_matching import FlowMatching
from model.vision_input import VisionBackbone
from model.feature_adaptation import create_feature_adapter
import torch
from torch import nn

# 生成动作模型（根据默认DiT尺寸）
def DiT_S(**kwargs):
    return DiT(depth=12, hidden_size=384, num_heads=6, **kwargs)


def DiT_B(**kwargs):
    return DiT(depth=12, hidden_size=768, num_heads=12, **kwargs)


def DiT_L(**kwargs):
    return DiT(depth=24, hidden_size=1024, num_heads=16, **kwargs)


def DiT_XL(**kwargs):
    return DiT(depth=28, hidden_size=1152, num_heads=16, **kwargs)


DiT_models = {'DiT-S': DiT_S, 'DiT-B': DiT_B, 'DiT-L': DiT_L, 'DiT-XL': DiT_XL}


class ActionModel(nn.Module):
    """
    Flow Matching Action Model for robot manipulation.

    基于 Rectified Flow / Conditional Flow Matching，参考:
    - NVIDIA Isaac-GR00T: Beta 时间采样 + Euler ODE
    - thu-ml/RDT-2: LogisticNormal 时间采样 + Euler ODE

    Flow Matching 公式:
        - 插值: x_t = (1-t) * noise + t * x_1  (t=0: 噪声, t=1: 数据)
        - 速度场目标: v = x_1 - noise
        - 损失: MSE(v_pred, v)
        - 推理: Euler ODE 从 t=0 积分到 t=1

    时序设计:
        n_obs_steps: 观测步数，用于视觉编码的历史帧数
        n_action_steps: 动作执行步数
        state: n_obs_steps 最后一帧 = action 第0帧时刻的机器人状态

        ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
        │O-1│ O │ A │ A │ A │ A │ A │ A │ A │ A │...
        └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
              │   │   └───────────────────────────┘
              │   │     predicted actions (T-1 帧)
              │   │
              │   └── state = action[0]，无噪音条件
              │
              └─── n_obs_steps 的最后一帧
    """

    def __init__(self,
                 token_size,
                 model_type,
                 in_channels,
                 future_action_window_size,
                 past_action_window_size,
                 use_vision_condition,
                 vision_backbone_type,
                 vision_pretrained,
                 num_cameras,
                 adapter_type,
                 # Flow matching parameters
                 time_sampling='logit_normal',
                 logit_normal_loc=0.0,
                 logit_normal_scale=1.0,
                 beta_alpha=1.5,
                 beta_beta=1.0,
                 num_timestep_buckets=1000,
                 # Legacy diffusion params (ignored, kept for checkpoint compat)
                 diffusion_steps=None,
                 noise_schedule=None,
                 freeze_vision_backbone=False,
                 class_dropout_prob=0.1,
                 n_obs_steps=1,
                 n_action_steps=None,
                 temporal_agg='last',
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.use_vision_condition = use_vision_condition
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps if n_action_steps is not None else future_action_window_size
        self.temporal_agg = temporal_agg

        # Flow Matching replaces GaussianDiffusion
        self.flow_matching = FlowMatching(
            time_sampling=time_sampling,
            logit_normal_loc=logit_normal_loc,
            logit_normal_scale=logit_normal_scale,
            beta_alpha=beta_alpha,
            beta_beta=beta_beta,
            num_timestep_buckets=num_timestep_buckets,
        )

        self.past_action_window_size = past_action_window_size
        self.future_action_window_size = future_action_window_size

        # Vision backbone and feature adapter
        if use_vision_condition:
            self.vision_backbone = VisionBackbone(
                backbone_type=vision_backbone_type,
                pretrained=vision_pretrained,
                num_cameras=num_cameras,
                freeze_backbone=freeze_vision_backbone,
                n_obs_steps=n_obs_steps,
                temporal_agg=temporal_agg,
            )

            vision_feature_dim = self.vision_backbone.get_output_dim()

            self.feature_adapter = create_feature_adapter(
                adapter_type=adapter_type,
                vision_feature_dim=vision_feature_dim,
                dit_hidden_size=token_size,
                num_layers=2,
                dropout=0.1
            )
        else:
            self.vision_backbone = None
            self.feature_adapter = None

        self.net = DiT_models[model_type](
            token_size=token_size,
            in_channels=in_channels,
            class_dropout_prob=class_dropout_prob,
            learn_sigma=False,
            future_action_window_size=future_action_window_size,
            past_action_window_size=past_action_window_size
        )

    def encode_vision_condition(self, images):
        """
        Encode images to vision condition features.

        Args:
            images: (batch_size, n_obs_steps, num_cameras, channels, height, width)
                   or (batch_size, num_cameras, channels, height, width)

        Returns:
            vision_condition: (batch_size, 1, token_size)
        """
        if not self.use_vision_condition:
            raise ValueError("Vision condition is not enabled")

        vision_features = self.vision_backbone(images)  # (B, 1, vision_dim)
        vision_condition = self.feature_adapter(vision_features)  # (B, 1, token_size)
        return vision_condition

    def loss(self, x, z=None, images=None, state=None):
        """
        Compute flow matching loss.

        训练目标: 预测速度场 v = x_1 - noise
        损失: MSE(v_pred, v_target)

        Args:
            x: (batch_size, future_action_window_size - 1, in_channels) - ground truth actions
            z: (batch_size, 1, token_size) - precomputed vision condition (optional)
            images: raw images (optional)
            state: (batch_size, in_channels) - 机器人当前状态

        Returns:
            loss: scalar loss value
        """
        if images is not None and self.use_vision_condition:
            z = self.encode_vision_condition(images)

        if z is None:
            raise ValueError("Either z or images must be provided")

        # Sample noise and continuous timesteps
        noise = torch.randn_like(x)  # (B, T-1, C)
        t = self.flow_matching.sample_time(x.size(0), x.device, dtype=x.dtype)  # (B,)

        # Compute noisy sample: x_t = (1-t)*noise + t*x_1
        x_t = self.flow_matching.q_sample(x, t, noise)

        # Discretize timestep for embedding
        t_discrete = self.flow_matching.discretize_timestep(t)

        # Predict velocity from x_t
        v_pred = self.net(x_t, t_discrete, z, state=state)

        # Compute target velocity: v = x_1 - noise
        v_target = self.flow_matching.compute_velocity(x, noise)

        assert v_pred.shape == v_target.shape == x.shape
        # MSE loss on velocity
        loss = ((v_pred - v_target) ** 2).mean()

        return loss

    @torch.no_grad()
    def sample(self, images, state=None, num_steps=10, cfg_scale=0, return_all=False,
               ode_solver='midpoint',
               # Legacy param kept for compatibility
               ddim_steps=None, use_ddim=None):
        """
        通过 ODE 积分生成动作序列。

        从 t=0 (纯噪声) 积分到 t=1 (数据)。

        Args:
            images: (B, n_obs_steps, num_cameras, C, H, W)
            state: (B, in_channels) - 机器人当前状态
            num_steps: ODE 积分步数 (default: 10)
            cfg_scale: Classifier-free guidance scale (default: 0, 无 guidance)
            return_all: 是否返回完整预测动作
            ode_solver: ODE 求解器类型 (default: 'midpoint')
                - 'euler': 一阶 Euler 方法，速度快但精度低
                - 'midpoint': 二阶中点法，精度高，推荐使用
            ddim_steps: Legacy alias for num_steps (backward compat)

        Returns:
            actions: (B, n_action_steps, in_channels)
        """
        # Legacy compatibility: ddim_steps -> num_steps
        if ddim_steps is not None:
            num_steps = ddim_steps

        device = next(self.parameters()).device
        batch_size = images.shape[0]

        # 1. Encode vision condition
        z = self.encode_vision_condition(images)  # (B, 1, token_size)

        # 2. Define model wrapper (with optional CFG)
        if cfg_scale > 1.0:
            def model_fn(x, t, **kwargs):
                return self.net.forward_with_cfg(x, t, kwargs['z'], cfg_scale, state=kwargs.get('state'))
        else:
            def model_fn(x, t, **kwargs):
                return self.net(x, t, kwargs['z'], state=kwargs.get('state'))

        # 3. ODE sampling
        predict_length = self.future_action_window_size - 1
        shape = (batch_size, predict_length, self.in_channels)

        if ode_solver == 'midpoint':
            sample_fn = self.flow_matching.midpoint_sample
        else:
            sample_fn = self.flow_matching.euler_sample

        actions = sample_fn(
            model_fn,
            shape,
            num_steps=num_steps,
            device=device,
            model_kwargs={'z': z, 'state': state},
        )  # (B, future_action_window_size - 1, in_channels)

        # 4. Truncate to n_action_steps
        if return_all:
            return actions
        else:
            return actions[:, :self.n_action_steps, :]
