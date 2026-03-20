from model.action_model.models import DiT
from model.action_model import create_diffusion
from . import gaussian_diffusion as gd
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
    Diffusion-based Action Model for robot manipulation.

    支持多帧观测输入 (n_obs_steps) 和 state 条件输入。

    时序设计:
        n_obs_steps: 观测步数，用于视觉编码的历史帧数
        n_action_steps: 动作执行步数，实际执行的动作数（比原来少1帧，因为 state 占了第0帧）
        state: n_obs_steps 最后一帧 = action 第0帧时刻的机器人状态

        ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
        │O-1│ O │ A │ A │ A │ A │ A │ A │ A │ A │...
        └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
              │   │   └───────────────────────────┘
              │   │     predicted actions (T-1 帧，有噪音)
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
                 diffusion_steps=100,
                 noise_schedule='squaredcos_cap_v2',
                 freeze_vision_backbone=False,
                 class_dropout_prob=0.1,
                 n_obs_steps=1,
                 n_action_steps=None,
                 temporal_agg='last',
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.noise_schedule = noise_schedule
        self.use_vision_condition = use_vision_condition
        self.n_obs_steps = n_obs_steps
        # n_action_steps: 推理时实际执行的动作步数，默认等于 future_action_window_size
        self.n_action_steps = n_action_steps if n_action_steps is not None else future_action_window_size
        self.temporal_agg = temporal_agg

        # GaussianDiffusion offers forward and backward functions q_sample and p_sample.
        self.diffusion_steps = diffusion_steps
        self.diffusion = create_diffusion(timestep_respacing="", noise_schedule=noise_schedule,
                                          diffusion_steps=self.diffusion_steps, sigma_small=True, learn_sigma=False)
        self.ddim_diffusion = None
        if self.diffusion.model_var_type in [gd.ModelVarType.LEARNED, gd.ModelVarType.LEARNED_RANGE]:
            learn_sigma = True
        else:
            learn_sigma = False
        self.past_action_window_size = past_action_window_size
        self.future_action_window_size = future_action_window_size

        # 如果引入其他的模态，将以此设计其他的模态融合函数，If Not则不使用任何Condition进行进行生成轨迹
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

            # Create feature adapter to project vision features to token_size
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
            learn_sigma=learn_sigma,
            future_action_window_size=future_action_window_size,
            past_action_window_size=past_action_window_size
        )

    def encode_vision_condition(self, images):
        """
        Encode images to vision condition features.

        支持多帧观测输入，将多相机图像编码为单个全局视觉条件向量。
        - 多帧聚合: 根据 temporal_agg 参数选择聚合方式 ('last', 'mean', 'concat')
        - ResNet: 通过 Global Average Pooling 提取全局特征
        - ViT: 通过 [CLS] token 提取全局特征
        - 多相机特征融合后投影到 token_size 维度

        Args:
            images: (batch_size, n_obs_steps, num_cameras, channels, height, width) - 多帧
                   or (batch_size, num_cameras, channels, height, width) - 单帧

        Returns:
            vision_condition: (batch_size, 1, token_size) - 单个全局视觉条件
        """
        if not self.use_vision_condition:
            raise ValueError("Vision condition is not enabled")

        # Extract vision features (已融合多相机)
        vision_features = self.vision_backbone(images)  # (B, 1, vision_dim)

        # Adapt features to token_size
        vision_condition = self.feature_adapter(vision_features)  # (B, 1, token_size)

        return vision_condition

    # Given condition z, state and ground truth token x, compute loss
    def loss(self, x, z=None, images=None, state=None):
        """
        Compute diffusion loss.

        噪音仅施加在 action[1:] 上 (即 x)，state (action[0]) 作为无噪音条件传入模型。

        Args:
            x: (batch_size, future_action_window_size - 1, in_channels) - ground truth actions (不含 state)
            z: (batch_size, 1, token_size) - precomputed vision condition (optional)
            images: (batch_size, n_obs_steps, num_cameras, 3, H, W) - raw images (optional)
                   or (batch_size, num_cameras, 3, H, W) for single frame
            state: (batch_size, in_channels) - 机器人当前状态 (action[0])，无噪音

        Returns:
            loss: scalar loss value
        """
        # Encode vision condition if images are provided
        if images is not None and self.use_vision_condition:
            z = self.encode_vision_condition(images)

        if z is None:
            raise ValueError("Either z or images must be provided")

        # sample random noise and timestep — 噪音仅施加在 actions 上，不影响 state
        noise = torch.randn_like(x)  # [B, T-1, C]
        timestep = torch.randint(0, self.diffusion.num_timesteps, (x.size(0),), device=x.device)

        # sample x_t from x (forward diffusion on actions only)
        x_t = self.diffusion.q_sample(x, timestep, noise)

        # predict noise from x_t, with state as clean conditioning
        noise_pred = self.net(x_t, timestep, z, state=state)

        assert noise_pred.shape == noise.shape == x.shape
        # Compute L2 loss
        loss = ((noise_pred - noise) ** 2).mean()
        # Optional: loss += loss_vlb

        return loss

    # Create DDIM sampler
    def create_ddim(self, ddim_step=10):
        self.ddim_diffusion = create_diffusion(timestep_respacing="ddim" + str(ddim_step),
                                               noise_schedule=self.noise_schedule,
                                               diffusion_steps=self.diffusion_steps,
                                               sigma_small=True,
                                               learn_sigma=False
                                               )
        return self.ddim_diffusion

    @torch.no_grad()
    def sample(self, images, state=None, ddim_steps=100, use_ddim=True, cfg_scale=0, return_all=False):
        """
        从观测图像和当前状态生成动作序列 (推理/采样)。

        推理流程:
        1. 编码视觉条件: images -> z (B, 1, token_size)
        2. 从高斯噪声开始，通过 DDIM 采样生成动作序列 (future_action_window - 1 帧)
        3. 截取前 n_action_steps 步动作用于执行

        Args:
            images: (B, n_obs_steps, num_cameras, C, H, W) - 多帧多相机观测
                   or (B, num_cameras, C, H, W) - 单帧多相机
            state: (B, in_channels) - 机器人当前状态 (n_obs_steps 最后一帧的动作值)
            ddim_steps: DDIM 采样步数，越大质量越好但速度越慢 (default: 100)
            use_ddim: 是否使用 DDIM 加速采样 (default: True)
            cfg_scale: Classifier-free guidance scale (default: 1.5, 无 guidance)
            return_all: 是否返回完整预测动作 (default: False)

        Returns:
            actions: (B, n_action_steps, in_channels) - 用于执行的动作序列
                    如果 return_all=True，返回 (B, future_action_window_size - 1, in_channels)
        """
        device = next(self.parameters()).device
        batch_size = images.shape[0]

        # 1. 编码视觉条件
        z = self.encode_vision_condition(images)  # (B, 1, token_size)

        # 2. 准备采样器
        if use_ddim:
            if self.ddim_diffusion is None or self.ddim_diffusion.num_timesteps != ddim_steps:
                self.create_ddim(ddim_steps)
            diffusion = self.ddim_diffusion
            sample_fn = diffusion.ddim_sample_loop
        else:
            diffusion = self.diffusion
            sample_fn = diffusion.p_sample_loop

        # 3. 定义模型包装器 (用于 CFG)
        if cfg_scale > 1.0:
            # Classifier-free guidance: 需要同时计算条件和无条件预测
            def model_fn(x, t, **kwargs):
                return self.net.forward_with_cfg(x, t, kwargs['z'], cfg_scale, state=kwargs.get('state'))
        else:
            # 无 guidance，直接使用模型
            def model_fn(x, t, **kwargs):
                return self.net(x, t, kwargs['z'], state=kwargs.get('state'))

        # 4. DDIM/DDPM 采样 — 生成 future_action_window - 1 帧 (不含 state)
        predict_length = self.future_action_window_size - 1
        shape = (batch_size, predict_length, self.in_channels)
        actions = sample_fn(
            model_fn,
            shape,
            clip_denoised=False,  # 动作空间不需要 clip 到 [-1, 1]
            model_kwargs={'z': z, 'state': state},
            device=device,
            progress=False,
        )  # (B, future_action_window_size - 1, in_channels)

        # 5. 截取 n_action_steps 步动作
        if return_all:
            return actions
        else:
            return actions[:, :self.n_action_steps, :]  # (B, n_action_steps, in_channels)
