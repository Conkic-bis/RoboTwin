from model.action_model.models import DiT
from model.action_model.flow_matching import FlowMatching
from model.vision_input import VisionBackbone
from model.feature_adaptation import create_feature_adapter
import torch
from torch import nn

# 生成动作模型（根据默认DiT尺寸）
def DiT_S(**kwargs):
    return DiT(depth=6, hidden_size=384, num_heads=4, **kwargs)


def DiT_B(**kwargs):
    return DiT(depth=12, hidden_size=768, num_heads=12, **kwargs)


def DiT_L(**kwargs):
    return DiT(depth=24, hidden_size=1024, num_heads=16, **kwargs)


def DiT_XL(**kwargs):
    return DiT(depth=28, hidden_size=1152, num_heads=16, **kwargs)


DiT_models = {'DiT-S': DiT_S, 'DiT-B': DiT_B, 'DiT-L': DiT_L, 'DiT-XL': DiT_XL}


class ActionModel(nn.Module):
    """
    Flow Matching-based Action Model for robot manipulation.

    使用 Conditional Flow Matching (CFM) / Rectified Flow 替代 DDIM 进行动作生成。
    模型学习速度场 v(x_t, t)，通过 Euler 积分从噪声生成动作序列。

    支持多帧观测输入 (n_obs_steps) 和 state 条件输入。

    时序设计:
        n_obs_steps: 观测步数，用于视觉编码的历史帧数
        n_action_steps: 动作执行步数，实际执行的动作数（比原来少1帧，因为 state 占了第0帧）
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
                 num_inference_steps=10,
                 freeze_vision_backbone=False,
                 class_dropout_prob=0.1,
                 n_obs_steps=1,
                 n_action_steps=None,
                 temporal_agg='last',
                 # Legacy params (ignored, kept for checkpoint compatibility)
                 diffusion_steps=None,
                 noise_schedule=None,
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.use_vision_condition = use_vision_condition
        self.n_obs_steps = n_obs_steps
        # n_action_steps: 推理时实际执行的动作步数，默认等于 future_action_window_size
        self.n_action_steps = n_action_steps if n_action_steps is not None else future_action_window_size
        self.temporal_agg = temporal_agg

        # Flow Matching: replaces GaussianDiffusion / DDIM
        self.num_inference_steps = num_inference_steps
        self.flow_matching = FlowMatching(
            num_inference_steps=num_inference_steps,
            timestep_loc=0.0,
            timestep_scale=1.0,
        )

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

        # Flow matching predicts velocity, not noise+variance, so learn_sigma=False
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

    # Given condition z, state and ground truth token x, compute flow matching loss
    def loss(self, x, z=None, images=None, state=None):
        """
        Compute flow matching loss.

        模型学习速度场 v(x_t, t) = x_1 - x_0，其中 x_1 为真实数据，x_0 为噪声。
        使用线性插值路径: x_t = (1-t)*x_0 + t*x_1。

        Args:
            x: (batch_size, future_action_window_size - 1, in_channels) - ground truth actions (不含 state)
            z: (batch_size, 1, token_size) - precomputed vision condition (optional)
            images: (batch_size, n_obs_steps, num_cameras, 3, H, W) - raw images (optional)
                   or (batch_size, num_cameras, 3, H, W) for single frame
            state: (batch_size, in_channels) - 机器人当前状态 (action[0])，无噪音

        Returns:
            loss: scalar flow matching loss value
        """
        # Encode vision condition if images are provided
        if images is not None and self.use_vision_condition:
            z = self.encode_vision_condition(images)

        if z is None:
            raise ValueError("Either z or images must be provided")

        # Compute flow matching loss: MSE(v_pred, x_1 - noise)
        loss = self.flow_matching.compute_loss(self.net, x_1=x, z=z, state=state)

        return loss

    @torch.no_grad()
    def sample(self, images, state=None, num_steps=None, cfg_scale=0, return_all=False,
               # Legacy params (kept for API compatibility)
               ddim_steps=None, use_ddim=None):
        """
        从观测图像和当前状态生成动作序列 (推理/采样)。

        使用 Flow Matching Euler 积分从噪声 (t=0) 到数据 (t=1) 生成动作。

        推理流程:
        1. 编码视觉条件: images -> z (B, 1, token_size)
        2. 从高斯噪声开始，通过 Euler 积分生成动作序列 (future_action_window - 1 帧)
        3. 截取前 n_action_steps 步动作用于执行

        Args:
            images: (B, n_obs_steps, num_cameras, C, H, W) - 多帧多相机观测
                   or (B, num_cameras, C, H, W) - 单帧多相机
            state: (B, in_channels) - 机器人当前状态 (n_obs_steps 最后一帧的动作值)
            num_steps: Euler 积分步数 (default: self.num_inference_steps)
            cfg_scale: Classifier-free guidance scale (default: 0, 无 guidance)
            return_all: 是否返回完整预测动作 (default: False)
            ddim_steps: [Legacy] 映射到 num_steps，保持向后兼容
            use_ddim: [Legacy] 忽略

        Returns:
            actions: (B, n_action_steps, in_channels) - 用于执行的动作序列
                    如果 return_all=True，返回 (B, future_action_window_size - 1, in_channels)
        """
        device = next(self.parameters()).device
        batch_size = images.shape[0]

        # Legacy compatibility: ddim_steps -> num_steps
        if num_steps is None and ddim_steps is not None:
            num_steps = ddim_steps

        # 1. 编码视觉条件
        z = self.encode_vision_condition(images)  # (B, 1, token_size)

        # 2. Flow Matching Euler 采样 — 生成 future_action_window - 1 帧 (不含 state)
        predict_length = self.future_action_window_size - 1
        shape = (batch_size, predict_length, self.in_channels)
        actions = self.flow_matching.sample(
            self.net,
            shape,
            z=z,
            state=state,
            device=device,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            progress=False,
        )  # (B, future_action_window_size - 1, in_channels)

        # 3. 截取 n_action_steps 步动作
        if return_all:
            return actions
        else:
            return actions[:, :self.n_action_steps, :]  # (B, n_action_steps, in_channels)
