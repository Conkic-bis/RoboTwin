import torch
import torch.nn as nn
import torchvision.models as models
from typing import Literal, Tuple


class VisionBackbone(nn.Module):
    """
    Vision backbone that extracts features from images using pretrained models.

    支持多帧观测输入 (n_obs_steps)，通过时间维度聚合得到单个全局视觉特征。

    Args:
        backbone_type: Type of backbone ('resnet18', 'resnet34', 'resnet50', 'vit_b_16', 'vit_b_32')
        pretrained: Whether to use pretrained weights
        num_cameras: Number of camera views to process
        freeze_backbone: Whether to freeze the backbone weights
        n_obs_steps: Number of observation steps (temporal frames)
        temporal_agg: Temporal aggregation method ('last', 'mean', 'concat')
            - 'last': 只使用最后一帧
            - 'mean': 对所有帧取平均
            - 'concat': 拼接所有帧特征后投影

    Returns:
        image_embeds: (batch_size, n_tokens, feature_dim) - 统一的视觉特征序列
    """

    def __init__(
        self,
        backbone_type: Literal['resnet18', 'resnet34', 'resnet50', 'vit_b_16', 'vit_b_32'] = 'resnet50',
        pretrained: bool = True,
        num_cameras: int = 4,
        freeze_backbone: bool = False,
        n_obs_steps: int = 1,
        temporal_agg: str = 'last',
    ):
        super().__init__()

        self.backbone_type = backbone_type
        self.num_cameras = num_cameras
        self.freeze_backbone = freeze_backbone
        self.n_obs_steps = n_obs_steps
        self.temporal_agg = temporal_agg

        assert temporal_agg in ['last', 'mean', 'concat'], \
            f"temporal_agg must be 'last', 'mean', or 'concat', got {temporal_agg}"

        # Initialize backbone based on type
        if 'resnet' in backbone_type:
            self.backbone, self.feature_dim = self._create_resnet_backbone(backbone_type, pretrained)
        elif 'vit' in backbone_type:
            self.backbone, self.feature_dim = self._create_vit_backbone(backbone_type, pretrained)
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")

        # Optionally freeze backbone
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Feature aggregation for multiple cameras
        # 多相机特征融合：将多个相机的全局特征拼接后投影回原维度
        if num_cameras > 1:
            self.camera_fusion = nn.Linear(self.feature_dim * num_cameras, self.feature_dim)
        else:
            self.camera_fusion = nn.Identity()

        # Temporal aggregation layer (for 'concat' mode)
        if temporal_agg == 'concat' and n_obs_steps > 1:
            self.temporal_fusion = nn.Linear(self.feature_dim * n_obs_steps, self.feature_dim)
        else:
            self.temporal_fusion = None

    def _create_resnet_backbone(self, backbone_type: str, pretrained: bool) -> Tuple[nn.Module, int]:
        """
        Create a ResNet backbone and extract features before final pooling.

        Returns:
            backbone: Modified ResNet model
            feature_dim: Dimension of output features
        """
        if backbone_type == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif backbone_type == 'resnet34':
            model = models.resnet34(pretrained=pretrained)
            feature_dim = 512
        elif backbone_type == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported ResNet type: {backbone_type}")

        # Remove the final average pooling and fully connected layer
        # We want to keep spatial features as "patches"
        backbone = nn.Sequential(*list(model.children())[:-2])

        return backbone, feature_dim

    def _create_vit_backbone(self, backbone_type: str, pretrained: bool) -> Tuple[nn.Module, int]:
        """
        Create a Vision Transformer backbone.

        Returns:
            backbone: ViT model
            feature_dim: Dimension of output features (hidden size)
        """
        if backbone_type == 'vit_b_16':
            model = models.vit_b_16(pretrained=pretrained)
            feature_dim = 768
        elif backbone_type == 'vit_b_32':
            model = models.vit_b_32(pretrained=pretrained)
            feature_dim = 768
        else:
            raise ValueError(f"Unsupported ViT type: {backbone_type}")

        # Remove the classification head
        # Keep encoder to get patch embeddings
        model.heads = nn.Identity()

        return model, feature_dim

    def _backbone_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        统一的 backbone forward，支持 batch 并行。

        Args:
            x: (N, C, H, W) - N 可以是 B * n_frames * num_cameras

        Returns:
            features: (N, n_tokens, feature_dim)
                - ResNet: n_tokens = H' * W' (spatial feature vectors)
                - ViT: n_tokens = 1 (CLS token)
        """
        if 'resnet' in self.backbone_type:
            features = self.backbone(x)  # (N, feat_dim, H', W')
            N, channels, H, W = features.shape
            features = features.permute(0, 2, 3, 1).reshape(N, H * W, channels)
        else:  # ViT
            x = self.backbone._process_input(x)
            n = x.shape[0]
            batch_class_token = self.backbone.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)
            x = self.backbone.encoder(x)  # (N, num_patches+1, feature_dim)
            features = x[:, 0:1, :]  # (N, 1, feature_dim) - CLS token
        return features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through vision backbone.

        将多帧、多相机全部 reshape 进 batch 维度，一次 backbone forward 完成，
        避免 for 循环串行调用，充分利用 GPU 并行。

        输入格式:
        - 多帧多相机: (B, n_obs_steps, num_cameras, C, H, W)
        - 单帧多相机: (B, num_cameras, C, H, W)
        - 单帧单相机: (B, C, H, W)

        Returns:
            image_embeds: (B, n_tokens, feature_dim) - 统一的视觉特征序列
        """
        # Normalize input to (B, n_obs_steps, num_cameras, C, H, W)
        if images.dim() == 4:
            images = images.unsqueeze(1).unsqueeze(1)
        elif images.dim() == 5:
            images = images.unsqueeze(1)

        batch_size, n_frames, num_cams, C, H, W = images.shape

        # ========== 将 (B, T, K, C, H, W) reshape 为 (B*T*K, C, H, W) 一次性 forward ==========
        flat_images = images.reshape(batch_size * n_frames * num_cams, C, H, W)
        flat_features = self._backbone_forward(flat_images)  # (B*T*K, n_tokens_per_cam, feat_dim)
        n_tokens_per_cam = flat_features.shape[1]
        feat_dim = flat_features.shape[2]

        # Reshape back: (B, T, K, n_tokens_per_cam, feat_dim)
        all_features = flat_features.reshape(batch_size, n_frames, num_cams, n_tokens_per_cam, feat_dim)

        # ========== 多相机融合 (沿 K 维度) ==========
        if num_cams > 1:
            if n_tokens_per_cam == 1:
                # ViT CLS: 在 feature 维度 concat 后 linear 融合
                # (B, T, K, 1, feat_dim) -> (B, T, 1, feat_dim * K)
                cam_features = all_features.squeeze(3)  # (B, T, K, feat_dim)
                cam_features = cam_features.reshape(batch_size, n_frames, 1, feat_dim * num_cams)
                cam_features = self.camera_fusion(cam_features)  # (B, T, 1, feat_dim)
            else:
                # ResNet spatial: 在 token 维度 concat
                # (B, T, K, n_tokens_per_cam, feat_dim) -> (B, T, K * n_tokens_per_cam, feat_dim)
                cam_features = all_features.reshape(batch_size, n_frames, num_cams * n_tokens_per_cam, feat_dim)
        else:
            cam_features = all_features.squeeze(2)  # (B, T, n_tokens_per_cam, feat_dim)

        # cam_features: (B, T, n_tokens, feat_dim)
        # n_tokens = K * n_tokens_per_cam (ResNet) 或 1 (ViT)

        # ========== 时间聚合 (沿 T 维度) ==========
        if n_frames == 1:
            image_embeds = cam_features[:, 0]  # (B, n_tokens, feat_dim)
        elif self.temporal_agg == 'last':
            image_embeds = cam_features[:, -1]  # (B, n_tokens, feat_dim)
        elif self.temporal_agg == 'mean':
            image_embeds = cam_features.mean(dim=1)  # (B, n_tokens, feat_dim)
        elif self.temporal_agg == 'concat':
            # (B, T, n_tokens, feat_dim) -> (B, n_tokens, feat_dim * T)
            image_embeds = cam_features.permute(0, 2, 1, 3).reshape(
                batch_size, cam_features.shape[2], feat_dim * n_frames
            )
            image_embeds = self.temporal_fusion(image_embeds)  # (B, n_tokens, feat_dim)
        else:
            raise ValueError(f"Unknown temporal_agg: {self.temporal_agg}")

        return image_embeds

    def get_output_dim(self) -> int:
        """Get the output feature dimension."""
        return self.feature_dim
