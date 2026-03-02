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

    def extract_resnet_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract global features from ResNet using Global Average Pooling.

        ResNet 通过层级卷积提取特征，最终使用全局平均池化得到单个全局特征向量。
        不同于 ViT 的 patch-based attention，ResNet 的特征是层级聚合的结果。

        Args:
            x: (batch_size, channels, height, width)

        Returns:
            features: (batch_size, n_tokens, feature_dim) - ResNet patch 特征序列
        """
        features = self.backbone(x)  # (B, C, H, W) e.g., (B, 2048, 7, 7)

        # print(f"ResNet features shape: {features.shape}")

        batch_size, channels, height, width = features.shape
        features = features.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)

        # print(f"ResNet patch features shape: {features.shape}")

        return features

    def extract_vit_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract global features from ViT using [CLS] token.

        ViT 通过 patch embedding 和 self-attention 提取特征。
        [CLS] token 通过与所有 patch tokens 的 attention 聚合了全局信息，
        可作为整张图像的全局表示。

        Args:
            x: (batch_size, channels, height, width)

        Returns:
            features: (batch_size, 1, feature_dim) - CLS token 作为全局特征
        """
        # Forward through ViT encoder
        x = self.backbone._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.backbone.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.backbone.encoder(x)  # (B, num_patches+1, feature_dim)

        # 只取 [CLS] token (index 0) 作为全局表示
        cls_token = x[:, 0:1, :]  # (B, 1, feature_dim)

        return cls_token

    def _extract_single_frame_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features from a single frame (multiple cameras).

        Args:
            images: (batch_size, num_cameras, C, H, W)

        Returns:
            features: (batch_size, n_tokens, feature_dim) - 融合后的单帧特征
        """
        batch_size, num_cams, C, H, W = images.shape

        # Process each camera view
        all_features = []
        for cam_idx in range(num_cams):
            cam_image = images[:, cam_idx]  # (B, C, H, W)

            # Extract features based on backbone type
            # 两种 backbone 都输出 (B, n_tokens, feature_dim)
            if 'resnet' in self.backbone_type:
                features = self.extract_resnet_features(cam_image)  # (B, n_tokens, feature_dim)
            else:  # ViT
                features = self.extract_vit_features(cam_image)  # (B, n_tokens, feature_dim)

            all_features.append(features)

        # Aggregate features from multiple cameras
        if num_cams > 1:
            if all_features[0].shape[1] == 1:
                frame_embeds = torch.cat(all_features, dim=2)  # (B, 1, feature_dim * num_cameras)
                frame_embeds = self.camera_fusion(frame_embeds)  # (B, 1, feature_dim)
            else:
                frame_embeds = torch.cat(all_features, dim=1)  # (B, n_tokens * num_cameras, feature_dim)
        else:
            frame_embeds = all_features[0]  # (B, n_tokens, feature_dim)

        return frame_embeds

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through vision backbone.

        支持多帧观测输入，提取多相机视觉特征并融合为单个全局表示。

        输入格式:
        - 多帧多相机: (B, n_obs_steps, num_cameras, C, H, W)
        - 单帧多相机: (B, num_cameras, C, H, W)
        - 单帧单相机: (B, C, H, W)

        处理流程:
        1. 对每帧提取多相机特征并融合: (B, n_tokens, feature_dim)
        2. 时间聚合 (temporal_agg):
           - 'last': 只用最后一帧
           - 'mean': 所有帧取平均
           - 'concat': 拼接后投影

        Args:
            images: (B, n_obs_steps, num_cameras, C, H, W) - 多帧多相机
                   or (B, num_cameras, C, H, W) - 单帧多相机
                   or (B, C, H, W) - 单帧单相机

        Returns:
            image_embeds: (B, n_tokens, feature_dim) - 统一的视觉特征序列
        """
        # Normalize input to (B, n_obs_steps, num_cameras, C, H, W)
        if images.dim() == 4:
            # (B, C, H, W) -> (B, 1, 1, C, H, W)
            images = images.unsqueeze(1).unsqueeze(1)
        elif images.dim() == 5:
            # (B, num_cameras, C, H, W) -> (B, 1, num_cameras, C, H, W)
            images = images.unsqueeze(1)
        # Now images is (B, n_obs_steps, num_cameras, C, H, W)

        batch_size, n_frames, num_cams, C, H, W = images.shape

        # Extract features for each frame
        frame_features = []
        for t in range(n_frames):
            frame_images = images[:, t]  # (B, num_cameras, C, H, W)
            frame_feat = self._extract_single_frame_features(frame_images)  # (B, 1, feature_dim)
            frame_features.append(frame_feat)

        # Temporal aggregation
        if n_frames == 1:
            # 单帧情况，直接返回
            image_embeds = frame_features[0]  # (B, 1, feature_dim)
        elif self.temporal_agg == 'last':
            # 只使用最后一帧
            image_embeds = frame_features[-1]  # (B, 1, feature_dim)
        elif self.temporal_agg == 'mean':
            # 对所有帧取平均
            stacked = torch.stack(frame_features, dim=1)  # (B, n_frames, n_tokens, feature_dim)
            image_embeds = stacked.mean(dim=1)  # (B, n_tokens, feature_dim)
        elif self.temporal_agg == 'concat':
            # 拼接所有帧特征后投影
            # frame_features: List of (B, n_tokens, feature_dim)
            concat_feat = torch.cat(frame_features, dim=-1)  # (B, n_tokens, feature_dim * n_frames)
            image_embeds = self.temporal_fusion(concat_feat)  # (B, n_tokens, feature_dim)
        else:
            raise ValueError(f"Unknown temporal_agg: {self.temporal_agg}")

        return image_embeds

    def get_output_dim(self) -> int:
        """Get the output feature dimension."""
        return self.feature_dim


# Example usage and testing
if __name__ == "__main__":
    # Test ResNet50 with single frame
    print("Testing ResNet50 backbone (single frame)...")
    resnet_backbone = VisionBackbone(
        backbone_type='resnet50',
        pretrained=False,
        num_cameras=4,
        freeze_backbone=False
    )

    # Test with 4 camera views, single frame
    dummy_images = torch.randn(2, 4, 3, 224, 224)  # (batch_size=2, 4 cameras, RGB, 224x224)
    output = resnet_backbone(dummy_images)
    print(f"ResNet50 output shape: {output.shape}")  # Expected: (2, 1, 2048)
    print(f"Feature dimension: {resnet_backbone.get_output_dim()}")
    assert output.shape == (2, 1, 2048), f"Expected (2, 1, 2048), got {output.shape}"

    # Test with multi-frame input
    print("\nTesting ResNet50 backbone (multi-frame, n_obs_steps=3)...")
    resnet_multi = VisionBackbone(
        backbone_type='resnet50',
        pretrained=False,
        num_cameras=4,
        freeze_backbone=False,
        n_obs_steps=3,
        temporal_agg='mean'
    )
    multi_frame_images = torch.randn(2, 3, 4, 3, 224, 224)  # (B, n_obs_steps, num_cameras, C, H, W)
    output = resnet_multi(multi_frame_images)
    print(f"Multi-frame output shape: {output.shape}")  # Expected: (2, 1, 2048)
    assert output.shape == (2, 1, 2048), f"Expected (2, 1, 2048), got {output.shape}"

    # Test temporal_agg='concat'
    print("\nTesting temporal_agg='concat'...")
    resnet_concat = VisionBackbone(
        backbone_type='resnet50',
        pretrained=False,
        num_cameras=4,
        freeze_backbone=False,
        n_obs_steps=2,
        temporal_agg='concat'
    )
    two_frame_images = torch.randn(2, 2, 4, 3, 224, 224)
    output = resnet_concat(two_frame_images)
    print(f"Concat output shape: {output.shape}")  # Expected: (2, 1, 2048)
    assert output.shape == (2, 1, 2048), f"Expected (2, 1, 2048), got {output.shape}"

    # Test ViT
    print("\nTesting ViT-B/16 backbone...")
    vit_backbone = VisionBackbone(
        backbone_type='vit_b_16',
        pretrained=False,
        num_cameras=4,
        freeze_backbone=False
    )

    output = vit_backbone(dummy_images)
    print(f"ViT-B/16 output shape: {output.shape}")  # Expected: (2, 1, 768)
    print(f"Feature dimension: {vit_backbone.get_output_dim()}")
    assert output.shape == (2, 1, 768), f"Expected (2, 1, 768), got {output.shape}"

    # Test single camera
    print("\nTesting single camera...")
    single_cam_backbone = VisionBackbone(
        backbone_type='resnet50',
        pretrained=False,
        num_cameras=1,
        freeze_backbone=False
    )
    single_cam_images = torch.randn(2, 1, 3, 224, 224)
    output = single_cam_backbone(single_cam_images)
    print(f"Single camera output shape: {output.shape}")  # Expected: (2, 1, 2048)
    assert output.shape == (2, 1, 2048), f"Expected (2, 1, 2048), got {output.shape}"

    print("\nVision backbone tests passed!")
