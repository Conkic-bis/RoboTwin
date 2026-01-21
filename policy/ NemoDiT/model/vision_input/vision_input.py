"""
vision_input.py

Vision backbone network for extracting visual features from robot camera observations.
Supports ResNet and ViT architectures from torchvision.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Literal, Tuple


class VisionBackbone(nn.Module):
    """
    Vision backbone that extracts features from images using pretrained models.

    Args:
        backbone_type: Type of backbone ('resnet18', 'resnet34', 'resnet50', 'vit_b_16', 'vit_b_32')
        pretrained: Whether to use pretrained weights
        num_cameras: Number of camera views to process
        freeze_backbone: Whether to freeze the backbone weights

    Returns:
        image_embeds: (batch_size, num_patches, hidden_size_of_vision_model)
    """

    def __init__(
        self,
        backbone_type: Literal['resnet18', 'resnet34', 'resnet50', 'vit_b_16', 'vit_b_32'] = 'resnet50',
        pretrained: bool = True,
        num_cameras: int = 4,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        self.backbone_type = backbone_type
        self.num_cameras = num_cameras
        self.freeze_backbone = freeze_backbone

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
        if num_cameras > 1:
            # Simple concatenation followed by a linear layer
            self.camera_fusion = nn.Linear(self.feature_dim * num_cameras, self.feature_dim)
        else:
            self.camera_fusion = None

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
        Extract spatial features from ResNet.

        Args:
            x: (batch_size, channels, height, width)

        Returns:
            features: (batch_size, num_patches, feature_dim)
        """
        features = self.backbone(x)  # (B, C, H, W)

        # Reshape to (B, num_patches, C)
        B, C, H, W = features.shape
        features = features.flatten(2).transpose(1, 2)  # (B, H*W, C)

        return features

    def extract_vit_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract patch features from ViT.

        Args:
            x: (batch_size, channels, height, width)

        Returns:
            features: (batch_size, num_patches+1, feature_dim)
                     Note: First token is [CLS] token
        """
        # Forward through ViT encoder
        x = self.backbone._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.backbone.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.backbone.encoder(x)

        return x  # (B, num_patches+1, feature_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through vision backbone.

        Args:
            images: (batch_size, num_cameras, channels, height, width)
                   or (batch_size, channels, height, width) if single camera

        Returns:
            image_embeds: (batch_size, num_patches, feature_dim)
        """
        # Handle single camera case
        if images.dim() == 4:
            images = images.unsqueeze(1)  # (B, 1, C, H, W)

        batch_size, num_cams, C, H, W = images.shape

        # Process each camera view
        all_features = []
        for cam_idx in range(num_cams):
            cam_image = images[:, cam_idx]  # (B, C, H, W)

            # Extract features based on backbone type
            if 'resnet' in self.backbone_type:
                features = self.extract_resnet_features(cam_image)
            else:  # ViT
                features = self.extract_vit_features(cam_image)

            all_features.append(features)

        # Aggregate features from multiple cameras
        if self.num_cameras > 1:
            # Option 1: Concatenate all camera features along feature dimension
            # Then project back to feature_dim
            # (B, num_patches, feature_dim) for each camera
            # Assuming all cameras have same num_patches, we concatenate along patch dimension
            image_embeds = torch.cat(all_features, dim=1)  # (B, num_patches*num_cameras, feature_dim)
        else:
            image_embeds = all_features[0]

        return image_embeds

    def get_output_dim(self) -> int:
        """Get the output feature dimension."""
        return self.feature_dim


# Example usage and testing
if __name__ == "__main__":
    # Test ResNet50
    print("Testing ResNet50 backbone...")
    resnet_backbone = VisionBackbone(
        backbone_type='resnet50',
        pretrained=False,
        num_cameras=4,
        freeze_backbone=False
    )

    # Test with 4 camera views
    dummy_images = torch.randn(2, 4, 3, 224, 224)  # (batch_size=2, 4 cameras, RGB, 224x224)
    output = resnet_backbone(dummy_images)
    print(f"ResNet50 output shape: {output.shape}")  # Expected: (2, num_patches*4, 2048)
    print(f"Feature dimension: {resnet_backbone.get_output_dim()}")

    # Test ViT
    print("\nTesting ViT-B/16 backbone...")
    vit_backbone = VisionBackbone(
        backbone_type='vit_b_16',
        pretrained=False,
        num_cameras=4,
        freeze_backbone=False
    )

    output = vit_backbone(dummy_images)
    print(f"ViT-B/16 output shape: {output.shape}")  # Expected: (2, num_patches*4, 768)
    print(f"Feature dimension: {vit_backbone.get_output_dim()}")

    print("\nVision backbone tests passed!")
