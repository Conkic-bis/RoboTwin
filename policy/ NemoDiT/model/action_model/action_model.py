"""
action_model.py

"""
from model.action_model.models import DiT
from model.action_model import create_diffusion
from . import gaussian_diffusion as gd
from model.vision_input import VisionBackbone
from model.feature_adaptation import create_feature_adapter
import torch
from torch import nn


# Create model sizes of ActionModels
def DiT_S(**kwargs):
    return DiT(depth=6, hidden_size=384, num_heads=4, **kwargs)


def DiT_B(**kwargs):
    return DiT(depth=12, hidden_size=768, num_heads=12, **kwargs)


def DiT_L(**kwargs):
    return DiT(depth=24, hidden_size=1024, num_heads=16, **kwargs)


# Model size
DiT_models = {'DiT-S': DiT_S, 'DiT-B': DiT_B, 'DiT-L': DiT_L}


# Create ActionModel with Vision Conditioning
class ActionModel(nn.Module):
    def __init__(self,
                 token_size,
                 model_type,
                 in_channels,
                 future_action_window_size,
                 past_action_window_size,
                 diffusion_steps=100,
                 noise_schedule='squaredcos_cap_v2',
                 use_vision_condition=True,
                 vision_backbone_type='resnet50',
                 vision_pretrained=True,
                 num_cameras=4,
                 freeze_vision_backbone=False,
                 adapter_type='mlp',
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.noise_schedule = noise_schedule
        self.use_vision_condition = use_vision_condition

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

        # Vision backbone and feature adapter
        if use_vision_condition:
            self.vision_backbone = VisionBackbone(
                backbone_type=vision_backbone_type,
                pretrained=vision_pretrained,
                num_cameras=num_cameras,
                freeze_backbone=freeze_vision_backbone
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
            class_dropout_prob=0.1,
            learn_sigma=learn_sigma,
            future_action_window_size=future_action_window_size,
            past_action_window_size=past_action_window_size
        )

    def encode_vision_condition(self, images):
        """
        Encode images to vision condition features.

        Args:
            images: (batch_size, num_cameras, channels, height, width)

        Returns:
            vision_condition: (batch_size, num_patches, token_size)
        """
        if not self.use_vision_condition:
            raise ValueError("Vision condition is not enabled")

        # Extract vision features
        vision_features = self.vision_backbone(images)  # (B, num_patches, vision_dim)

        # Adapt features to token_size
        vision_condition = self.feature_adapter(vision_features)  # (B, num_patches, token_size)

        return vision_condition

    # Given condition z and ground truth token x, compute loss
    def loss(self, x, z=None, images=None):
        """
        Compute diffusion loss.

        Args:
            x: (batch_size, future_action_window_size, in_channels) - ground truth actions
            z: (batch_size, num_patches, token_size) - precomputed vision condition (optional)
            images: (batch_size, num_cameras, 3, H, W) - raw images (optional)

        Returns:
            loss: scalar loss value
        """
        # Encode vision condition if images are provided
        if images is not None and self.use_vision_condition:
            z = self.encode_vision_condition(images)

        if z is None:
            raise ValueError("Either z or images must be provided")

        # sample random noise and timestep
        noise = torch.randn_like(x)  # [B, T, C]
        timestep = torch.randint(0, self.diffusion.num_timesteps, (x.size(0),), device=x.device)

        # sample x_t from x
        x_t = self.diffusion.q_sample(x, timestep, noise)

        # predict noise from x_t
        noise_pred = self.net(x_t, timestep, z)

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
