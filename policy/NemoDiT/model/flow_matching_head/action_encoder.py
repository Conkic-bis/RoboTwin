# Action Encoder for Flow Matching
#
# Encodes noisy actions + sinusoidal timestep embedding through an MLP.
# Reference: ABot-Manipulation flow_matching_head/action_encoder.py

import torch
import torch.nn as nn
import math


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for continuous timesteps.
    """

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) continuous or discrete timestep values

        Returns:
            encoding: (B, dim) sinusoidal encoding
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class ActionEncoder(nn.Module):
    """
    Encodes noisy action tokens with timestep information.

    Takes noisy actions (B, T, action_dim) and a timestep embedding,
    projects them to the DiT hidden dimension.

    Args:
        action_dim: Dimension of action space
        hidden_size: DiT hidden dimension
        timestep_dim: Dimension of timestep embedding (default: 256)
    """

    def __init__(self, action_dim: int, hidden_size: int, timestep_dim: int = 256):
        super().__init__()

        self.timestep_encoder = SinusoidalPositionalEncoding(timestep_dim)
        self.timestep_mlp = nn.Sequential(
            nn.Linear(timestep_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Action projection: action_dim -> hidden_size
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, noisy_actions: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Encode noisy actions with timestep conditioning.

        Args:
            noisy_actions: (B, T, action_dim) noisy action sequence
            t: (B,) timestep values

        Returns:
            action_tokens: (B, T, hidden_size) encoded action tokens
        """
        # Encode timestep
        t_emb = self.timestep_encoder(t)  # (B, timestep_dim)
        t_emb = self.timestep_mlp(t_emb)  # (B, hidden_size)

        # Encode actions
        action_tokens = self.action_proj(noisy_actions)  # (B, T, hidden_size)

        # Add timestep embedding to each action token
        action_tokens = action_tokens + t_emb.unsqueeze(1)  # (B, T, hidden_size)

        return action_tokens
