# Cross-Attention DiT for Flow Matching Action Generation
#
# DiT transformer with cross-attention to VLM embeddings.
# Action tokens attend to VLM hidden states via cross-attention,
# with AdaLayerNorm timestep conditioning.
#
# Reference: ABot-Manipulation flow_matching_head/cross_attention_dit.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.vision_transformer import Mlp, use_fused_attn
from typing import Optional
from torch.jit import Final


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_freq = t_freq.to(next(self.mlp.parameters()).dtype)
        return self.mlp(t_freq)


class CrossAttention(nn.Module):
    """
    Cross-attention layer: action tokens (query) attend to VLM embeddings (key/value).
    """
    fused_attn: Final[bool]

    def __init__(self, hidden_size: int, num_heads: int = 8, qkv_bias: bool = True):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)

        self.fused_attn = use_fused_attn()

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) action tokens (query)
            context: (B, L, D) VLM hidden states (key/value)
            key_padding_mask: (B, L) 1=real token, 0=pad. When provided, pad
                keys are masked with a large negative logit so they contribute
                zero softmax weight (and zero value mixing).

        Returns:
            output: (B, T, D)
        """
        B, T, D = x.shape
        _, L, _ = context.shape

        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(context).reshape(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(context).reshape(B, L, self.num_heads, self.head_dim)

        q = self.q_norm(q).transpose(1, 2)  # (B, H, T, D_h)
        k = self.k_norm(k).transpose(1, 2)  # (B, H, L, D_h)
        v = v.transpose(1, 2)               # (B, H, L, D_h)

        # Build additive attn_mask once so that pad keys get logit ≈ -inf.
        # Using finfo.min instead of -inf keeps numerics well-defined under
        # bf16/fp16 (real -inf can produce NaN when 0 * -inf appears later).
        attn_mask = None
        if key_padding_mask is not None:
            neg_inf = torch.finfo(q.dtype).min
            attn_mask = (1.0 - key_padding_mask.to(q.dtype)) * neg_inf
            attn_mask = attn_mask[:, None, None, :]   # (B, 1, 1, L) broadcasts

        if self.fused_attn:
            output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, scale=self.scale
            )
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if attn_mask is not None:
                scores = scores + attn_mask
            attn = F.softmax(scores, dim=-1)
            output = torch.matmul(attn, v)

        output = output.transpose(1, 2).reshape(B, T, D)
        return self.out_proj(output)


class DiTCrossAttnBlock(nn.Module):
    """
    DiT block with self-attention + cross-attention to VLM features.

    Architecture:
        1. AdaLN-modulated self-attention over action tokens
        2. Cross-attention to VLM context
        3. AdaLN-modulated feed-forward MLP
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()

        # Self-attention
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )

        # Cross-attention to VLM context
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = CrossAttention(hidden_size, num_heads)
        self.norm_context = nn.LayerNorm(hidden_size, eps=1e-6)

        # Feed-forward
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )

        # AdaLN modulation: timestep -> (shift, scale) * 3 pairs for each sub-layer
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        t_emb: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) action tokens
            context: (B, L, D) VLM hidden states
            t_emb: (B, D) timestep embedding
            key_padding_mask: (B, L) 1=valid VLM token, 0=pad

        Returns:
            x: (B, T, D) updated action tokens
        """
        # Compute AdaLN modulation parameters
        shift_sa, scale_sa, shift_ca, scale_ca, shift_ff, scale_ff = \
            self.adaLN_modulation(t_emb).chunk(6, dim=-1)

        # Self-attention with AdaLN
        x_norm = modulate(self.norm1(x), shift_sa.unsqueeze(1), scale_sa.unsqueeze(1))
        x = x + self.self_attn(x_norm, x_norm, x_norm, need_weights=False)[0]

        # Cross-attention with AdaLN
        x_norm = modulate(self.norm2(x), shift_ca.unsqueeze(1), scale_ca.unsqueeze(1))
        context_norm = self.norm_context(context)
        x = x + self.cross_attn(x_norm, context_norm, key_padding_mask=key_padding_mask)

        # Feed-forward with AdaLN
        x_norm = modulate(self.norm3(x), shift_ff.unsqueeze(1), scale_ff.unsqueeze(1))
        x = x + self.mlp(x_norm)

        return x


class CrossAttentionDiT(nn.Module):
    """
    DiT backbone with cross-attention to VLM features for flow matching.

    Takes noisy action tokens and VLM context, outputs predicted velocity field.
    Supports state conditioning by prepending state as an additional token.

    Architecture:
        - ActionEmbedder: projects noisy actions to hidden_size
        - StateEmbedder: projects robot state to hidden_size (1 token)
        - TimestepEmbedder: sinusoidal + MLP timestep embedding
        - N x DiTCrossAttnBlock: self-attn + cross-attn + MLP
        - FinalLayer: project back to action_dim

    Args:
        hidden_size: Transformer hidden dimension
        depth: Number of DiT blocks
        num_heads: Number of attention heads
        action_dim: Action space dimension
        future_action_window_size: Number of future actions to predict
        mlp_ratio: MLP expansion ratio
    """

    def __init__(
        self,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        action_dim: int = 14,
        future_action_window_size: int = 13,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.future_action_window_size = future_action_window_size

        # Action embedding
        self.action_embedder = nn.Linear(action_dim, hidden_size)

        # State embedding (clean state token)
        self.state_embedder = nn.Linear(action_dim, hidden_size)

        # Timestep embedding
        self.t_embedder = TimestepEmbedder(hidden_size)

        # Positional embedding for action sequence
        # Sequence: [state, action_1, ..., action_{T-1}]
        # Length = 1 + (future_action_window_size - 1) = future_action_window_size
        scale = hidden_size ** -0.5
        self.pos_embed = nn.Parameter(
            scale * torch.randn(future_action_window_size, hidden_size)
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTCrossAttnBlock(hidden_size, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        # Final output projection
        self.final_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.final_proj = nn.Linear(hidden_size, action_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Zero-init final projection for stable training start
        nn.init.constant_(self.final_proj.weight, 0)
        nn.init.constant_(self.final_proj.bias, 0)

        # Initialize timestep embedder
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass: predict velocity from noisy actions + VLM context.

        Args:
            noisy_actions: (B, T, action_dim) noisy action sequence
                T = future_action_window_size - 1
            t: (B,) discretized timestep indices
            context: (B, L, hidden_size) VLM hidden states (after projection)
            state: (B, action_dim) current robot state
            context_mask: (B, L) 1=valid VLM token, 0=pad — forwarded to
                cross-attention so pad keys contribute zero softmax weight.

        Returns:
            v_pred: (B, T, action_dim) predicted velocity field
        """
        # Embed actions
        x = self.action_embedder(noisy_actions)  # (B, T, D)

        # Embed state as prefix token
        if state is not None:
            s = self.state_embedder(state).unsqueeze(1)  # (B, 1, D)
            x = torch.cat([s, x], dim=1)  # (B, 1+T, D)

        # Add positional embedding
        x = x + self.pos_embed[:x.shape[1]]

        # Embed timestep
        t_emb = self.t_embedder(t)  # (B, D)

        # Pass through DiT blocks with cross-attention to VLM context
        for block in self.blocks:
            x = block(x, context, t_emb, key_padding_mask=context_mask)

        # Final projection
        x = self.final_norm(x)
        x = self.final_proj(x)  # (B, 1+T, action_dim)

        # Skip state token, return action predictions only
        if state is not None:
            x = x[:, 1:, :]  # (B, T, action_dim)

        return x


# Model size presets (matching ABot naming)
def DiT_B_CrossAttn(**kwargs):
    return CrossAttentionDiT(hidden_size=768, depth=12, num_heads=12, **kwargs)

def DiT_L_CrossAttn(**kwargs):
    return CrossAttentionDiT(hidden_size=1024, depth=24, num_heads=16, **kwargs)

def DiT_S_CrossAttn(**kwargs):
    return CrossAttentionDiT(hidden_size=384, depth=12, num_heads=6, **kwargs)

def DiT_XL_CrossAttn(**kwargs):
    return CrossAttentionDiT(hidden_size=1152, depth=28, num_heads=16, **kwargs)


DiT_CrossAttn_models = {
    'DiT-S': DiT_S_CrossAttn,
    'DiT-B': DiT_B_CrossAttn,
    'DiT-L': DiT_L_CrossAttn,
    'DiT-XL': DiT_XL_CrossAttn,
}
