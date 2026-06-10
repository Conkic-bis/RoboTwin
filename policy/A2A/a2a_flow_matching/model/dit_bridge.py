"""DiT bridge backbone: a diffusion-transformer used as a flow vector field.

Instead of denoising from N(0, I), the DiT here regresses the bridge velocity
u_t = z1 - z0 along a history-centered transport path (the source distribution
is part of the training objective — no train/test source mismatch).

Token layout:

    [latent tokens (z_t chunked), condition tokens (RobotMAF output, optional)]

with the flow time t and the pooled global condition injected through AdaLN
modulation (as in DiT / the existing AdaLNBlock). The condition tokens are
in-context tokens only; their output positions are discarded, so the future
action target never leaks back into the condition stream — this fixes the
naive Nemo-style integration where MAF refined the noisy action tokens.

Interface matches SimpleFlowNet so TorchFlowMatcher can drive either backbone:

    forward(x, t, global_cond=None, cond_tokens=None) -> velocity (B, latent_dim)
"""

import torch
import torch.nn as nn

from a2a_flow_matching.model.flow_net import AdaLNBlock
from a2a_flow_matching.model.positional_embedding import SinusoidalPosEmb


class DiTBridge(nn.Module):
    def __init__(
        self,
        latent_dim,
        cond_dim,
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        n_latent_tokens=8,
        cond_token_dim=None,
        mlp_ratio=4.0,
        dropout=0.0,
        time_embed_dim=256,
        max_cond_tokens=32,
    ):
        super().__init__()
        assert latent_dim % n_latent_tokens == 0, \
            "latent_dim must be divisible by n_latent_tokens"
        self.latent_dim = latent_dim
        self.n_latent_tokens = n_latent_tokens
        self.chunk_dim = latent_dim // n_latent_tokens

        self.input_proj = nn.Linear(self.chunk_dim, hidden_dim)
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.Mish(),
            nn.Linear(time_embed_dim * 4, hidden_dim),
        )
        self.cond_embed = nn.Linear(cond_dim, hidden_dim)
        self.cond_token_proj = (
            nn.Linear(cond_token_dim, hidden_dim) if cond_token_dim is not None else None
        )

        self.blocks = nn.ModuleList([
            AdaLNBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                max_seq_len=n_latent_tokens + max_cond_tokens,
            ) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, self.chunk_dim)

        self._init_weights()

    def _init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        nn.init.normal_(self.time_embed[1].weight, std=0.02)
        nn.init.normal_(self.time_embed[3].weight, std=0.02)
        # AdaLN modulation layers re-zero themselves in AdaLNBlock._init_weights,
        # but self.apply above overwrote them — restore the zero init.
        for block in self.blocks:
            block._init_weights()

    def forward(self, x, t, global_cond=None, cond_tokens=None):
        B = x.shape[0]
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=x.device)
        if t.ndim == 0:
            t = t.expand(B)

        tokens = self.input_proj(x.view(B, self.n_latent_tokens, self.chunk_dim))

        if cond_tokens is not None and self.cond_token_proj is not None:
            tokens = torch.cat([tokens, self.cond_token_proj(cond_tokens)], dim=1)

        t_emb = self.time_embed(t)
        if global_cond is not None:
            t_emb = t_emb + self.cond_embed(global_cond)

        # AdaLNBlock expects two modulation inputs (t, c) that it sums; the
        # condition is already folded into t_emb so pass zeros for c.
        zero_c = torch.zeros_like(t_emb)
        for block in self.blocks:
            tokens = block(tokens, t_emb, zero_c)

        tokens = self.norm(tokens[:, : self.n_latent_tokens])
        return self.out_proj(tokens).view(B, self.latent_dim)
