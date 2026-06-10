"""RobotMAF: reliability-aware multimodal condition encoder.

Adapts the AudioX-style Multimodal Adaptive Fusion (MAF) idea to robot
visuomotor control. Three heterogeneous signals are tokenized into a shared
space and fused with noise-aware gates:

    visual tokens   scene semantics, object pose, appearance
    history tokens  dynamics trend, temporal continuity
    state tokens    current kinematic anchor

Unlike the naive "MAF inside the DiT token flow" integration, this module only
produces the *condition* — the future action target z1 is never touched, so it
remains a pure flow target rather than leaking into the condition.

Outputs both a pooled global condition vector ``c`` (for the MLP flow net /
AdaLN modulation) and the fused token sequence ``c_tokens`` (for token-level
conditioning of a DiT bridge backbone), plus the per-modality gate weights for
interpretability metrics (gate value / entropy under perturbations).

The gate is made aware of source uncertainty by feeding it the history noise
intensity sigma_a and latent source noise sigma_z:

    gate = G(visual_global, history_global, state_global, sigma_a, sigma_z)

Flow time t is intentionally NOT a gate input here: in this codebase the
condition is encoded once per action chunk and reused across all ODE steps,
so a t-dependent gate would require re-encoding the condition at every step.
"""

import torch
import torch.nn as nn

from a2a_flow_matching.model.flow_net import Attention
from a2a_flow_matching.model.layers import Mlp


class MAFBlock(nn.Module):
    """Pre-LN transformer block over the concatenated multimodal tokens."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0, max_seq_len=64):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            attn_drop=dropout,
            proj_drop=dropout,
            max_seq_len=max_seq_len,
        )
        self.norm2 = nn.LayerNorm(dim)

        def approx_gelu(): return nn.GELU(approximate="tanh")

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=approx_gelu,
            drop=dropout,
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class RobotMAFConditionEncoder(nn.Module):
    """Fuse visual / history / state tokens into an action-generative condition.

    Args:
        obs_feature_dim: per-timestep visual feature dim from the obs encoder.
        state_dim: proprioceptive state dim (== action_dim on RoboTwin).
        n_obs_steps: number of history timesteps (visual + proprio).
        token_dim: shared token width inside the MAF block.
        latent_dim: output dim of the pooled global condition ``c``.
        num_layers/num_heads/mlp_ratio/dropout: MAF transformer hyperparams.
    """

    NUM_MODALITIES = 3  # visual, history, state

    def __init__(
        self,
        obs_feature_dim: int,
        state_dim: int,
        n_obs_steps: int,
        token_dim: int = 256,
        latent_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_obs_steps = n_obs_steps
        self.token_dim = token_dim
        self.latent_dim = latent_dim

        # Per-modality tokenizers: project each timestep into the shared space.
        self.obs_proj = nn.Sequential(nn.Linear(obs_feature_dim, token_dim), nn.LayerNorm(token_dim))
        self.hist_proj = nn.Sequential(nn.Linear(state_dim, token_dim), nn.LayerNorm(token_dim))
        self.state_proj = nn.Sequential(nn.Linear(state_dim, token_dim), nn.LayerNorm(token_dim))

        self.modality_embed = nn.Parameter(torch.zeros(self.NUM_MODALITIES, token_dim))
        nn.init.normal_(self.modality_embed, std=0.02)

        # tokens = n_obs_steps visual + n_obs_steps history + 1 state
        total_tokens = 2 * n_obs_steps + 1
        self.blocks = nn.ModuleList([
            MAFBlock(
                dim=token_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                max_seq_len=total_tokens,
            ) for _ in range(num_layers)
        ])

        # Noise-aware modality gate: pooled modality globals + (sigma_a, sigma_z).
        self.gate = nn.Sequential(
            nn.Linear(self.NUM_MODALITIES * token_dim + 2, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, self.NUM_MODALITIES),
        )

        self.out_proj = nn.Sequential(
            nn.Linear(token_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def output_token_dim(self):
        return self.token_dim

    def forward(self, obs_features, history_states, current_state, sigma_a=0.0, sigma_z=0.0):
        """
        Args:
            obs_features: (B, n_obs_steps, obs_feature_dim) per-step visual features
            history_states: (B, n_obs_steps, state_dim) proprio history
            current_state: (B, state_dim) current kinematic anchor
            sigma_a: history (raw-space) noise std actually applied this pass
            sigma_z: latent source noise std actually applied this pass

        Returns:
            c: (B, latent_dim) pooled global condition
            c_tokens: (B, 2*n_obs_steps+1, token_dim) fused condition tokens
            gates: (B, 3) modality gate weights (visual, history, state)
        """
        B = obs_features.shape[0]

        obs_tok = self.obs_proj(obs_features) + self.modality_embed[0]
        hist_tok = self.hist_proj(history_states) + self.modality_embed[1]
        state_tok = self.state_proj(current_state).unsqueeze(1) + self.modality_embed[2]

        tokens = torch.cat([obs_tok, hist_tok, state_tok], dim=1)
        for block in self.blocks:
            tokens = block(tokens)

        n = self.n_obs_steps
        obs_global = tokens[:, :n].mean(dim=1)
        hist_global = tokens[:, n:2 * n].mean(dim=1)
        state_global = tokens[:, 2 * n]

        sigmas = tokens.new_tensor([float(sigma_a), float(sigma_z)]).expand(B, 2)
        gate_logits = self.gate(torch.cat([obs_global, hist_global, state_global, sigmas], dim=-1))
        gates = torch.softmax(gate_logits, dim=-1)

        fused = (
            gates[:, 0:1] * obs_global
            + gates[:, 1:2] * hist_global
            + gates[:, 2:3] * state_global
        )
        c = self.out_proj(fused)
        return c, tokens, gates
