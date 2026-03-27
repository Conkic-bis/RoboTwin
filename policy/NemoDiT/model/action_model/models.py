# Modified from facebookresearch's DiT repos
# DiT: https://github.com/facebookresearch/DiT/blob/main/models.py

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from timm.models.vision_transformer import Attention, Mlp, use_fused_attn, RmsNorm
from typing import Optional
from torch.jit import Final

def modulate(x, shift, scale):
    return x * (1 + scale) + shift


#################################################################################
#               Embedding Layers for Timesteps and conditions                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
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
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(next(self.mlp.parameters()).dtype)
        t_emb = self.mlp(t_freq)
        return t_emb

class LabelEmbedder(nn.Module):
    """
    Embeds conditions into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, in_size, hidden_size, dropout_prob=0.1, conditions_shape=(1, 1, 4096)):
        super().__init__()
        self.linear = nn.Linear(in_size, hidden_size)
        self.dropout_prob = dropout_prob
        if dropout_prob > 0:
            self.uncondition = nn.Parameter(torch.empty(conditions_shape[1:]))

    def token_drop(self, conditions, force_drop_ids=None):
        """
        Drops conditions to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(conditions.shape[0], device=conditions.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        conditions = torch.where(drop_ids.unsqueeze(1).unsqueeze(1).expand(conditions.shape[0], *self.uncondition.shape), self.uncondition, conditions)
        return conditions


    def forward(self, conditions, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            conditions = self.token_drop(conditions, force_drop_ids)
        embeddings = self.linear(conditions)
        return embeddings

#################################################################################
#                      Embedding Layers for Actions and State                   #
#################################################################################
class ActionEmbedder(nn.Module):
    def __init__(self, action_size, hidden_size):
        super().__init__()
        self.linear = nn.Linear(action_size, hidden_size)

    def forward(self, x):
        x = self.linear(x)
        return x

# Action_History is not used now
class HistoryEmbedder(nn.Module):
    def __init__(self, action_size, hidden_size):
        super().__init__()
        self.linear = nn.Linear(action_size, hidden_size)

    def forward(self, x):
        x = self.linear(x)
        return x

class StateEmbedder(nn.Module):
    """
    Embeds the robot's current state (action at frame 0) into hidden representations.
    State is the last frame of n_obs_steps = frame 0 of action sequence.
    It is provided as clean (no noise) conditioning to the model.
    """
    def __init__(self, state_size, hidden_size):
        super().__init__()
        self.linear = nn.Linear(state_size, hidden_size)

    def forward(self, x):
        x = self.linear(x)
        return x

#################################################################################
#                          Cross Attention Layers                               #
#################################################################################
class CrossAttention(nn.Module):
    """
    A cross-attention layer with flash attention.
    
    Paper:
    https://arxiv.org/abs/1706.03762
    
    Reference:
    https://github.com/meta-llama/llama3/blob/main/llama/model.py
    """
    fused_attn: Final[bool]
    def __init__(self, hidden_size: int, num_heads: int = 8, n_kv_heads = None, norm_eps: float = 1e-6, qkv_bias: bool = False, qk_norm: bool = False, attn_drop: float = 0, proj_drop: float = 0,) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0, 'hidden_size should be divisible by num_heads'
        self.n_heads = num_heads
        self.n_kv_heads = self.n_heads if n_kv_heads is None else n_kv_heads
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError('num_heads should be divisible by num_kv_heads')
        self.n_rep = self.n_heads // self.n_kv_heads
        self.hidden_size = hidden_size
        if self.hidden_size % self.n_heads != 0:
            raise ValueError('hidden_size should be divisible by num_heads')
        self.head_size = self.hidden_size // self.n_heads

        self.wq = nn.Linear(
            self.hidden_size, 
            self.n_heads * self.head_size, 
            bias=False
        )
        self.wkv = nn.Linear(
            self.hidden_size, 
            self.n_kv_heads * self.head_size * 2, 
            bias=False
        )
        self.wo = nn.Linear(
            self.n_heads * self.head_size, 
            self.hidden_size, 
            bias=False
        )

        self.norm_eps = norm_eps
        self.norm_q = RmsNorm(self.head_size, eps=self.norm_eps)
        self.norm_k = RmsNorm(self.head_size, eps=self.norm_eps)

        self.fused_attn = use_fused_attn()

        self.attn_scale = 1.0 / math.sqrt(self.head_size)

    def forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        ck: Optional[torch.Tensor] = None,
        cv: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        bs, seq_len, _ = x.shape   # (bs, seq_len, hidden_size), batch size, sequence length, hidden size
        
        xq = self.wq(x)
        xq = xq.view(bs, seq_len, self.n_heads, self.head_size)
        xq = self.norm_q(xq)

        if c is not None:
            _, c_len, _ = c.shape     # (bs, c_len, hidden_size), batch size, condition length, hidden size

            ckv = self.wkv(c)
            ckv = ckv.view(bs, c_len, self.n_kv_heads, self.head_size, 2)
            ck, cv = ckv.unbind(-1)

            ck = self.norm_k(ck)

        # Repeat k/v heads if n_kv_heads < n_heads
        ck = repeat_kv(
            ck, self.n_rep
        )  # (bs, c_len, n_heads, head_size)
        cv = repeat_kv(
            cv, self.n_rep
        )  # (bs, c_len, n_heads, head_size)

        xq = xq.transpose(1, 2)  # (bs, n_heads, seq_len, head_size)
        ck = ck.transpose(1, 2)  # (bs, n_heads, c_len, head_size)
        cv = cv.transpose(1, 2)  # (bs, n_heads, c_len, head_size)

        # Prepare attn mask (bs, c_len) to mask the condition
        if mask is not None:
            mask = mask.reshape(bs, 1, 1, -1)
            mask = mask.expand(-1, -1, seq_len, -1)

        if self.fused_attn:
            output = F.scaled_dot_product_attention(
                query=xq,
                key=ck,
                value=cv,
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=False,
                scale=self.attn_scale,
            )
        else:
            scores = torch.matmul(xq, ck.transpose(2, 3)) * self.attn_scale
            if mask is not None:
                attn = attn.masked_fill_(mask.logical_not(), float('-inf'))
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = torch.matmul(scores, cv)   # (bs, n_heads, seq_len, head_size)

        output = output.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        return self.wo(output)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )
    
#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with self-attention and cross-attention conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = RmsNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = RmsNorm(hidden_size, eps=1e-6)
        self.cross_attn = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm3 = RmsNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x, z):
        x = x + self.attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), z)
        x = x + self.mlp(self.norm3(x))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = RmsNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, x):
        x = self.norm_final(x)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.

    支持 state 条件输入:
        state 是机器人当前状态 (n_obs_steps 最后一帧 = action 第0帧)，
        作为无噪音的条件 token 参与 transformer 计算。

        序列结构: [condition(t+z), state, noisy_action_1, ..., noisy_action_{T-1}]
        - condition: timestep + vision condition (1 token)
        - state: 机器人当前状态，无噪音 (1 token)
        - noisy actions: 需要去噪的未来动作序列 (T-1 tokens)
        总长度 = 1 + 1 + (T-1) = T+1，与原来的位置编码大小一致
    """
    def __init__(
        self,
        hidden_size,
        depth,
        num_heads,
        in_channels,
        token_size,
        future_action_window_size,
        past_action_window_size=0,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        learn_sigma=False,
    ):
        super().__init__()

        assert past_action_window_size == 0, "Error: action_history is not used now"

        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.class_dropout_prob = class_dropout_prob
        self.num_heads = num_heads
        self.past_action_window_size = past_action_window_size
        self.future_action_window_size = future_action_window_size

        # Action history is not used now.
        self.history_embedder = HistoryEmbedder(action_size=in_channels, hidden_size=hidden_size)

        self.x_embedder = ActionEmbedder(action_size=in_channels, hidden_size=hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.z_embedder = LabelEmbedder(in_size=token_size, hidden_size=hidden_size, dropout_prob=class_dropout_prob, conditions_shape=(1, 1, token_size))
        # State embedder: 将机器人当前状态嵌入为条件 token
        self.state_embedder = StateEmbedder(state_size=in_channels, hidden_size=hidden_size)
        scale = hidden_size ** -0.5

        # Learnable positional embeddings
        # 序列: [condition, state, action_1, ..., action_{T-1}]
        # 总长度 = 1 + 1 + (future_action_window_size - 1) = future_action_window_size + 1
        self.positional_embedding = nn.Parameter(
                scale * torch.randn(future_action_window_size + past_action_window_size + 1, hidden_size))

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # # Initialize token_embed like nn.Linear
        nn.init.normal_(self.x_embedder.linear.weight, std=0.02)
        nn.init.constant_(self.x_embedder.linear.bias, 0)

        nn.init.normal_(self.history_embedder.linear.weight, std=0.02)
        nn.init.constant_(self.history_embedder.linear.bias, 0)

        # Initialize state embedder
        nn.init.normal_(self.state_embedder.linear.weight, std=0.02)
        nn.init.constant_(self.state_embedder.linear.bias, 0)

        # Initialize label embedding table:
        if self.class_dropout_prob > 0:
            nn.init.normal_(self.z_embedder.uncondition, std=0.02)
        nn.init.normal_(self.z_embedder.linear.weight, std=0.02)
        nn.init.constant_(self.z_embedder.linear.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, z, state=None):
        """
        Forward pass of DiT.

        Args:
            x: (N, T, in_channels) - noisy action sequence to denoise
               T = future_action_window_size - 1 (state 不参与去噪)
            t: (N,) - diffusion timesteps
            z: (N, 1, vision_feature_dim) - 原生视觉条件特征 (无 adapter 投影)
               通过 ResNet GAP 或 ViT CLS token 得到的全局视觉特征，由 z_embedder 直接投影到 hidden_size
            state: (N, in_channels) - 机器人当前状态 (n_obs_steps 最后一帧的动作值)
                   作为无噪音的条件 token

        Returns:
            noise_pred: (N, T, in_channels) - predicted noise (仅预测 action[1:] 的噪音)

        序列结构: [condition(t+z), state, noisy_action_1, ..., noisy_action_{T-1}]
        """
        x = self.x_embedder(x)                              # (N, T, D)  T = future_action_window - 1
        t = self.t_embedder(t)                              # (N, D)
        z = self.z_embedder(z, self.training)               # (N, 1, D)
        t = t.unsqueeze(1)                                  # (N, 1, D)


        s = self.state_embedder(state)                      # (N, D)
        s = s.unsqueeze(1)                                  # (N, 1, D)
        x = torch.cat((t, s, x), dim=1)                     # (N, 1+1+T, D) = (N, T+2, D)

        x = x + self.positional_embedding                   # (N, T+1+1, D)
        for block in self.blocks:
            x = block(x, z)
        x = self.final_layer(x)

        return x[:, 2:, :]  # (N, T, C) 跳过 timestep 和 state tokens

