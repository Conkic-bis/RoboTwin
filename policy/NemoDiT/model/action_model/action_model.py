# Flow Matching Action Model with Qwen3-VL Conditioning
#
# Integrates Qwen3-VL VLM with a cross-attention DiT for flow matching
# action generation. The VLM provides rich vision-language features that
# condition the action model via cross-attention.
#
# Architecture:
#   Qwen3-VL (images + instruction) -> hidden_states (B, L, H_vlm)
#       -> VLM projection (H_vlm -> H_dit) -> context (B, L, H_dit)
#   FlowMatching + CrossAttentionDiT:
#       noisy_actions (B, T, action_dim) + context -> v_pred (B, T, action_dim)
#
# Reference: ABot-Manipulation ABot_M0 framework

import torch
import torch.nn as nn
from typing import Optional, List, Any, Dict

from model.vlm.qwen3_vl import Qwen3VLInterface
from model.flow_matching_head.cross_attention_dit import (
    CrossAttentionDiT, DiT_CrossAttn_models
)
from model.action_model.flow_matching import FlowMatching


class Qwen3VLActionModel(nn.Module):
    """
    Qwen3-VL + Flow Matching Action Model.

    Data flow:
        1. Images + instruction -> Qwen3-VL -> hidden_states (B, L, H_vlm)
        2. hidden_states -> vlm_proj -> context (B, L, H_dit)
        3. Flow matching training/sampling:
           - Training: noise actions, predict velocity v = x_1 - noise
           - Inference: ODE integration from noise to clean actions

    Args:
        vlm_model_name: HuggingFace model name for Qwen3-VL
        freeze_vlm: Whether to freeze VLM weights
        use_lora: Apply LoRA to VLM
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        dit_model_type: DiT size ('DiT-S', 'DiT-B', 'DiT-L', 'DiT-XL')
        action_dim: Action space dimension
        future_action_window_size: Number of future actions to predict
        n_action_steps: Number of action steps to execute during inference
        time_sampling: Flow matching time sampling strategy
        logit_normal_loc: LogisticNormal location parameter
        logit_normal_scale: LogisticNormal scale parameter
        beta_alpha: Beta distribution alpha
        beta_beta: Beta distribution beta
        num_timestep_buckets: Timestep discretization buckets
    """

    def __init__(
        self,
        # VLM parameters
        vlm_model_name: str = "Qwen/Qwen3-VL-4B-Instruct",
        freeze_vlm: bool = True,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        # DiT parameters
        dit_model_type: str = 'DiT-B',
        action_dim: int = 14,
        future_action_window_size: int = 13,
        n_action_steps: int = 8,
        # Flow matching parameters
        time_sampling: str = 'logit_normal',
        logit_normal_loc: float = 0.0,
        logit_normal_scale: float = 1.0,
        beta_alpha: float = 1.5,
        beta_beta: float = 1.0,
        num_timestep_buckets: int = 1000,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.future_action_window_size = future_action_window_size
        self.n_action_steps = n_action_steps

        # 1. Vision-Language Model
        self.vlm = Qwen3VLInterface(
            model_name=vlm_model_name,
            freeze=freeze_vlm,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
        )

        vlm_hidden_size = self.vlm.get_hidden_size()

        # 2. VLM-to-DiT projection
        dit_builder = DiT_CrossAttn_models[dit_model_type]
        # Create a temporary instance to get hidden_size
        _tmp = dit_builder(action_dim=action_dim, future_action_window_size=future_action_window_size)
        dit_hidden_size = _tmp.hidden_size
        del _tmp

        self.vlm_proj = nn.Sequential(
            nn.Linear(vlm_hidden_size, dit_hidden_size),
            nn.LayerNorm(dit_hidden_size),
            nn.GELU(),
            nn.Linear(dit_hidden_size, dit_hidden_size),
            nn.LayerNorm(dit_hidden_size),
        )

        # 3. Cross-Attention DiT for action generation
        self.dit = dit_builder(
            action_dim=action_dim,
            future_action_window_size=future_action_window_size,
        )

        # 4. Flow Matching
        self.flow_matching = FlowMatching(
            time_sampling=time_sampling,
            logit_normal_loc=logit_normal_loc,
            logit_normal_scale=logit_normal_scale,
            beta_alpha=beta_alpha,
            beta_beta=beta_beta,
            num_timestep_buckets=num_timestep_buckets,
        )

    def encode_vlm(
        self,
        images: Optional[List[Any]] = None,
        instruction: str = "Predict the next robot actions.",
        robot_state: Optional[torch.Tensor] = None,
        preprocessed_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Encode images + instruction through Qwen3-VL and project to DiT space.

        Returns:
            context: (B, L, dit_hidden_size)
        """
        # Get VLM hidden states
        hidden_states = self.vlm(
            images=images,
            instruction=instruction,
            robot_state=robot_state,
            preprocessed_inputs=preprocessed_inputs,
        )  # (B, L, vlm_hidden_size)

        # The VLM runs in bfloat16 but vlm_proj / DiT are kept in float32,
        # so cast the context back to the projection's dtype before the
        # Linear layer to avoid a mat1/mat2 dtype mismatch.
        proj_dtype = next(self.vlm_proj.parameters()).dtype
        if hidden_states.dtype != proj_dtype:
            hidden_states = hidden_states.to(proj_dtype)

        # Project to DiT dimension
        context = self.vlm_proj(hidden_states)  # (B, L, dit_hidden_size)

        return context

    def loss(
        self,
        actions: torch.Tensor,
        context: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute flow matching loss.

        Args:
            actions: (B, T, action_dim) ground truth future actions
                T = future_action_window_size - 1
            context: (B, L, dit_hidden_size) VLM context (pre-computed)
            state: (B, action_dim) current robot state

        Returns:
            loss: scalar MSE loss on velocity prediction
        """
        # Sample noise and timesteps
        noise = torch.randn_like(actions)
        t = self.flow_matching.sample_time(actions.size(0), actions.device, dtype=actions.dtype)

        # Create noisy actions: x_t = (1-t)*noise + t*x_1
        x_t = self.flow_matching.q_sample(actions, t, noise)

        # Discretize timestep for embedding
        t_discrete = self.flow_matching.discretize_timestep(t)

        # Predict velocity
        v_pred = self.dit(x_t, t_discrete, context, state=state)

        # Target velocity: v = x_1 - noise
        v_target = self.flow_matching.compute_velocity(actions, noise)

        assert v_pred.shape == v_target.shape == actions.shape
        loss = ((v_pred - v_target) ** 2).mean()

        return loss

    @torch.no_grad()
    def sample(
        self,
        context: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        num_steps: int = 10,
        ode_solver: str = 'midpoint',
        return_all: bool = False,
    ) -> torch.Tensor:
        """
        Generate action sequence via ODE integration.

        Args:
            context: (B, L, dit_hidden_size) VLM context
            state: (B, action_dim) current robot state
            num_steps: ODE integration steps
            ode_solver: 'euler' or 'midpoint'
            return_all: Return full prediction or truncate to n_action_steps

        Returns:
            actions: (B, n_action_steps, action_dim)
        """
        device = context.device
        batch_size = context.shape[0]
        predict_length = self.future_action_window_size - 1
        shape = (batch_size, predict_length, self.action_dim)

        def model_fn(x, t, **kwargs):
            return self.dit(x, t, kwargs['context'], state=kwargs.get('state'))

        if ode_solver == 'midpoint':
            sample_fn = self.flow_matching.midpoint_sample
        else:
            sample_fn = self.flow_matching.euler_sample

        actions = sample_fn(
            model_fn,
            shape,
            num_steps=num_steps,
            device=device,
            model_kwargs={'context': context, 'state': state},
        )

        if return_all:
            return actions
        else:
            return actions[:, :self.n_action_steps, :]
