# Flow Matching Action Model with Qwen3-VL Conditioning
#
# Replaces the ResNet-based VisionBackbone + LabelEmbedder pathway with a
# Qwen3-VL vision-language backbone feeding a cross-attention DiT head.
# Flow matching (logit_normal / beta / uniform time sampling, midpoint ODE)
# from feature_update_flowmatching2.0 is preserved verbatim.
#
# Architecture (one training step):
#   images + instruction  ──► Qwen3-VL  ──► (B, L_max, H_vlm)
#                                                │  + attention_mask (B, L_max)
#                                                ▼
#                                       vlm_proj (H_vlm → H_dit)
#                                                │
#                                                ▼
#   noisy_actions (B, T-1, action_dim) ──► CrossAttentionDiT  ──► v_pred
#                                                │
#                                                └── cross-attn masks pad keys

from typing import List, Optional, Any, Dict, Tuple

import torch
from torch import nn

from model.vlm.qwen3_vl import Qwen3VLInterface
from model.flow_matching_head.cross_attention_dit import (
    CrossAttentionDiT, DiT_CrossAttn_models,
)
from model.action_model.flow_matching import FlowMatching


class ActionModel(nn.Module):
    """
    Qwen3-VL + Flow Matching action model.

    Kept the class name ``ActionModel`` so downstream deploy/eval code from
    feature_update_flowmatching2.0 stays import-compatible. The constructor
    signature is a superset of the 2.0 version: new VLM kwargs were added,
    deprecated ResNet kwargs are accepted and ignored (logged once) to keep
    old training shell scripts working during the transition.
    """

    def __init__(
        self,
        # Core action spec
        model_type: str = 'DiT-B',
        in_channels: int = 14,
        future_action_window_size: int = 13,
        past_action_window_size: int = 0,
        n_obs_steps: int = 1,
        n_action_steps: Optional[int] = None,
        # VLM
        vlm_model_name: str = 'Qwen/Qwen3-VL-4B-Instruct',
        freeze_vlm: bool = True,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        # Flow matching (identical to 2.0 branch)
        time_sampling: str = 'logit_normal',
        logit_normal_loc: float = 0.0,
        logit_normal_scale: float = 1.0,
        beta_alpha: float = 1.5,
        beta_beta: float = 1.0,
        num_timestep_buckets: int = 1000,
        # Legacy params kept for checkpoint / shell-script compatibility
        diffusion_steps: Optional[int] = None,
        noise_schedule: Optional[str] = None,
        token_size: Optional[int] = None,         # old ResNet path
        use_vision_condition: bool = True,        # ignored, VLM is mandatory
        vision_backbone_type: Optional[str] = None,
        vision_pretrained: bool = True,
        num_cameras: int = 4,
        adapter_type: Optional[str] = None,
        freeze_vision_backbone: bool = False,
        class_dropout_prob: float = 0.1,
        temporal_agg: str = 'last',
    ):
        super().__init__()

        self.in_channels = in_channels
        self.action_dim = in_channels
        self.future_action_window_size = future_action_window_size
        self.past_action_window_size = past_action_window_size
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps if n_action_steps is not None else future_action_window_size
        self.num_cameras = num_cameras
        self.class_dropout_prob = class_dropout_prob
        self.temporal_agg = temporal_agg

        # 1. Qwen3-VL backbone
        self.vlm = Qwen3VLInterface(
            model_name=vlm_model_name,
            freeze=freeze_vlm,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
        )
        vlm_hidden_size = self.vlm.get_hidden_size()

        # 2. Cross-attention DiT head (sized by model_type preset)
        dit_builder = DiT_CrossAttn_models[model_type]
        self.dit = dit_builder(
            action_dim=in_channels,
            future_action_window_size=future_action_window_size,
        )
        dit_hidden_size = self.dit.hidden_size

        # 3. VLM → DiT projection
        self.vlm_proj = nn.Sequential(
            nn.Linear(vlm_hidden_size, dit_hidden_size),
            nn.LayerNorm(dit_hidden_size),
            nn.GELU(),
            nn.Linear(dit_hidden_size, dit_hidden_size),
            nn.LayerNorm(dit_hidden_size),
        )

        # 4. Flow matching utility (time sampling + ODE solvers)
        self.flow_matching = FlowMatching(
            time_sampling=time_sampling,
            logit_normal_loc=logit_normal_loc,
            logit_normal_scale=logit_normal_scale,
            beta_alpha=beta_alpha,
            beta_beta=beta_beta,
            num_timestep_buckets=num_timestep_buckets,
        )

        # Expose a ``net`` alias so the 2.0 branch's EMA path
        # (``EMAModel(model.net)``) still works without modification.
        self.net = self.dit

    # ------------------------------------------------------------------ #
    # VLM encoding (batched)
    # ------------------------------------------------------------------ #

    def encode_vlm_batch(
        self,
        images_batch: List[List[Any]],
        instructions: List[str],
        robot_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batched VLM → projection. Returns (context, context_mask).

        Args:
            images_batch: length-B list; each entry is a per-camera image list.
            instructions: length-B list of strings.
            robot_states: optional (B, action_dim) tensor.

        Returns:
            context      : (B, L_max, H_dit)
            context_mask : (B, L_max) 1=valid, 0=pad
        """
        hidden_states, attention_mask = self.vlm.forward_batch(
            images_batch, instructions, robot_states,
        )
        proj_dtype = next(self.vlm_proj.parameters()).dtype
        if hidden_states.dtype != proj_dtype:
            hidden_states = hidden_states.to(proj_dtype)
        context = self.vlm_proj(hidden_states)
        return context, attention_mask

    # ------------------------------------------------------------------ #
    # Training loss
    # ------------------------------------------------------------------ #

    def loss(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        # legacy kwargs
        images: Optional[List[List[Any]]] = None,
        instructions: Optional[List[str]] = None,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Flow matching loss. Target velocity: v = x_1 - noise.

        Either supply precomputed ``context`` (+ optional ``context_mask``),
        or supply ``images`` + ``instructions`` and we will encode via VLM.

        Args:
            x: (B, T-1, action_dim) ground truth future actions.
            context: (B, L, H_dit) precomputed VLM context (optional).
            context_mask: (B, L) valid-token mask for ``context`` (optional).
            state: (B, action_dim) current robot state.
            images, instructions: raw inputs for on-the-fly VLM encoding.
            z: legacy alias of ``context`` from the 2.0 branch.
        """
        if context is None:
            if z is not None:                       # legacy alias
                context = z
            elif images is not None and instructions is not None:
                context, context_mask = self.encode_vlm_batch(
                    images_batch=images,
                    instructions=instructions,
                    robot_states=state,
                )
            else:
                raise ValueError(
                    "ActionModel.loss requires either `context` or "
                    "(`images` + `instructions`)."
                )

        # Sample noise and continuous timestep
        noise = torch.randn_like(x)
        t = self.flow_matching.sample_time(x.size(0), x.device, dtype=x.dtype)

        # Forward interpolation: x_t = (1-t)*noise + t*x_1
        x_t = self.flow_matching.q_sample(x, t, noise)
        t_discrete = self.flow_matching.discretize_timestep(t)

        v_pred = self.dit(
            x_t, t_discrete, context,
            state=state, context_mask=context_mask,
        )
        v_target = self.flow_matching.compute_velocity(x, noise)

        assert v_pred.shape == v_target.shape == x.shape, (
            f"velocity shape mismatch: pred={v_pred.shape}, "
            f"target={v_target.shape}, actions={x.shape}"
        )

        return ((v_pred - v_target) ** 2).mean()

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def sample(
        self,
        images: Optional[List[List[Any]]] = None,
        instructions: Optional[List[str]] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        num_steps: int = 10,
        cfg_scale: float = 0.0,
        return_all: bool = False,
        ode_solver: str = 'midpoint',
        ddim_steps: Optional[int] = None,          # legacy alias
        use_ddim: Optional[bool] = None,           # legacy, ignored
    ) -> torch.Tensor:
        """
        ODE integration from t=0 (noise) to t=1 (data).

        CFG note: the VLM path does not currently support classifier-free
        guidance (there is no ``uncondition`` embedding as in the 2.0
        ResNet+LabelEmbedder path). ``cfg_scale`` is accepted for API
        compatibility but ignored when > 1. Keep at 0 / 1.

        Args:
            images, instructions: raw inputs for VLM (optional if
                ``context`` is already supplied).
            context, context_mask: precomputed batch context & mask.
            state: (B, action_dim) clean robot state token.
            num_steps: ODE integration step count.
            ode_solver: 'euler' | 'midpoint' (default: midpoint).
            ddim_steps: legacy alias for num_steps.
        """
        if ddim_steps is not None:
            num_steps = ddim_steps

        if context is None:
            if images is None or instructions is None:
                raise ValueError(
                    "ActionModel.sample requires either `context` or "
                    "(`images` + `instructions`)."
                )
            context, context_mask = self.encode_vlm_batch(
                images_batch=images,
                instructions=instructions,
                robot_states=state,
            )

        device = context.device
        batch_size = context.shape[0]
        predict_length = self.future_action_window_size - 1
        shape = (batch_size, predict_length, self.action_dim)

        def model_fn(x, t, **kwargs):
            return self.dit(
                x, t, kwargs['context'],
                state=kwargs.get('state'),
                context_mask=kwargs.get('context_mask'),
            )

        if ode_solver == 'midpoint':
            sample_fn = self.flow_matching.midpoint_sample
        else:
            sample_fn = self.flow_matching.euler_sample

        actions = sample_fn(
            model_fn,
            shape,
            num_steps=num_steps,
            device=device,
            model_kwargs={
                'context': context,
                'state': state,
                'context_mask': context_mask,
            },
        )

        if return_all:
            return actions
        return actions[:, :self.n_action_steps, :]


# Legacy factory helpers (kept so `from action_model import DiT_S` etc.
# inside 2.0-style shell scripts keep resolving).
def DiT_S(**kwargs):
    return DiT_CrossAttn_models['DiT-S'](**kwargs)


def DiT_B(**kwargs):
    return DiT_CrossAttn_models['DiT-B'](**kwargs)


def DiT_L(**kwargs):
    return DiT_CrossAttn_models['DiT-L'](**kwargs)


def DiT_XL(**kwargs):
    return DiT_CrossAttn_models['DiT-XL'](**kwargs)


DiT_models = {'DiT-S': DiT_S, 'DiT-B': DiT_B, 'DiT-L': DiT_L, 'DiT-XL': DiT_XL}
