"""Unified Bridge Policy: latent-aligned multimodal action bridge.

Unifies noise-to-action diffusion/flow and action-to-action (A2A) flow
matching as one family of *source distributions* in a shared action latent
space, with an uncertainty-aware multimodal condition encoder and explicit
latent alignment objectives.

    z0 = E_hist(h + sigma_a * eps_a) + sigma_z * eps_z     (history-centered)
    z1 = E_act(a+)                                          (flow target)
    c  = RobotMAF(obs, history, state)  or  concat baseline (condition)

    flow:  z_t = (1-t) z0 + t z1,  v_theta(z_t, t, c) -> z1 - z0

Source modes (the unified-source study in the proposal):

    gaussian          z0 ~ N(0, I)                          noise-to-action
    clean_history     z0 = E_hist(h)                        original A2A
    noised_history    z0 = E_hist(h + sigma_a eps)           A2A-Noise
    mixed_bridge      z0 = E_hist(h + sigma_a eps) + sigma_z eps_z
    residual_history  z0 ~ N(0, I), target r1 = z1 - E_hist(h),
                      z1_hat = E_hist(h) + r1_hat            history-shifted
                      residual generation (flow-form of the Nemo-style
                      residual diffusion baseline)

Backbones:

    mlp   SimpleFlowNet (lightweight, continuous with A2A)
    dit   DiTBridge (high-capacity transformer flow/bridge vector field)

Latent alignment losses (beyond A2A's recon + consistency):

    L_jepa   JEPA predictor P(c, z0) -> stopgrad(z1)
    L_align  InfoNCE or VICReg between condition c and target z1

The source distribution is part of the *training* objective — never train on
a Gaussian source and only swap the inference initial point (train/test
source mismatch, proposal section 8.1).
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from a2a_flow_matching.common.normalizer import LinearNormalizer
from a2a_flow_matching.common.pytorch_util import dict_apply
from a2a_flow_matching.policy.base_image_policy import BaseImagePolicy

from a2a_flow_matching.model.flow_net import SimpleFlowNet
from a2a_flow_matching.model.dit_bridge import DiTBridge
from a2a_flow_matching.model.robot_maf import RobotMAFConditionEncoder
from a2a_flow_matching.model.latent_alignment import (
    JEPAPredictor,
    jepa_loss,
    info_nce_loss,
    vicreg_loss,
    latent_collapse_metrics,
)
from a2a_flow_matching.model.action_ae import CNNActionEncoder, SimpleActionDecoder
from a2a_flow_matching.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from a2a_flow_matching.model.flow_matchers import TorchFlowMatcher

SOURCE_MODES = ("gaussian", "clean_history", "noised_history", "mixed_bridge", "residual_history")
CONDITION_MODES = ("concat", "maf")
BACKBONES = ("mlp", "dit")


class UnifiedBridgePolicy(BaseImagePolicy):
    def __init__(
        self,
        shape_meta: dict,
        obs_encoder: MultiImageObsEncoder,
        horizon,
        n_action_steps,
        n_obs_steps,
        flow_matcher: TorchFlowMatcher,
        # backbone selection
        backbone="mlp",
        flow_net=None,            # mlp backbone cfg (hidden_dim/num_layers/...)
        dit=None,                 # dit backbone cfg (hidden_dim/num_layers/...)
        # unified source distribution
        source_mode="mixed_bridge",
        history_noise_std=0.02,   # sigma_a, raw action/state space
        latent_noise_std=0.0,     # sigma_z, latent space
        # condition encoder
        condition_mode="maf",
        maf=None,                 # maf cfg (token_dim/num_layers/...)
        # latent alignment
        jepa_weight=0.0,
        align_weight=0.0,
        align_type="infonce",     # infonce | vicreg
        # A2A-inherited losses
        decode_flow_latents=True,
        consistency_weight=1.0,
        latent_dim=512,
        action_ae=None,
        **kwargs,
    ):
        super().__init__()

        assert source_mode in SOURCE_MODES, f"source_mode must be one of {SOURCE_MODES}"
        assert condition_mode in CONDITION_MODES, f"condition_mode must be one of {CONDITION_MODES}"
        assert backbone in BACKBONES, f"backbone must be one of {BACKBONES}"
        assert align_type in ("infonce", "vicreg")

        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_feature_dim = obs_encoder.output_shape()[0]

        self.source_mode = source_mode
        self.history_noise_std = history_noise_std
        self.latent_noise_std = latent_noise_std
        self.condition_mode = condition_mode
        self.backbone = backbone
        self.jepa_weight = jepa_weight
        self.align_weight = align_weight
        self.align_type = align_type
        self.decode_flow_latents = decode_flow_latents
        self.consistency_weight = consistency_weight
        self.latent_dim = latent_dim
        self.num_sampling_steps = flow_matcher.num_sampling_steps
        self.flow_matcher = flow_matcher
        self.action_ae = action_ae
        self.last_metrics = {}

        # ---- condition encoder ----
        self.obs_encoder = obs_encoder
        if condition_mode == "maf":
            maf = maf or {}
            self.maf_encoder = RobotMAFConditionEncoder(
                obs_feature_dim=obs_feature_dim,
                state_dim=action_dim,
                n_obs_steps=n_obs_steps,
                token_dim=maf.get("token_dim", 256),
                latent_dim=latent_dim,
                num_layers=maf.get("num_layers", 2),
                num_heads=maf.get("num_heads", 8),
                mlp_ratio=maf.get("mlp_ratio", 4.0),
                dropout=maf.get("dropout", 0.0),
            )
            cond_token_dim = self.maf_encoder.output_token_dim()
        else:
            # Concat baseline: same inputs as MAF (obs + history + state), one
            # linear fusion — isolates the contribution of adaptive fusion.
            self.cond_projector = nn.Linear(
                obs_feature_dim * n_obs_steps + action_dim * n_obs_steps + action_dim,
                latent_dim,
            )
            cond_token_dim = None

        # ---- generative backbone (bridge vector field) ----
        if backbone == "dit":
            dit = dit or {}
            self.flow_net = DiTBridge(
                latent_dim=latent_dim,
                cond_dim=latent_dim,
                hidden_dim=dit.get("hidden_dim", 512),
                num_layers=dit.get("num_layers", 6),
                num_heads=dit.get("num_heads", 8),
                n_latent_tokens=dit.get("n_latent_tokens", 8),
                cond_token_dim=cond_token_dim if dit.get("use_cond_tokens", True) else None,
                mlp_ratio=dit.get("mlp_ratio", 4.0),
                dropout=dit.get("dropout", 0.0),
                max_cond_tokens=2 * n_obs_steps + 1,
            )
        else:
            self.flow_net = SimpleFlowNet(
                input_dim=latent_dim,
                hidden_dim=flow_net.hidden_dim,
                output_dim=latent_dim,
                num_layers=flow_net.num_layers,
                mlp_ratio=flow_net.mlp_ratio,
                dropout=flow_net.dropout,
                condition_dim=latent_dim,
            )

        # ---- action latent codecs (same as A2A, shared across all variants) ----
        self.history_action_encoder = CNNActionEncoder(
            pred_horizon=n_obs_steps,
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_dim=action_ae.net.enc_hidden_dim,
        )
        future_horizon = n_action_steps
        self.future_horizon = future_horizon
        self.action_encoder = CNNActionEncoder(
            pred_horizon=future_horizon,
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_dim=action_ae.net.enc_hidden_dim,
        )
        self.action_decoder = SimpleActionDecoder(
            dec_hidden_dim=action_ae.net.dec_hidden_dim,
            latent_dim=latent_dim,
            pred_horizon=future_horizon,
            action_dim=action_dim,
            num_layers=action_ae.net.num_layers,
            dropout=action_ae.net.dropout,
        )

        # ---- latent alignment heads ----
        if jepa_weight > 0:
            self.jepa_predictor = JEPAPredictor(
                cond_dim=latent_dim,
                latent_dim=latent_dim,
                hidden_dim=latent_dim,
            )

        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.kwargs = kwargs

    # ------------------------------------------------------------------
    # condition / source construction
    # ------------------------------------------------------------------
    def _encode_condition(self, nobs, batch_size):
        """Returns (c, cond_kwargs, gates). cond_kwargs feed the flow net."""
        this_nobs = dict_apply(nobs, lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)  # (B*T, obs_feature_dim)
        obs_features = nobs_features.reshape(batch_size, self.n_obs_steps, -1)

        history_states = nobs["agent_pos"][:, : self.n_obs_steps, :]
        current_state = history_states[:, -1, :]

        gates = None
        if self.condition_mode == "maf":
            sigma_a = self.history_noise_std if self.source_mode in ("noised_history", "mixed_bridge") else 0.0
            sigma_z = self.latent_noise_std if self.source_mode == "mixed_bridge" else 0.0
            c, c_tokens, gates = self.maf_encoder(
                obs_features, history_states, current_state,
                sigma_a=sigma_a, sigma_z=sigma_z,
            )
        else:
            flat = torch.cat([
                obs_features.reshape(batch_size, -1),
                history_states.reshape(batch_size, -1),
                current_state,
            ], dim=-1)
            c = self.cond_projector(flat)
            c_tokens = None

        cond_kwargs = {"global_cond": c}
        if self.backbone == "dit" and c_tokens is not None:
            cond_kwargs["cond_tokens"] = c_tokens
        return c, cond_kwargs, gates

    def _build_source(self, history_states):
        """Build (z_source, mu_h) per source_mode.

        mu_h is the clean history latent, only used by residual_history to
        shift the generated residual back to an absolute latent.
        """
        B = history_states.shape[0]
        device = history_states.device

        if self.source_mode == "gaussian":
            return torch.randn(B, self.latent_dim, device=device), None

        if self.source_mode == "residual_history":
            mu_h = self.history_action_encoder(history_states)
            return torch.randn(B, self.latent_dim, device=device), mu_h

        h = history_states
        if self.source_mode in ("noised_history", "mixed_bridge") and self.history_noise_std > 0:
            h = h + torch.randn_like(h) * self.history_noise_std
        z_source = self.history_action_encoder(h)
        if self.source_mode == "mixed_bridge" and self.latent_noise_std > 0:
            z_source = z_source + torch.randn_like(z_source) * self.latent_noise_std
        return z_source, None

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    def compute_loss(self, batch):
        assert "valid_mask" not in batch
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]

        c, cond_kwargs, gates = self._encode_condition(nobs, batch_size)

        history_states = nobs["agent_pos"][:, : self.n_obs_steps, :]
        future_start = self.n_obs_steps - 1
        future_end = future_start + self.n_action_steps
        future_actions = nactions[:, future_start:future_end, :]
        z1 = self.action_encoder(future_actions)

        z_source, mu_h = self._build_source(history_states)
        # residual_history generates r = z1 - mu_h from a Gaussian source
        flow_target = z1 - mu_h if mu_h is not None else z1

        flow_loss, metrics = self.flow_matcher.compute_loss(
            self.flow_net,
            target=flow_target,
            start=z_source,
            **cond_kwargs,
        )
        loss = flow_loss
        metrics["flow_loss"] = flow_loss.item()

        # ---- latent alignment ----
        if self.jepa_weight > 0:
            l_jepa = jepa_loss(self.jepa_predictor, c, z_source.detach(), z1)
            loss = loss + self.jepa_weight * l_jepa
            metrics["jepa_loss"] = l_jepa.item()

        if self.align_weight > 0:
            if self.align_type == "infonce":
                l_align = info_nce_loss(c, z1)
            else:
                l_align = vicreg_loss(c, z1)
            loss = loss + self.align_weight * l_align
            metrics["align_loss"] = l_align.item()

        # ---- inference-consistency losses (A2A-inherited) ----
        if self.decode_flow_latents:
            latents_pred = self.flow_matcher.sample(
                self.flow_net,
                shape=(batch_size, self.latent_dim),
                device=c.device,
                start=z_source,
                num_steps=self.num_sampling_steps,
                **cond_kwargs,
            )
            if mu_h is not None:
                latents_pred = latents_pred + mu_h

            if self.consistency_weight > 0:
                consistency_loss = F.mse_loss(latents_pred, z1)
                loss = loss + self.consistency_weight * consistency_loss
                metrics["consistency_loss"] = consistency_loss.item()

            if self.action_ae["flow_recon_weight"] > 0:
                actions_recon = self.action_decoder(latents_pred)
                recon = F.l1_loss(actions_recon, future_actions)
                loss = loss + self.action_ae["flow_recon_weight"] * recon
                metrics["flow_action_recon_loss"] = recon.item()

        # ---- autoencoder preservation ----
        if self.action_ae["enc_recon_weight"] > 0:
            actions_recon = self.action_decoder(z1)
            recon = F.l1_loss(actions_recon, future_actions)
            loss = loss + self.action_ae["enc_recon_weight"] * recon
            metrics["enc_action_recon_loss"] = recon.item()

        # ---- diagnostics (no gradient) ----
        with torch.no_grad():
            src_for_dist = z_source + mu_h if mu_h is not None else z_source
            metrics["source_target_dist"] = (src_for_dist - z1).norm(dim=-1).mean().item()
            metrics.update(latent_collapse_metrics(z1, prefix="z1_"))
            if gates is not None:
                metrics["gate_visual"] = gates[:, 0].mean().item()
                metrics["gate_history"] = gates[:, 1].mean().item()
                metrics["gate_state"] = gates[:, 2].mean().item()
                metrics["gate_entropy"] = (
                    -(gates * torch.log(gates + 1e-8)).sum(dim=-1).mean().item()
                )

        self.last_metrics = metrics
        return loss

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B = value.shape[0]

        c, cond_kwargs, _ = self._encode_condition(nobs, B)
        history_states = nobs["agent_pos"][:, : self.n_obs_steps, :]
        z_source, mu_h = self._build_source(history_states)

        latents_pred = self.flow_matcher.sample(
            self.flow_net,
            shape=(B, self.latent_dim),
            device=c.device,
            num_steps=self.num_sampling_steps,
            start=z_source,
            return_traces=False,
            **cond_kwargs,
        )
        if mu_h is not None:
            latents_pred = latents_pred + mu_h

        with torch.no_grad():
            action_pred = self.action_decoder(latents_pred)

        action_pred = self.normalizer["action"].unnormalize(action_pred)
        action = action_pred[:, : self.n_action_steps]
        return {"action": action, "action_pred": action_pred}

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    # ------------------------------------------------------------------
    # visualization / latent metrics helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_latents_for_visualization(self, batch):
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])

        history_states = nobs["agent_pos"][:, : self.n_obs_steps, :]
        z_source, mu_h = self._build_source(history_states)
        if mu_h is not None:
            z_source = z_source + mu_h

        future_start = self.n_obs_steps - 1
        future_end = future_start + self.n_action_steps
        future_actions = nactions[:, future_start:future_end, :]
        future_latents = self.action_encoder(future_actions)
        return z_source, future_latents

    @torch.no_grad()
    def get_flow_trajectories(self, batch, num_steps=None, n_samples=5):
        if num_steps is None:
            num_steps = self.num_sampling_steps

        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]
        n_samples = min(n_samples, batch_size)

        nobs = dict_apply(nobs, lambda x: x[:n_samples])
        c, cond_kwargs, _ = self._encode_condition(nobs, n_samples)

        history_states = nobs["agent_pos"][:, : self.n_obs_steps, :]
        z_source, mu_h = self._build_source(history_states)

        future_start = self.n_obs_steps - 1
        future_end = future_start + self.n_action_steps
        future_actions = nactions[:n_samples, future_start:future_end, :]
        future_latents = self.action_encoder(future_actions)

        _, (traj_history, _) = self.flow_matcher.sample(
            self.flow_net,
            shape=(n_samples, self.latent_dim),
            device=c.device,
            num_steps=num_steps,
            start=z_source,
            return_traces=True,
            **cond_kwargs,
        )

        traj_history_cpu = []
        for t in traj_history:
            if hasattr(t, "cpu"):
                traj_history_cpu.append(t.cpu())
            else:
                traj_history_cpu.append(torch.tensor(t))

        traj_stacked = torch.stack(traj_history_cpu, dim=0)
        if mu_h is not None:
            traj_stacked = traj_stacked + mu_h.cpu().unsqueeze(0)

        trajectories = [traj_stacked[:, i, :].numpy() for i in range(n_samples)]
        return trajectories, future_latents.cpu().numpy()
