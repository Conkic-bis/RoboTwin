# Inference wrapper for the Qwen3-VL + Flow Matching ActionModel.
#
# What changed vs the 2.0 branch:
#   - The ResNet vision path was replaced by a Qwen3-VL VLM. The processor
#     wants RAW per-camera images (uint8 RGB) plus a plain-text instruction,
#     not ImageNet-normalized tensors. So the obs cache now stores raw
#     numpy frames and build_inputs is handled inside the VLM.
#   - checkpoint['model_state_dict'] written by train.py is already the
#     *unwrapped* tree (see train.save_checkpoint), so we just call
#     load_state_dict once without touching any `_orig_mod.` prefix.
#   - For single-step inference we call encode_vlm_batch + loss-free
#     sampling once per decision; batch dim is always 1 here (a single
#     robot) but the batch path still applies as a degenerate B=1 case.

import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

parent_dir = str(Path(__file__).parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from model.action_model.action_model import ActionModel
from utils.rotation_utils import convert_endpose_9d_to_7d


class NemoDiT:
    """
    Runtime wrapper around ActionModel for policy deployment.

    The model stores a rolling window of RAW per-camera images (uint8 RGB),
    plus an optional robot state vector, and a language instruction.
    encode_vlm_batch() is called once per decision with B=1 to produce the
    conditioning context for the DiT sampler.
    """

    def __init__(
        self,
        ckpt_file: str,
        n_obs_steps: int = 1,
        n_action_steps: int = 10,
        num_inference_steps: int = 10,
        ode_solver: str = 'midpoint',
        device: str = 'cuda:0',
        quat_convention: str = 'wxyz',
        use_both_arms: bool = True,
        action_type: str = 'endpose',
        instruction: str = 'Predict the next robot actions.',
        # Legacy param
        ddim_steps: Optional[int] = None,
    ):
        self.device = device
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.num_inference_steps = ddim_steps if ddim_steps is not None else num_inference_steps
        self.ode_solver = ode_solver
        self.quat_convention = quat_convention
        self.use_both_arms = use_both_arms
        self.action_type = action_type
        self.instruction = instruction

        self.model = self._load_model(ckpt_file)
        self.model.eval()

        self.obs_cache: Optional[Dict[str, Any]] = None
        self.action_queue: List[np.ndarray] = []

        # Per-arm action-dim bookkeeping used by the RoboTwin format
        # converter. The model itself is agnostic; these live here.
        if action_type == 'endpose':
            self.single_arm_dim = 10
        else:
            self.single_arm_dim = 7

    # ------------------------------------------------------------------ #
    # Model loading
    # ------------------------------------------------------------------ #

    def _load_model(self, ckpt_file: str) -> ActionModel:
        print(f"[NemoDiT] loading checkpoint: {ckpt_file}")
        checkpoint = torch.load(ckpt_file, map_location='cpu')
        train_args = checkpoint.get('args', {}) or {}

        print(
            f"[NemoDiT] model_type={train_args.get('model_type', 'DiT-B')} "
            f"action_dim={train_args.get('action_dim', 'N/A')} "
            f"n_obs_steps={train_args.get('n_obs_steps', 1)} "
            f"n_action_steps={train_args.get('n_action_steps', 'N/A')}"
        )

        model = ActionModel(
            model_type=train_args.get('model_type', 'DiT-B'),
            in_channels=train_args.get('action_dim', 20 if self.use_both_arms else 10),
            future_action_window_size=train_args.get('future_action_window', 10),
            past_action_window_size=train_args.get('past_action_window', 0),
            n_obs_steps=train_args.get('n_obs_steps', 1),
            n_action_steps=train_args.get('n_action_steps', self.n_action_steps),
            # VLM
            vlm_model_name=train_args.get('vlm_model_name', 'Qwen/Qwen3-VL-4B-Instruct'),
            freeze_vlm=train_args.get('freeze_vlm', True),
            use_lora=train_args.get('use_lora', False),
            lora_r=train_args.get('lora_r', 16),
            lora_alpha=train_args.get('lora_alpha', 32),
            # Flow matching
            time_sampling=train_args.get('time_sampling', 'logit_normal'),
            logit_normal_loc=train_args.get('logit_normal_loc', 0.0),
            logit_normal_scale=train_args.get('logit_normal_scale', 1.0),
            beta_alpha=train_args.get('beta_alpha', 1.5),
            beta_beta=train_args.get('beta_beta', 1.0),
            num_timestep_buckets=train_args.get('num_timestep_buckets', 1000),
            # legacy knobs — ignored, kept for argparse compat
            token_size=train_args.get('token_size', 2048),
            vision_backbone_type=train_args.get('vision_backbone', None),
            vision_pretrained=False,
            num_cameras=train_args.get('num_cameras', 4),
            adapter_type=train_args.get('adapter_type', None),
            class_dropout_prob=0.0,
            temporal_agg=train_args.get('temporal_agg', 'last'),
        )

        # train.save_checkpoint() always unwraps the compiled modules, so
        # keys in model_state_dict match the eager module tree exactly.
        # strict=False because frozen VLM weights are typically not saved.
        missing, unexpected = model.load_state_dict(
            checkpoint['model_state_dict'], strict=False,
        )
        # Report only non-VLM mismatches — missing VLM keys are expected
        # whenever freeze_vlm=True and we reload from the HF hub.
        missing_non_vlm = [k for k in missing if not k.startswith('vlm.')]
        if missing_non_vlm:
            print(f"[NemoDiT] WARN missing keys (non-vlm): {missing_non_vlm[:8]}"
                  f"{' ...' if len(missing_non_vlm) > 8 else ''}")
        if unexpected:
            print(f"[NemoDiT] WARN unexpected keys: {unexpected[:8]}"
                  f"{' ...' if len(unexpected) > 8 else ''}")

        model = model.to(self.device)

        # Update local knobs to whatever the checkpoint was trained with.
        self.n_obs_steps = train_args.get('n_obs_steps', self.n_obs_steps)
        self.n_action_steps = train_args.get('n_action_steps', self.n_action_steps)
        print(f"[NemoDiT] loaded epoch={checkpoint.get('epoch', 'N/A')}")
        return model

    # ------------------------------------------------------------------ #
    # Observation cache
    # ------------------------------------------------------------------ #

    def reset_obs(self):
        self.obs_cache = None
        self.action_queue = []

    def update_obs(self, obs: Dict[str, np.ndarray]):
        """Append the latest observation to the rolling cache.

        obs['images_raw'] is a list of per-camera uint8 RGB arrays; we
        duplicate the first frame to fill the n_obs_steps window.
        """
        if self.obs_cache is None:
            self.obs_cache = {
                'images_raw': deque(maxlen=self.n_obs_steps),
            }
            for _ in range(self.n_obs_steps):
                self.obs_cache['images_raw'].append(obs['images_raw'])
        else:
            self.obs_cache['images_raw'].append(obs['images_raw'])

        if 'agent_pos' in obs:
            self.obs_cache['agent_pos'] = obs['agent_pos']
        if 'instruction' in obs and obs['instruction']:
            self.obs_cache['instruction'] = obs['instruction']

    # ------------------------------------------------------------------ #
    # Input prep
    # ------------------------------------------------------------------ #

    def _current_images(self) -> List[np.ndarray]:
        """Latest frame's per-camera uint8 images (what Qwen3-VL expects)."""
        if self.obs_cache is None or len(self.obs_cache['images_raw']) == 0:
            raise ValueError("obs cache empty — call update_obs() first")
        return self.obs_cache['images_raw'][-1]

    def _current_state(self) -> Optional[torch.Tensor]:
        if self.obs_cache is None or 'agent_pos' not in self.obs_cache:
            return None
        agent_pos = self.obs_cache['agent_pos']
        return torch.from_numpy(np.asarray(agent_pos, dtype=np.float32))[None].to(self.device)

    def _current_instruction(self) -> str:
        if self.obs_cache is not None and 'instruction' in self.obs_cache:
            return self.obs_cache['instruction']
        return self.instruction

    # ------------------------------------------------------------------ #
    # RoboTwin action format conversion (unchanged)
    # ------------------------------------------------------------------ #

    def _convert_action_to_robotwin(self, action: np.ndarray) -> np.ndarray:
        if self.action_type == 'joint':
            if self.use_both_arms:
                left = action[:, :7]
                right = action[:, 7:14]
                return np.concatenate([left, right], axis=-1)
            return action
        # endpose
        if self.use_both_arms:
            left = self._convert_single_arm_action(action[:, :10])
            right = self._convert_single_arm_action(action[:, 10:20])
            return np.concatenate([left, right], axis=-1)
        return self._convert_single_arm_action(action)

    def _convert_single_arm_action(self, action: np.ndarray) -> np.ndarray:
        translation = action[:, :3]
        rot6d = action[:, 3:9]
        gripper = action[:, 9:10]
        pose_9d = np.concatenate([translation, rot6d], axis=-1)
        pose_7d = convert_endpose_9d_to_7d(pose_9d, self.quat_convention)
        return np.concatenate([pose_7d, gripper], axis=-1)

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def get_action(self, obs: Dict[str, np.ndarray]) -> List[np.ndarray]:
        self.update_obs(obs)

        if len(self.action_queue) > 0:
            return [self.action_queue.pop(0)]

        images_raw = self._current_images()
        state = self._current_state()
        instruction = self._current_instruction()

        # B=1 degenerate batch: images wrapped in an outer list of length 1.
        action_pred = self.model.sample(
            images=[images_raw],
            instructions=[instruction],
            state=state,
            num_steps=self.num_inference_steps,
            ode_solver=self.ode_solver,
            cfg_scale=1.0,
            return_all=False,
        )  # (1, n_action_steps, action_dim)

        action_pred = action_pred.cpu().numpy()[0]
        action_converted = self._convert_action_to_robotwin(action_pred)

        actions_to_execute = min(self.n_action_steps, len(action_converted))
        for i in range(1, actions_to_execute):
            self.action_queue.append(action_converted[i])

        return [action_converted[0]]

    @torch.no_grad()
    def get_all_actions(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        self.update_obs(obs)
        images_raw = self._current_images()
        state = self._current_state()
        instruction = self._current_instruction()

        action_pred = self.model.sample(
            images=[images_raw],
            instructions=[instruction],
            state=state,
            num_steps=self.num_inference_steps,
            ode_solver=self.ode_solver,
            cfg_scale=1.0,
            return_all=False,
        )
        action_pred = action_pred.cpu().numpy()[0]
        return self._convert_action_to_robotwin(action_pred)
