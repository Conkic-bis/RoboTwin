# Qwen3VL + Flow Matching Inference Model
#
# Wraps the Qwen3VLActionModel for deployment in RoboTwin.
# Handles observation caching, action queueing, and format conversion.

import torch
import numpy as np
from typing import Dict, List, Optional, Any
from collections import deque

import sys
from pathlib import Path

parent_dir = str(Path(__file__).parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from model.action_model.action_model import Qwen3VLActionModel
from utils.rotation_utils import convert_endpose_9d_to_7d


class Qwen3VLFlowMatchingModel:
    """
    Inference wrapper for Qwen3VL + Flow Matching policy.

    Handles:
    - Checkpoint loading with full training config restore
    - Observation history caching (n_obs_steps)
    - Action queueing for receding horizon control
    - Action format conversion (rot6d -> quaternion for endpose)

    Args:
        ckpt_file: Path to training checkpoint
        n_obs_steps: Number of observation history steps
        n_action_steps: Number of actions to execute per inference
        num_inference_steps: ODE integration steps
        ode_solver: 'euler' or 'midpoint'
        device: Torch device
        quat_convention: Output quaternion convention
        use_both_arms: Dual arm mode
        action_type: 'endpose' or 'joint'
        instruction: Task instruction for VLM
    """

    def __init__(
        self,
        ckpt_file: str,
        n_obs_steps: int = 1,
        n_action_steps: int = 8,
        num_inference_steps: int = 10,
        ode_solver: str = "midpoint",
        device: str = "cuda:0",
        quat_convention: str = "wxyz",
        use_both_arms: bool = True,
        action_type: str = "joint",
        instruction: str = "Predict the next robot actions.",
    ):
        self.device = device
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.num_inference_steps = num_inference_steps
        self.ode_solver = ode_solver
        self.quat_convention = quat_convention
        self.use_both_arms = use_both_arms
        self.action_type = action_type
        self.instruction = instruction

        # Load model
        self.model = self._load_model(ckpt_file)
        self.model.eval()

        # Observation cache
        self.obs_cache: Optional[Dict[str, deque]] = None
        self.action_queue: List[np.ndarray] = []

        if action_type == "endpose":
            self.single_arm_dim = 10
        else:
            self.single_arm_dim = 7

    def _load_model(self, ckpt_file: str) -> Qwen3VLActionModel:
        """Load model from checkpoint with saved training config."""
        print(f"Loading checkpoint from: {ckpt_file}")
        checkpoint = torch.load(ckpt_file, map_location='cpu')

        train_args = checkpoint.get('args', {})

        print(f"VLM: {train_args.get('vlm_model_name', 'Qwen/Qwen3-VL-4B-Instruct')}")
        print(f"DiT: {train_args.get('dit_model_type', 'DiT-B')}, "
              f"action_dim={train_args.get('action_dim', 'N/A')}")

        model = Qwen3VLActionModel(
            vlm_model_name=train_args.get('vlm_model_name', 'Qwen/Qwen3-VL-4B-Instruct'),
            freeze_vlm=True,  # Always frozen during inference
            use_lora=train_args.get('use_lora', False),
            lora_r=train_args.get('lora_r', 16),
            lora_alpha=train_args.get('lora_alpha', 32),
            dit_model_type=train_args.get('dit_model_type', 'DiT-B'),
            action_dim=train_args.get('action_dim', 14),
            future_action_window_size=train_args.get('future_action_window', 13),
            n_action_steps=train_args.get('n_action_steps', self.n_action_steps),
            time_sampling=train_args.get('time_sampling', 'logit_normal'),
            logit_normal_loc=train_args.get('logit_normal_loc', 0.0),
            logit_normal_scale=train_args.get('logit_normal_scale', 1.0),
            beta_alpha=train_args.get('beta_alpha', 1.5),
            beta_beta=train_args.get('beta_beta', 1.0),
            num_timestep_buckets=train_args.get('num_timestep_buckets', 1000),
        )

        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model = model.to(self.device)

        self.n_obs_steps = train_args.get('n_obs_steps', self.n_obs_steps)
        self.n_action_steps = train_args.get('n_action_steps', self.n_action_steps)

        print(f"Model loaded from epoch {checkpoint.get('epoch', 'N/A')}")
        return model

    def reset_obs(self):
        """Reset observation cache at the beginning of each episode."""
        self.obs_cache = None
        self.action_queue = []

    def update_obs(self, obs: Dict[str, Any]):
        """
        Update observation cache with new observation.

        Args:
            obs: Dictionary containing:
                - 'images': List[np.ndarray] - raw images per camera (H, W, 3) uint8
                - 'agent_pos': (action_dim,) current robot state
        """
        if self.obs_cache is None:
            self.obs_cache = {
                'images': deque(maxlen=self.n_obs_steps),
            }
            for _ in range(self.n_obs_steps):
                self.obs_cache['images'].append(obs['images'])
        else:
            self.obs_cache['images'].append(obs['images'])

        if 'agent_pos' in obs:
            self.obs_cache['agent_pos'] = obs['agent_pos']

    def _prepare_state_input(self) -> Optional[torch.Tensor]:
        """Prepare state tensor from observation cache."""
        if self.obs_cache is None or 'agent_pos' not in self.obs_cache:
            return None
        agent_pos = self.obs_cache['agent_pos']
        state = np.expand_dims(agent_pos, axis=0)
        return torch.from_numpy(state).float().to(self.device)

    def _convert_action_to_robotwin(self, action: np.ndarray) -> np.ndarray:
        """Convert model output to RoboTwin format."""
        if self.action_type == "joint":
            if self.use_both_arms:
                left_action = action[:, :7]
                right_action = action[:, 7:14]
                return np.concatenate([left_action, right_action], axis=-1)
            else:
                return action
        else:
            if self.use_both_arms:
                left_action = action[:, :10]
                right_action = action[:, 10:20]
                left_converted = self._convert_single_arm_action(left_action)
                right_converted = self._convert_single_arm_action(right_action)
                return np.concatenate([left_converted, right_converted], axis=-1)
            else:
                return self._convert_single_arm_action(action)

    def _convert_single_arm_action(self, action: np.ndarray) -> np.ndarray:
        """Convert single arm action from rot6d to quaternion."""
        translation = action[:, :3]
        rot6d = action[:, 3:9]
        gripper = action[:, 9:10]
        pose_9d = np.concatenate([translation, rot6d], axis=-1)
        pose_7d = convert_endpose_9d_to_7d(pose_9d, self.quat_convention)
        return np.concatenate([pose_7d, gripper], axis=-1)

    @torch.no_grad()
    def get_action(self, obs: Dict[str, Any]) -> List[np.ndarray]:
        """
        Get action sequence from current observation.

        Args:
            obs: Dictionary containing:
                - 'images': List[np.ndarray] - raw images per camera
                - 'agent_pos': robot state (optional)

        Returns:
            List of actions to execute
        """
        self.update_obs(obs)

        if len(self.action_queue) > 0:
            action = self.action_queue.pop(0)
            return [action]

        # Use latest observation images for VLM
        latest_images = list(self.obs_cache['images'])[-1]

        # Prepare state
        state = self._prepare_state_input()

        # Encode through VLM
        context = self.model.encode_vlm(
            images=latest_images,
            instruction=self.instruction,
            robot_state=state[0] if state is not None else None,
        )

        # Sample actions via flow matching ODE
        action_pred = self.model.sample(
            context=context,
            state=state,
            num_steps=self.num_inference_steps,
            ode_solver=self.ode_solver,
            return_all=False,
        )  # (1, n_action_steps, action_dim)

        action_pred = action_pred.cpu().numpy()[0]
        action_converted = self._convert_action_to_robotwin(action_pred)

        # Queue remaining actions
        for i in range(1, min(self.n_action_steps, len(action_converted))):
            self.action_queue.append(action_converted[i])

        return [action_converted[0]]

    @torch.no_grad()
    def get_all_actions(self, obs: Dict[str, Any]) -> np.ndarray:
        """Get all predicted actions at once (without queueing)."""
        self.update_obs(obs)

        latest_images = list(self.obs_cache['images'])[-1]
        state = self._prepare_state_input()

        context = self.model.encode_vlm(
            images=latest_images,
            instruction=self.instruction,
            robot_state=state[0] if state is not None else None,
        )

        action_pred = self.model.sample(
            context=context,
            state=state,
            num_steps=self.num_inference_steps,
            ode_solver=self.ode_solver,
            return_all=False,
        )

        action_pred = action_pred.cpu().numpy()[0]
        return self._convert_action_to_robotwin(action_pred)
