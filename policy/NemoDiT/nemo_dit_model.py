"""
nemo_dit_model.py

Model wrapper class for NemoDiT policy inference in RoboTwin.
Provides observation caching and DDIM-based action generation.

Refactored DDIM sampling based on rdt_runner.py style for better
transparency and control over the denoising process.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any
from collections import deque

import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from model.action_model.action_model import ActionModel
from utils.rotation_utils import convert_endpose_9d_to_7d


class NemoDiT:
    """
    NemoDiT model wrapper for RoboTwin policy deployment.

    This class wraps the ActionModel and provides:
    - Checkpoint loading
    - Observation caching for temporal context
    - DDIM-based action generation
    - Action format conversion (rot6d -> quaternion)
    """

    def __init__(
        self,
        ckpt_file: str,
        n_obs_steps: int = 1,
        n_action_steps: int = 10,
        ddim_steps: int = 10,
        device: str = "cuda:0",
        quat_convention: str = "wxyz",
        use_both_arms: bool = True,
        action_type: str = "endpose",
    ):
        """
        Initialize NemoDiT model.

        Args:
            ckpt_file: Path to model checkpoint file
            n_obs_steps: Number of observation steps to cache
            n_action_steps: Number of action steps to execute per inference
            ddim_steps: Number of DDIM sampling steps
            device: Device to run inference on
            quat_convention: Output quaternion convention ("wxyz" or "xyzw")
            use_both_arms: Whether to use dual arm mode
            action_type: Action type - "endpose" or "joint"
        """
        self.device = device
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.ddim_steps = ddim_steps
        self.quat_convention = quat_convention
        self.use_both_arms = use_both_arms
        self.action_type = action_type

        # Load model
        self.model = self._load_model(ckpt_file)
        self.model.eval()

        # Create DDIM sampler
        self.model.create_ddim(ddim_step=ddim_steps)

        # Observation cache
        self.obs_cache: Optional[Dict[str, deque]] = None
        self.action_queue: List[np.ndarray] = []

        # Action dimension info based on action_type
        # endpose: Single arm: 10D (3 translation + 6 rot6d + 1 gripper), Dual arm: 20D
        # joint: Single arm: 7D (6 joint + 1 gripper), Dual arm: 14D
        if action_type == "endpose":
            self.single_arm_dim = 10
        else:  # joint
            self.single_arm_dim = 7

    def _load_model(self, ckpt_file: str) -> ActionModel:
        """Load model from checkpoint."""
        print(f"Loading checkpoint from: {ckpt_file}")
        checkpoint = torch.load(ckpt_file, map_location=self.device)

        # Get model args from checkpoint
        args = checkpoint.get('args', {})

        # Create model with saved args
        model = ActionModel(
            token_size=args.get('token_size', 2048),
            model_type=args.get('model_type', 'DiT-B'),
            in_channels=args.get('action_dim', 20 if self.use_both_arms else 10),
            future_action_window_size=args.get('future_action_window', 10),
            past_action_window_size=args.get('past_action_window', 0),
            diffusion_steps=args.get('diffusion_steps', 100),
            noise_schedule=args.get('noise_schedule', 'squaredcos_cap_v2'),
            use_vision_condition=True,
            vision_backbone_type=args.get('vision_backbone', 'resnet50'),
            vision_pretrained=args.get('vision_pretrained', True),
            num_cameras=args.get('num_cameras', 4),
            freeze_vision_backbone=args.get('freeze_vision', False),
            adapter_type=args.get('adapter_type', 'attention_pooling'),
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)

        print(f"Model loaded successfully. Epoch: {checkpoint.get('epoch', 'N/A')}")
        return model

    def reset_obs(self):
        """Reset observation cache at the beginning of each episode."""
        self.obs_cache = None
        self.action_queue = []

    def update_obs(self, obs: Dict[str, np.ndarray]):
        """
        Update observation cache with new observation.

        Args:
            obs: Dictionary containing:
                - 'images': (num_cameras, 3, H, W) normalized images
                - 'agent_pos': (action_dim,) current joint/ee positions (optional)
        """
        if self.obs_cache is None:
            # Initialize cache
            self.obs_cache = {
                'images': deque(maxlen=self.n_obs_steps),
            }

        self.obs_cache['images'].append(obs['images'])

    def _prepare_vision_input(self) -> torch.Tensor:
        """Prepare vision input from observation cache."""
        if self.obs_cache is None or len(self.obs_cache['images']) == 0:
            raise ValueError("No observations in cache. Call update_obs first.")

        # Use the latest observation
        images = self.obs_cache['images'][-1]  # (num_cameras, 3, H, W)

        # Add batch dimension
        images = np.expand_dims(images, axis=0)  # (1, num_cameras, 3, H, W)

        # Convert to tensor
        images_tensor = torch.from_numpy(images).float().to(self.device)

        return images_tensor

    def _ddim_sample(
        self,
        vision_condition: torch.Tensor,
        clip_denoised: bool = True,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        DDIM sampling with explicit step-by-step denoising.

        This method is refactored based on rdt_runner.py's conditional_sample style,
        providing better transparency and control over the denoising process.

        Args:
            vision_condition: (batch_size, 1, token_size) vision condition from encoder.
            clip_denoised: Whether to clip predicted x0 to [-1, 1].
            eta: DDIM eta parameter (0 for deterministic, >0 for stochastic).

        Returns:
            action_pred: (batch_size, future_window, action_dim) denoised action sequence.
        """
        batch_size = vision_condition.shape[0]
        action_dim = self.model.in_channels
        future_window = self.model.future_action_window_size
        device = vision_condition.device

        # Ensure DDIM diffusion is created
        if self.model.ddim_diffusion is None:
            self.model.create_ddim(ddim_step=self.ddim_steps)

        ddim_diffusion = self.model.ddim_diffusion

        # Initialize from random noise
        noisy_action = torch.randn(
            batch_size, future_window, action_dim,
            device=device
        )

        # Get timestep indices (reversed for denoising: T -> 0)
        timestep_indices = list(range(ddim_diffusion.num_timesteps))[::-1]

        # Model kwargs for conditional generation
        model_kwargs = {'y': vision_condition}

        # DDIM sampling loop (step-by-step denoising)
        for i in timestep_indices:
            # Create timestep tensor for current batch
            t = torch.tensor([i] * batch_size, device=device)

            # Single DDIM step: x_t -> x_{t-1}
            out = ddim_diffusion.ddim_sample(
                model=self.model.net,
                x=noisy_action,
                t=t,
                clip_denoised=clip_denoised,
                model_kwargs=model_kwargs,
                eta=eta,
            )

            # Update noisy_action for next iteration
            noisy_action = out["sample"]

        return noisy_action

    def _convert_action_to_robotwin(self, action: np.ndarray) -> np.ndarray:
        """
        Convert model output to RoboTwin format.

        For endpose action_type:
            - Converts rot6d to quaternion
            - Input: Single arm (T, 10) [x,y,z, r1-r6, gripper], Dual arm (T, 20)
            - Output: Single arm (T, 8) [x,y,z, qw,qx,qy,qz, gripper], Dual arm (T, 16)

        For joint action_type:
            - No conversion needed, pass through directly
            - Input/Output: Single arm (T, 7) [j1-j6, gripper], Dual arm (T, 14)

        Args:
            action: (T, action_dim) action sequence

        Returns:
            Converted action in RoboTwin format
        """
        # For joint action type, no conversion needed
        if self.action_type == "joint":
            return action

        # For endpose action type, convert rot6d to quaternion
        T = action.shape[0]

        if self.use_both_arms:
            # Split into left and right arm
            left_action = action[:, :10]  # (T, 10)
            right_action = action[:, 10:20]  # (T, 10)

            # Convert each arm
            left_converted = self._convert_single_arm_action(left_action)
            right_converted = self._convert_single_arm_action(right_action)

            # Combine
            return np.concatenate([left_converted, right_converted], axis=-1)
        else:
            return self._convert_single_arm_action(action)

    def _convert_single_arm_action(self, action: np.ndarray) -> np.ndarray:
        """
        Convert single arm action from rot6d to quaternion format.

        Args:
            action: (T, 10) - [x,y,z, r1-r6, gripper]

        Returns:
            (T, 8) - [x,y,z, qw,qx,qy,qz, gripper]
        """
        T = action.shape[0]

        # Extract components
        translation = action[:, :3]  # (T, 3)
        rot6d = action[:, 3:9]  # (T, 6)
        gripper = action[:, 9:10]  # (T, 1)

        # Convert rot6d to 7d pose (translation + quaternion)
        pose_9d = np.concatenate([translation, rot6d], axis=-1)  # (T, 9)
        pose_7d = convert_endpose_9d_to_7d(pose_9d, self.quat_convention)  # (T, 7)

        # Combine with gripper
        return np.concatenate([pose_7d, gripper], axis=-1)  # (T, 8)

    @torch.no_grad()
    def get_action(self, obs: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """
        Get action sequence from current observation.

        Uses DDIM sampling with explicit step-by-step denoising
        (refactored based on rdt_runner.py style).

        Args:
            obs: Dictionary containing:
                - 'images': (num_cameras, 3, H, W) normalized images

        Returns:
            List of actions to execute
        """
        # Update observation cache
        self.update_obs(obs)

        # Check if we have queued actions
        if len(self.action_queue) > 0:
            # Return remaining queued actions
            action = self.action_queue.pop(0)
            return [action]

        # Prepare input
        images = self._prepare_vision_input()

        # Encode vision condition
        vision_condition = self.model.encode_vision_condition(images)

        # DDIM sampling with step-by-step denoising
        action_pred = self._ddim_sample(
            vision_condition=vision_condition,
            clip_denoised=True,
            eta=0.0,
        )

        # Convert to numpy
        action_pred = action_pred.cpu().numpy()[0]  # (T, action_dim)

        # Convert to RoboTwin format
        action_converted = self._convert_action_to_robotwin(action_pred)

        # Queue actions (execute first n_action_steps)
        actions_to_execute = min(self.n_action_steps, len(action_converted))
        for i in range(1, actions_to_execute):
            self.action_queue.append(action_converted[i])

        return [action_converted[0]]

    @torch.no_grad()
    def get_all_actions(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Get all predicted actions at once (without queueing).

        Uses DDIM sampling with explicit step-by-step denoising
        (refactored based on rdt_runner.py style).

        Args:
            obs: Dictionary containing images

        Returns:
            (T, action_dim) action sequence in RoboTwin format
        """
        # Update observation cache
        self.update_obs(obs)

        # Prepare input
        images = self._prepare_vision_input()

        # Encode vision condition
        vision_condition = self.model.encode_vision_condition(images)

        # DDIM sampling with step-by-step denoising
        action_pred = self._ddim_sample(
            vision_condition=vision_condition,
            clip_denoised=True,
            eta=0.0,
        )

        # Convert to numpy
        action_pred = action_pred.cpu().numpy()[0]  # (T, action_dim)

        # Convert to RoboTwin format
        return self._convert_action_to_robotwin(action_pred)
