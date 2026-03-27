import torch
import numpy as np
from typing import Dict, List, Optional, Any
from collections import deque

import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from model.action_model.action_model import ActionModel
from utils.rotation_utils import convert_endpose_9d_to_7d


class NemoDiT:

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
        """
        Load model from checkpoint.

        参考 eval.py 的 load_model 函数，使用 checkpoint 中保存的完整训练参数。0204
        这确保了模型结构与训练时完全一致。
        """
        print(f"Loading checkpoint from: {ckpt_file}")
        checkpoint = torch.load(ckpt_file, map_location='cpu')

        # 获取训练时的参数
        train_args = checkpoint.get('args', {})

        # 打印模型配置信息
        print(f"Model config: {train_args.get('model_type', 'DiT-B')}, "
              f"action_dim={train_args.get('action_dim', 'N/A')}")
        print(f"Temporal config: n_obs_steps={train_args.get('n_obs_steps', 1)}, "
              f"n_action_steps={train_args.get('n_action_steps', 'N/A')}, "
              f"future_action_window={train_args.get('future_action_window', 'N/A')}")

        model = ActionModel(
            token_size=train_args.get('token_size', 2048),
            model_type=train_args.get('model_type', 'DiT-B'),
            in_channels=train_args.get('action_dim', 20 if self.use_both_arms else 10),
            future_action_window_size=train_args.get('future_action_window', 10),
            past_action_window_size=train_args.get('past_action_window', 0),
            diffusion_steps=train_args.get('diffusion_steps', 100),
            noise_schedule=train_args.get('noise_schedule', 'squaredcos_cap_v2'),
            use_vision_condition=True,
            vision_backbone_type=train_args.get('vision_backbone', 'resnet50'),
            vision_pretrained=False,  # 不需要预训练权重，我们会加载训练好的
            num_cameras=train_args.get('num_cameras', 4),
            freeze_vision_backbone=False,
            adapter_type=train_args.get('adapter_type', 'mlp'),
            class_dropout_prob=0.0,  # 推理时关闭 dropout
            n_obs_steps=train_args.get('n_obs_steps', 1),
            n_action_steps=train_args.get('n_action_steps', self.n_action_steps),
            temporal_agg=train_args.get('temporal_agg', 'last'),
        )

        # 加载权重
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model = model.to(self.device)

        # 更新实例变量以匹配训练配置
        self.n_obs_steps = train_args.get('n_obs_steps', self.n_obs_steps)
        self.n_action_steps = train_args.get('n_action_steps', self.n_action_steps)

        print(f"Model loaded from epoch {checkpoint.get('epoch', 'N/A')}")
        return model

    def reset_obs(self):
        """Reset observation cache at the beginning of each episode."""
        self.obs_cache = None
        self.action_queue = []

    def update_obs(self, obs: Dict[str, np.ndarray]):
        """
        Update observation cache with new observation.

        When the cache is empty (e.g. first call after reset), the observation
        is duplicated to fill the entire n_obs_steps window so that
        _prepare_vision_input always produces the shape the model expects.

        Args:
            obs: Dictionary containing:
                - 'images': (num_cameras, 3, H, W) normalized images
                - 'agent_pos': (action_dim,) current joint/ee positions (optional)
        """
        if self.obs_cache is None:
            # Initialize cache and pad with the first observation
            self.obs_cache = {
                'images': deque(maxlen=self.n_obs_steps),
            }
            for _ in range(self.n_obs_steps):
                self.obs_cache['images'].append(obs['images'])
        else:
            self.obs_cache['images'].append(obs['images'])

        # Cache current robot state for state conditioning
        if 'agent_pos' in obs:
            self.obs_cache['agent_pos'] = obs['agent_pos']

    def _prepare_vision_input(self) -> torch.Tensor:
        """Prepare vision input from observation cache."""
        if self.obs_cache is None or len(self.obs_cache['images']) == 0:
            raise ValueError("No observations in cache. Call update_obs first.")

        # Stack all cached observations along a temporal dimension
        # Each element in the deque is (num_cameras, 3, H, W)
        images_list = list(self.obs_cache['images'])  # list of (num_cameras, 3, H, W)
        images = np.stack(images_list, axis=0)  # (n_obs_steps, num_cameras, 3, H, W)

        # Add batch dimension
        images = np.expand_dims(images, axis=0)  # (1, n_obs_steps, num_cameras, 3, H, W)

        # Convert to tensor
        images_tensor = torch.from_numpy(images).float().to(self.device)

        return images_tensor

    def _prepare_state_input(self) -> Optional[torch.Tensor]:
        """Prepare state tensor from observation cache for state conditioning."""
        if self.obs_cache is None or 'agent_pos' not in self.obs_cache:
            return None

        agent_pos = self.obs_cache['agent_pos']  # (action_dim,)
        # Add batch dimension and convert to tensor
        state = np.expand_dims(agent_pos, axis=0)  # (1, action_dim)
        state_tensor = torch.from_numpy(state).float().to(self.device)
        return state_tensor

    def _convert_action_to_robotwin(self, action: np.ndarray) -> np.ndarray:
        """
        Convert model output to RoboTwin format.

        take_action() always expects dual-arm layout:
            [left_arm, left_gripper, right_arm, right_gripper]

        For endpose action_type:
            - Converts rot6d to quaternion per arm
            - Dual arm input (T, 20) → output (T, 16)

        For joint action_type:
            - No rotation conversion needed
            - Dual arm input (T, 14) → output (T, 14)

        Args:
            action: (T, action_dim) action sequence

        Returns:
            Converted action in RoboTwin format
        """
        if self.action_type == "joint":
            if self.use_both_arms:
                # (T, 14) = [left_joints(6), left_gripper(1),
                #             right_joints(6), right_gripper(1)]
                left_action = action[:, :7]    # (T, 7)
                right_action = action[:, 7:14] # (T, 7)
                return np.concatenate([left_action, right_action], axis=-1)
            else:
                # Single arm (T, 7) = [joints(6), gripper(1)]
                return action
        else:
            # endpose: need rot6d → quaternion conversion
            if self.use_both_arms:
                # (T, 20) = [left(10), right(10)]
                left_action = action[:, :10]
                right_action = action[:, 10:20]
                left_converted = self._convert_single_arm_action(left_action)
                right_converted = self._convert_single_arm_action(right_action)
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

        使用 model.sample() 方法进行推理，与 eval.py 保持一致。

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

        # Prepare input: (1, n_obs_steps, num_cameras, C, H, W)
        images = self._prepare_vision_input()

        # Prepare state for conditioning: (1, action_dim)
        state = self._prepare_state_input()

        # 使用 model.sample() 进行推理，与 eval.py 一致
        action_pred = self.model.sample(
            images,
            state=state,
            ddim_steps=self.ddim_steps,
            cfg_scale=1.0,  # 无 classifier-free guidance
            return_all=False  # 只返回 n_action_steps 步
        )  # (1, n_action_steps, action_dim)

        # Convert to numpy
        action_pred = action_pred.cpu().numpy()[0]  # (n_action_steps, action_dim)

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

        使用 model.sample() 方法进行推理，与 eval.py 保持一致。

        Args:
            obs: Dictionary containing images

        Returns:
            (T, action_dim) action sequence in RoboTwin format
        """
        # Update observation cache
        self.update_obs(obs)

        # Prepare input: (1, n_obs_steps, num_cameras, C, H, W)
        images = self._prepare_vision_input()

        # Prepare state for conditioning: (1, action_dim)
        state = self._prepare_state_input()

        # 使用 model.sample() 进行推理，截取 n_action_steps 步用于执行
        action_pred = self.model.sample(
            images,
            state=state,
            ddim_steps=self.ddim_steps,
            cfg_scale=1.0,  # 无 classifier-free guidance
            return_all=False  # 只返回 n_action_steps 步
        )  # (1, n_action_steps, action_dim)

        # Convert to numpy
        action_pred = action_pred.cpu().numpy()[0]  # (T, action_dim)

        # Convert to RoboTwin format
        return self._convert_action_to_robotwin(action_pred)
