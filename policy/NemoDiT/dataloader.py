import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Callable, List, Dict
import glob
import cv2

from utils.rotation_utils import convert_endpose_7d_to_9d


class RobotDataset(Dataset):
    """
    Dataset for loading robot manipulation data from HDF5 files.

    Data structure for each episode (raw HDF5 format):
    ├── episode_X.hdf5
    │   ├── /endpose
    │   │   ├── /endpose/left_endpose    (T, 7)  - 3D translation + 4D quaternion
    │   │   ├── /endpose/left_gripper    (T, 1)  - gripper state (1-DoF)
    │   │   ├── /endpose/right_endpose   (T, 7)  - 3D translation + 4D quaternion
    │   │   └── /endpose/right_gripper   (T, 1)  - gripper state (1-DoF)
    │   ├── /joint_action
    │   │   ├── /joint_action/left_arm     (T, 6)  - 6-DoF joint angles
    │   │   ├── /joint_action/left_gripper (T, 1)  - gripper state
    │   │   ├── /joint_action/right_arm    (T, 6)  - 6-DoF joint angles
    │   │   └── /joint_action/right_gripper(T, 1)  - gripper state
    │   └── /observation
    │       ├── /observation/front_camera/rgb   (T, H, W, 3)
    │       ├── /observation/head_camera/rgb    (T, H, W, 3)
    │       ├── /observation/left_camera/rgb    (T, H, W, 3)
    │       └── /observation/right_camera/rgb   (T, H, W, 3)

    Output action format (after conversion):
        action_type='endpose':
            - Single arm: (T, 10) = 3D translation + 6D rot6d + 1D gripper
            - Dual arm:   (T, 20) = (3D + 6D + 1D) * 2
        action_type='joint':
            - Single arm: (T, 7) = 6D joint angles + 1D gripper
            - Dual arm:   (T, 14) = (6D + 1D) * 2

    Diffusion Policy 时序设计:
        n_obs_steps: 观测步数，用于视觉编码的历史帧数
        n_action_steps: 动作执行步数，实际执行的动作数（< future_action_window）

        时间轴示例 (n_obs_steps=2, n_action_steps=8, future_action_window=16):

        ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
        │O-1│ O │ A │ A │ A │ A │ A │ A │ A │ A │...
        └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
              │   │   └───────────────────────────┘
              │   │     predicted actions (future_action_window - 1)
              │   │
              │   └── state = action[0]，机器人当前状态（无噪音）
              │
              └─── n_obs_steps 的最后一帧 = state = action 的第0帧时刻

    Args:
        data_path: Path to directory containing episode HDF5 files
        future_action_window: Number of future action steps to predict
        past_action_window: Number of past action steps as context (deprecated, use n_obs_steps)
        transform: Optional transform to apply to images
        num_cameras: Number of camera views to use (default: 4)
        camera_names: List of camera names to use (default: all 4 cameras)
        use_both_arms: Whether to use both arms (default: False, only left arm)
        action_type: 'endpose' for end-effector pose, 'joint' for joint angles
        quat_convention: Quaternion convention in HDF5 data, "wxyz" or "xyzw"
        n_obs_steps: Number of observation steps for visual conditioning (default: 1)
    """

    def __init__(
        self,
        data_path: str,
        future_action_window: int = 10,
        past_action_window: int = 0,
        transform: Optional[Callable] = None,
        num_cameras: int = 4,
        camera_names: Optional[List[str]] = None,
        use_both_arms: bool = False,
        action_type: str = 'endpose',  # 'endpose' or 'joint'
        quat_convention: str = 'wxyz',
        n_obs_steps: int = 1,  # 观测步数
    ):
        super().__init__()

        self.data_path = data_path
        self.future_action_window = future_action_window
        self.past_action_window = past_action_window
        self.transform = transform
        self.num_cameras = num_cameras
        self.use_both_arms = use_both_arms
        self.action_type = action_type
        self.quat_convention = quat_convention
        self.n_obs_steps = n_obs_steps

        # Validate parameters
        assert action_type in ['endpose', 'joint'], \
            f"action_type must be 'endpose' or 'joint', got {action_type}"
        assert n_obs_steps >= 1, f"n_obs_steps must be >= 1, got {n_obs_steps}"

        # Default camera names
        if camera_names is None:
            self.camera_names = ['front_camera', 'head_camera', 'left_camera', 'right_camera']
        else:
            self.camera_names = camera_names

        assert len(self.camera_names) >= num_cameras, \
            f"Requested {num_cameras} cameras but only {len(self.camera_names)} names provided"

        self.camera_names = self.camera_names[:num_cameras]

        # Load episode file paths
        self.episode_files = self._load_episode_files()
        print(f"Found {len(self.episode_files)} episode files")
        print(f"Action type: {action_type}")

        # Build index: (episode_idx, timestep)
        self.indices = self._build_indices()
        print(f"Total valid samples: {len(self.indices)}")

    def _load_episode_files(self) -> List[str]:
        """Load all episode HDF5 file paths."""
        episode_pattern = os.path.join(self.data_path, "episode*.hdf5")
        episode_files = sorted(glob.glob(episode_pattern))

        if len(episode_files) == 0:
            raise ValueError(f"No episode files found in {self.data_path}")

        return episode_files

    def _build_indices(self) -> List[tuple]:
        """
        Build indices for sampling.

        所有 timestep 均为合法 sample，边界处通过 padding 补齐:
        - 左侧不足 n_obs_steps 帧时，用第 0 帧重复填充
        - 右侧不足 future_action_window 帧时，用最后一帧重复填充

        Returns:
            List of (episode_idx, action_start_timestep) tuples
            action_start_timestep 是动作序列的起始帧（也是 n_obs_steps 的最后一帧）
        """
        indices = []

        for ep_idx, ep_file in enumerate(self.episode_files):
            with h5py.File(ep_file, 'r') as f:
                # Get episode length from action data
                if self.action_type == 'endpose':
                    data_key = 'endpose/left_endpose'
                else:  # joint
                    data_key = 'joint_action/left_arm'

                episode_length = f[data_key].shape[0]

                # 所有 timestep 都可作为 sample，边界处用 padding 补齐
                for t in range(episode_length):
                    indices.append((ep_idx, t))

        return indices

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.indices)

    def _load_actions(self, f: h5py.File, start_idx: int) -> np.ndarray:
        """
        Load action sequences from HDF5 file.

        Args:
            f: Open HDF5 file handle
            start_idx: Starting timestep index

        Returns:
            actions: (future_action_window, action_dim) numpy array
                     action_type='endpose':
                         Single arm: (T, 10) = 3D translation + 6D rot6d + 1D gripper
                         Dual arm:   (T, 20) = (3D + 6D + 1D) * 2
                     action_type='joint':
                         Single arm: (T, 7) = 6D joint angles + 1D gripper
                         Dual arm:   (T, 14) = (6D + 1D) * 2
        """
        if self.action_type == 'endpose':
            actions = self._load_endpose_actions(f, start_idx)
        else:  # joint
            actions = self._load_joint_actions(f, start_idx)

        # 右侧 padding: 当 start_idx + future_action_window 超出 episode 长度时，
        # 用最后一帧的值重复填充
        if actions.shape[0] < self.future_action_window:
            pad_len = self.future_action_window - actions.shape[0]
            padding = np.repeat(actions[-1:], pad_len, axis=0)
            actions = np.concatenate([actions, padding], axis=0)

        return actions

    def _load_endpose_actions(self, f: h5py.File, start_idx: int) -> np.ndarray:
        """Load end-effector pose actions and convert to rot6d representation."""
        # Load left arm actions
        left_endpose_7d = f['endpose/left_endpose'][
            start_idx : start_idx + self.future_action_window
        ]  # (T, 7) - 3D translation + 4D quaternion
        left_gripper = f['endpose/left_gripper'][
            start_idx : start_idx + self.future_action_window
        ]  # (T, 1) - gripper state

        # Convert quaternion to rot6d: (T, 7) -> (T, 9)
        left_endpose_9d = convert_endpose_7d_to_9d(
            left_endpose_7d, quat_convention=self.quat_convention
        )  # (T, 9) - 3D translation + 6D rot6d

        # Fix: ensure gripper is 2D (T, 1)
        if left_gripper.ndim == 1:
            left_gripper = left_gripper[:, np.newaxis]

        if self.use_both_arms:
            right_endpose_7d = f['endpose/right_endpose'][
                start_idx : start_idx + self.future_action_window
            ]
            right_gripper = f['endpose/right_gripper'][
                start_idx : start_idx + self.future_action_window
            ]

            # Convert quaternion to rot6d: (T, 7) -> (T, 9)
            right_endpose_9d = convert_endpose_7d_to_9d(
                right_endpose_7d, quat_convention=self.quat_convention
            )

            if right_gripper.ndim == 1:
                right_gripper = right_gripper[:, np.newaxis]

            # Dual arm: (T, 20) = (9 + 1) * 2
            actions = np.concatenate([
                left_endpose_9d, left_gripper,
                right_endpose_9d, right_gripper
            ], axis=-1)
        else:
            # Single arm: (T, 10) = 9 + 1
            actions = np.concatenate([
                left_endpose_9d, left_gripper
            ], axis=-1)

        return actions

    def _load_joint_actions(self, f: h5py.File, start_idx: int) -> np.ndarray:
        """Load joint angle actions."""
        # Load left arm joint angles
        left_joints = f['joint_action/left_arm'][
            start_idx : start_idx + self.future_action_window
        ]  # (T, 6) - 6-DoF joint angles
        left_gripper = f['joint_action/left_gripper'][
            start_idx : start_idx + self.future_action_window
        ]  # (T,) or (T, 1) - gripper state

        # Ensure correct shape
        if left_joints.ndim == 1:
            left_joints = left_joints[np.newaxis, :]
        if left_gripper.ndim == 1:
            left_gripper = left_gripper[:, np.newaxis]

        if self.use_both_arms:
            right_joints = f['joint_action/right_arm'][
                start_idx : start_idx + self.future_action_window
            ]  # (T, 6)
            right_gripper = f['joint_action/right_gripper'][
                start_idx : start_idx + self.future_action_window
            ]  # (T,) or (T, 1)

            if right_joints.ndim == 1:
                right_joints = right_joints[np.newaxis, :]
            if right_gripper.ndim == 1:
                right_gripper = right_gripper[:, np.newaxis]

            # Dual arm: (T, 14) = (6 + 1) * 2
            actions = np.concatenate([
                left_joints, left_gripper,
                right_joints, right_gripper
            ], axis=-1)
        else:
            # Single arm: (T, 7) = 6 + 1
            actions = np.concatenate([
                left_joints, left_gripper
            ], axis=-1)

        return actions

    def _load_images(self, f: h5py.File, timestep: int) -> List[np.ndarray]:
        """
        Load images from all cameras at a given timestep.

        Handles both raw numpy arrays and JPEG/PNG encoded bytes.

        Args:
            f: HDF5 file handle
            timestep: Single timestep to load

        Returns:
            List of images, one per camera, each (H, W, 3)
        """
        images = []

        for cam_name in self.camera_names:
            img_path = f'observation/{cam_name}/rgb'
            img = f[img_path][timestep]

            # Handle different storage formats
            if isinstance(img, bytes):
                # Decode from bytes (JPEG/PNG encoded)
                img_array = np.frombuffer(img, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV loads as BGR
            elif isinstance(img, np.ndarray):
                # Already a numpy array
                pass
            else:
                raise TypeError(f"Unexpected image type: {type(img)}")

            images.append(img)

        return images

    def _load_obs_images(self, f: h5py.File, action_start_timestep: int) -> List[List[np.ndarray]]:
        """
        Load observation images for n_obs_steps frames.

        n_obs_steps 的最后一帧对应 action_start_timestep。
        例如 n_obs_steps=2, action_start_timestep=10:
            加载 timestep 9 和 10 的图像

        Args:
            f: HDF5 file handle
            action_start_timestep: 动作序列的起始帧（也是观测序列的最后一帧）

        Returns:
            List of length n_obs_steps, each element is List of images per camera
            [[cam0_t0, cam1_t0, ...], [cam0_t1, cam1_t1, ...], ...]
        """
        obs_images = []

        # 观测序列: 从 (action_start - n_obs_steps + 1) 到 action_start (包含)
        obs_start = action_start_timestep - self.n_obs_steps + 1

        for t in range(obs_start, action_start_timestep + 1):
            # 左侧 padding: 当 t < 0 时，用第 0 帧重复填充
            t_clamped = max(0, t)
            frame_images = self._load_images(f, t_clamped)
            obs_images.append(frame_images)

        return obs_images


    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.

        Returns:
            Dictionary containing:
                - 'images': (n_obs_steps, num_cameras, 3, H, W) tensor - 多帧观测图像
                - 'actions': (future_action_window, action_dim) tensor - 动作序列
                - 'episode_idx': episode index
                - 'timestep': action start timestep within episode
        """
        episode_idx, action_start_timestep = self.indices[idx]
        episode_file = self.episode_files[episode_idx]

        with h5py.File(episode_file, 'r') as f:
            # Load actions starting from action_start_timestep
            actions = self._load_actions(f, action_start_timestep)  # (T, action_dim)

            # Load observation images for n_obs_steps frames
            # obs_images: List[List[np.ndarray]] - [n_obs_steps][num_cameras]
            obs_images = self._load_obs_images(f, action_start_timestep)

        # Convert to torch tensors and apply transforms
        action_tensor = torch.from_numpy(actions).float()

        # Process images: obs_images[t][cam] -> (n_obs_steps, num_cameras, 3, H, W)
        all_frame_tensors = []
        for frame_images in obs_images:  # 遍历每个时间步
            frame_tensors = []
            for img in frame_images:  # 遍历每个相机
                # Ensure uint8 format
                if img.dtype != np.uint8:
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)

                # Apply transform
                if self.transform is not None:
                    img_tensor = self.transform(img)  # (3, H, W)
                else:
                    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

                frame_tensors.append(img_tensor)

            # Stack cameras for this frame: (num_cameras, 3, H, W)
            frame_tensor = torch.stack(frame_tensors, dim=0)
            all_frame_tensors.append(frame_tensor)

        # Stack all frames: (n_obs_steps, num_cameras, 3, H, W)
        images_tensor = torch.stack(all_frame_tensors, dim=0)

        # 提取 state: action 的第0帧 = n_obs_steps 的最后一帧时刻的机器人状态
        # state 不参与噪音扩散，作为模型的额外条件输入
        state_tensor = action_tensor[0]  # (action_dim,)
        # 预测目标: action[1:] (future_action_window - 1 帧)
        actions_to_predict = action_tensor[1:]  # (future_action_window - 1, action_dim)

        return {
            'images': images_tensor,
            'state': state_tensor,
            'actions': actions_to_predict,
            'episode_idx': episode_idx,
            'timestep': action_start_timestep,
        }


class RobotDatasetLazy(Dataset):
    """
    Memory-efficient lazy loading version of RobotDataset.
    Opens HDF5 files only when needed and doesn't keep them in memory.

    Useful for very large datasets that don't fit in memory.
    """

    def __init__(self, *args, **kwargs):
        # Initialize with same arguments as RobotDataset
        self.dataset = RobotDataset(*args, **kwargs)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Directly call RobotDataset's __getitem__ which already does lazy loading
        return self.dataset[idx]


# Example usage and testing
if __name__ == "__main__":
    print("Testing RobotDataset...")

    # Example: Test dataset creation
    # Uncomment and modify path to test with actual data
    """
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    dataset = RobotDataset(
        data_path='path/to/robot/data',
        future_action_window=10,
        past_action_window=0,
        transform=transform,
        num_cameras=4,
        use_both_arms=False
    )

    print(f"Dataset size: {len(dataset)}")

    # Test loading a sample
    sample = dataset[0]
    print(f"Images shape: {sample['images'].shape}")
    print(f"Actions shape: {sample['actions'].shape}")
    print(f"Episode idx: {sample['episode_idx']}")
    print(f"Timestep: {sample['timestep']}")

    # Test dataloader
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2
    )

    for batch in dataloader:
        print(f"Batch images shape: {batch['images'].shape}")
        print(f"Batch actions shape: {batch['actions'].shape}")
        break

    print("Dataset test completed!")
    """

    print("RobotDataset implementation complete.")
    print("Uncomment the test code above and provide data path to test.")
