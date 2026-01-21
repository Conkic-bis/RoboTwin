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
    │   └── /observation
    │       ├── /observation/front_camera/rgb   (T, H, W, 3)
    │       ├── /observation/head_camera/rgb    (T, H, W, 3)
    │       ├── /observation/left_camera/rgb    (T, H, W, 3)
    │       └── /observation/right_camera/rgb   (T, H, W, 3)

    Output action format (after conversion):
        - Single arm: (T, 10) = 3D translation + 6D rot6d + 1D gripper
        - Dual arm:   (T, 20) = (3D + 6D + 1D) * 2

    Args:
        data_path: Path to directory containing episode HDF5 files
        future_action_window: Number of future action steps to predict
        past_action_window: Number of past action steps as context
        transform: Optional transform to apply to images
        num_cameras: Number of camera views to use (default: 4)
        camera_names: List of camera names to use (default: all 4 cameras)
        use_both_arms: Whether to use both arms (default: False, only left arm)
        action_mode: 'absolute' or 'relative' actions
        quat_convention: Quaternion convention in HDF5 data, "wxyz" or "xyzw"
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
        action_mode: str = 'absolute',
        quat_convention: str = 'wxyz',
    ):
        super().__init__()

        self.data_path = data_path
        self.future_action_window = future_action_window
        self.past_action_window = past_action_window
        self.transform = transform
        self.num_cameras = num_cameras
        self.use_both_arms = use_both_arms
        self.action_mode = action_mode
        self.quat_convention = quat_convention

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
        Build valid indices for sampling.

        Returns:
            List of (episode_idx, start_timestep) tuples
        """
        indices = []

        for ep_idx, ep_file in enumerate(self.episode_files):
            with h5py.File(ep_file, 'r') as f:
                # Get episode length from action data
                left_endpose = f['endpose/left_endpose']
                episode_length = left_endpose.shape[0]

                # Valid start timesteps
                # Need past_action_window before and future_action_window after
                for t in range(self.past_action_window,
                              episode_length - self.future_action_window + 1):
                    indices.append((ep_idx, t))

        return indices

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.indices)

    def _load_actions(self, f: h5py.File, start_idx: int) -> np.ndarray:
        """
        Load action sequences from HDF5 file and convert to rot6d representation.

        Args:
            f: Open HDF5 file handle
            start_idx: Starting timestep index

        Returns:
            actions: (future_action_window, action_dim) numpy array
                     Single arm: (T, 10) = 3D translation + 6D rot6d + 1D gripper
                     Dual arm:   (T, 20) = (3D + 6D + 1D) * 2
        """
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

    def _load_images(self, f: h5py.File, timestep: int) -> List[np.ndarray]:
        """
        Load images from all cameras at a given timestep.
        """

        images = []

        for cam_name in self.camera_names:
            img_path = f'observation/{cam_name}/rgb'
            img = f[img_path][timestep]
            
            # Debug: print type and shape
            print(f"[DEBUG] {cam_name} type: {type(img)}, ", end="")
            
            # Handle different storage formats
            if isinstance(img, bytes):
                # Decode from bytes (JPEG/PNG encoded)
                img_array = np.frombuffer(img, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV loads as BGR
                print(f"decoded shape: {img.shape}")
            elif isinstance(img, np.ndarray):
                print(f"shape: {img.shape}, dtype: {img.dtype}")
            else:
                raise TypeError(f"Unexpected image type: {type(img)}")
            
            images.append(img)

        return images


    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            """
            Get a sample from the dataset.

            Returns:
                Dictionary containing:
                    - 'images': (num_cameras, 3, H, W) tensor
                    - 'actions': (future_action_window, action_dim) tensor
                    - 'episode_idx': episode index
                    - 'timestep': timestep within episode
            """
            episode_idx, timestep = self.indices[idx]
            episode_file = self.episode_files[episode_idx]

            with h5py.File(episode_file, 'r') as f:
                # Load actions
                actions = self._load_actions(f, timestep)  # (T, action_dim)

                # Load images from current timestep
                images = self._load_images(f, timestep)  # List of (H, W, 3)

            # Convert to torch tensors and apply transforms
            action_tensor = torch.from_numpy(actions).float()

            # Process images
            image_tensors = []
            for img in images:
                # Debug
                print(f"[DEBUG] Processing img type: {type(img)}, shape: {img.shape}, dtype: {img.dtype}")
                
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

                image_tensors.append(img_tensor)

            # Stack images: (num_cameras, 3, H, W)
            images_tensor = torch.stack(image_tensors, dim=0)

            return {
                'images': images_tensor,
                'actions': action_tensor,
                'episode_idx': episode_idx,
                'timestep': timestep,
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
