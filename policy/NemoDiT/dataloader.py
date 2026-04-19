# Dataloader for Qwen3VL + Flow Matching Policy
#
# Extends the NemoDiT dataloader to also provide task instructions
# and PIL images suitable for the Qwen3-VL processor.
#
# The key difference from NemoDiT's dataloader:
#   - Returns raw PIL images (no ImageNet normalization) for Qwen3-VL processor
#   - Includes task instruction text for VLM conditioning
#   - Supports pre-tokenized VLM inputs for training efficiency

import os
import re
import json
import random
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Callable, List, Dict
import glob
import cv2

from utils.rotation_utils import convert_endpose_7d_to_9d


_EPISODE_NUM_RE = re.compile(r'episode(\d+)')


def _parse_episode_index(path: str) -> Optional[int]:
    """Extract the integer episode id from a filename like ``episode12.hdf5``."""
    m = _EPISODE_NUM_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else None


class Qwen3VLRobotDataset(Dataset):
    """
    Dataset for Qwen3VL + Flow Matching policy training.

    Returns both raw images (for VLM) and action data (for flow matching).

    HDF5 structure:
        episode_X.hdf5
        ├── /endpose/            (T, 7) per arm
        ├── /joint_action/       (T, 6+1) per arm
        └── /observation/
            ├── front_camera/rgb (T, H, W, 3)
            ├── head_camera/rgb  (T, H, W, 3)
            ├── left_camera/rgb  (T, H, W, 3)
            └── right_camera/rgb (T, H, W, 3)

    Args:
        data_path: Path to HDF5 episode directory
        future_action_window: Number of future action steps to predict
        num_cameras: Number of camera views
        use_both_arms: Use dual arm mode
        action_type: 'endpose' or 'joint'
        quat_convention: 'wxyz' or 'xyzw'
        n_obs_steps: Number of observation history steps
        instruction: Fallback task instruction if per-episode JSON files are
            unavailable.
        camera_names: Camera name list
        instructions_path: Optional path to a directory containing per-episode
            JSON instruction files (``episode{N}.json``). Each JSON is expected
            to have ``"seen"`` / ``"unseen"`` keys with a list of instruction
            strings. If ``None``, it is auto-derived as
            ``<data_path>/../instructions``. Pass an empty string to fully
            disable auto-loading.
        instruction_split: Which split in the JSON to sample from during
            training (``"seen"`` by default).
        random_instruction: If ``True`` (default when instructions are
            available), randomly pick one instruction per sample; otherwise
            always use the first one.
    """

    def __init__(
        self,
        data_path: str,
        future_action_window: int = 13,
        num_cameras: int = 4,
        use_both_arms: bool = True,
        action_type: str = 'joint',
        quat_convention: str = 'wxyz',
        n_obs_steps: int = 1,
        instruction: str = "Predict the next robot actions.",
        camera_names: Optional[List[str]] = None,
        instructions_path: Optional[str] = None,
        instruction_split: str = 'seen',
        random_instruction: bool = True,
    ):
        super().__init__()

        self.data_path = data_path
        self.future_action_window = future_action_window
        self.num_cameras = num_cameras
        self.use_both_arms = use_both_arms
        self.action_type = action_type
        self.quat_convention = quat_convention
        self.n_obs_steps = n_obs_steps
        self.instruction = instruction
        self.instruction_split = instruction_split
        self.random_instruction = random_instruction

        assert action_type in ['endpose', 'joint']
        assert n_obs_steps >= 1

        if camera_names is None:
            self.camera_names = ['front_camera', 'head_camera', 'left_camera', 'right_camera']
        else:
            self.camera_names = camera_names
        self.camera_names = self.camera_names[:num_cameras]

        # Load episode files
        self.episode_files = sorted(glob.glob(os.path.join(data_path, "episode*.hdf5")))
        if len(self.episode_files) == 0:
            raise ValueError(f"No episode files found in {data_path}")
        print(f"Found {len(self.episode_files)} episode files")

        # Resolve instructions directory.
        #   - None / empty string -> auto-derive as <data_path>/../instructions
        #   - "disable" (case-insensitive) -> turn off auto-loading entirely
        #   - any other value -> use it as-is
        if instructions_path is None or instructions_path == '':
            instructions_path = os.path.join(os.path.dirname(os.path.abspath(data_path)),
                                             'instructions')
        elif instructions_path.lower() == 'disable':
            instructions_path = None
        self.instructions_path = instructions_path

        # Pre-load per-episode instruction lists (indexed by dataset episode_idx).
        self.episode_instructions: List[List[str]] = self._load_episode_instructions()
        num_with_instr = sum(1 for lst in self.episode_instructions if lst)
        if num_with_instr > 0:
            print(f"Loaded instructions for {num_with_instr}/{len(self.episode_files)} "
                  f"episodes from '{self.instructions_path}' (split='{self.instruction_split}', "
                  f"random={self.random_instruction})")
        else:
            print(f"No per-episode instructions found; falling back to single instruction.")

        # Build indices
        self.indices = self._build_indices()
        print(f"Total valid samples: {len(self.indices)}")
        print(f"Action type: {action_type}, Fallback instruction: '{instruction[:60]}...'")

    def _load_episode_instructions(self) -> List[List[str]]:
        """Map each episode in ``self.episode_files`` to a list of instruction
        strings loaded from ``<instructions_path>/episode{N}.json``.

        Returns a list of length ``len(self.episode_files)``; entries are empty
        lists when no JSON file exists for that episode.
        """
        per_episode: List[List[str]] = [[] for _ in self.episode_files]
        if not self.instructions_path or not os.path.isdir(self.instructions_path):
            return per_episode

        for ep_idx, ep_file in enumerate(self.episode_files):
            episode_num = _parse_episode_index(ep_file)
            if episode_num is None:
                continue
            json_path = os.path.join(self.instructions_path, f'episode{episode_num}.json')
            if not os.path.isfile(json_path):
                continue
            try:
                with open(json_path, 'r', encoding='utf-8') as jf:
                    data = json.load(jf)
            except (OSError, json.JSONDecodeError) as e:
                print(f"[Qwen3VLRobotDataset] Failed to read {json_path}: {e}")
                continue

            instructions: List[str] = []
            if isinstance(data, dict):
                if self.instruction_split in data and isinstance(data[self.instruction_split], list):
                    instructions = [s for s in data[self.instruction_split] if isinstance(s, str) and s.strip()]
                elif 'seen' in data and isinstance(data['seen'], list):
                    instructions = [s for s in data['seen'] if isinstance(s, str) and s.strip()]
            elif isinstance(data, list):
                instructions = [s for s in data if isinstance(s, str) and s.strip()]

            per_episode[ep_idx] = instructions

        return per_episode

    def _get_instruction(self, episode_idx: int) -> str:
        """Return an instruction for the given episode, sampling randomly when
        multiple are available."""
        candidates = self.episode_instructions[episode_idx] if episode_idx < len(self.episode_instructions) else []
        if candidates:
            if self.random_instruction and len(candidates) > 1:
                return random.choice(candidates)
            return candidates[0]
        return self.instruction

    def _build_indices(self) -> List[tuple]:
        """Build (episode_idx, timestep) indices. All timesteps valid with padding."""
        indices = []
        for ep_idx, ep_file in enumerate(self.episode_files):
            with h5py.File(ep_file, 'r') as f:
                if self.action_type == 'endpose':
                    data_key = 'endpose/left_endpose'
                else:
                    data_key = 'joint_action/left_arm'
                episode_length = f[data_key].shape[0]
                for t in range(episode_length):
                    indices.append((ep_idx, t))
        return indices

    def __len__(self) -> int:
        return len(self.indices)

    def _load_actions(self, f: h5py.File, start_idx: int) -> np.ndarray:
        """Load and pad action sequence."""
        if self.action_type == 'endpose':
            actions = self._load_endpose_actions(f, start_idx)
        else:
            actions = self._load_joint_actions(f, start_idx)

        if actions.shape[0] < self.future_action_window:
            pad_len = self.future_action_window - actions.shape[0]
            padding = np.repeat(actions[-1:], pad_len, axis=0)
            actions = np.concatenate([actions, padding], axis=0)

        return actions

    def _load_endpose_actions(self, f: h5py.File, start_idx: int) -> np.ndarray:
        """Load endpose actions with rot6d conversion."""
        left_ep = f['endpose/left_endpose'][start_idx:start_idx + self.future_action_window]
        left_gr = f['endpose/left_gripper'][start_idx:start_idx + self.future_action_window]
        left_9d = convert_endpose_7d_to_9d(left_ep, quat_convention=self.quat_convention)
        if left_gr.ndim == 1:
            left_gr = left_gr[:, np.newaxis]

        if self.use_both_arms:
            right_ep = f['endpose/right_endpose'][start_idx:start_idx + self.future_action_window]
            right_gr = f['endpose/right_gripper'][start_idx:start_idx + self.future_action_window]
            right_9d = convert_endpose_7d_to_9d(right_ep, quat_convention=self.quat_convention)
            if right_gr.ndim == 1:
                right_gr = right_gr[:, np.newaxis]
            return np.concatenate([left_9d, left_gr, right_9d, right_gr], axis=-1)
        else:
            return np.concatenate([left_9d, left_gr], axis=-1)

    def _load_joint_actions(self, f: h5py.File, start_idx: int) -> np.ndarray:
        """Load joint angle actions."""
        left_j = f['joint_action/left_arm'][start_idx:start_idx + self.future_action_window]
        left_g = f['joint_action/left_gripper'][start_idx:start_idx + self.future_action_window]
        if left_j.ndim == 1:
            left_j = left_j[np.newaxis, :]
        if left_g.ndim == 1:
            left_g = left_g[:, np.newaxis]

        if self.use_both_arms:
            right_j = f['joint_action/right_arm'][start_idx:start_idx + self.future_action_window]
            right_g = f['joint_action/right_gripper'][start_idx:start_idx + self.future_action_window]
            if right_j.ndim == 1:
                right_j = right_j[np.newaxis, :]
            if right_g.ndim == 1:
                right_g = right_g[:, np.newaxis]
            return np.concatenate([left_j, left_g, right_j, right_g], axis=-1)
        else:
            return np.concatenate([left_j, left_g], axis=-1)

    def _load_image(self, f: h5py.File, cam_name: str, timestep: int) -> np.ndarray:
        """Load a single camera image as uint8 RGB array."""
        img_path = f'observation/{cam_name}/rgb'
        img = f[img_path][timestep]

        if isinstance(img, bytes):
            img_array = np.frombuffer(img, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(img, np.ndarray):
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)

        return img

    def _load_obs_images(self, f: h5py.File, action_start_timestep: int) -> List[List[np.ndarray]]:
        """
        Load observation images for n_obs_steps frames.
        Returns: [[cam0_t0, cam1_t0, ...], [cam0_t1, cam1_t1, ...], ...]
        """
        obs_images = []
        obs_start = action_start_timestep - self.n_obs_steps + 1

        for t in range(obs_start, action_start_timestep + 1):
            t_clamped = max(0, t)
            frame_images = [self._load_image(f, cam, t_clamped) for cam in self.camera_names]
            obs_images.append(frame_images)

        return obs_images

    def __getitem__(self, idx: int) -> Dict[str, any]:
        """
        Get a training sample.

        Returns:
            Dictionary containing:
                - 'images_raw': List[np.ndarray] - raw images for VLM (last obs step, all cameras)
                - 'images_tensor': (n_obs_steps, num_cameras, 3, H, W) - normalized for fallback
                - 'actions': (future_action_window - 1, action_dim) - target actions
                - 'state': (action_dim,) - current robot state
                - 'instruction': str - task instruction
                - 'episode_idx': int
                - 'timestep': int
        """
        episode_idx, action_start_timestep = self.indices[idx]
        episode_file = self.episode_files[episode_idx]

        with h5py.File(episode_file, 'r') as f:
            actions = self._load_actions(f, action_start_timestep)
            obs_images = self._load_obs_images(f, action_start_timestep)

        # Convert actions to tensor
        action_tensor = torch.from_numpy(actions).float()
        state_tensor = action_tensor[0]  # (action_dim,)
        actions_to_predict = action_tensor[1:]  # (T-1, action_dim)

        # Raw images for VLM (last observation step, all cameras)
        # These are uint8 numpy arrays that will be processed by Qwen3-VL processor
        images_raw = obs_images[-1]  # List[np.ndarray], len = num_cameras

        # Also prepare normalized tensor images (for compatibility / fallback)
        all_frame_tensors = []
        for frame_images in obs_images:
            frame_tensors = []
            for img in frame_images:
                img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                frame_tensors.append(img_t)
            all_frame_tensors.append(torch.stack(frame_tensors, dim=0))
        images_tensor = torch.stack(all_frame_tensors, dim=0)

        return {
            'images_raw': images_raw,
            'images_tensor': images_tensor,
            'actions': actions_to_predict,
            'state': state_tensor,
            'instruction': self._get_instruction(episode_idx),
            'episode_idx': episode_idx,
            'timestep': action_start_timestep,
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for Qwen3VL dataset.

    Handles mixed types (raw images as lists, tensors for stacking).
    """
    collated = {
        'images_raw': [item['images_raw'] for item in batch],  # List[List[np.ndarray]]
        'images_tensor': torch.stack([item['images_tensor'] for item in batch]),
        'actions': torch.stack([item['actions'] for item in batch]),
        'state': torch.stack([item['state'] for item in batch]),
        'instruction': [item['instruction'] for item in batch],
        'episode_idx': [item['episode_idx'] for item in batch],
        'timestep': [item['timestep'] for item in batch],
    }
    return collated
