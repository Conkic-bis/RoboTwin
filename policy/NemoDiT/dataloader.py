# Dataloader for Qwen3VL + Flow Matching policy.
#


import glob
import json
import os
import random
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.rotation_utils import convert_endpose_7d_to_9d


_EPISODE_NUM_RE = re.compile(r'episode(\d+)')


def _parse_episode_index(path: str) -> Optional[int]:
    """Extract the integer episode id from a filename like episode12.hdf5."""
    m = _EPISODE_NUM_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else None


class RobotDataset(Dataset):
    """
    RoboTwin HDF5 dataset adapted for the Qwen3-VL + Flow Matching pipeline.

    Each sample yields:
        - images_raw      : List[np.ndarray] uint8 — raw per-camera images
                            of the last obs frame, passed straight to the
                            Qwen3-VL processor (no ImageNet normalization).
        - images_tensor   : (n_obs_steps, num_cameras, 3, H, W) float tensor
                            (kept for downstream compatibility / fallback).
        - actions         : (future_action_window - 1, action_dim) target.
        - state           : (action_dim,) — first action frame used as the
                            clean state conditioning token.
        - instruction     : str.
        - episode_idx, timestep: ints (for debugging / logging).

    Args:
        data_path: directory containing episode*.hdf5.
        future_action_window: number of future action steps to load.
        past_action_window: kept for CLI compatibility (must be 0).
        transform: optional torchvision transform for ``images_tensor``.
        num_cameras: number of camera views to use.
        camera_names: override for camera key names.
        use_both_arms: dual-arm vs single-arm.
        action_type: 'endpose' or 'joint'.
        quat_convention: 'wxyz' or 'xyzw' (endpose only).
        n_obs_steps: observation window length (frames).
        instruction: fallback instruction when per-episode JSON is missing.
        instructions_path: directory containing episode{N}.json.
            - None / '' → auto-derive ``<data_path>/../instructions``.
            - 'disable' → fully off, always use ``instruction``.
            - else → use as-is.
        instruction_split: JSON split to sample from ('seen' | 'unseen').
        random_instruction: randomly pick one per sample when True.
    """

    def __init__(
        self,
        data_path: str,
        future_action_window: int = 13,
        past_action_window: int = 0,
        transform: Optional[Callable] = None,
        num_cameras: int = 4,
        camera_names: Optional[List[str]] = None,
        use_both_arms: bool = False,
        action_type: str = 'endpose',
        quat_convention: str = 'wxyz',
        n_obs_steps: int = 1,
        instruction: str = 'Predict the next robot actions.',
        instructions_path: Optional[str] = None,
        instruction_split: str = 'seen',
        random_instruction: bool = True,
    ):
        super().__init__()

        assert action_type in ('endpose', 'joint'), f"unknown action_type {action_type}"
        assert n_obs_steps >= 1, f"n_obs_steps must be >= 1, got {n_obs_steps}"
        assert past_action_window == 0, "past_action_window != 0 is not supported"

        self.data_path = data_path
        self.future_action_window = future_action_window
        self.past_action_window = past_action_window
        self.transform = transform
        self.num_cameras = num_cameras
        self.use_both_arms = use_both_arms
        self.action_type = action_type
        self.quat_convention = quat_convention
        self.n_obs_steps = n_obs_steps
        self.instruction = instruction
        self.instruction_split = instruction_split
        self.random_instruction = random_instruction

        if camera_names is None:
            camera_names = ['front_camera', 'head_camera', 'left_camera', 'right_camera']
        assert len(camera_names) >= num_cameras
        self.camera_names = camera_names[:num_cameras]

        self.episode_files = sorted(glob.glob(os.path.join(data_path, 'episode*.hdf5')))
        if not self.episode_files:
            raise ValueError(f"No episode files found in {data_path}")
        print(f"[RobotDataset] {len(self.episode_files)} episode files")

        # Resolve instructions directory policy.
        if instructions_path is None or instructions_path == '':
            instructions_path = os.path.join(
                os.path.dirname(os.path.abspath(data_path)), 'instructions',
            )
        elif instructions_path.lower() == 'disable':
            instructions_path = None
        self.instructions_path = instructions_path

        self.episode_instructions: List[List[str]] = self._load_episode_instructions()
        n_with = sum(1 for lst in self.episode_instructions if lst)
        if n_with:
            print(
                f"[RobotDataset] instructions loaded for {n_with}/{len(self.episode_files)} "
                f"episodes (split='{self.instruction_split}', "
                f"random={self.random_instruction})"
            )
        else:
            print(f"[RobotDataset] no per-episode instructions; using fallback")

        # _build_indices opens and closes HDF5 files with a `with` context
        # so no open handles survive past __init__ — fork-safe.
        self.indices = self._build_indices()
        print(f"[RobotDataset] {len(self.indices)} samples  (action_type={action_type})")

        # Fix A — the None sentinel. Handles are opened lazily inside
        # __getitem__ the first time each worker accesses a file, so handles
        # never cross the fork boundary. Declared here only so that
        # *accidental* eager opens in __init__ would crash fast (NoneType
        # subscript), acting as a type-level guard rail.
        self._hdf5_handles: Optional[Dict[int, h5py.File]] = None

    # -------------------------------------------------------------- #
    # Instruction handling
    # -------------------------------------------------------------- #

    def _load_episode_instructions(self) -> List[List[str]]:
        per_episode: List[List[str]] = [[] for _ in self.episode_files]
        if not self.instructions_path or not os.path.isdir(self.instructions_path):
            return per_episode

        for ep_idx, ep_file in enumerate(self.episode_files):
            ep_num = _parse_episode_index(ep_file)
            if ep_num is None:
                continue
            json_path = os.path.join(self.instructions_path, f'episode{ep_num}.json')
            if not os.path.isfile(json_path):
                continue
            try:
                with open(json_path, 'r', encoding='utf-8') as jf:
                    data = json.load(jf)
            except (OSError, json.JSONDecodeError) as exc:
                print(f"[RobotDataset] failed to read {json_path}: {exc}")
                continue

            strings: List[str] = []
            if isinstance(data, dict):
                chosen = data.get(self.instruction_split) or data.get('seen')
                if isinstance(chosen, list):
                    strings = [s for s in chosen if isinstance(s, str) and s.strip()]
            elif isinstance(data, list):
                strings = [s for s in data if isinstance(s, str) and s.strip()]
            per_episode[ep_idx] = strings

        return per_episode

    def _get_instruction(self, episode_idx: int) -> str:
        pool = (
            self.episode_instructions[episode_idx]
            if episode_idx < len(self.episode_instructions) else []
        )
        if pool:
            if self.random_instruction and len(pool) > 1:
                return random.choice(pool)
            return pool[0]
        return self.instruction

    # -------------------------------------------------------------- #
    # Indexing
    # -------------------------------------------------------------- #

    def _build_indices(self) -> List[tuple]:
        indices: List[tuple] = []
        for ep_idx, ep_file in enumerate(self.episode_files):
            with h5py.File(ep_file, 'r') as f:
                if self.action_type == 'endpose':
                    data_key = 'endpose/left_endpose'
                else:
                    data_key = 'joint_action/left_arm'
                ep_len = f[data_key].shape[0]
            for t in range(ep_len):
                indices.append((ep_idx, t))
        return indices

    def __len__(self) -> int:
        return len(self.indices)

    # -------------------------------------------------------------- #
    # Per-worker persistent HDF5 handles (Fix A core)
    # -------------------------------------------------------------- #

    def _get_h5(self, episode_idx: int) -> h5py.File:
        """Return an open HDF5 file handle, opening it lazily on first use.

        The ``_hdf5_handles`` dict is created inside the worker process so
        that per-file chunk caches and BTree state are strictly process-
        private. h5py is not fork-safe: sharing the same libhdf5 state
        across forked children leads to lseek races on the shared kernel
        file offset and corrupted reads.
        """
        handles = self._hdf5_handles
        if handles is None:
            handles = {}
            self._hdf5_handles = handles

        h = handles.get(episode_idx)
        if h is None:
            # libver='latest' enables cached chunk lookups on SWMR-layout
            # files. swmr=False because we are read-only and don't need
            # concurrent writer coordination.
            h = h5py.File(
                self.episode_files[episode_idx], 'r',
                libver='latest', swmr=False,
            )
            handles[episode_idx] = h
        return h

    def __del__(self):
        handles = getattr(self, '_hdf5_handles', None) or {}
        for h in handles.values():
            try:
                h.close()
            except Exception:
                pass

    # -------------------------------------------------------------- #
    # Loading primitives
    # -------------------------------------------------------------- #

    def _load_actions(self, f: h5py.File, start_idx: int) -> np.ndarray:
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
        lep = f['endpose/left_endpose'][start_idx:start_idx + self.future_action_window]
        lgr = f['endpose/left_gripper'][start_idx:start_idx + self.future_action_window]
        l9d = convert_endpose_7d_to_9d(lep, quat_convention=self.quat_convention)
        if lgr.ndim == 1:
            lgr = lgr[:, np.newaxis]

        if self.use_both_arms:
            rep = f['endpose/right_endpose'][start_idx:start_idx + self.future_action_window]
            rgr = f['endpose/right_gripper'][start_idx:start_idx + self.future_action_window]
            r9d = convert_endpose_7d_to_9d(rep, quat_convention=self.quat_convention)
            if rgr.ndim == 1:
                rgr = rgr[:, np.newaxis]
            return np.concatenate([l9d, lgr, r9d, rgr], axis=-1)
        return np.concatenate([l9d, lgr], axis=-1)

    def _load_joint_actions(self, f: h5py.File, start_idx: int) -> np.ndarray:
        lj = f['joint_action/left_arm'][start_idx:start_idx + self.future_action_window]
        lg = f['joint_action/left_gripper'][start_idx:start_idx + self.future_action_window]
        if lj.ndim == 1:
            lj = lj[np.newaxis, :]
        if lg.ndim == 1:
            lg = lg[:, np.newaxis]

        if self.use_both_arms:
            rj = f['joint_action/right_arm'][start_idx:start_idx + self.future_action_window]
            rg = f['joint_action/right_gripper'][start_idx:start_idx + self.future_action_window]
            if rj.ndim == 1:
                rj = rj[np.newaxis, :]
            if rg.ndim == 1:
                rg = rg[:, np.newaxis]
            return np.concatenate([lj, lg, rj, rg], axis=-1)
        return np.concatenate([lj, lg], axis=-1)

    def _load_image(self, f: h5py.File, cam_name: str, timestep: int) -> np.ndarray:
        img_path = f'observation/{cam_name}/rgb'
        img = f[img_path][timestep]
        if isinstance(img, bytes):
            arr = np.frombuffer(img, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(img, np.ndarray):
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        else:
            raise TypeError(f"unexpected image type: {type(img)}")
        return img

    def _load_obs_images(
        self, f: h5py.File, action_start_timestep: int,
    ) -> List[List[np.ndarray]]:
        """Load n_obs_steps frames * num_cameras images.

        Left-pad with the first frame when action_start_timestep is at the
        beginning of the episode.
        """
        obs_start = action_start_timestep - self.n_obs_steps + 1
        frames: List[List[np.ndarray]] = []
        for t in range(obs_start, action_start_timestep + 1):
            t_clamped = max(0, t)
            frames.append([
                self._load_image(f, cam, t_clamped) for cam in self.camera_names
            ])
        return frames

    # -------------------------------------------------------------- #
    # Sample assembly
    # -------------------------------------------------------------- #

    def __getitem__(self, idx: int) -> Dict[str, object]:
        episode_idx, action_start_timestep = self.indices[idx]
        f = self._get_h5(episode_idx)

        actions = self._load_actions(f, action_start_timestep)
        obs_images = self._load_obs_images(f, action_start_timestep)

        action_tensor = torch.from_numpy(actions).float()
        state_tensor = action_tensor[0]                # (action_dim,)
        actions_to_predict = action_tensor[1:]         # (T-1, action_dim)

        # images_raw: last-obs-step camera images for the VLM processor
        images_raw = obs_images[-1]

        # images_tensor: full (n_obs_steps, num_cameras, 3, H, W) stack
        all_frame_tensors = []
        for frame_images in obs_images:
            frame_tensors = []
            for img in frame_images:
                if self.transform is not None:
                    t = self.transform(img)
                else:
                    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                frame_tensors.append(t)
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
    """Stack tensor fields; keep variable-shape fields as plain lists."""
    return {
        'images_raw': [item['images_raw'] for item in batch],
        'images_tensor': torch.stack([item['images_tensor'] for item in batch]),
        'actions': torch.stack([item['actions'] for item in batch]),
        'state': torch.stack([item['state'] for item in batch]),
        'instruction': [item['instruction'] for item in batch],
        'episode_idx': [item['episode_idx'] for item in batch],
        'timestep': [item['timestep'] for item in batch],
    }


# Back-compat alias.
RobotDatasetLazy = RobotDataset
Qwen3VLRobotDataset = RobotDataset
