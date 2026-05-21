"""RoboTwin-multi-camera variant of A2A's RobotImageDataset.

Reads a zarr produced by policy/A2A/process_data.py with groups:
    data/head_camera  : (N, 3, H, W) uint8
    data/left_camera  : (N, 3, H, W) uint8
    data/right_camera : (N, 3, H, W) uint8
    data/state        : (N, D)       float32   -- joint_action.vector[t]
    data/action       : (N, D)       float32   -- joint_action.vector[t+1]
    meta/episode_ends : (E,)         int64

Yields, per sample, a sequence of length `horizon`:
    obs:
        head_cam  : (T, 3, H, W) float32 in [0,1]
        left_cam  : (T, 3, H, W) float32 in [0,1]
        right_cam : (T, 3, H, W) float32 in [0,1]
        agent_pos : (T, D)       float32
    action        : (T, D)       float32
"""

import copy
from typing import Dict, List

import numba
import numpy as np
import torch

from a2a_flow_matching.common.normalize_util import get_image_range_normalizer
from a2a_flow_matching.common.normalizer import LinearNormalizer
from a2a_flow_matching.common.pytorch_util import dict_apply
from a2a_flow_matching.common.replay_buffer import ReplayBuffer
from a2a_flow_matching.common.sampler import (
    SequenceSampler,
    downsample_mask,
    get_val_mask,
)
from a2a_flow_matching.dataset.base_dataset import BaseImageDataset


DEFAULT_CAM_KEYS: List[str] = ["head_camera", "left_camera", "right_camera"]
CAM_KEY_TO_OBS_KEY = {
    "head_camera": "head_cam",
    "left_camera": "left_cam",
    "right_camera": "right_cam",
    "front_camera": "front_cam",
}


class RobotImageDataset(BaseImageDataset):
    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        batch_size=64,
        max_train_episodes=None,
        cam_keys: List[str] = None,
    ):
        super().__init__()

        if cam_keys is None:
            cam_keys = list(DEFAULT_CAM_KEYS)

        # Open just to discover which cameras are actually present.
        import zarr  # local import: avoid hard dep at module load time
        with zarr.open(zarr_path, mode="r") as root:
            available = set(root["data"].keys())
        present_cams = [k for k in cam_keys if k in available]
        if not present_cams:
            raise ValueError(
                f"No camera streams from {cam_keys} found in {zarr_path}; "
                f"available data keys: {sorted(available)}"
            )
        self.cam_keys = present_cams

        load_keys = self.cam_keys + ["state", "action"]
        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=load_keys)

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        self.batch_size = batch_size
        sequence_length = self.sampler.sequence_length
        self.buffers = {
            k: np.zeros((batch_size, sequence_length, *v.shape[1:]), dtype=v.dtype)
            for k, v in self.sampler.replay_buffer.items()
        }
        self.buffers_torch = {k: torch.from_numpy(v) for k, v in self.buffers.items()}
        for v in self.buffers_torch.values():
            v.pin_memory()

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "action": self.replay_buffer["action"],
            "agent_pos": self.replay_buffer["state"],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        img_norm = get_image_range_normalizer()
        for cam in self.cam_keys:
            normalizer[CAM_KEY_TO_OBS_KEY[cam]] = img_norm
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        obs = {}
        for cam in self.cam_keys:
            obs_key = CAM_KEY_TO_OBS_KEY[cam]
            obs[obs_key] = sample[cam].astype(np.float32) / 255.0  # T,3,H,W
        obs["agent_pos"] = sample["state"].astype(np.float32)
        return {"obs": obs, "action": sample["action"].astype(np.float32)}

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        if isinstance(idx, slice):
            raise NotImplementedError
        elif isinstance(idx, int):
            sample = self.sampler.sample_sequence(idx)
            sample = dict_apply(sample, torch.from_numpy)
            return sample
        elif isinstance(idx, np.ndarray):
            assert len(idx) == self.batch_size
            for k, v in self.sampler.replay_buffer.items():
                batch_sample_sequence(
                    self.buffers[k],
                    v,
                    self.sampler.indices,
                    idx,
                    self.sampler.sequence_length,
                )
            return self.buffers_torch
        else:
            raise ValueError(idx)

    def postprocess(self, samples, device):
        agent_pos = samples["state"].to(device, non_blocking=True)
        action = samples["action"].to(device, non_blocking=True)
        obs = {"agent_pos": agent_pos}
        for cam in self.cam_keys:
            obs_key = CAM_KEY_TO_OBS_KEY[cam]
            obs[obs_key] = samples[cam].to(device, non_blocking=True) / 255.0
        return {"obs": obs, "action": action}


def _batch_sample_sequence(
    data: np.ndarray,
    input_arr: np.ndarray,
    indices: np.ndarray,
    idx: np.ndarray,
    sequence_length: int,
):
    for i in numba.prange(len(idx)):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = indices[idx[i]]
        data[i, sample_start_idx:sample_end_idx] = input_arr[buffer_start_idx:buffer_end_idx]
        if sample_start_idx > 0:
            data[i, :sample_start_idx] = data[i, sample_start_idx]
        if sample_end_idx < sequence_length:
            data[i, sample_end_idx:] = data[i, sample_end_idx - 1]


_batch_sample_sequence_sequential = numba.jit(_batch_sample_sequence, nopython=True, parallel=False)
_batch_sample_sequence_parallel = numba.jit(_batch_sample_sequence, nopython=True, parallel=True)


def batch_sample_sequence(
    data: np.ndarray,
    input_arr: np.ndarray,
    indices: np.ndarray,
    idx: np.ndarray,
    sequence_length: int,
):
    batch_size = len(idx)
    assert data.shape == (batch_size, sequence_length, *input_arr.shape[1:])
    if batch_size >= 16 and data.nbytes // batch_size >= 2 ** 16:
        _batch_sample_sequence_parallel(data, input_arr, indices, idx, sequence_length)
    else:
        _batch_sample_sequence_sequential(data, input_arr, indices, idx, sequence_length)
