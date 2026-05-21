"""Multi-camera observation buffer + chunked action retrieval for A2A.

Modelled on policy/DP/diffusion_policy/env_runner/dp_runner.py but parameterised
by the cam keys actually present in the policy's shape_meta, so it gracefully
handles head-only / head+left+right / head+left+right+front layouts.

A2A's predict_action returns
    {"action": (B, n_action_steps, D), "action_pred": (B, n_action_steps, D)}
The runner exposes get_action which yields the n_action_steps chunk for the
deploy_policy.eval() loop to iterate over.
"""

from collections import deque
from typing import Iterable, Optional

import numpy as np
import torch

from a2a_flow_matching.common.pytorch_util import dict_apply


class A2ARunner:
    def __init__(
        self,
        n_obs_steps: int = 8,
        n_action_steps: int = 8,
        cam_keys: Optional[Iterable[str]] = None,
    ):
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.cam_keys = tuple(cam_keys) if cam_keys is not None else ("head_cam", "left_cam", "right_cam")
        self.obs: deque = deque(maxlen=n_obs_steps + 1)

    # ---------- obs buffer ----------
    def reset_obs(self):
        self.obs.clear()

    def update_obs(self, current_obs):
        self.obs.append(current_obs)

    def _stack_last_n(self, all_obs, n_steps):
        all_obs = list(all_obs)
        head = all_obs[-1]
        if isinstance(head, np.ndarray):
            result = np.zeros((n_steps,) + head.shape, dtype=head.dtype)
            start_idx = -min(n_steps, len(all_obs))
            result[start_idx:] = np.array(all_obs[start_idx:])
            if n_steps > len(all_obs):
                result[:start_idx] = result[start_idx]
        elif isinstance(head, torch.Tensor):
            result = torch.zeros((n_steps,) + head.shape, dtype=head.dtype)
            start_idx = -min(n_steps, len(all_obs))
            result[start_idx:] = torch.stack(all_obs[start_idx:])
            if n_steps > len(all_obs):
                result[:start_idx] = result[start_idx]
        else:
            raise RuntimeError(f"unsupported obs type: {type(head)}")
        return result

    def _get_n_steps_obs(self):
        assert len(self.obs) > 0, "no observation recorded; call update_obs first"
        out = {}
        for key in self.obs[0].keys():
            out[key] = self._stack_last_n([o[key] for o in self.obs], self.n_obs_steps)
        return out

    # ---------- inference ----------
    def get_action(self, policy, observation=None):
        device, _ = policy.device, policy.dtype
        if observation is not None:
            self.obs.append(observation)
        obs = self._get_n_steps_obs()

        obs_dict = dict_apply(obs, lambda x: torch.from_numpy(x).to(device=device))

        obs_dict_input = {}
        for cam_key in self.cam_keys:
            if cam_key in obs_dict:
                obs_dict_input[cam_key] = obs_dict[cam_key].unsqueeze(0)
        obs_dict_input["agent_pos"] = obs_dict["agent_pos"].unsqueeze(0)

        with torch.no_grad():
            action_dict = policy.predict_action(obs_dict_input)

        np_action_dict = dict_apply(action_dict, lambda x: x.detach().to("cpu").numpy())
        action = np_action_dict["action"].squeeze(0)[: self.n_action_steps]
        return action
