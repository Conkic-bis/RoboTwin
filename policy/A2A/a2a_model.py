"""Checkpoint loader + lightweight inference wrapper for A2A on RoboTwin.

Mirrors policy/DP/dp_model.py but delegates obs stacking to A2ARunner instead
of DPRunner. The checkpoint payload schema (cfg + state_dicts) is identical to
DP's, so this class loads a model produced by a2a_flow_matching/workspace/
a2a_workspace.py without any extra plumbing.
"""

import os
import sys

import dill
import hydra
import torch

CURRENT_FILE = os.path.abspath(__file__)
PARENT_DIR = os.path.dirname(CURRENT_FILE)
sys.path.insert(0, PARENT_DIR)

from a2a_flow_matching.env_runner.a2a_runner import A2ARunner  # noqa: E402
from a2a_flow_matching.workspace.a2a_workspace import A2AWorkspace  # noqa: E402


class A2A:
    def __init__(
        self,
        ckpt_file: str,
        n_obs_steps: int,
        n_action_steps: int,
        cam_keys=None,
        device: str = "cuda:0",
    ):
        self.policy = self._load_policy(ckpt_file, device)
        self.runner = A2ARunner(
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            cam_keys=cam_keys,
        )

    def update_obs(self, observation):
        self.runner.update_obs(observation)

    def reset_obs(self):
        self.runner.reset_obs()

    def get_action(self, observation=None):
        return self.runner.get_action(self.policy, observation)

    def get_last_obs(self):
        return self.runner.obs[-1]

    @staticmethod
    def _load_policy(ckpt_path: str, device: str):
        with open(ckpt_path, "rb") as f:
            payload = torch.load(f, pickle_module=dill, map_location="cpu")
        cfg = payload["cfg"]
        workspace_cls = hydra.utils.get_class(cfg._target_)
        workspace: A2AWorkspace = workspace_cls(cfg, output_dir=None)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        using_ema = bool(cfg.training.use_ema and workspace.ema_model is not None)
        policy = workspace.ema_model if using_ema else workspace.model

        # Surface the *actual* class & variant-defining hyperparameters from the
        # loaded ckpt so the user can verify which model was reconstructed.
        # RoboTwin's eval_policy.py prints "Policy Name: A2A" (= the folder
        # name), which is unrelated to the variant; this block disambiguates.
        try:
            policy_target = str(cfg.policy._target_)
            flow_target = str(cfg.policy.flow_matcher._target_)
            history_noise_std = float(getattr(cfg.policy, "history_noise_std", 0.0))
            print(
                f"[A2A] reconstructed policy class: {policy_target}\n"
                f"[A2A]   flow_matcher          : {flow_target}\n"
                f"[A2A]   history_noise_std     : {history_noise_std}\n"
                f"[A2A]   ema weights loaded    : {using_ema}\n"
                f"[A2A]   num_sampling_steps    : {int(cfg.policy.flow_matcher.num_sampling_steps)}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[A2A] (could not introspect cfg for variant info: {exc})")

        policy.to(torch.device(device))
        policy.eval()
        return policy
