"""RoboTwin evaluation entrypoint for A2A.

Conforms to the contract enforced by script/eval_policy.py:
    encode_obs(observation) -> dict suitable for the policy
    get_model(usr_args)     -> stateful model with .get_action / .update_obs / .reset_obs
    eval(TASK_ENV, model, observation)  -> rollout one action chunk
    reset_model(model)      -> per-episode reset hook
"""

import glob
import os
import re

import numpy as np
import yaml

from .a2a_model import A2A


CAM_KEYS = ("head_cam", "left_cam", "right_cam")


def _resolve_ckpt(ckpt_dir, ckpt_num):
    """Return the requested checkpoint, or fall back to the latest available.

    Training writes <epoch>.ckpt at every `checkpoint_every` interval plus the
    final epoch (e.g. 50, 100, ..., 1000). A smoke-test / shorter run won't have
    1000.ckpt, so rather than failing with an opaque torch.load error we pick
    the highest-numbered checkpoint actually present.
    """
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(
            f"[A2A] checkpoint directory does not exist:\n  {ckpt_dir}\n"
            f"Make sure ckpt_setting/expert_data_num/seed match the values "
            f"used by train.sh (ckpt_setting must equal the training task_config)."
        )

    exact = os.path.join(ckpt_dir, f"{ckpt_num}.ckpt")
    if os.path.isfile(exact):
        return exact

    candidates = []
    for path in glob.glob(os.path.join(ckpt_dir, "*.ckpt")):
        stem = os.path.splitext(os.path.basename(path))[0]
        if re.fullmatch(r"\d+", stem):
            candidates.append((int(stem), path))
    if not candidates:
        raise FileNotFoundError(
            f"[A2A] no <epoch>.ckpt found in:\n  {ckpt_dir}\n"
            f"Directory exists but contains no numbered checkpoints."
        )

    latest_epoch, latest_path = max(candidates, key=lambda x: x[0])
    print(
        f"[A2A] requested checkpoint {ckpt_num}.ckpt not found; "
        f"falling back to latest available: {latest_epoch}.ckpt"
    )
    return latest_path


def encode_obs(observation):
    head = np.moveaxis(observation["observation"]["head_camera"]["rgb"], -1, 0) / 255.0
    left = np.moveaxis(observation["observation"]["left_camera"]["rgb"], -1, 0) / 255.0
    right = np.moveaxis(observation["observation"]["right_camera"]["rgb"], -1, 0) / 255.0
    obs = {
        "head_cam": head,
        "left_cam": left,
        "right_cam": right,
        "agent_pos": observation["joint_action"]["vector"],
    }
    return obs


def get_model(usr_args):
    task_name = usr_args["task_name"]
    ckpt_setting = usr_args["ckpt_setting"]
    expert_data_num = usr_args["expert_data_num"]
    seed = usr_args["seed"]
    ckpt_num = usr_args["checkpoint_num"]

    ckpt_dir = (
        f"./policy/A2A/checkpoints/"
        f"{task_name}-{ckpt_setting}-{expert_data_num}-{seed}"
    )
    ckpt_file = _resolve_ckpt(ckpt_dir, ckpt_num)
    print(f"[A2A] loading checkpoint: {ckpt_file}")

    # n_obs_steps / n_action_steps come straight from the policy yaml — they're
    # not embodiment-dependent so a single config suffices.
    config_path = "./policy/A2A/a2a_flow_matching/config/robot_a2a.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return A2A(
        ckpt_file,
        n_obs_steps=cfg["n_obs_steps"],
        n_action_steps=cfg["n_action_steps"],
        cam_keys=CAM_KEYS,
    )


def eval(TASK_ENV, model, observation):
    """Run one action chunk against TASK_ENV.

    Mirrors policy/DP/deploy_policy.py: the runner's stack_last_n() pads
    early frames with the first observed obs, so no explicit pre-update is
    needed and would actually duplicate the first frame in the deque.
    """
    obs = encode_obs(observation)
    _ = TASK_ENV.get_instruction()  # A2A is language-agnostic.

    actions = model.get_action(obs)

    for action in actions:
        TASK_ENV.take_action(action, action_type="qpos")
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        model.update_obs(obs)


def reset_model(model):
    model.reset_obs()
