"""RoboTwin evaluation entrypoint for A2A.

Conforms to the contract enforced by script/eval_policy.py:
    encode_obs(observation) -> dict suitable for the policy
    get_model(usr_args)     -> stateful model with .get_action / .update_obs / .reset_obs
    eval(TASK_ENV, model, observation)  -> rollout one action chunk
    reset_model(model)      -> per-episode reset hook
"""

import os

import numpy as np
import yaml

from .a2a_model import A2A


CAM_KEYS = ("head_cam", "left_cam", "right_cam")


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

    action_dim = usr_args["left_arm_dim"] + usr_args["right_arm_dim"] + 2

    ckpt_file = (
        f"./policy/A2A/checkpoints/{task_name}-{ckpt_setting}-"
        f"{expert_data_num}-{seed}/{ckpt_num}.ckpt"
    )

    config_path = (
        f"./policy/A2A/a2a_flow_matching/config/robot_a2a_{action_dim}.yaml"
    )
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"No A2A config for action_dim={action_dim} at {config_path}. "
            f"Add a robot_a2a_{action_dim}.yaml + matching task config."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    n_obs_steps = cfg["n_obs_steps"]
    n_action_steps = cfg["n_action_steps"]

    return A2A(
        ckpt_file,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        cam_keys=CAM_KEYS,
    )


def eval(TASK_ENV, model, observation):
    """Run one action chunk against TASK_ENV."""
    obs = encode_obs(observation)
    _ = TASK_ENV.get_instruction()  # A2A is language-agnostic; we still consume it.

    if len(model.runner.obs) == 0:
        model.update_obs(obs)

    actions = model.get_action(obs)

    for action in actions:
        TASK_ENV.take_action(action, action_type="qpos")
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        model.update_obs(obs)


def reset_model(model):
    model.reset_obs()
