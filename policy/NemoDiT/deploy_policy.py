# Deployment script for Qwen3VL + Flow Matching policy in RoboTwin
#
# Provides encode_obs(), get_model(), eval(), and reset_model() functions
# compatible with the RoboTwin evaluation pipeline.

import numpy as np
import yaml
from typing import Dict, Any

from .qwen3vl_fm_model import Qwen3VLFlowMatchingModel
from .utils.rotation_utils import convert_endpose_7d_to_9d


# Global action type for eval function
_ACTION_TYPE = "joint"


def encode_obs(observation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Post-process observation from RoboTwin environment.

    For Qwen3VL, we return raw images (not ImageNet-normalized) since
    the Qwen3-VL processor handles its own normalization.

    Args:
        observation: Raw observation from RoboTwin environment

    Returns:
        Processed observation dictionary:
            - "images": List[np.ndarray] - raw uint8 images per camera
            - "agent_pos": current joint/ee positions
    """
    head_cam = observation["observation"]["head_camera"]["rgb"]
    left_cam = observation["observation"]["left_camera"]["rgb"]
    right_cam = observation["observation"]["right_camera"]["rgb"]

    if "front_camera" in observation["observation"]:
        front_cam = observation["observation"]["front_camera"]["rgb"]
    else:
        front_cam = head_cam

    # Return raw images as list (Qwen3-VL processor handles normalization)
    images = [front_cam, head_cam, left_cam, right_cam]

    obs = {"images": images}

    # Extract robot state
    if _ACTION_TYPE == "joint" and "joint_action" in observation:
        joint_action = observation["joint_action"]
        left_arm = np.array(joint_action["left_arm"], dtype=np.float32)
        left_gripper = np.array([joint_action["left_gripper"]], dtype=np.float32)
        right_arm = np.array(joint_action["right_arm"], dtype=np.float32)
        right_gripper = np.array([joint_action["right_gripper"]], dtype=np.float32)
        obs["agent_pos"] = np.concatenate([left_arm, left_gripper, right_arm, right_gripper])
    elif _ACTION_TYPE == "endpose" and "endpose" in observation:
        endpose = observation["endpose"]
        left_7d = np.array(endpose["left_endpose"], dtype=np.float32).reshape(1, 7)
        right_7d = np.array(endpose["right_endpose"], dtype=np.float32).reshape(1, 7)
        left_9d = convert_endpose_7d_to_9d(left_7d, quat_convention="wxyz")[0]
        right_9d = convert_endpose_7d_to_9d(right_7d, quat_convention="wxyz")[0]
        left_gripper = np.array([endpose["left_gripper"]], dtype=np.float32)
        right_gripper = np.array([endpose["right_gripper"]], dtype=np.float32)
        obs["agent_pos"] = np.concatenate([left_9d, left_gripper, right_9d, right_gripper])

    return obs


def get_model(usr_args: Dict[str, Any]) -> Qwen3VLFlowMatchingModel:
    """
    Load and initialize the Qwen3VL + Flow Matching policy model.

    Args:
        usr_args: Configuration dictionary from deploy_policy.yml

    Returns:
        Initialized model
    """
    global _ACTION_TYPE

    task_name = usr_args.get('task_name', 'default_task')
    ckpt_setting = usr_args.get('ckpt_setting', 'default')
    seed = usr_args.get('seed', 0)
    checkpoint_num = usr_args.get('checkpoint_num', 500)
    expert_data_num = usr_args.get('expert_data_num', 100)

    from pathlib import Path
    _policy_dir = str(Path(__file__).parent)
    ckpt_file = (
        f"{_policy_dir}/checkpoints/"
        f"{task_name}-{ckpt_setting}-{expert_data_num}-{seed}/{checkpoint_num}.pt"
    )

    if 'checkpoint_path' in usr_args:
        ckpt_file = usr_args['checkpoint_path']

    action_type = usr_args.get('action_type', 'joint')
    _ACTION_TYPE = action_type

    model = Qwen3VLFlowMatchingModel(
        ckpt_file=ckpt_file,
        n_obs_steps=usr_args.get('n_obs_steps', 1),
        n_action_steps=usr_args.get('n_action_steps', 8),
        num_inference_steps=usr_args.get('num_inference_steps', 10),
        ode_solver=usr_args.get('ode_solver', 'midpoint'),
        device="cuda:0",
        quat_convention=usr_args.get('quat_convention', 'wxyz'),
        use_both_arms=usr_args.get('use_both_arms', True),
        action_type=action_type,
        instruction=usr_args.get('instruction', 'Predict the next robot actions.'),
    )

    return model


def eval(TASK_ENV, model: Qwen3VLFlowMatchingModel, observation: Dict[str, Any]):
    """
    Main evaluation loop for policy deployment.

    Args:
        TASK_ENV: RoboTwin task environment instance
        model: Initialized Qwen3VLFlowMatchingModel
        observation: Initial observation from environment
    """
    global _ACTION_TYPE

    obs = encode_obs(observation)
    actions = model.get_action(obs)

    if _ACTION_TYPE == "joint":
        control_mode = "qpos"
    else:
        control_mode = "ee"

    for action in actions:
        TASK_ENV.take_action(action, action_type=control_mode)


def reset_model(model: Qwen3VLFlowMatchingModel):
    """Reset model cache at the beginning of every evaluation episode."""
    model.reset_obs()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)