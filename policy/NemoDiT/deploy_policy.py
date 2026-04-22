# Deployment entry point for the Qwen3-VL + Flow Matching policy.
#
# Key change vs the 2.0 branch:
#   The Qwen3-VL processor expects RAW per-camera images (uint8 HxWxC RGB)
#   and applies its own preprocessing. The ImageNet normalization that the
#   ResNet path needed is gone. We also forward the task instruction so
#   the VLM has language grounding for each decision.

from typing import Any, Dict

import numpy as np
import yaml

from .nemo_dit_model import NemoDiT
from .utils.rotation_utils import convert_endpose_7d_to_9d


# Global action type tag so eval() can map to the correct control mode.
_ACTION_TYPE = 'endpose'


def encode_obs(observation: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Convert a RoboTwin env observation into the dict the model consumes.

    Output:
        images_raw : List[np.ndarray(H, W, 3) uint8]
                     Camera order must match the training dataloader:
                     [front, head, left, right]
        agent_pos  : np.ndarray  — current robot state (joint or endpose 9d)
    """
    head_cam = observation['observation']['head_camera']['rgb']
    left_cam = observation['observation']['left_camera']['rgb']
    right_cam = observation['observation']['right_camera']['rgb']
    front_cam = observation['observation'].get('front_camera', {}).get('rgb', head_cam)

    def _to_uint8_rgb(img: np.ndarray) -> np.ndarray:
        """Guarantee HxWxC uint8 contiguous RGB, since some RoboTwin tasks
        return float [0,1] and others uint8 [0,255]."""
        if img.dtype != np.uint8:
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8) if img.max() <= 1.0 \
                else img.astype(np.uint8)
        return np.ascontiguousarray(img)

    images_raw = [
        _to_uint8_rgb(front_cam),
        _to_uint8_rgb(head_cam),
        _to_uint8_rgb(left_cam),
        _to_uint8_rgb(right_cam),
    ]

    obs = {'images_raw': images_raw}

    # Robot state vector. Must match the action_dim the model was trained
    # on, byte-for-byte — the dataloader uses actions[0] as state, so the
    # layouts below mirror _load_{endpose,joint}_actions.
    if _ACTION_TYPE == 'joint' and 'joint_action' in observation:
        ja = observation['joint_action']
        obs['agent_pos'] = np.concatenate([
            np.asarray(ja['left_arm'], dtype=np.float32),
            np.asarray([ja['left_gripper']], dtype=np.float32),
            np.asarray(ja['right_arm'], dtype=np.float32),
            np.asarray([ja['right_gripper']], dtype=np.float32),
        ])
    elif _ACTION_TYPE == 'endpose' and 'endpose' in observation:
        ep = observation['endpose']
        le7 = np.asarray(ep['left_endpose'], dtype=np.float32).reshape(1, 7)
        re7 = np.asarray(ep['right_endpose'], dtype=np.float32).reshape(1, 7)
        le9 = convert_endpose_7d_to_9d(le7, quat_convention='wxyz')[0]
        re9 = convert_endpose_7d_to_9d(re7, quat_convention='wxyz')[0]
        obs['agent_pos'] = np.concatenate([
            le9,
            np.asarray([ep['left_gripper']], dtype=np.float32),
            re9,
            np.asarray([ep['right_gripper']], dtype=np.float32),
        ])

    return obs


def get_model(usr_args: Dict[str, Any]) -> NemoDiT:
    global _ACTION_TYPE

    task_name = usr_args.get('task_name', 'default_task')
    ckpt_setting = usr_args.get('ckpt_setting', 'default')
    seed = usr_args.get('seed', 0)
    checkpoint_num = usr_args.get('checkpoint_num', 600)
    expert_data_num = usr_args.get('expert_data_num', 100)

    ckpt_file = (
        f"./policy/NemoDiT/checkpoints/"
        f"{task_name}-{ckpt_setting}-{expert_data_num}-{seed}/{checkpoint_num}.pt"
    )
    if 'checkpoint_path' in usr_args:
        ckpt_file = usr_args['checkpoint_path']

    action_type = usr_args.get('action_type', 'endpose')
    _ACTION_TYPE = action_type

    return NemoDiT(
        ckpt_file=ckpt_file,
        n_obs_steps=usr_args.get('n_obs_steps', 1),
        n_action_steps=usr_args.get('n_action_steps', 10),
        num_inference_steps=usr_args.get('num_inference_steps', 10),
        ode_solver=usr_args.get('ode_solver', 'midpoint'),
        device=usr_args.get('device', 'cuda:0'),
        quat_convention=usr_args.get('quat_convention', 'wxyz'),
        use_both_arms=usr_args.get('use_both_arms', True),
        action_type=action_type,
        instruction=usr_args.get('instruction', 'Predict the next robot actions.'),
    )


def eval(TASK_ENV, model: NemoDiT, observation: Dict[str, Any]):
    global _ACTION_TYPE

    obs = encode_obs(observation)
    # Attach the task instruction so the VLM has language grounding. Falls
    # back to the default inside NemoDiT if the env returns an empty string.
    obs['instruction'] = TASK_ENV.get_instruction() or model.instruction

    actions = model.get_action(obs)

    control_mode = 'qpos' if _ACTION_TYPE == 'joint' else 'ee'
    for action in actions:
        TASK_ENV.take_action(action, action_type=control_mode)


def reset_model(model: NemoDiT):
    model.reset_obs()


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
