import numpy as np
import yaml
from typing import Dict, Any

from .nemo_dit_model import NemoDiT
from .utils.rotation_utils import convert_endpose_7d_to_9d


# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

# Global variable to store action type for eval function
_ACTION_TYPE = "endpose"


def encode_obs(observation: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Post-process observation from RoboTwin environment.

    Converts raw environment observations to the format expected by the model.

    Args:
        observation: Raw observation from RoboTwin environment
            - observation["observation"]["head_camera"]["rgb"]: (H, W, 3) uint8
            - observation["observation"]["left_camera"]["rgb"]: (H, W, 3) uint8
            - observation["observation"]["right_camera"]["rgb"]: (H, W, 3) uint8
            - observation["joint_action"]["vector"]: current joint positions

    Returns:
        Processed observation dictionary:
            - "images": (num_cameras, 3, H, W) normalized float32 array
            - "agent_pos": current joint/ee positions
    """
    # Extract camera images
    head_cam = observation["observation"]["head_camera"]["rgb"]
    left_cam = observation["observation"]["left_camera"]["rgb"]
    right_cam = observation["observation"]["right_camera"]["rgb"]

    # Check if front camera exists, otherwise duplicate head camera
    if "front_camera" in observation["observation"]:
        front_cam = observation["observation"]["front_camera"]["rgb"]
    else:
        front_cam = head_cam

    # Process each camera image
    # Convert from (H, W, C) uint8 to (C, H, W) normalized float32
    def process_image(img):
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        # Transpose to (C, H, W)
        img = np.transpose(img, (2, 0, 1))
        # Apply ImageNet normalization
        img = (img - IMAGENET_MEAN.reshape(3, 1, 1)) / IMAGENET_STD.reshape(3, 1, 1)
        return img

    # Stack all cameras in the same order as training dataloader:
    # ['front_camera', 'head_camera', 'left_camera', 'right_camera']
    images = np.stack([
        process_image(front_cam),
        process_image(head_cam),
        process_image(left_cam),
        process_image(right_cam),
    ], axis=0).astype(np.float32)

    # Build observation dictionary
    obs = {
        "images": images,
    }

    # Extract robot state (agent_pos) for state conditioning
    # Format must match model's action representation:
    #   joint mode:   [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)] = 14D
    #   endpose mode: [left_9d(9), left_gripper(1), right_9d(9), right_gripper(1)] = 20D
    if _ACTION_TYPE == "joint" and "joint_action" in observation:
        joint_action = observation["joint_action"]
        left_arm = np.array(joint_action["left_arm"], dtype=np.float32)
        left_gripper = np.array([joint_action["left_gripper"]], dtype=np.float32)
        right_arm = np.array(joint_action["right_arm"], dtype=np.float32)
        right_gripper = np.array([joint_action["right_gripper"]], dtype=np.float32)
        obs["agent_pos"] = np.concatenate([left_arm, left_gripper, right_arm, right_gripper])
    elif _ACTION_TYPE == "endpose" and "endpose" in observation:
        endpose = observation["endpose"]
        # Environment returns 7D: [x, y, z, qw, qx, qy, qz] (wxyz convention)
        left_endpose_7d = np.array(endpose["left_endpose"], dtype=np.float32).reshape(1, 7)
        right_endpose_7d = np.array(endpose["right_endpose"], dtype=np.float32).reshape(1, 7)
        # Convert quaternion -> rot6d: (1, 7) -> (1, 9)
        left_endpose_9d = convert_endpose_7d_to_9d(left_endpose_7d, quat_convention="wxyz")[0]
        right_endpose_9d = convert_endpose_7d_to_9d(right_endpose_7d, quat_convention="wxyz")[0]
        left_gripper = np.array([endpose["left_gripper"]], dtype=np.float32)
        right_gripper = np.array([endpose["right_gripper"]], dtype=np.float32)
        obs["agent_pos"] = np.concatenate([
            left_endpose_9d, left_gripper, right_endpose_9d, right_gripper
        ])

    return obs


def get_model(usr_args: Dict[str, Any]) -> NemoDiT:
    """
    Load and initialize the NemoDiT policy model.

    Args:
        usr_args: Configuration dictionary from deploy_policy.yml
            Required keys:
            - task_name: Name of the task
            - ckpt_setting: Checkpoint configuration name
            - seed: Random seed used during training
            - checkpoint_num: Checkpoint number to load (e.g., 600)
            Optional keys:
            - n_obs_steps: Number of observation steps (default: 1)
            - n_action_steps: Number of actions to execute (default: 10)
            - ddim_steps: DDIM sampling steps (default: 10)
            - use_both_arms: Whether to use dual arm (default: True)
            - quat_convention: Output quaternion convention (default: "wxyz")
            - action_type: Action type - "endpose" or "joint" (default: "endpose")

    Returns:
        Initialized NemoDiT model
    """
    global _ACTION_TYPE

    # Build checkpoint path
    task_name = usr_args.get('task_name', 'default_task')
    ckpt_setting = usr_args.get('ckpt_setting', 'default')
    seed = usr_args.get('seed', 0)
    checkpoint_num = usr_args.get('checkpoint_num', 600)
    expert_data_num = usr_args.get('expert_data_num', 100)

    # Checkpoint path pattern
    # You can customize this path based on your checkpoint directory structure
    ckpt_file = (
        f"./policy/NemoDiT/checkpoints/"
        f"{task_name}-{ckpt_setting}-{expert_data_num}-{seed}/{checkpoint_num}.pt"
    )

    # Alternative: use direct checkpoint path if provided
    if 'checkpoint_path' in usr_args:
        ckpt_file = usr_args['checkpoint_path']

    # Model configuration
    n_obs_steps = usr_args.get('n_obs_steps', 1)
    n_action_steps = usr_args.get('n_action_steps', 10)
    ddim_steps = usr_args.get('ddim_steps', 10)
    use_both_arms = usr_args.get('use_both_arms', True)
    quat_convention = usr_args.get('quat_convention', 'wxyz')
    action_type = usr_args.get('action_type', 'endpose')

    # Set global action type for eval function
    _ACTION_TYPE = action_type

    # Initialize model
    model = NemoDiT(
        ckpt_file=ckpt_file,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        ddim_steps=ddim_steps,
        device="cuda:0",
        quat_convention=quat_convention,
        use_both_arms=use_both_arms,
        action_type=action_type,
    )

    return model


def eval(TASK_ENV, model: NemoDiT, observation: Dict[str, Any]):
    """
    Main evaluation loop for policy deployment.

    This function:
    1. Encodes the current observation
    2. Gets action from the model
    3. Executes actions sequentially
    4. Updates model with new observations

    Args:
        TASK_ENV: RoboTwin task environment instance
        model: Initialized NemoDiT model
        observation: Initial observation from environment

    Control Modes Supported:
        - "qpos": Joint position control (for action_type="joint")
        - "ee": End-effector pose control (for action_type="endpose")
        - "delta_ee": Delta end-effector control

    The control mode is automatically selected based on the action_type
    configured during training:
        - action_type="endpose" -> control_mode="ee"
        - action_type="joint" -> control_mode="qpos"
    """
    global _ACTION_TYPE

    # Encode initial observation
    obs = encode_obs(observation)

    # Get task instruction (if using language conditioning)
    instruction = TASK_ENV.get_instruction()

    # Get actions from the model针对step by step推理必须要如下设置)
    actions = model.get_action(obs)

    # Map policy action_type to environment action_type parameter
    if _ACTION_TYPE == "joint":
        control_mode = "qpos"
    else:  # endpose
        control_mode = "ee"

    # Execute each action (open-loop: no re-planning within this batch)
    for action in actions:
        TASK_ENV.take_action(action, action_type=control_mode)


def reset_model(model: NemoDiT):
    """
    Clean the model cache at the beginning of every evaluation episode.

    This resets:
    - Observation history buffer
    - Action queue

    Args:
        model: NemoDiT model instance
    """
    model.reset_obs()


# ============================================================================
# Alternative evaluation function with more control
# ============================================================================

def eval_with_action_queue(TASK_ENV, model: NemoDiT, observation: Dict[str, Any]):
    """
    Alternative evaluation loop using action queueing.

    Instead of executing all predicted actions at once, this version:
    1. Predicts a sequence of actions
    2. Executes them one by one from the queue
    3. Only re-predicts when the queue is empty

    This can be more responsive to environment changes.
    """
    global _ACTION_TYPE

    obs = encode_obs(observation)
    instruction = TASK_ENV.get_instruction()

    # Map policy action_type to environment action_type parameter
    if _ACTION_TYPE == "joint":
        control_mode = "qpos"
    else:  # endpose
        control_mode = "ee"

    max_steps = 1000  # Maximum steps per episode

    for step in range(max_steps):
        # Get single action (model handles queueing internally)
        actions = model.get_action(obs)
        action = actions[0]

        # Execute action
        TASK_ENV.take_action(action, action_type=control_mode)

        # Check if done
        observation = TASK_ENV.get_obs()
        if TASK_ENV.is_done():
            break

        # Update observation
        obs = encode_obs(observation)


# ============================================================================
# Utility functions
# ============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
