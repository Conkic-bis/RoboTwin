"""Double-env (model-server / eval-client) variant of A2A's deploy entry.

Mirrors policy/DP/deploy_policy_double_env.py — the server side wraps the model
inside a `model.call(func_name=..., obs=...)` RPC façade. We rely on the same
A2A wrapper as the single-env path; only the dispatch shape is different.
"""

import numpy as np

try:
    from .a2a_model import A2A
except Exception:  # pragma: no cover - server-side may not import sibling modules
    A2A = None


def encode_obs(observation):
    head = np.moveaxis(observation["observation"]["head_camera"]["rgb"], -1, 0) / 255.0
    left = np.moveaxis(observation["observation"]["left_camera"]["rgb"], -1, 0) / 255.0
    right = np.moveaxis(observation["observation"]["right_camera"]["rgb"], -1, 0) / 255.0
    return {
        "head_cam": head,
        "left_cam": left,
        "right_cam": right,
        "agent_pos": observation["joint_action"]["vector"],
    }


def get_model(usr_args):
    from .deploy_policy import _resolve_ckpt, _VARIANT_TO_SUFFIX

    task_name = usr_args["task_name"]
    ckpt_setting = usr_args["ckpt_setting"]
    expert_data_num = usr_args["expert_data_num"]
    seed = usr_args["seed"]
    ckpt_num = usr_args["checkpoint_num"]
    variant = usr_args.get("variant", "a2a")

    if variant not in _VARIANT_TO_SUFFIX:
        raise ValueError(
            f"[A2A] unknown variant '{variant}'; expected one of "
            f"{list(_VARIANT_TO_SUFFIX)}"
        )
    suffix = _VARIANT_TO_SUFFIX[variant]

    ckpt_dir = (
        f"./policy/A2A/checkpoints/"
        f"{task_name}-{ckpt_setting}-{expert_data_num}-{seed}{suffix}"
    )
    ckpt_file = _resolve_ckpt(ckpt_dir, ckpt_num)
    print(f"[A2A] variant={variant} loading checkpoint: {ckpt_file}")

    import yaml
    config_path = f"./policy/A2A/a2a_flow_matching/config/robot_{variant}.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return A2A(
        ckpt_file,
        n_obs_steps=cfg["n_obs_steps"],
        n_action_steps=cfg["n_action_steps"],
        cam_keys=("head_cam", "left_cam", "right_cam"),
    )


def eval(TASK_ENV, model, observation):
    obs = encode_obs(observation)
    _ = TASK_ENV.get_instruction()

    actions = model.call(func_name="get_action", obs=obs)

    for action in actions:
        TASK_ENV.take_action(action, action_type="qpos")
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        model.call(func_name="update_obs", obs=obs)


def reset_model(model):
    model.call(func_name="reset_obs")
