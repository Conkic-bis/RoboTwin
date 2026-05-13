"""A2A training entrypoint for RoboTwin.

Usage (via train.sh):
    python train.py --config-name=robot_a2a_14.yaml \
        task.name=<task> \
        task.dataset.zarr_path=data/<task>-<config>-<N>.zarr \
        training.seed=<seed> \
        setting=<config> expert_data_num=<N> head_camera_type=D435

The head_camera_type field is resolved against ../../task_config/_camera_config.yml
to inject the correct [3, H, W] shape into every camera entry of shape_meta,
matching policy/DP/train.py behaviour.
"""

import os
import pathlib
import sys

import hydra
import yaml
from omegaconf import OmegaConf

# Make `a2a_flow_matching` importable when invoked as a script.
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

OmegaConf.register_new_resolver("eval", eval, replace=True)


def _resolve_camera_shape(head_camera_type: str):
    cam_cfg_path = HERE.parent.parent / "task_config" / "_camera_config.yml"
    if not cam_cfg_path.is_file():
        raise FileNotFoundError(
            f"Camera config not found at {cam_cfg_path}; "
            "must run from policy/A2A inside a RoboTwin checkout."
        )
    with open(cam_cfg_path, "r", encoding="utf-8") as f:
        cams = yaml.safe_load(f)
    if head_camera_type not in cams:
        raise KeyError(
            f"camera {head_camera_type} not defined; available: {list(cams.keys())}"
        )
    cam = cams[head_camera_type]
    return [3, int(cam["h"]), int(cam["w"])]


@hydra.main(
    version_base=None,
    config_path=str(HERE / "a2a_flow_matching" / "config"),
)
def main(cfg: OmegaConf):
    head_camera_type = cfg.get("head_camera_type", None)
    if head_camera_type is not None:
        img_shape = _resolve_camera_shape(str(head_camera_type))
        for cam_key in ("head_cam", "left_cam", "right_cam"):
            if cam_key in cfg.task.shape_meta.obs:
                cfg.task.shape_meta.obs[cam_key].shape = img_shape

    OmegaConf.resolve(cfg)

    workspace_cls = hydra.utils.get_class(cfg._target_)
    workspace = workspace_cls(cfg)
    print(f"[A2A train] task={cfg.task_name} zarr={cfg.task.dataset.zarr_path}")
    workspace.run()


if __name__ == "__main__":
    main()
