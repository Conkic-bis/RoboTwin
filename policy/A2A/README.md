# A2A: Action-to-Action Flow Matching Policy (RoboTwin port)

A2A is a flow-matching policy that learns to map a history of states to a future
action chunk, conditioned on multi-view RGB observations. This directory is a
self-contained port of [A2A_Flow_Matching](https://github.com/wyxfoe/A2A_Flow_Matching)
onto the RoboTwin 2.0 benchmark, mirroring the structure of `policy/DP/`.

## Status

This is an incremental port. Implemented so far:

- **Stage A — Algorithm code**: `a2a_flow_matching/` contains the ported model,
  flow-matching utilities, action autoencoder, multi-image observation encoder,
  and policy classes (`A2AImagePolicy`, `A2ANoiseImagePolicy`) with imports
  rewritten to be self-contained inside this directory.
- **Stage B — Data pipeline**: `process_data.py` converts RoboTwin HDF5 demos
  (head + left + right cameras + `joint_action/vector`) into a multi-camera
  ZARR dataset, and `a2a_flow_matching/dataset/robot_image_dataset.py` reads it.

Still to do (next PR): Hydra training entrypoint (`train.py`, `train.sh`), the
A2A workspace and env runner, `deploy_policy.py` / `eval.sh`, and per-action-dim
config YAMLs.

## Layout

```
policy/A2A/
├── process_data.py / process_data.sh   # HDF5 -> multi-cam ZARR
├── pyproject.toml
└── a2a_flow_matching/
    ├── common/                         # normalizer, replay buffer, sampler, utils
    ├── model/                          # flow_net, layers, action_ae, flow_matchers
    │   ├── vision/                     # multi_image_obs_encoder, ResNet getter, crop
    │   └── diffusion/ema_model.py
    ├── policy/                         # base + A2A + A2A-noise image policies
    └── dataset/robot_image_dataset.py  # multi-cam zarr reader
```

## Usage (current stages)

### 1. Collect demos with RoboTwin's native pipeline

```bash
# from repo root
bash collect_data.sh beat_block_hammer demo_randomized 0
```

### 2. Convert HDF5 -> ZARR

```bash
cd policy/A2A
bash process_data.sh beat_block_hammer demo_randomized 50
# writes ./data/beat_block_hammer-demo_randomized-50.zarr
```

The ZARR holds three RGB streams (`head_camera`, `left_camera`, `right_camera`,
NCHW uint8), `state` and `action` (`float32`, both equal to RoboTwin's
`joint_action/vector` shifted by one step, matching `policy/DP/process_data.py`),
and `meta/episode_ends`.

## Notes on the port

- `n_obs_steps >= 8` is required by the default `CNNActionEncoder` (3 stride-2
  convs). For short-history experiments use `MLPActionEncoder` (already
  available in `model/action_ae.py`).
- All imports under `a2a_flow_matching/` are rewritten from
  `roboverse_learn.il.*` to `a2a_flow_matching.*`. No `metasim`, `curobo`,
  `pytorch3d`, or IsaacSim dependencies are introduced.
- The dataset reads any subset of `{head_camera, left_camera, right_camera}`
  actually present in the zarr, exposing them as `head_cam`, `left_cam`,
  `right_cam` keys to the obs encoder.

## Reference

Original paper / code: [wyxfoe/A2A_Flow_Matching](https://github.com/wyxfoe/A2A_Flow_Matching).
