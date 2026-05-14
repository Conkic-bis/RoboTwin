# A2A: Action-to-Action Flow Matching Policy (RoboTwin port)

A2A is a flow-matching policy that maps a history of states to a future
action chunk, conditioned on multi-view RGB observations. This directory is a
self-contained port of [wyxfoe/A2A_Flow_Matching](https://github.com/wyxfoe/A2A_Flow_Matching)
onto the RoboTwin 2.0 benchmark.

> **Hyperparameter fidelity**: every algorithmic hyperparameter — horizon,
> n_obs_steps, n_action_steps, optimizer, EMA, flow matcher, encoder/decoder
> sizes, sampling steps, batch size, lr schedule — is taken **verbatim** from
> the original A2A repo's `default_runner.yaml`, `default_train.yaml`, and
> `policy_config/a2a.yaml`. Only paths, the `_target_` namespace, and the
> embodiment-determined `agent_pos`/`action` dimensions (and the image shape,
> which is injected at runtime from `task_config/_camera_config.yml`) are
> rebound for RoboTwin. RoboTwin's DP project was used only as a deployment
> integration reference, not as a source of hyperparameters.

## Layout

```
policy/A2A/
├── process_data.py / .sh         # RoboTwin HDF5 -> multi-cam ZARR
├── train.py / .sh                # Hydra training entrypoint
├── eval.sh / eval_double_env.sh  # Wrappers around script/eval_policy.py
├── deploy_policy.py / .yml       # get_model/eval/reset_model contract
├── deploy_policy_double_env.py   # model-server flavour
├── a2a_model.py                  # checkpoint loader + inference wrapper
├── pyproject.toml / .gitignore / README.md
└── a2a_flow_matching/            # self-contained ML package
    ├── common/                   # normalizer, replay buffer, sampler, utils
    ├── model/                    # flow_net, layers, action_ae, flow matchers,
    │   ├── vision/               #   multi_image_obs_encoder, ResNet getter, crop
    │   └── diffusion/ema_model.py
    ├── policy/                   # base + A2A + A2A-Noise image policies
    ├── dataset/robot_image_dataset.py   # multi-cam zarr reader
    ├── env_runner/a2a_runner.py  # obs deque + chunked action retrieval
    ├── workspace/                # base_workspace, a2a_workspace (train loop)
    └── config/                   # robot_a2a_14.yaml, robot_a2a_16.yaml + task/
```

## Installation

```bash
cd policy/A2A
pip install -e .
```

## End-to-end usage

### 1. Collect demos (RoboTwin's native pipeline)

```bash
# from repo root
bash collect_data.sh beat_block_hammer demo_randomized 0
```

### 2. Convert HDF5 -> multi-camera ZARR

```bash
cd policy/A2A
bash process_data.sh beat_block_hammer demo_randomized 50
# writes ./data/beat_block_hammer-demo_randomized-50.zarr
```

The ZARR groups: `data/head_camera`, `data/left_camera`, `data/right_camera`
(uint8 NCHW), `data/state` and `data/action` (float32; state[t] =
joint_action.vector[t], action[t] = vector[t+1]), `meta/episode_ends`.

### 3. Train

```bash
bash train.sh beat_block_hammer demo_randomized 50 0 0
#               task              config         N seed gpu
# checkpoints land at ./checkpoints/<task>-<config>-<N>-<seed>/<epoch>.ckpt
```

`action_dim` is read automatically from the zarr's `meta/action_dim` attr
(written by `process_data.py`), so a single `robot_a2a.yaml` config works for
14-dim aloha-agilex, 16-dim dual-Franka, or any other bimanual embodiment.

#### Logging with Weights & Biases (optional)

Training logs `train_loss` / `val_loss` / `lr` / `global_step` / `epoch` per
batch to a local `logs.json.txt` *and*, if available, to wandb. Behaviour:

| Condition                         | wandb mode | What happens                          |
| --------------------------------- | ---------- | ------------------------------------- |
| `WANDB_API_KEY` set, `DEBUG=False`| `online`   | live run streamed to wandb.ai         |
| API key unset (or no `~/.netrc`)  | `offline`  | run saved under `<output_dir>/wandb/` |
| `DEBUG=True` in `train.sh`        | `offline`  | same as above                         |
| `logging.mode=disabled` override  | disabled   | JSON file only, no wandb at all       |

The fallback is automatic — `train.sh` checks for an API key before launching
and switches to `offline` if missing, so no run ever blocks on `wandb login`.
To customise project / entity / tags, override in CLI or edit
`a2a_flow_matching/config/robot_a2a.yaml`'s `logging:` block.

```bash
# Force-disable wandb entirely:
bash train.sh beat_block_hammer demo_randomized 50 0 0  # then in CLI pass
python train.py ... logging.mode=disabled
# Or for a team account:
python train.py ... logging.entity=my-org logging.project=robotwin-a2a
```

### 4. Evaluate (drives `script/eval_policy.py` with `policy_name=A2A`)

```bash
bash eval.sh beat_block_hammer demo_randomized demo_randomized 50 0 0
# argv: task task_config ckpt_setting expert_data_num seed gpu_id
```

For client/server-style evaluation (matches DP's `eval_double_env.sh`):

```bash
bash eval_double_env.sh beat_block_hammer demo_randomized demo_randomized 50 0 0 a2a
```

## Key design decisions

- **Algorithm code is 1:1 with upstream**: every file under
  `a2a_flow_matching/{common,model,policy}/` is the original A2A source with
  imports rewritten from `roboverse_learn.il.*` to `a2a_flow_matching.*`.
  No `metasim` / `curobo` / `pytorch3d` / IsaacSim dependencies are introduced.
- **`n_obs_steps == 8`** is preserved (A2A's CNN action encoder requires it).
- **`MultiImageObsEncoder` natively supports multi-cam**, so the only
  additions over upstream are extra entries in `shape_meta.obs` (head_cam /
  left_cam / right_cam) and the dataset loader.
- **Checkpoint payload schema** = `{cfg, state_dicts}` with `cfg._target_`
  pointing to `A2AWorkspace`; `a2a_model.py` reconstructs the workspace via
  `hydra.utils.get_class(cfg._target_)`, matching DP's pattern so RoboTwin's
  `script/eval_policy.py` works unmodified.

## Variants

Single config: `a2a_flow_matching/config/robot_a2a.yaml` (action_dim
auto-detected at training time from the zarr).

Policies (both already ported under `a2a_flow_matching/policy/`):

- `A2AImagePolicy` (default).
- `A2ANoiseImagePolicy` — adds Gaussian noise to history actions; flip the
  config's policy `_target_` to switch.

## Reference

Original paper / code: [wyxfoe/A2A_Flow_Matching](https://github.com/wyxfoe/A2A_Flow_Matching).
