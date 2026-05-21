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
bash train.sh beat_block_hammer demo_randomized 50 0 0            # plain a2a
bash train.sh beat_block_hammer demo_randomized 50 0 0 a2a_noise  # noise variant
#               task              config         N seed gpu [variant]
#
# plain a2a   ckpt: ./checkpoints/<task>-<config>-<N>-<seed>/<epoch>.ckpt
# a2a_noise   ckpt: ./checkpoints/<task>-<config>-<N>-<seed>-noise/<epoch>.ckpt
```

`action_dim` is read automatically from the zarr's `meta/action_dim` attr
(written by `process_data.py`), so a single config works for 14-dim
aloha-agilex, 16-dim dual-Franka, or any other bimanual embodiment.

`variant` (optional 6th arg, default `a2a`) selects which A2A variant to
train. Supported: `a2a` (plain), `a2a_noise` (history noise std=0.02,
OT-coupled flow matcher — the upstream README's recommended deployment
variant; mitigates compounding-error / jitter in closed-loop rollouts).
The two variants save to **separate** checkpoint directories so you can
train both for the same task/seed without overwriting.

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
# argv: task task_config ckpt_setting expert_data_num seed gpu_id [checkpoint_num]
```

To evaluate the `a2a_noise` variant, either:

- (one-off) set the `A2A_VARIANT` env var on the eval.sh call:
  ```bash
  A2A_VARIANT=a2a_noise bash eval.sh beat_block_hammer demo_clean demo_clean 50 0 0
  # combine with checkpoint_num if needed:
  A2A_VARIANT=a2a_noise bash eval.sh beat_block_hammer demo_clean demo_clean 50 0 0 500
  ```
- (persistent) edit `deploy_policy.yml` and set `variant: a2a_noise`.

It selects which ckpt directory to load:
`<...>-seed/` for `a2a`, `<...>-seed-noise/` for `a2a_noise`.

- `task_config` (arg 2) = eval-time scene config (domain randomization etc.).
- `ckpt_setting` (arg 3) = the training `task_config` baked into the
  checkpoint directory name. These can differ (e.g. train on `demo_clean`,
  evaluate on `demo_randomized`).
- `checkpoint_num` (arg 7, optional) = which epoch's `.ckpt` to load.
  Defaults to `1000` (the final-epoch checkpoint of a full run). If that
  exact file is absent — common for shorter / smoke-test runs —
  `get_model()` automatically loads the **highest-numbered** `.ckpt` present
  in the directory and prints which one it used, so eval always adapts to
  the model you actually trained. To pin a specific epoch:

  ```bash
  bash eval.sh beat_block_hammer demo_randomized demo_randomized 50 0 0 200
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

Two ready-to-use variants. Pick at training time via train.sh's 6th arg, and
at eval time via the `variant` field in `deploy_policy.yml`.

| Variant     | Config file              | Policy class           | Key difference from plain a2a                                                                                                                                                  | When to use |
| ----------- | ------------------------ | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------- |
| `a2a`       | `robot_a2a.yaml`         | `A2AImagePolicy`       | — (paper baseline; `ConditionalFlowMatcher`)                                                                                                                                   | reproduce paper / clean baseline |
| `a2a_noise` | `robot_a2a_noise.yaml`   | `A2ANoiseImagePolicy`  | adds `history_noise_std=0.02` Gaussian noise to history states (both at train and inference) **and** switches the flow matcher to `ExactOptimalTransportConditionalFlowMatcher` | **closed-loop deployment** — the upstream README's explicit recommendation; mitigates compounding error / jitter when the flow source is commanded signal (which RoboTwin's `vector` is) |

Two upstream variants are intentionally **not** ported:

- `a2a_mini` — upstream `yaml` references `vita.a2a_mini_policy` but that
  Python file does not exist in the upstream repo. Dead / unreleased
  ablation config (a model-size sweep that drops the action AE and uses an
  8-layer flow_net).
- `a2a_reg` — upstream `yaml` references `vita.a2a_reg_policy`, also missing.
  Reverse ablation: replaces flow matching with a single MLP regression
  pass (1 NFE instead of 6). Useful as a paper baseline only.

If you need either of these later, the missing Python classes have to be
authored from scratch (the upstream maintainers seem to have shipped the
yaml but not the implementation).

## Reference

Original paper / code: [wyxfoe/A2A_Flow_Matching](https://github.com/wyxfoe/A2A_Flow_Matching).
