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
    ├── policy/                   # base + A2A + A2A-Noise + UnifiedBridge policies
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
bash train.sh beat_block_hammer demo_randomized 50 0 0             # plain a2a
bash train.sh beat_block_hammer demo_randomized 50 0 0 a2a_noise   # noise variant
bash train.sh beat_block_hammer demo_randomized 50 0 0 bridge      # unified bridge (MLP)
bash train.sh beat_block_hammer demo_randomized 50 0 0 bridge_dit  # unified bridge (DiT)
#               task              config         N seed gpu [variant] [hydra overrides...]
#
# plain a2a   ckpt: ./checkpoints/<task>-<config>-<N>-<seed>/<epoch>.ckpt
# a2a_noise   ckpt: ./checkpoints/<task>-<config>-<N>-<seed>-noise/<epoch>.ckpt
# bridge      ckpt: ./checkpoints/<task>-<config>-<N>-<seed>-bridge/<epoch>.ckpt
# bridge_dit  ckpt: ./checkpoints/<task>-<config>-<N>-<seed>-bridge_dit/<epoch>.ckpt
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

To evaluate a non-default variant (`a2a_noise` / `bridge` / `bridge_dit`), either:

- (one-off) set the `A2A_VARIANT` env var on the eval.sh call:
  ```bash
  A2A_VARIANT=a2a_noise bash eval.sh beat_block_hammer demo_clean demo_clean 50 0 0
  A2A_VARIANT=bridge    bash eval.sh beat_block_hammer demo_clean demo_clean 50 0 0
  # combine with checkpoint_num if needed:
  A2A_VARIANT=bridge_dit bash eval.sh beat_block_hammer demo_clean demo_clean 50 0 0 500
  ```
- (persistent) edit `deploy_policy.yml` and set `variant: <name>`.

It selects which ckpt directory to load: `<...>-seed/` for `a2a`,
`<...>-seed-noise/` for `a2a_noise`, `<...>-seed-bridge/` for `bridge`,
`<...>-seed-bridge_dit/` for `bridge_dit`.

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

Four ready-to-use variants. Pick at training time via train.sh's 6th arg, and
at eval time via the `variant` field in `deploy_policy.yml` (or `A2A_VARIANT`).

| Variant      | Config file              | Policy class           | Key difference from plain a2a                                                                                                                                                  | When to use |
| ------------ | ------------------------ | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------- |
| `a2a`        | `robot_a2a.yaml`         | `A2AImagePolicy`       | — (paper baseline; `ConditionalFlowMatcher`)                                                                                                                                   | reproduce paper / clean baseline |
| `a2a_noise`  | `robot_a2a_noise.yaml`   | `A2ANoiseImagePolicy`  | adds `history_noise_std=0.02` Gaussian noise to history states (both at train and inference) **and** switches the flow matcher to `ExactOptimalTransportConditionalFlowMatcher` | **closed-loop deployment** — the upstream README's explicit recommendation; mitigates compounding error / jitter when the flow source is commanded signal (which RoboTwin's `vector` is) |
| `bridge`     | `robot_bridge.yaml`      | `UnifiedBridgePolicy`  | unified source distribution (`gaussian`/`clean_history`/`noised_history`/`mixed_bridge`/`residual_history`) + RobotMAF multimodal condition + latent alignment (JEPA / InfoNCE / VICReg), MLP backbone | the unified-bridge framework on the lightweight backbone |
| `bridge_dit` | `robot_bridge_dit.yaml`  | `UnifiedBridgePolicy`  | same as `bridge` but the vector field is a `DiTBridge` transformer with token-level MAF conditioning                                                                              | the unified-bridge framework on a high-capacity DiT backbone |

### Unified Bridge Framework (`bridge` / `bridge_dit`)

`UnifiedBridgePolicy` treats noise-to-action diffusion and action-to-action
flow matching as endpoints of one family of *source distributions* in the
shared action latent space:

```
z0 = E_hist(h + sigma_a*eps_a) + sigma_z*eps_z    (history-centered source)
z1 = E_act(a+)                                    (future action latent, flow target)
c  = RobotMAF(obs_tokens, hist_tokens, state_token)  (condition; never touches a+)

z_t = (1-t) z0 + t z1,   v_theta(z_t, t, c) -> z1 - z0
```

Everything in the experimental matrix is a hydra override on the same code
path (same action AE, same condition encoder, same parameter budget):

```bash
# source-distribution study (MLP group)
bash train.sh <task> <cfg> 50 0 0 bridge policy.source_mode=gaussian        # noise-to-action flow
bash train.sh <task> <cfg> 50 0 0 bridge policy.source_mode=clean_history   # A2A clean
bash train.sh <task> <cfg> 50 0 0 bridge policy.source_mode=noised_history  # A2A-Noise
bash train.sh <task> <cfg> 50 0 0 bridge                                    # ours: mixed_bridge
bash train.sh <task> <cfg> 50 0 0 bridge policy.source_mode=residual_history # history-shifted residual

# fusion ablation: concat vs MAF (noise-aware gated fusion)
bash train.sh <task> <cfg> 50 0 0 bridge policy.condition_mode=concat

# alignment ablation
bash train.sh <task> <cfg> 50 0 0 bridge policy.jepa_weight=0 policy.align_weight=0
bash train.sh <task> <cfg> 50 0 0 bridge policy.align_type=vicreg

# source noise sweeps (sigma_a / sigma_z)
bash train.sh <task> <cfg> 50 0 0 bridge policy.history_noise_std=0.1 policy.latent_noise_std=0.1

# DiT group
bash train.sh <task> <cfg> 50 0 0 bridge_dit policy.source_mode=gaussian    # Gaussian DiT flow
bash train.sh <task> <cfg> 50 0 0 bridge_dit                                # ours on DiT
```

Training logs include per-loss metrics (`train_flow_loss`, `train_jepa_loss`,
`train_align_loss`, `train_consistency_loss`), latent diagnostics
(`train_source_target_dist`, `train_z1_var`, `train_z1_effective_rank`) and
MAF gate interpretability values (`train_gate_visual` / `_history` / `_state`,
`train_gate_entropy`) — the latent and gate metrics of the experimental plan.

Note the bridge configs use `ConditionalFlowMatcher` (independent coupling)
instead of the OT matcher: OT minibatch re-pairing would couple sample *i*'s
history latent with sample *j*'s future latent and break the physical
`(z0_i, z1_i, c_i)` correspondence the bridge depends on.

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
