# DiT for Robot Action Generation with Vision Conditioning

This project implements a Diffusion Transformer (DiT) model for robot action generation, conditioned on visual observations from multiple cameras. It supports multi-frame temporal observations, dual-arm manipulation, multiple action representations (end-effector pose / joint angles), and classifier-free guidance.

## Architecture Overview

### Core Components

1. **Vision Backbone** (`model/vision_input/vision_input.py`)
   - Extracts visual features from robot camera observations
   - Supports multiple backbone architectures:
     - ResNet (ResNet18, ResNet34, ResNet50)
     - Vision Transformer (ViT-B/16, ViT-B/32)
   - Handles multiple camera views (front, head, left, right) via learned camera fusion
   - Supports multi-frame observations (`n_obs_steps`) with temporal aggregation (`last`, `mean`, `concat`)
   - Output: `(batch_size, 1, vision_feature_dim)` — single global feature token per sample

2. **Feature Adapter** (`model/feature_adaptation.py`)
   - Projects vision features to DiT's expected token dimension
   - Three adapter types:
     - **Linear**: Simple linear projection with optional LayerNorm
     - **MLP**: Multi-layer perceptron with configurable depth, GELU activation, LayerNorm, and dropout
     - **Attention Pooling**: Learnable query-based cross-attention feature aggregation
   - Output: `(batch_size, 1, token_size)`

3. **DiT Model** (`model/action_model/models.py` & `action_model.py`)
   - Transformer-based diffusion model for action prediction
   - Accepts visual conditions and generates robot actions
   - Supports four model sizes: **DiT-S, DiT-B, DiT-L, DiT-XL**
   - Supports classifier-free guidance (CFG) with configurable dropout probability
   - Predicts future action sequences with temporal coherence

4. **Data Loader** (`dataloader.py`)
   - Loads robot manipulation data from HDF5 files
   - Supports both single-arm and dual-arm configurations
   - Supports two action types: `endpose` (end-effector pose) and `joint` (joint angles)
   - Handles multiple camera views with multi-frame temporal windowing
   - Automatic quaternion-to-rot6d conversion for end-effector poses
   - Efficient lazy loading for large datasets

5. **Rotation Utilities** (`utils/rotation_utils.py`)
   - Quaternion ↔ rotation matrix ↔ rot6d conversions
   - Supports both `wxyz` and `xyzw` quaternion conventions
   - `convert_endpose_7d_to_9d()`: Converts (translation + quaternion) to (translation + rot6d)

## Data Format

Robot data should be organized in HDF5 files with the following structure:

```
    Data structure for each episode (raw HDF5 format):
    ├── episode_X.hdf5
    │   ├── /endpose
    │   │   ├── /endpose/left_endpose    (T, 7)  - 3D translation + 4D quaternion
    │   │   ├── /endpose/left_gripper    (T, 1)  - gripper state (1-DoF)
    │   │   ├── /endpose/right_endpose   (T, 7)  - 3D translation + 4D quaternion
    │   │   └── /endpose/right_gripper   (T, 1)  - gripper state (1-DoF)
    │   ├── /joint_action
    │   │   ├── /joint_action/left_arm     (T, 6)  - 6-DoF joint angles
    │   │   ├── /joint_action/left_gripper (T, 1)  - gripper state
    │   │   ├── /joint_action/right_arm    (T, 6)  - 6-DoF joint angles
    │   │   └── /joint_action/right_gripper(T, 1)  - gripper state
    │   └── /observation
    │       ├── /observation/front_camera/rgb   (T, H, W, 3)
    │       ├── /observation/head_camera/rgb    (T, H, W, 3)
    │       ├── /observation/left_camera/rgb    (T, H, W, 3)
    │       └── /observation/right_camera/rgb   (T, H, W, 3)
```

### Action Representations

When loading data, the action representation depends on `--action_type`:

- **`joint`** (default): 6D joint angles + 1D gripper = **7D per arm** (14D dual-arm)
- **`endpose`**: 3D translation + 6D rot6d (converted from quaternion) + 1D gripper = **10D per arm** (20D dual-arm)

The `action_dim` is automatically computed from `action_type` and `use_both_arms`.

## Installation

```bash
pip install torch torchvision h5py numpy tqdm timm
```

## Training

### Basic Usage

```bash
python train.py \
  --data_path /path/to/robot/data \
  --model_type DiT-B \
  --action_type joint \
  --num_cameras 4 \
  --vision_backbone resnet50 \
  --batch_size 16 \
  --epochs 500 \
  --lr 1e-4
```

### Key Arguments

**Data Arguments:**
- `--data_path`: Path to directory containing episode HDF5 files
- `--num_cameras`: Number of camera views (default: 4)
- `--use_both_arms`: Use both arms data (default: True)
- `--action_type`: Action type — `endpose` or `joint` (default: `joint`)
- `--quat_convention`: Quaternion convention in HDF5 data — `wxyz` or `xyzw` (default: `wxyz`)

**Model Arguments:**
- `--model_type`: DiT model size — `DiT-S`, `DiT-B`, `DiT-L`, `DiT-XL` (default: `DiT-B`)
- `--action_dim`: Action dimension (auto-computed from `action_type` and `use_both_arms`)
- `--future_action_window`: Number of future actions to predict (default: 12)
- `--past_action_window`: Number of past actions (default: 0)
- `--token_size`: Token size for conditioning (default: 2048)
- `--dropout_prob`: Class dropout probability for classifier-free guidance (default: 0.1)

**Temporal / Observation Arguments:**
- `--n_obs_steps`: Number of observation frames for visual conditioning (default: 2)
- `--n_action_steps`: Number of action steps to execute during inference / receding horizon (default: 8)
- `--temporal_agg`: Temporal aggregation method for multi-frame observations — `last`, `mean`, `concat` (default: `concat`)

> **Constraint:** `n_obs_steps + n_action_steps <= future_action_window`

**Vision Arguments:**
- `--vision_backbone`: Vision backbone type — `resnet18`, `resnet34`, `resnet50`, `vit_b_16`, `vit_b_32` (default: `resnet50`)
- `--vision_pretrained`: Use pretrained weights (default: True)
- `--freeze_vision`: Freeze vision backbone (default: False)
- `--adapter_type`: Feature adapter type — `linear`, `mlp`, `attention_pooling` (default: `mlp`)
- `--image_size`: Image size for vision backbone (default: 224)
- `--no_resize`: Skip image resizing, use original image size

**Diffusion Arguments:**
- `--diffusion_steps`: Number of diffusion timesteps (default: 500)
- `--noise_schedule`: Noise schedule type (default: `squaredcos_cap_v2`)

**Training Arguments:**
- `--batch_size`: Batch size (default: 16)
- `--epochs`: Number of epochs (default: 500)
- `--lr`: Learning rate (default: 1e-4)
- `--weight_decay`: Weight decay for AdamW (default: 0.01)
- `--grad_clip`: Gradient clipping max norm (default: 1.0)
- `--use_amp`: Enable Automatic Mixed Precision (FP16) training
- `--warmup_epochs`: Number of warmup epochs (default: 0)
- `--warmup_type`: Warmup schedule type — `linear` or `cosine` (default: `linear`)
- `--checkpoint_dir`: Directory to save checkpoints (default: `checkpoints`)
- `--save_every`: Save checkpoint every N epochs (default: 50)
- `--resume`: Path to checkpoint to resume training from

### Example Training Commands

**Joint angles, dual-arm, ResNet50 backbone:**
```bash
python train.py \
  --data_path ./robot_data \
  --model_type DiT-B \
  --action_type joint \
  --vision_backbone resnet50 \
  --adapter_type mlp \
  --n_obs_steps 2 \
  --temporal_agg concat \
  --batch_size 16 \
  --epochs 500
```

**End-effector pose, Vision Transformer with frozen backbone:**
```bash
python train.py \
  --data_path ./robot_data \
  --model_type DiT-L \
  --action_type endpose \
  --vision_backbone vit_b_16 \
  --adapter_type attention_pooling \
  --freeze_vision \
  --batch_size 16 \
  --epochs 500
```

**With AMP and warmup scheduler:**
```bash
python train.py \
  --data_path ./robot_data \
  --model_type DiT-B \
  --action_type joint \
  --vision_backbone resnet50 \
  --use_amp \
  --warmup_epochs 10 \
  --warmup_type cosine \
  --batch_size 16 \
  --epochs 500
```

## Evaluation / Inference

Use `eval.py` to run inference with a trained checkpoint:

```bash
python eval.py \
  --checkpoint checkpoints/final.pt \
  --data_path ./robot_data \
  --ddim_steps 10 \
  --cfg_scale 1.0 \
  --num_samples 5
```

**Eval Arguments:**
- `--checkpoint`: Path to model checkpoint (required)
- `--data_path`: Path to dataset for evaluation (optional — runs demo inference if omitted)
- `--num_samples`: Number of samples to evaluate (default: 5)
- `--ddim_steps`: DDIM sampling steps (default: 10)
- `--cfg_scale`: Classifier-free guidance scale (default: 1.0; set >1.0 to enable CFG)

The evaluation script reports MSE and L1 metrics between predicted and ground-truth actions. By default, only the first `n_action_steps` actions are compared (receding horizon control).

## Model Architecture Details

### Information Flow

```
Camera Images (B, n_obs_steps, num_cams, 3, H, W)
    ↓
Vision Backbone (ResNet/ViT) — per frame, per camera
    ↓
Camera Fusion — concat + linear projection
    ↓
Per-frame Features (B, n_obs_steps, 1, feature_dim)
    ↓
Temporal Aggregation (last / mean / concat)
    ↓
Global Vision Feature (B, 1, feature_dim)
    ↓
Feature Adapter (Linear/MLP/AttentionPooling)
    ↓
Vision Condition (B, 1, token_size)
    ↓
DiT Model (with Diffusion Process + optional CFG)
    ↓
Predicted Actions (B, future_window, action_dim)
```

### DiT Model Components

- **ActionEmbedder**: Projects noisy action sequences from `action_dim` to `hidden_size`
- **TimestepEmbedder**: Sinusoidal embeddings + MLP for diffusion timesteps
- **LabelEmbedder**: Projects vision condition to `hidden_size`; supports label dropout for CFG
- **DiTBlock**: Transformer blocks with self-attention and adaptive layer norm
- **FinalLayer**: Output layer for action prediction

### Diffusion Process

- **Forward (q_sample)**: Adds noise to ground truth actions
- **Training**: Model learns to predict the noise (epsilon)
- **Inference**: Iteratively denoises random noise to generate actions
- **DDIM Sampling**: Fast deterministic sampling for inference
- **Classifier-Free Guidance**: Conditional/unconditional interpolation via `cfg_scale`

## File Structure

```
Nemo-Diffusion-Transformer/
├── model/
│   ├── action_model/
│   │   ├── action_model.py        # ActionModel with vision conditioning & sample()
│   │   ├── models.py              # Core DiT architecture (S/B/L/XL)
│   │   ├── gaussian_diffusion.py  # Diffusion process & DDIM
│   │   └── ...
│   ├── vision_input/
│   │   ├── vision_input.py        # Vision backbone (ResNet/ViT) + temporal aggregation
│   │   └── __init__.py
│   └── feature_adaptation.py      # Feature projection layers (Linear/MLP/AttentionPooling)
├── utils/
│   ├── __init__.py
│   └── rotation_utils.py          # Quaternion / rot6d / rotation matrix conversions
├── train.py                       # Training script (AMP, warmup, CFG dropout)
├── eval.py                        # Evaluation / inference script
├── dataloader.py                  # Robot dataset loader (multi-frame, endpose/joint)
└── README.md                      # This file
```

## Key Features

1. **No VAE Required**: Direct action prediction without VAE encoding
2. **Multi-Camera Support**: Integrates observations from multiple viewpoints with learned camera fusion
3. **Multi-Frame Observations**: Temporal visual conditioning via `n_obs_steps` with configurable aggregation
4. **Flexible Vision Backbones**: Choose between ResNet and ViT architectures
5. **Multiple Adapter Strategies**: Linear, MLP, or attention-based feature adaptation
6. **Dual Action Types**: End-effector pose (with automatic rot6d conversion) or joint angles
7. **Dual-Arm Support**: Single-arm and dual-arm configurations with automatic `action_dim` calculation
8. **Classifier-Free Guidance**: Conditional generation with configurable guidance scale
9. **Mixed Precision Training**: Optional FP16 AMP for faster training
10. **Warmup Scheduling**: Linear or cosine learning rate warmup
11. **Receding Horizon Control**: `n_action_steps` for executing a subset of predicted actions
12. **Efficient Data Loading**: HDF5-based lazy loading for large datasets
13. **Pretrained Weights**: Leverages ImageNet pretrained vision models

## Checkpoints

Checkpoints are saved to the `checkpoints/` directory:
- `{epoch}.pt`: Periodic checkpoints (every `--save_every` epochs)
- `latest.pt`: Most recent checkpoint
- `final.pt`: Final model after training

Each checkpoint contains model weights, optimizer state, scheduler state, scaler state (if AMP), and all training arguments.

To resume training:
```bash
python train.py --resume checkpoints/latest.pt --data_path ./robot_data
```

## Inference (Python API)

```python
import torch
from model.action_model.action_model import ActionModel

# Load model
model = ActionModel(
    token_size=2048,
    model_type='DiT-B',
    in_channels=14,  # e.g. dual-arm joint: (6+1)*2 = 14
    future_action_window_size=12,
    past_action_window_size=0,
    n_obs_steps=2,
    n_action_steps=8,
    temporal_agg='concat',
    use_vision_condition=True,
    vision_backbone_type='resnet50',
    num_cameras=4,
    class_dropout_prob=0.1,
    diffusion_steps=500,
    noise_schedule='squaredcos_cap_v2',
)

# Load checkpoint
checkpoint = torch.load('checkpoints/final.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval().cuda()

# Generate actions from images
with torch.no_grad():
    # images: (1, n_obs_steps, num_cameras, 3, 224, 224)
    images = torch.randn(1, 2, 4, 3, 224, 224).cuda()

    # sample() handles vision encoding, DDIM sampling, and CFG internally
    actions = model.sample(
        images,
        ddim_steps=10,
        use_ddim=True,
        cfg_scale=1.0,
        return_all=False,  # returns only n_action_steps actions
    )
    # actions: (1, 8, 14) — (batch, n_action_steps, action_dim)
```

## Customization

### Adding New Vision Backbones

Edit `model/vision_input/vision_input.py` to add new backbone architectures:

```python
def _create_custom_backbone(self, pretrained):
    model = YourCustomModel(pretrained=pretrained)
    feature_dim = model.feature_dim
    return model, feature_dim
```

### Custom Feature Adapters

Create new adapter classes in `model/feature_adaptation.py` following the base adapter interface.

## Troubleshooting

1. **Out of Memory**: Reduce `--batch_size`, use smaller model (`DiT-S`), freeze vision backbone (`--freeze_vision`), or enable AMP (`--use_amp`)
2. **Slow Training**: Use `--freeze_vision` to freeze pretrained weights, reduce `--num_cameras`, or enable `--use_amp`
3. **Data Loading Errors**: Verify HDF5 file structure matches expected format; check `--action_type` and `--quat_convention` match your data

## Citation

If you use this code, please cite:

- DiT: Peebles & Xie, "Scalable Diffusion Models with Transformers", ICCV 2023
- Your robot learning work

## License

This code is built upon the DiT implementation from Meta/Facebook Research.
