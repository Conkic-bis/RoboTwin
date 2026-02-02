# DiT for Robot Action Generation with Vision Conditioning

This project implements a Diffusion Transformer (DiT) model for robot action generation, conditioned on visual observations from multiple cameras.

## Architecture Overview

### Core Components

1. **Vision Backbone** (`model/vision_input/vision_input.py`)
   - Extracts visual features from robot camera observations
   - Supports multiple backbone architectures:
     - ResNet (ResNet18, ResNet34, ResNet50)
     - Vision Transformer (ViT-B/16, ViT-B/32)
   - Handles multiple camera views (front, head, left, right)
   - Output: `(batch_size, num_patches, vision_feature_dim)`

2. **Feature Adapter** (`model/feature_adaptation.py`)
   - Projects vision features to DiT's expected token dimension
   - Three adapter types:
     - **Linear**: Simple linear projection
     - **MLP**: Multi-layer perceptron for more expressive transformation
     - **Attention Pooling**: Learnable query-based feature aggregation
   - Output: `(batch_size, num_patches, token_size)`

3. **DiT Model** (`model/action_model/models.py` & `action_model.py`)
   - Transformer-based diffusion model for action prediction
   - Accepts visual conditions and generates robot actions
   - Supports three model sizes: DiT-S, DiT-B, DiT-L
   - Predicts future action sequences with temporal coherence

4. **Data Loader** (`dataloader.py`)
   - Loads robot manipulation data from HDF5 files
   - Supports both single-arm and dual-arm configurations
   - Handles multiple camera views
   - Efficient lazy loading for large datasets

## Data Format

Robot data should be organized in HDF5 files with the following structure:

```
episode_X.hdf5
├── /endpose
│   ├── /endpose/left_endpose     (T, 6)  - 6-DoF end-effector pose
│   ├── /endpose/left_gripper     (T, 1)  - gripper state
│   ├── /endpose/right_endpose    (T, 6)  - 6-DoF end-effector pose (optional)
│   └── /endpose/right_gripper    (T, 1)  - gripper state (optional)
└── /observation
    ├── /observation/front_camera/rgb   (T, H, W, 3)
    ├── /observation/head_camera/rgb    (T, H, W, 3)
    ├── /observation/left_camera/rgb    (T, H, W, 3)
    └── /observation/right_camera/rgb   (T, H, W, 3)
```

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
  --action_dim 7 \
  --future_action_window 10 \
  --num_cameras 4 \
  --vision_backbone resnet50 \
  --batch_size 32 \
  --epochs 1000 \
  --lr 1e-4
```

### Key Arguments

**Data Arguments:**
- `--data_path`: Path to directory containing episode HDF5 files
- `--num_cameras`: Number of camera views (default: 4)

**Model Arguments:**
- `--model_type`: DiT model size (DiT-S, DiT-B, DiT-L)
- `--action_dim`: Action dimension (default: 7 for 6-DoF + gripper)
- `--future_action_window`: Number of future actions to predict (default: 10)
- `--token_size`: Token size for conditioning (default: 2048)

**Vision Arguments:**
- `--vision_backbone`: Vision backbone type (resnet18, resnet34, resnet50, vit_b_16, vit_b_32)
- `--vision_pretrained`: Use pretrained weights (default: True)
- `--freeze_vision`: Freeze vision backbone (default: False)
- `--adapter_type`: Feature adapter type (linear, mlp, attention_pooling)

**Training Arguments:**
- `--batch_size`: Batch size (default: 32)
- `--epochs`: Number of epochs (default: 1000)
- `--lr`: Learning rate (default: 1e-4)
- `--checkpoint_dir`: Directory to save checkpoints

### Example Training Commands

**Using ResNet50 backbone:**
```bash
python train.py \
  --data_path ./robot_data \
  --model_type DiT-B \
  --vision_backbone resnet50 \
  --adapter_type mlp \
  --batch_size 32 \
  --epochs 1000
```

**Using Vision Transformer:**
```bash
python train.py \
  --data_path ./robot_data \
  --model_type DiT-L \
  --vision_backbone vit_b_16 \
  --adapter_type attention_pooling \
  --batch_size 16 \
  --epochs 1000 \
  --freeze_vision
```

**Dual-arm robot:**
```bash
python train.py \
  --data_path ./robot_data \
  --model_type DiT-B \
  --action_dim 14 \
  --vision_backbone resnet50 \
  --batch_size 32
```

## Model Architecture Details

### Information Flow

```
Camera Images (B, num_cams, 3, H, W)
    ↓
Vision Backbone (ResNet/ViT)
    ↓
Vision Features (B, num_patches, vision_dim)
    ↓
Feature Adapter (Linear/MLP/AttentionPooling)
    ↓
Adapted Features (B, num_patches, token_size)
    ↓
DiT Model (with Diffusion Process)
    ↓
Predicted Actions (B, future_window, action_dim)
```

### DiT Model Components

- **ActionEmbedder**: Embeds noisy action sequences
- **TimestepEmbedder**: Embeds diffusion timesteps
- **LabelEmbedder**: Processes vision conditions
- **DiTBlock**: Transformer blocks with self-attention
- **FinalLayer**: Output layer for action prediction

### Diffusion Process

- **Forward (q_sample)**: Adds noise to ground truth actions
- **Training**: Model learns to predict the noise
- **Inference**: Iteratively denoises random noise to generate actions
- **DDIM Sampling**: Fast deterministic sampling for inference

## File Structure

```
Nemo-Diffusion-Transformer/
├── model/
│   ├── action_model/
│   │   ├── action_model.py        # ActionModel with vision conditioning
│   │   ├── models.py              # Core DiT architecture
│   │   ├── gaussian_diffusion.py  # Diffusion process
│   │   └── ...
│   ├── vision_input/
│   │   ├── vision_input.py        # Vision backbone (ResNet/ViT)
│   │   └── __init__.py
│   └── feature_adaptation.py      # Feature projection layers
├── train.py                       # Training script
├── dataloader.py                  # Robot dataset loader
└── README_ROBOT_ACTION.md         # This file
```

## Key Features

1. **No VAE Required**: Direct action prediction without VAE encoding
2. **Multi-Camera Support**: Integrates observations from multiple viewpoints
3. **Flexible Vision Backbones**: Choose between ResNet and ViT architectures
4. **Multiple Adapter Strategies**: Linear, MLP, or attention-based feature adaptation
5. **Temporal Action Prediction**: Generates coherent action sequences
6. **Efficient Data Loading**: HDF5-based lazy loading for large datasets
7. **Pretrained Weights**: Leverages ImageNet pretrained vision models

## Checkpoints

Checkpoints are saved to the `checkpoints/` directory:
- `dit_action_checkpoint_epoch_X_step_Y.pt`: Periodic checkpoints
- `latest.pt`: Most recent checkpoint
- `final.pt`: Final model after training

To resume training:
```bash
python train.py --resume checkpoints/latest.pt --data_path ./robot_data
```

## Inference

For inference, you can use the trained model to generate action sequences:

```python
import torch
from model.action_model.action_model import ActionModel

# Load model
model = ActionModel(
    token_size=2048,
    model_type='DiT-B',
    in_channels=7,
    future_action_window_size=10,
    past_action_window_size=0,
    use_vision_condition=True,
    vision_backbone_type='resnet50',
    num_cameras=4
)

# Load checkpoint
checkpoint = torch.load('checkpoints/final.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Create DDIM sampler for fast inference
model.create_ddim(ddim_step=10)

# Generate actions from images
with torch.no_grad():
    # images: (1, 4, 3, 224, 224) - batch of 1, 4 cameras
    vision_condition = model.encode_vision_condition(images)

    # Sample actions using DDIM
    # Initial random noise
    x_T = torch.randn(1, 10, 7)  # (batch, future_window, action_dim)

    # Denoise to generate actions
    # ... (use model.ddim_diffusion.p_sample_loop)
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

1. **Out of Memory**: Reduce `--batch_size`, use smaller model (`DiT-S`), or freeze vision backbone
2. **Slow Training**: Use `--freeze_vision` to freeze pretrained weights, reduce `--num_cameras`
3. **Data Loading Errors**: Verify HDF5 file structure matches expected format

## Citation

If you use this code, please cite:

- DiT: Peebles & Xie, "Scalable Diffusion Models with Transformers", ICCV 2023
- Your robot learning work

## License

This code is built upon the DiT implementation from Meta/Facebook Research.
