#!/bin/bash
# ============================================================================
# NemoDiT Policy Training Script for RoboTwin
# ============================================================================
#
# Usage:
#   bash train.sh ${task_name} ${task_config} ${expert_data_num} ${seed} ${gpu_id}
#
# Example:
#   bash train.sh beat_block_hammer default 50 0 0
#   这会从 data/beat_block_hammer/demo_clean_50/data/ 加载数据
#
# Arguments:
#   task_name       - Name of the task to train on
#   task_config     - Task configuration (e.g., default)
#   expert_data_num - Number of expert demonstrations to use
#   seed            - Random seed for reproducibility
#   gpu_id          - GPU device ID
#
# ============================================================================

# Parse command line arguments
task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
gpu_id=${5}

# ============================================================================
# Training Configuration
# ============================================================================

# Model configuration
model_type="DiT-B"           # Model size: DiT-S, DiT-B, DiT-L, DiT-XL
vision_backbone="resnet50"    # Vision backbone: resnet18, resnet34, resnet50, vit_b_16, vit_b_32
adapter_type="mlp"            # Feature adapter: linear, mlp, attention_pooling

# Action configuration
action_type="joint"         # Action type: endpose or joint
use_both_arms="--use_both_arms"  # Use dual arm (comment out for single arm)
quat_convention="wxyz"        # Quaternion convention in HDF5 data

# Temporal configuration
n_obs_steps=2                 # Number of observation history steps
n_action_steps=8              # Number of action steps to execute per inference
future_action_window=12       # Number of future action steps to predict
past_action_window=0          # Number of past action steps as context
temporal_agg="concat"           # Temporal aggregation: last, mean, concat

# Diffusion configuration
diffusion_steps=100           # Number of diffusion steps
noise_schedule="squaredcos_cap_v2"  # Noise schedule type
ddim_steps=10                 # DDIM sampling steps (for inference)

# Training configuration
epochs=500                   # Number of training epochs
batch_size=32                 # Batch size per GPU
lr=1e-4                       # Learning rate
weight_decay=0.01             # Weight decay for L2 regularization
grad_clip=1.0                 # Gradient clipping max norm
num_workers=4                 # Number of data loading workers
save_every=50                # Save checkpoint every N epochs

# Camera configuration
num_cameras=4                 # Number of camera views
head_camera_type="D435"       # Head camera type

# ============================================================================
# WandB Configuration
# ============================================================================
use_wandb="--use_wandb"                  # Set to "--use_wandb" to enable WandB logging
wandb_project="robotwin_nemodit" # WandB project name
wandb_entity=""               # WandB entity (username or team name)
                              # Leave empty to use your default entity

# ============================================================================
# Setup
# ============================================================================

# Set GPU device
export CUDA_VISIBLE_DEVICES=${gpu_id}

# Print training info
echo -e "\033[33m============================================\033[0m"
echo -e "\033[33m[NemoDiT] Policy Training\033[0m"
echo -e "\033[33m============================================\033[0m"
echo -e "\033[33m[NemoDiT] GPU: ${gpu_id}\033[0m"
echo -e "\033[33m[NemoDiT] Task: ${task_name}\033[0m"
echo -e "\033[33m[NemoDiT] Config: ${task_config}\033[0m"
echo -e "\033[33m[NemoDiT] Expert Data: ${expert_data_num}\033[0m"
echo -e "\033[33m[NemoDiT] Seed: ${seed}\033[0m"
echo -e "\033[33m[NemoDiT] Model: ${model_type}\033[0m"
echo -e "\033[33m[NemoDiT] Action Type: ${action_type}\033[0m"
echo -e "\033[33m============================================\033[0m"

# Data path
# 格式: data/{task_name}/demo_clean_{expert_data_num}/data
# 例如: data/beat_block_hammer/demo_clean_50/data
data_path="../../data/${task_name}/demo_clean_${expert_data_num}/data"
# Checkpoint directory
checkpoint_dir="checkpoints/${task_name}-${task_config}-${expert_data_num}-${seed}"

# ============================================================================
# Run Training
# ============================================================================

python train.py \
    --data_path ${data_path} \
    --num_cameras ${num_cameras} \
    ${use_both_arms} \
    --action_type ${action_type} \
    --quat_convention ${quat_convention} \
    --model_type ${model_type} \
    --vision_backbone ${vision_backbone} \
    --vision_pretrained \
    --adapter_type ${adapter_type} \
    --n_obs_steps ${n_obs_steps} \
    --n_action_steps ${n_action_steps} \
    --future_action_window ${future_action_window} \
    --past_action_window ${past_action_window} \
    --temporal_agg ${temporal_agg} \
    --diffusion_steps ${diffusion_steps} \
    --noise_schedule ${noise_schedule} \
    --epochs ${epochs} \
    --batch_size ${batch_size} \
    --lr ${lr} \
    --weight_decay ${weight_decay} \
    --grad_clip ${grad_clip} \
    ${use_wandb} \
    --wandb_project ${wandb_project} \
    --wandb_entity ${wandb_entity} \
    --num_workers ${num_workers} \
    --checkpoint_dir ${checkpoint_dir} \
    --save_every ${save_every} \
    --device cuda:0

echo -e "\033[32m[NemoDiT] Training completed!\033[0m"
echo -e "\033[32m[NemoDiT] Checkpoints saved to: ${checkpoint_dir}\033[0m"
