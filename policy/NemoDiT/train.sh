#!/bin/bash
# ============================================================================
# NemoDiT (Qwen3VL + Flow Matching) Policy Training Script for RoboTwin
# ============================================================================
#
# Architecture:
#   Qwen3-VL (frozen or LoRA) -> VLM hidden states
#   -> VLM projection -> Cross-Attention DiT (flow matching) -> actions
#
# Usage:
#   bash train.sh ${task_name} ${task_config} ${expert_data_num} ${seed} ${gpu_id}
#
# Example:
#   bash train.sh beat_block_hammer default 50 0 0
#   从 data/beat_block_hammer/demo_clean/data/ 加载数据
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
# VLM Configuration
# ============================================================================
vlm_model_name="Qwen/Qwen3-VL-4B-Instruct"   # Options: Qwen/Qwen3-VL-4B-Instruct, Qwen/Qwen3-VL-7B-Instruct
freeze_vlm="--freeze_vlm"             # Freeze VLM weights (recommended)
# use_lora="--use_lora"               # Uncomment to enable LoRA fine-tuning of VLM
use_lora=""
lora_r=16
lora_alpha=32

# Task instruction for VLM conditioning.
# NOTE: This is only used as a fallback when per-episode instruction JSON files
# are not found. By default the dataloader will look for
#   ${data_path}/../instructions/episode{N}.json
# and randomly sample one string from the chosen split (default: "seen") for
# every training step.
instruction="Pick the silver curved hammer head from the table and strike"
# Leave instructions_path empty to auto-derive as <data_path>/../instructions.
# Set to a path to override, or set to the literal string "disable" to turn off
# auto-loading and always use the fallback instruction above.
instructions_path=""
instruction_split="seen"        # Which JSON split to sample from: seen / unseen
# no_random_instruction="--no_random_instruction"   # Uncomment to disable random sampling
no_random_instruction=""

# ============================================================================
# DiT Action Model Configuration
# ============================================================================
dit_model_type="DiT-S"        # DiT size: DiT-S, DiT-B, DiT-L, DiT-XL

# Action configuration
action_type="joint"           # Action type: endpose or joint
use_both_arms="--use_both_arms"   # Use dual arm (comment out for single arm)
quat_convention="wxyz"        # Quaternion convention in HDF5 data

# Temporal configuration
n_obs_steps=1                 # Number of observation history steps
n_action_steps=8              # Number of action steps to execute per inference
future_action_window=13       # Number of future action steps to predict

# ============================================================================
# Flow Matching Configuration
# ============================================================================
time_sampling="logit_normal"  # Time sampling: logit_normal, beta, uniform
logit_normal_loc=0.0
logit_normal_scale=1.0
beta_alpha=1.5
beta_beta=1.0
num_timestep_buckets=1000
num_inference_steps=10        # ODE integration steps (used at inference)

# ============================================================================
# Training Configuration
# ============================================================================
epochs=500                    # Number of training epochs
batch_size=32                  # Batch size per GPU (smaller due to VLM memory)
lr=1e-4                       # Learning rate for action head
vlm_lr=1e-5                   # Learning rate for VLM params (if unfrozen/LoRA)
weight_decay=0.01             # Weight decay for L2 regularization
grad_clip=1.0                 # Gradient clipping max norm
num_workers=4                 # Number of data loading workers
gradient_accumulation_steps=1
# use_amp="--use_amp"         # Uncomment to enable AMP mixed precision
use_amp=""

# Warmup
warmup_epochs=0
warmup_type="linear"          # linear or cosine

# EMA (exponential moving average)
# use_ema="--use_ema"         # Uncomment to enable EMA
use_ema="--use_ema"
ema_max_value=0.9999

# Checkpointing
save_every=50                 # Save checkpoint every N epochs
# resume="checkpoints/.../latest.pt"   # Uncomment to resume training
resume_arg=""

# Camera configuration
num_cameras=4                 # Number of camera views

# ============================================================================
# WandB Configuration
# ============================================================================
use_wandb="--use_wandb"                           # Set empty to disable WandB
wandb_project="robotwin_nemodit"
wandb_entity="conkicx-xi-an-jiaotong-liverpool-university"
wandb_name="nemodit_${task_name}_${dit_model_type}_${action_type}_${expert_data_num}_seed${seed}"

# ============================================================================
# Setup
# ============================================================================

# Set GPU device
export CUDA_VISIBLE_DEVICES=${gpu_id}

# Print training info
echo -e "\033[33m============================================\033[0m"
echo -e "\033[33m[NemoDiT] Qwen3VL + Flow Matching Training\033[0m"
echo -e "\033[33m============================================\033[0m"
echo -e "\033[33m[NemoDiT] GPU:           ${gpu_id}\033[0m"
echo -e "\033[33m[NemoDiT] Task:          ${task_name}\033[0m"
echo -e "\033[33m[NemoDiT] Config:        ${task_config}\033[0m"
echo -e "\033[33m[NemoDiT] Expert Data:   ${expert_data_num}\033[0m"
echo -e "\033[33m[NemoDiT] Seed:          ${seed}\033[0m"
echo -e "\033[33m[NemoDiT] VLM:           ${vlm_model_name}\033[0m"
echo -e "\033[33m[NemoDiT] DiT:           ${dit_model_type}\033[0m"
echo -e "\033[33m[NemoDiT] Action Type:   ${action_type}\033[0m"
echo -e "\033[33m============================================\033[0m"

# Data path
# Format: data/{task_name}/demo_clean/data
data_path="../../data/${task_name}/demo_clean/data"

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
    --instruction "${instruction}" \
    ${instructions_path:+--instructions_path "${instructions_path}"} \
    --instruction_split ${instruction_split} \
    ${no_random_instruction} \
    --vlm_model_name ${vlm_model_name} \
    ${freeze_vlm} \
    ${use_lora} \
    --lora_r ${lora_r} \
    --lora_alpha ${lora_alpha} \
    --dit_model_type ${dit_model_type} \
    --future_action_window ${future_action_window} \
    --n_obs_steps ${n_obs_steps} \
    --n_action_steps ${n_action_steps} \
    --time_sampling ${time_sampling} \
    --logit_normal_loc ${logit_normal_loc} \
    --logit_normal_scale ${logit_normal_scale} \
    --beta_alpha ${beta_alpha} \
    --beta_beta ${beta_beta} \
    --num_timestep_buckets ${num_timestep_buckets} \
    --num_inference_steps ${num_inference_steps} \
    --batch_size ${batch_size} \
    --epochs ${epochs} \
    --lr ${lr} \
    --vlm_lr ${vlm_lr} \
    --weight_decay ${weight_decay} \
    --grad_clip ${grad_clip} \
    --num_workers ${num_workers} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    ${use_amp} \
    --warmup_epochs ${warmup_epochs} \
    --warmup_type ${warmup_type} \
    ${use_ema} \
    --ema_max_value ${ema_max_value} \
    --checkpoint_dir ${checkpoint_dir} \
    --save_every ${save_every} \
    ${resume_arg} \
    ${use_wandb} \
    --wandb_project ${wandb_project} \
    --wandb_entity ${wandb_entity} \
    --wandb_name ${wandb_name} \
    --device cuda:0

echo -e "\033[32m[NemoDiT] Training completed!\033[0m"
echo -e "\033[32m[NemoDiT] Checkpoints saved to: ${checkpoint_dir}\033[0m"
