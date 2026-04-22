#!/bin/bash
# ============================================================================
# NemoDiT (Qwen3-VL + Flow Matching) Training Script for RoboTwin
# ============================================================================
#
# Usage:
#   bash train.sh ${task_name} ${task_config} ${expert_data_num} ${seed} ${gpu_id}
#
# Example:
#   bash train.sh beat_block_hammer default 50 0 0
#   → data from ../../data/beat_block_hammer/demo_clean_50/data
#
# ============================================================================

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
gpu_id=${5}

# ---- Model ---------------------------------------------------------------
model_type="DiT-S"                                # DiT-S | DiT-B | DiT-L | DiT-XL

# ---- VLM -----------------------------------------------------------------
vlm_model_name="Qwen/Qwen3-VL-4B-Instruct"
freeze_vlm="--freeze_vlm"                         # "" disables freezing
use_lora=""                                       # "--use_lora" to enable
lora_r=16
lora_alpha=32

# ---- Action spec ---------------------------------------------------------
action_type="joint"                               # joint | endpose
use_both_arms="--use_both_arms"
quat_convention="wxyz"

# ---- Temporal config -----------------------------------------------------
n_obs_steps=2
n_action_steps=8
future_action_window=12
past_action_window=0
temporal_agg="concat"                             # last | mean | concat

# ---- Flow matching -------------------------------------------------------
time_sampling="beta"                      # logit_normal | beta | uniform
logit_normal_loc=0.0
logit_normal_scale=1.0
beta_alpha=1.5
beta_beta=1.0
num_timestep_buckets=1000
num_inference_steps=10                            # inference-only

# ---- torch.compile (Fix B) ----------------------------------------------
use_compile="--use_compile"                       # "--no_compile" to disable
compile_mode="default"                            # default | reduce-overhead | max-autotune

# ---- Optimization --------------------------------------------------------
epochs=500
batch_size=16
lr=1e-4
vlm_lr=1e-5
weight_decay=0.01
grad_clip=1.0
num_workers=4
save_every=50

# ---- Camera --------------------------------------------------------------
num_cameras=4

# ---- WandB ---------------------------------------------------------------
use_wandb="--use_wandb"
wandb_project="robotwin_qwen3vl_fm"
wandb_entity=""

# ---- Instructions --------------------------------------------------------
# Leave instructions_path empty to auto-derive <data_path>/../instructions.
# Set to "disable" to use only the fallback instruction string.
instructions_path=""
instruction_split="seen"

# ==========================================================================
export CUDA_VISIBLE_DEVICES=${gpu_id}

echo -e "\033[33m============================================\033[0m"
echo -e "\033[33m[NemoDiT] Qwen3-VL + Flow Matching Training\033[0m"
echo -e "\033[33m============================================\033[0m"
echo -e "\033[33m  GPU=${gpu_id}  Task=${task_name}  Config=${task_config}\033[0m"
echo -e "\033[33m  Seed=${seed}  Model=${model_type}  Action=${action_type}\033[0m"
echo -e "\033[33m  VLM=${vlm_model_name}\033[0m"
echo -e "\033[33m============================================\033[0m"

data_path="../../data/${task_name}/demo_clean_${expert_data_num}/data"
checkpoint_dir="checkpoints/${task_name}-${task_config}-${expert_data_num}-${seed}"

# Optional: pass --instructions_path only if set to something non-empty.
instructions_flag=""
if [ -n "${instructions_path}" ]; then
    instructions_flag="--instructions_path ${instructions_path}"
fi

python train.py \
    --data_path ${data_path} \
    --num_cameras ${num_cameras} \
    ${use_both_arms} \
    --action_type ${action_type} \
    --quat_convention ${quat_convention} \
    --model_type ${model_type} \
    --vlm_model_name ${vlm_model_name} \
    ${freeze_vlm} \
    ${use_lora} \
    --lora_r ${lora_r} \
    --lora_alpha ${lora_alpha} \
    --n_obs_steps ${n_obs_steps} \
    --n_action_steps ${n_action_steps} \
    --future_action_window ${future_action_window} \
    --past_action_window ${past_action_window} \
    --temporal_agg ${temporal_agg} \
    --time_sampling ${time_sampling} \
    --logit_normal_loc ${logit_normal_loc} \
    --logit_normal_scale ${logit_normal_scale} \
    --beta_alpha ${beta_alpha} \
    --beta_beta ${beta_beta} \
    --num_timestep_buckets ${num_timestep_buckets} \
    --num_inference_steps ${num_inference_steps} \
    ${use_compile} \
    --compile_mode ${compile_mode} \
    --epochs ${epochs} \
    --batch_size ${batch_size} \
    --lr ${lr} \
    --vlm_lr ${vlm_lr} \
    --weight_decay ${weight_decay} \
    --grad_clip ${grad_clip} \
    --num_workers ${num_workers} \
    --save_every ${save_every} \
    --checkpoint_dir ${checkpoint_dir} \
    ${use_wandb} \
    --wandb_project ${wandb_project} \
    --wandb_entity "${wandb_entity}" \
    --instruction_split ${instruction_split} \
    ${instructions_flag} \
    --device cuda:0

echo -e "\033[32m[NemoDiT] Training completed\033[0m"
echo -e "\033[32m[NemoDiT] Checkpoints: ${checkpoint_dir}\033[0m"
