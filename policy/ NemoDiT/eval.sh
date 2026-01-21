#!/bin/bash
# ============================================================================
# NemoDiT Policy Evaluation Script for RoboTwin
# ============================================================================
#
# Usage:
#   bash eval.sh ${task_name} ${task_config} ${ckpt_setting} ${seed} ${gpu_id}
#
# Example:
#   bash eval.sh pick_place default default 0 0
#
# ============================================================================

# Policy name - should match the directory name in RoboTwin/policy/
policy_name=NemoDiT

# Parse command line arguments
task_name=${1}
task_config=${2}
ckpt_setting=${3}
seed=${4}
gpu_id=${5}

# Set GPU device
export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33m[NemoDiT] Using GPU: ${gpu_id}\033[0m"
echo -e "\033[33m[NemoDiT] Task: ${task_name}\033[0m"
echo -e "\033[33m[NemoDiT] Config: ${task_config}\033[0m"
echo -e "\033[33m[NemoDiT] Checkpoint: ${ckpt_setting}\033[0m"
echo -e "\033[33m[NemoDiT] Seed: ${seed}\033[0m"

# Navigate to RoboTwin root directory
cd ../..

# Run evaluation
PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/${policy_name}/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --seed ${seed} \
    --policy_name ${policy_name}

echo -e "\033[32m[NemoDiT] Evaluation completed!\033[0m"
