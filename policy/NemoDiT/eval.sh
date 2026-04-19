#!/bin/bash
# ============================================================================
# NemoDiT (Qwen3VL + Flow Matching) Evaluation Script (RoboTwin)
# ============================================================================
#
# Usage:
#   bash eval.sh ${task_name} ${task_config} ${ckpt_setting} ${expert_data_num} ${seed} ${gpu_id}
#
# Example:
#   bash eval.sh beat_block_hammer demo_clean demo_clean 50 0 0
#
# ============================================================================

policy_name="$(basename "$(cd "$(dirname "$0")" && pwd)")"

task_name=${1}
task_config=${2}
ckpt_setting=${3}
expert_data_num=${4}
seed=${5}
gpu_id=${6}

export CUDA_VISIBLE_DEVICES=${gpu_id}

echo -e "\033[33m============================================\033[0m"
echo -e "\033[33m[NemoDiT] Policy Evaluation\033[0m"
echo -e "\033[33m============================================\033[0m"
echo -e "\033[33m[NemoDiT] GPU:           ${gpu_id}\033[0m"
echo -e "\033[33m[NemoDiT] Task:          ${task_name}\033[0m"
echo -e "\033[33m[NemoDiT] Config:        ${task_config}\033[0m"
echo -e "\033[33m[NemoDiT] Ckpt Setting:  ${ckpt_setting}\033[0m"
echo -e "\033[33m[NemoDiT] Expert Data:   ${expert_data_num}\033[0m"
echo -e "\033[33m[NemoDiT] Seed:          ${seed}\033[0m"
echo -e "\033[33m============================================\033[0m"

cd ../..

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/${policy_name}/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --expert_data_num ${expert_data_num} \
    --seed ${seed} \
    --policy_name ${policy_name}

echo -e "\033[32m[NemoDiT] Evaluation completed!\033[0m"