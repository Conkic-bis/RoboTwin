# Usage:
#   bash eval.sh ${task_name} ${task_config} ${ckpt_setting} ${expert_data_num} ${seed} ${gpu_id}
#
# Example:
#   bash eval.sh pick_place default default 100 0 0
#
# Arguments:
#   task_name       - Name of the task to evaluate
#   task_config     - Task configuration (e.g., default)
#   ckpt_setting    - Checkpoint setting name
#   expert_data_num - Number of expert demonstrations used for training
#   seed            - Random seed
#   gpu_id          - GPU device ID
#
# ============================================================================

# Policy name - should match the directory name in RoboTwin/policy/
policy_name=NemoDiT

# Parse command line arguments
task_name=${1}
task_config=${2}
ckpt_setting=${3}
expert_data_num=${4}
seed=${5}
gpu_id=${6}

export CUDA_VISIBLE_DEVICES=${gpu_id}

# Print evaluation info
echo -e "\033[33m============================================\033[0m"
echo -e "\033[33m[NemoDiT] Policy Evaluation\033[0m"
echo -e "\033[33m============================================\033[0m"
echo -e "\033[33m[NemoDiT] GPU: ${gpu_id}\033[0m"
echo -e "\033[33m[NemoDiT] Task: ${task_name}\033[0m"
echo -e "\033[33m[NemoDiT] Config: ${task_config}\033[0m"
echo -e "\033[33m[NemoDiT] Checkpoint: ${ckpt_setting}\033[0m"
echo -e "\033[33m[NemoDiT] Expert Data: ${expert_data_num}\033[0m"
echo -e "\033[33m[NemoDiT] Seed: ${seed}\033[0m"
echo -e "\033[33m============================================\033[0m"

# Navigate to RoboTwin root directory
cd ../..

# Run evaluation
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
