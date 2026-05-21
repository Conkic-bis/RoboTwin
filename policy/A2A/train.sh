#!/bin/bash
# Usage:
#   bash train.sh <task_name> <task_config> <expert_data_num> <seed> <gpu_id> [variant]
# Example:
#   bash train.sh beat_block_hammer demo_randomized 50 0 0            # plain a2a
#   bash train.sh beat_block_hammer demo_randomized 50 0 0 a2a_noise  # noise variant
#
# variant         (optional, default "a2a") which A2A variant to train.
#                 Currently supported: a2a, a2a_noise.
#                 Selects ./a2a_flow_matching/config/robot_${variant}.yaml.
#                 Noise variant saves to a separate ckpt dir (suffix "-noise").
#
# action_dim is auto-detected from the zarr's meta attrs — no need to pass it.

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
gpu_id=${5}
variant=${6:-a2a}

head_camera_type=D435
DEBUG=False

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

if [ "$DEBUG" = "True" ]; then
    wandb_mode=offline
    echo -e "\033[33mDebug mode (wandb offline)\033[0m"
elif [ -z "${WANDB_API_KEY:-}" ] && [ ! -f "$HOME/.netrc" ]; then
    # No API key and no cached login -> fall back to offline so training
    # doesn't block on `wandb login`.
    wandb_mode=offline
    echo -e "\033[33mWANDB_API_KEY not set; running wandb offline\033[0m"
else
    wandb_mode=online
    echo -e "\033[33mTrain mode (wandb online)\033[0m"
fi

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

if [ ! -d "./data/${task_name}-${task_config}-${expert_data_num}.zarr" ]; then
    bash process_data.sh "${task_name}" "${task_config}" "${expert_data_num}"
fi

config_file="robot_${variant}.yaml"
if [ ! -f "./a2a_flow_matching/config/${config_file}" ]; then
    echo -e "\033[31m[A2A] unknown variant '${variant}'; no ./a2a_flow_matching/config/${config_file}\033[0m"
    echo -e "\033[31m      available: $(ls ./a2a_flow_matching/config/robot_*.yaml | sed 's|.*/robot_||; s|\.yaml||' | tr '\n' ' ')\033[0m"
    exit 1
fi
echo -e "\033[33mvariant: ${variant} -> ${config_file}\033[0m"

python train.py --config-name=${config_file} \
    task_name=${task_name} \
    task.dataset.zarr_path="data/${task_name}-${task_config}-${expert_data_num}.zarr" \
    training.debug=${DEBUG} \
    training.seed=${seed} \
    training.device="cuda:0" \
    exp_name=${task_name}-${variant}-train \
    logging.mode=${wandb_mode} \
    setting=${task_config} \
    expert_data_num=${expert_data_num} \
    head_camera_type=${head_camera_type}
