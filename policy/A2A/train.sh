#!/bin/bash
# Usage:
#   bash train.sh <task_name> <task_config> <expert_data_num> <seed> <action_dim> <gpu_id>
# Example:
#   bash train.sh beat_block_hammer demo_randomized 50 0 14 0

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
action_dim=${5}
gpu_id=${6}

head_camera_type=D435
DEBUG=False

alg_name=robot_a2a_${action_dim}

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

if [ "$DEBUG" = "True" ]; then
    wandb_mode=offline
    echo -e "\033[33mDebug mode\033[0m"
else
    wandb_mode=online
    echo -e "\033[33mTrain mode\033[0m"
fi

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

if [ ! -d "./data/${task_name}-${task_config}-${expert_data_num}.zarr" ]; then
    bash process_data.sh "${task_name}" "${task_config}" "${expert_data_num}"
fi

python train.py --config-name=${alg_name}.yaml \
    task_name=${task_name} \
    task.dataset.zarr_path="data/${task_name}-${task_config}-${expert_data_num}.zarr" \
    training.debug=${DEBUG} \
    training.seed=${seed} \
    training.device="cuda:0" \
    exp_name=${task_name}-a2a-train \
    logging.mode=${wandb_mode} \
    setting=${task_config} \
    expert_data_num=${expert_data_num} \
    head_camera_type=${head_camera_type}
