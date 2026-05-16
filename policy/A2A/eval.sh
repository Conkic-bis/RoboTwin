#!/bin/bash
# Usage:
#   bash eval.sh <task_name> <task_config> <ckpt_setting> <expert_data_num> <seed> <gpu_id> [checkpoint_num]
#
# Args:
#   task_name        task to evaluate, e.g. beat_block_hammer
#   task_config      eval-time scene config (domain randomization etc.),
#                    e.g. demo_randomized / demo_clean
#   ckpt_setting     which trained checkpoint to load: the `setting`
#                    (== training task_config) baked into the ckpt dir name
#   expert_data_num  number of demos the checkpoint was trained on
#   seed             training seed (part of the ckpt dir name)
#   gpu_id           CUDA device
#   checkpoint_num   (optional) epoch number of the .ckpt to load.
#                    Defaults to 1000 (the final-epoch checkpoint). If the
#                    exact file is missing, get_model() auto-falls-back to
#                    the highest-numbered .ckpt in the directory.
#
# The trained checkpoint is expected at:
#   ./policy/A2A/checkpoints/<task_name>-<ckpt_setting>-<expert_data_num>-<seed>/<checkpoint_num>.ckpt
# which is exactly where train.sh writes it (ckpt_setting must equal the
# task_config used during training).

policy_name=A2A
task_name=${1}
task_config=${2}
ckpt_setting=${3}
expert_data_num=${4}
seed=${5}
gpu_id=${6}
checkpoint_num=${7:-}

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../..

extra_overrides=""
if [ -n "${checkpoint_num}" ]; then
    extra_overrides="--checkpoint_num ${checkpoint_num}"
fi

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/${policy_name}/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --expert_data_num ${expert_data_num} \
    --seed ${seed} \
    ${extra_overrides}
