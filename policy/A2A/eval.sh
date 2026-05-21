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
#                    Defaults to deploy_policy.yml's checkpoint_num (1000). If
#                    the exact file is missing, get_model() auto-falls-back to
#                    the highest-numbered .ckpt in the directory.
#
# Env vars:
#   A2A_VARIANT      (optional) which A2A variant to evaluate.
#                    Defaults to deploy_policy.yml's `variant` field ("a2a").
#                    Set to "a2a_noise" to load checkpoints from the
#                    "-noise"-suffixed directory written by
#                    `train.sh ... a2a_noise`.
#                    Example:
#                      A2A_VARIANT=a2a_noise bash eval.sh \
#                          beat_block_hammer demo_clean demo_clean 50 0 0
#
# The trained checkpoint is expected at:
#   a2a       -> ./policy/A2A/checkpoints/<task>-<ckpt_setting>-<N>-<seed>/<ckpt_num>.ckpt
#   a2a_noise -> ./policy/A2A/checkpoints/<task>-<ckpt_setting>-<N>-<seed>-noise/<ckpt_num>.ckpt

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
    extra_overrides="${extra_overrides} --checkpoint_num ${checkpoint_num}"
fi
if [ -n "${A2A_VARIANT:-}" ]; then
    extra_overrides="${extra_overrides} --variant ${A2A_VARIANT}"
    echo -e "\033[33mA2A_VARIANT=${A2A_VARIANT} (overrides deploy_policy.yml)\033[0m"
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
