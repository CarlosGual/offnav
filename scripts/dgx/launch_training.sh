#!/bin/bash

# Get number of GPUs
if [ -z "$CUDA_VISIBLE_DEVICES" ]
then
    echo "CUDA_VISIBLE_DEVICES is not set"
else
    IFS=',' read -ra ADDR <<< "$CUDA_VISIBLE_DEVICES"
    num_gpus=${#ADDR[@]}
    echo "Number of GPUs: $num_gpus"
fi
num_cpus=$(nproc)

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export OMP_NUM_THREADS=$((num_cpus/num_gpus))

setup="full"
exp_name="cyclic_lr_navrl"

config="configs/experiments/off_objectnav.yaml"
DATA_PATH="data/datasets/objectnav/objectnav_hm3d_hd" # _${setup}"
TENSORBOARD_DIR="tb/${exp_name}_${setup}"
CHECKPOINT_DIR="data/checkpoints/offnav/${exp_name}_${setup}"


echo "In ObjectNav OFFNAV"
python -u -m torch.distributed.launch \
    --use_env \
    --nproc_per_node $num_gpus \
    run.py \
    --exp-config $config \
    --run-type train \
    TENSORBOARD_DIR $TENSORBOARD_DIR \
    CHECKPOINT_FOLDER $CHECKPOINT_DIR \
    NUM_UPDATES 400000 \
    WANDB_ENABLED True \
    NUM_ENVIRONMENTS 4 \
    OFFLINE.IQL.num_mini_batch 1 \
    RL.DDPPO.force_distributed True \
    TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
