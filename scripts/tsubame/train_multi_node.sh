#!/bin/bash

# shellcheck disable=SC1090
source ~/miniconda3/etc/profile.d/conda.sh
conda activate habitat

python run.py \
    --exp-config $config \
    --run-type train \
    TENSORBOARD_DIR $TENSORBOARD_DIR \
    CHECKPOINT_FOLDER $CHECKPOINT_DIR \
    NUM_UPDATES 100000 \
    WANDB_ENABLED True \
    NUM_ENVIRONMENTS 32 \
    OFFLINE.IQL.num_mini_batch 8 \
    RL.DDPPO.force_distributed True \
    RL.DDPPO.distrib_backend 'NCCL' \
    TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz"