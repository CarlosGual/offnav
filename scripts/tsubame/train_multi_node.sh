#!/bin/bash

# shellcheck disable=SC1090
source ~/miniconda3/etc/profile.d/conda.sh
conda activate habitat
torchrun \
  --nnodes $NNODES \
  --nproc_per_node $NPERNODE \
  --max_restarts 3 \
  --rdzv-id=$JOB_ID \
  --rdzv-backend=c10d \
  --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
  run.py \
    --exp-config $config \
    --run-type train \
    TENSORBOARD_DIR $TENSORBOARD_DIR \
    CHECKPOINT_FOLDER $CHECKPOINT_DIR \
    NUM_UPDATES 50000 \
    WANDB_ENABLED True \
    NUM_ENVIRONMENTS 8 \
    OFFLINE.IQL.num_mini_batch 2 \
    RL.DDPPO.force_distributed True \
    RL.DDPPO.distrib_backend 'NCCL' \
    TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz"