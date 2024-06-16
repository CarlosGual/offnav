#!/bin/bash

# shellcheck disable=SC1090
source ~/miniconda3/etc/profile.d/conda.sh
conda activate habitat

printenv | grep NHOSTS, NGPU_PER_NODE, MASTER_IP_ADDR, RDZV_PORT

torchrun --nnodes="${NHOSTS}" \
  --nproc_per_node="${NGPU_PER_NODE}" \
  --max_restarts 3 \
  --rdzv_backend=c10d \
  --rdzv_id="${JOB_ID}" \
  --rdzv_endpoint="${MASTER_IP_ADDR}":"${RDZV_PORT}" \
  run.py \
    --exp-config "$CONFIG" \
    --run-type train \
    TENSORBOARD_DIR "$TENSORBOARD_DIR" \
    CHECKPOINT_FOLDER "$CHECKPOINT_DIR" \
    NUM_UPDATES 50000 \
    WANDB_ENABLED True \
    NUM_ENVIRONMENTS 64 \
    OFFLINE.IQL.num_mini_batch 2 \
    RL.DDPPO.force_distributed True \
    RL.DDPPO.distrib_backend 'NCCL' \
    TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz"