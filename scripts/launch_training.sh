#!/bin/bash
export NUM_GPUS=5
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

config="configs/experiments/off_objectnav.yaml"
DATA_PATH="data/datasets/objectnav/objectnav_hm3d_hd"
TENSORBOARD_DIR="tb/from_scratch_full_dataset_3e4_wd"
CHECKPOINT_DIR="data/from_scratch_full_dataset_3e4_wd"


echo "In ObjectNav OFFNAV"
python -u -m torch.distributed.launch \
    --use_env \
    --nproc_per_node $NUM_GPUS \
    run.py \
    --exp-config $config \
    --run-type train \
    TENSORBOARD_DIR $TENSORBOARD_DIR \
    CHECKPOINT_FOLDER $CHECKPOINT_DIR \
    NUM_UPDATES 200000 \
    WANDB_ENABLED True \
    NUM_ENVIRONMENTS 24 \
    RL.DDPPO.force_distributed True \
    TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
