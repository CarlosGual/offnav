#!/bin/bash
export NUM_GPUS=4
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

config="configs/experiments/off_objectnav.yaml"
DATA_PATH="data/datasets/objectnav/objectnav_hm3d_hd_minimal"
TENSORBOARD_DIR="tb/eval_minimal/"
CHECKPOINT_DIR="data/new_checkpoints"


echo "In ObjectNav IL DDP"
python -u -m run \
    --exp-config $config \
    --run-type eval \
    TENSORBOARD_DIR $TENSORBOARD_DIR \
    CHECKPOINT_FOLDER $CHECKPOINT_DIR \
    NUM_UPDATES 50000 \
    NUM_ENVIRONMENTS 16 \
    RL.DDPPO.force_distributed True \
    TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
