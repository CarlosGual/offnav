#!/bin/bash
# export NUM_GPUS=1
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

config="configs/experiments/off_objectnav.yaml"
DATA_PATH="data/datasets/objectnav/hm3d/v1"
TENSORBOARD_DIR="tb/small_test_minival"
CHECKPOINT_DIR="data/from_scratch_beta_3_with_weighting_and_lr_3e6_minimal"

echo "In ObjectNav IL DDP"
python -u -m run \
    --exp-config $config \
    --run-type eval \
    TENSORBOARD_DIR $TENSORBOARD_DIR \
    EVAL_CKPT_PATH_DIR $CHECKPOINT_DIR \
    NUM_UPDATES 50000 \
    NUM_ENVIRONMENTS 8 \
    RL.DDPPO.force_distributed True \
    EVAL.SPLIT "val_mini" \
    TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
