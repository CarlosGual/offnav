#!/bin/bash
export NUM_GPUS=2
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

config="configs/experiments/off_objectnav.yaml"
DATA_PATH="data/datasets/objectnav/objectnav_hm3d_hd_minimal"
TENSORBOARD_DIR="tb/initialized_bc/from_scratch_beta_3_with_weighting_and_lr_3e6_minimal"
CHECKPOINT_DIR="data/from_scratch_beta_3_with_weighting_and_lr_3e6_minimal"


echo "In ObjectNav IL DDP"
python -u -m torch.distributed.launch \
    --use_env \
    --nproc_per_node $NUM_GPUS \
    run.py \
    --exp-config $config \
    --run-type train \
    TENSORBOARD_DIR $TENSORBOARD_DIR \
    CHECKPOINT_FOLDER $CHECKPOINT_DIR \
    NUM_UPDATES 100000 \
    WANDB_ENABLED True \
    NUM_ENVIRONMENTS 6 \
    RL.DDPPO.force_distributed True \
    TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
