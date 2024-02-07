#!/bin/bash
export NUM_GPUS=4
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

config="configs/experiments/off_objectnav.yaml"
DATA_PATH="data/datasets/objectnav/objectnav_hm3d_hd"
TENSORBOARD_DIR="tb/initialized_bc/with_inflection_weight_full_dataset"
CHECKPOINT_DIR="data/with_inflection_weight_full_dataset"


echo "In ObjectNav IL DDP"
python -u -m torch.distributed.launch \
    --use_env \
    --nproc_per_node $NUM_GPUS \
    run.py \
    --exp-config $config \
    --run-type train \
    TENSORBOARD_DIR $TENSORBOARD_DIR \
    CHECKPOINT_FOLDER $CHECKPOINT_DIR \
    NUM_UPDATES 500000 \
    WANDB_ENABLED True \
    NUM_ENVIRONMENTS 4 \
    WANDB_ENABLED True \
    RL.DDPPO.force_distributed True \
    TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
