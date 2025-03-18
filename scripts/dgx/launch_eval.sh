#!/bin/bash
# export NUM_GPUS=1
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

setup="setup2"
exp_name="pruebas_grad2_sampled5"

config="configs/experiments/mil_objectnav.yaml"
DATA_PATH="data/datasets/objectnav/objectnav_hm3d_hd_${setup}"
TENSORBOARD_DIR="tb/${exp_name}_${setup}"
CHECKPOINT_DIR="data/checkpoints/metanav/${exp_name}_${setup}"

echo "In Meta IL DDP"
python -u -m run \
    --exp-config $config \
    --run-type eval \
    TENSORBOARD_DIR "$TENSORBOARD_DIR" \
    EVAL_CKPT_PATH_DIR "$CHECKPOINT_DIR" \
    NUM_UPDATES 50000 \
    NUM_ENVIRONMENTS 8 \
    RL.DDPPO.force_distributed True \
    EVAL.SPLIT "val" \
    TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
    POLICY.RGB_ENCODER.backbone 'resnet18' \
    WANDB_ENABLED True
