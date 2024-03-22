#!/bin/bash
#AKBATCH -r troll_2
#SBATCH -N 1
#SBATCH -J test_shared_heads
#SBATCH -o slurm-%J.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate habitat

export NUM_GPUS=2
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

exp_name="test_shared_heads"
config="configs/experiments/off_objectnav.yaml"
DATA_PATH="data/datasets/objectnav/objectnav_hm3d_hd"
TENSORBOARD_DIR="tb/${exp_name}"
CHECKPOINT_DIR="data/checkpoints/offnav/${exp_name}"

mkdir -p $TENSORBOARD_DIR
mkdir -p $CHECKPOINT_DIR
set -x

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
    NUM_ENVIRONMENTS 16 \
    RL.DDPPO.force_distributed True \
    TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
