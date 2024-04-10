#!/bin/bash
#AKBATCH -r troll_2
#SBATCH -N 1
#SBATCH -J default_cyclic_lr_resnet18
#SBATCH --ntasks-per-node=2
#SBATCH --output=slurm_logs/%x-%j.out

# shellcheck disable=SC1090
source ~/miniconda3/etc/profile.d/conda.sh
conda activate habitat

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

setup="setup2"

config="configs/experiments/off_objectnav.yaml"
DATA_PATH="data/datasets/objectnav/objectnav_hm3d_hd_${setup}"
TENSORBOARD_DIR="tb/${SLURM_JOB_NAME}_${setup}"
CHECKPOINT_DIR="data/checkpoints/offnav/${SLURM_JOB_NAME}_${setup}"

mkdir -p $TENSORBOARD_DIR
mkdir -p $CHECKPOINT_DIR
mkdir -p slurm_logs

# ******************* These are read internally it seems ***********************************
# ******** Master port, address and world size MUST be passed as variables for DDP to work
MAIN_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MAIN_ADDR
# ******************************************************************************************

echo "In ObjectNav OFFNAV"
set -x
srun --pty python -u -m run \
    --exp-config $config \
    --run-type train \
    TENSORBOARD_DIR $TENSORBOARD_DIR \
    CHECKPOINT_FOLDER $CHECKPOINT_DIR \
    NUM_UPDATES 200000 \
    WANDB_ENABLED True \
    NUM_ENVIRONMENTS 32 \
    OFFLINE.IQL.num_mini_batch 4 \
    RL.DDPPO.force_distributed True \
    TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz"
