#!/bin/bash
#AKBATCH -r slime_3
#SBATCH -N 1
#SBATCH -J shared_long_test
#SBATCH --output=slurm_logs/%x-%j.out

# shellcheck disable=SC1090
source ~/miniconda3/etc/profile.d/conda.sh
conda activate habitat

# Get number of GPUs
if [ -z "$NVIDIA_VISIBLE_DEVICES" ]
then
    echo "NVIDIA_VISIBLE_DEVICES is not set"
else
    IFS=',' read -ra ADDR <<< "$NVIDIA_VISIBLE_DEVICES"
    num_gpus=${#ADDR[@]}
    echo "Number of GPUs: $num_gpus"
fi
# Get number of CPUs
num_cpus=$(nproc)

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export OMP_NUM_THREADS=$((num_cpus/num_gpus))

setup="setup1"
config="configs/experiments/off_objectnav.yaml"
DATA_PATH="data/datasets/objectnav/objectnav_hm3d_hd_${setup}"
TENSORBOARD_DIR="tb/${SLURM_JOB_NAME}_${setup}"
CHECKPOINT_DIR="data/checkpoints/offnav/${SLURM_JOB_NAME}_${setup}"

mkdir -p $TENSORBOARD_DIR
mkdir -p $CHECKPOINT_DIR
mkdir -p slurm_logs

echo "In ObjectNav OFFNAV"
torchrun --nproc_per_node $num_gpus run.py \
    --exp-config $config \
    --run-type train \
    TENSORBOARD_DIR $TENSORBOARD_DIR \
    CHECKPOINT_FOLDER $CHECKPOINT_DIR \
    NUM_UPDATES 200000 \
    WANDB_ENABLED False \
    NUM_ENVIRONMENTS 8 \
    OFFLINE.IQL.num_mini_batch 4 \
    RL.DDPPO.force_distributed True \
    TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz"
