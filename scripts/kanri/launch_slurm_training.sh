#!/bin/bash
#AKBATCH -r wyvern_2
#SBATCH -N 1
#SBATCH -J test_shared_heads
#SBATCH --output=slurm_logs/ddoff-train-%j.out

# shellcheck disable=SC1090
source ~/miniconda3/etc/profile.d/conda.sh
conda activate habitat

# Get number of GPUs
if [ -z "$GPU_DEVICE_ORDINAL" ]
then
    echo "CUDA_VISIBLE_DEVICES is not set"
else
    IFS=',' read -ra ADDR <<< "$GPU_DEVICE_ORDINAL"
    NUM_GPUS=${#ADDR[@]}
    echo "Number of GPUs: $NUM_GPUS"
fi

# Use the number of GPUs for the --ntasks-per-node option
#SBATCH --ntasks-per-node=$NUM_GPUS

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
mkdir -p slurm_logs

echo "In ObjectNav OFFNAV"
srun python -u -m run \
    --exp-config $config \
    --run-type train \
    TENSORBOARD_DIR $TENSORBOARD_DIR \
    CHECKPOINT_FOLDER $CHECKPOINT_DIR \
    NUM_UPDATES 200000 \
    WANDB_ENABLED True \
    NUM_ENVIRONMENTS 120 \
    OFFLINE.IQL.num_mini_batch 30 \
    RL.DDPPO.force_distributed True \
    TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz"
