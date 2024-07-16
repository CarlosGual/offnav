#!/bin/bash
#AKBATCH -r troll_2
#SBATCH -N 1
#SBATCH --requeue
#SBATCH -J mil_resnet18_higher_one_loop
#SBATCH --output=slurm_logs/%x-%j.out

# shellcheck disable=SC1090
source ~/miniconda3/etc/profile.d/conda.sh
conda activate habitat

# Get number of GPUs
if [ -z "$CUDA_VISIBLE_DEVICES" ]
then
    echo "CUDA_VISIBLE_DEVICES is not set"
else
    IFS=',' read -ra ADDR <<< "$CUDA_VISIBLE_DEVICES"
    num_gpus=${#ADDR[@]}
    echo "Number of GPUs: $num_gpus"
fi
# Get number of CPUs
num_cpus=$(nproc)

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export OMP_NUM_THREADS=$((num_cpus/num_gpus))
export WANDB_MODE=offline

setup="setup1"
exp_name="mil_resnet18_higher_one_loop"
config="configs/experiments/mil_objectnav.yaml"
DATA_PATH="data/datasets/objectnav/objectnav_hm3d_hd_${setup}"
TENSORBOARD_DIR="tb/${exp_name}_${setup}"
CHECKPOINT_DIR="data/checkpoints/mil/${exp_name}_${setup}"

mkdir -p $TENSORBOARD_DIR
mkdir -p $CHECKPOINT_DIR
mkdir -p slurm_logs

echo "In ObjectNav OFFNAV"
torchrun --nproc_per_node $num_gpus run.py \
    --exp-config $config \
    --run-type train \
    TENSORBOARD_DIR $TENSORBOARD_DIR \
    CHECKPOINT_FOLDER $CHECKPOINT_DIR \
    NUM_UPDATES 50000 \
    WANDB_ENABLED True \
    NUM_ENVIRONMENTS 10 \
    OFFLINE.IQL.num_mini_batch 10 \
    RL.DDPPO.force_distributed True \
    TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz"
