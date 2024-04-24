#!/bin/bash
#AKBATCH -r golem_1
#SBATCH -N 1
#SBATCH -J default_cyclic_lr_resnet18
#SBATCH --output=slurm_logs/%x-%j.out

# shellcheck disable=SC1090
source ~/miniconda3/etc/profile.d/conda.sh
conda activate habitat

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

setup="setup3"
exp_name="default_cyclic_lr_resnet18"
config="configs/experiments/off_objectnav.yaml"
DATA_PATH="data/datasets/objectnav/objectnav_hm3d_hd_${setup}"
TENSORBOARD_DIR="tb/${exp_name}_${setup}"
CHECKPOINT_DIR="data/checkpoints/offnav/${exp_name}_${setup}"

echo "In ObjectNav IL DDP"
python -u -m run \
    --exp-config $config \
    --run-type eval \
    TENSORBOARD_DIR $TENSORBOARD_DIR \
    EVAL_CKPT_PATH_DIR $CHECKPOINT_DIR \
    NUM_UPDATES 50000 \
    NUM_ENVIRONMENTS 4 \
    WANDB_ENABLED True \
    EVAL.SPLIT "val" \
    TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz"
