#!/bin/bash

# ******************* Setup dirs *************************************************
setup="full"
exp_name="dgx_1nodo_semifreezed_0.0001"

CONFIG="configs/experiments/mil_objectnav.yaml"
DATA_PATH="data/datasets/objectnav/objectnav_hm3d_hd_${setup}"
TENSORBOARD_DIR="tb/${exp_name}_${setup}"
CHECKPOINT_DIR="data/checkpoints/metanav/${exp_name}_${setup}"

# ******************* Set nvidia-smi to log GPU usage ******************************
mkdir -p "$TENSORBOARD_DIR"
mkdir -p "$CHECKPOINT_DIR"

nvidia-smi --query-gpu=timestamp,name,gpu_bus_id,utilization.gpu,utilization.memory,memory.used,memory.free \
    --format=csv -l 1 > "${TENSORBOARD_DIR}"/gpu-usage.log &
NVIDIA_SMI_PID=$!

# ******************* Setup number of cpus and gpus *******************************
NUM_CPUS=$(nproc)
NGPU_PER_NODE=$(nvidia-smi -L | wc -l)
NHOSTS=1

# ******************* Export variables ********************************************
export CONFIG
export DATA_PATH
export TENSORBOARD_DIR
export CHECKPOINT_DIR
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export LOG_DIR
export NGPU_PER_NODE
export OMP_NUM_THREADS=$((NUM_CPUS/NGPU_PER_NODE))

# ******************* Run the training script *******************************
echo "Getting number of cpus and cpus per node..."
echo "NUM_CPUS: $NUM_CPUS", "NGPU_PER_NODE: $NGPU_PER_NODE", "CPUS PER GPU: $OMP_NUM_THREADS"
echo "Running meta imitation learning..."

torchrun --nnodes="${NHOSTS}" \
  --nproc_per_node="${NGPU_PER_NODE}" \
  --max_restarts 3 \
  run.py \
    --exp-config "$CONFIG" \
    --run-type train \
    TENSORBOARD_DIR "$TENSORBOARD_DIR" \
    CHECKPOINT_FOLDER "$CHECKPOINT_DIR" \
    NUM_UPDATES 30000 \
    WANDB_ENABLED True \
    NUM_ENVIRONMENTS 30 \
    IL.BehaviorCloning.num_mini_batch 2 \
    IL.BehaviorCloning.lr 0.0001 \
    META.MIL.num_tasks 2 \
    META.MIL.num_gradient_updates 2 \
    META.MIL.num_updates_per_sampled_tasks 5 \
    RL.DDPPO.force_distributed True \
    RL.DDPPO.distrib_backend 'NCCL' \
    TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
    POLICY.RGB_ENCODER.backbone 'resnet50' \
#    POLICY.RGB_ENCODER.pretrained_encoder 'data/visual_encoders/omnidata_DINO_02.pth' \

kill $NVIDIA_SMI_PID