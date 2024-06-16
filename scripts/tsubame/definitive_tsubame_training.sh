#!/bin/bash
#$ -cwd
#$ -l gpu_1=2
#$ -j y
#$ -l h_rt=00:10:00
#$ -o slurm_logs/$JOB_NAME_$JOB_ID/log.out
#$ -N tests_nuevo_script

# ******************* Setup dirs ***********************************
setup="full"
# Define paths
CONFIG="configs/experiments/il_objectnav.yaml"
DATA_PATH="data/datasets/objectnav/objectnav_hm3d_hd_${setup}"
TENSORBOARD_DIR="tb/${JOB_NAME}_${setup}"
CHECKPOINT_DIR="data/checkpoints/offnav/${JOB_NAME}_${setup}"
LOG_DIR="slurm_logs/${JOB_NAME}_${JOB_ID}"

mkdir -p "$TENSORBOARD_DIR"
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$LOG_DIR"

# ******************* Setup openmpi *******************************
module load openmpi/5.0.2-intel

NUM_CPUS=$(nproc)
NGPU_PER_NODE=$(nvidia-smi -L | wc -l)
# shellcheck disable=SC2207
NODES=( $(cat $PE_HOSTFILE | cut -d' ' -f1) )
MASTER_IP_ADDR=$(hostname -i)
RDZV_PORT=$(expr ${JOB_ID} % 50000 + 10000)

# shellcheck disable=SC2145
echo "Assigned nodes: ${NODES[@]}, nnodes=${NHOSTS}, nGPUspernode=${NGPU_PER_NODE}"
echo "Master node: ${HOSTNAME}($MASTER_IP_ADDR)"

# Set nvidia-smi to log GPU usage
nvidia-smi --query-gpu=timestamp,name,gpu_bus_id,utilization.gpu,utilization.memory,memory.used,memory.free \
    --format=csv -l 1 > "${LOG_DIR}"/gpu-usage.log &
NVIDIA_SMI_PID=$!

# ******************* Export variables *******************************
export CONFIG
export DATA_PATH
export TENSORBOARD_DIR
export CHECKPOINT_DIR
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export LOG_DIR
export NGPU_PER_NODE
export MASTER_IP_ADDR
export RDZV_PORT
export OMP_NUM_THREADS=$((NUM_CPUS/NGPU_PER_NODE))

# ******************* Run the training script *******************************
# shellcheck disable=SC1090
source ~/miniconda3/etc/profile.d/conda.sh
conda activate habitat

mpirun -npernode 1 -n "$NHOSTS" \
  apptainer run --nv \
    --bind /gs/fs/tga-aklab/data \
    --bind /gs/fs/tga-aklab/carlos/repositorios \
    --bind /gs/fs/tga-aklab/carlos/miniconda3 \
    --env NCCL_DEBUG=INFO \
    apptainer/offnav.sif
    torchrun --nnodes="${NHOSTS}" \
      --nproc_per_node="${NGPU_PER_NODE}" \
      --max_restarts 3 \
      --rdzv_backend=c10d \
      --rdzv_id="${JOB_ID}" \
      --rdzv_endpoint="${MASTER_IP_ADDR}":"${RDZV_PORT}" \
      run.py \
        --exp-config "$CONFIG" \
        --run-type train \
        TENSORBOARD_DIR "$TENSORBOARD_DIR" \
        CHECKPOINT_FOLDER "$CHECKPOINT_DIR" \
        NUM_UPDATES 50000 \
        WANDB_ENABLED True \
        NUM_ENVIRONMENTS 32 \
        OFFLINE.IQL.num_mini_batch 2 \
        RL.DDPPO.force_distributed True \
        RL.DDPPO.distrib_backend 'NCCL' \
        TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz"

kill $NVIDIA_SMI_PID
