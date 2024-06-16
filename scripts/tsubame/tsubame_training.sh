#!/bin/sh
#$ -cwd
#$ -l node_f=2
#$ -j y
#$ -l h_rt=24:00:00
#$ -o slurm_logs/$JOB_NAME_$JOB_ID/log.out
#$ -N pirlnav

# ******************* Setup dirs ***********************************
setup="full"
# Define paths
config="configs/experiments/il_objectnav.yaml"
DATA_PATH="data/datasets/objectnav/objectnav_hm3d_hd_${setup}"
TENSORBOARD_DIR="tb/${JOB_NAME}_${setup}"
CHECKPOINT_DIR="data/checkpoints/offnav/${JOB_NAME}_${setup}"
LOG_DIR="slurm_logs/${JOB_NAME}_${JOB_ID}"

mkdir -p $TENSORBOARD_DIR
mkdir -p $CHECKPOINT_DIR
mkdir -p "$LOG_DIR"

export config
export DATA_PATH
export TENSORBOARD_DIR
export CHECKPOINT_DIR
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

# ******************* Setup openmpi *******************************
module load openmpi/5.0.2-intel
# Get number of GPUs
if [ -z "$NVIDIA_VISIBLE_DEVICES" ]
then
    echo "NVIDIA_VISIBLE_DEVICES is not set"
else
    IFS=',' read -ra ADDR <<< "$NVIDIA_VISIBLE_DEVICES"
    num_gpus=${#ADDR[@]}
fi
# Get number of CPUs
num_cpus=$(nproc)
export OMP_NUM_THREADS=$((num_cpus/num_gpus))
export NNODES=$NHOSTS
export NPERNODE=$num_gpus
export NP=$(($NPERNODE * $NNODES))
export MASTER_ADDR=$(head -n 1 "$SGE_JOB_SPOOL_DIR"/pe_hostfile | cut -d " " -f 1)
export MASTER_PORT=$((10000+ ($JOB_ID % 50000)))
echo NNODES=$NNODES
echo NPERNODE=$NPERNODE
echo NP=$NP
echo MASTERADDR=$MASTER_ADDR
echo MASTERPORT=$MASTER_PORT

# Set nvidia-smi to log GPU usage
nvidia-smi --query-gpu=timestamp,name,gpu_bus_id,utilization.gpu,utilization.memory,memory.used,memory.free \
    --format=csv -l 1 > "${LOG_DIR}"/gpu-usage.log &
NVIDIA_SMI_PID=$!
# ******************************************************************

echo "In ObjectNav OFFNAV"
mpirun \
  -np $NP \
  -npernode $NPERNODE \
  -x LD_LIBRARY_PATH \
  -bind-to none \
  -mca pml ob1 \
  -mca btl ^openib \
    apptainer exec --nv \
      --bind /gs/fs/tga-aklab/data \
      --bind /gs/fs/tga-aklab/carlos/repositorios \
      --bind /gs/fs/tga-aklab/carlos/miniconda3 \
      apptainer/offnav.sif \
      bash scripts/tsubame/train_multi_node.sh

kill $NVIDIA_SMI_PID