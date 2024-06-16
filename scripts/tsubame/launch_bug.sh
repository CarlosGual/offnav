#!/bin/bash
#$ -cwd
#$ -l node_h=2
#$ -j y
#$ -l h_rt=00:10:00
#$ -o slurm_logs/$JOB_NAME_$JOB_ID/log.out
#$ -N test_bug

module load openmpi/5.0.2-intel
source ~/miniconda3/etc/profile.d/conda.sh
conda activate habitat

NGPU_PER_NODE=$(nvidia-smi -L | wc -l)
MASTER_IP_ADDR=$(hostname -i)
RDZV_PORT=$(expr ${JOB_ID} % 50000 + 10000)

mpirun -npernode 1 -n 2 \
      torchrun --nnodes=2 \
        --nproc_per_node="${NGPU_PER_NODE}" \
        --max_restarts 3 \
        --rdzv_backend=c10d \
        --rdzv_id=123456 \
        --rdzv_endpoint="${MASTER_IP_ADDR}":"${RDZV_PORT}" \
        test.py
