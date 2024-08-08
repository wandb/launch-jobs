#!/bin/bash

#SBATCH --job-name=example1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm-%x-%j.out

source ~/.bashrc

set -x -e

echo "START TIME: $(date)"

LOG_PATH="helloworld.log"

conda activate helloworld
export CMD="python helloworld.py"

srun --jobid $SLURM_JOBID bash -c "$CMD" 2>&1 | tee -a $LOG_PATH

echo "END TIME: $(date)"