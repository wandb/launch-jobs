#!/bin/bash

#SBATCH --job-name=example1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

CMD="python main.py"
srun --jobid $SLURM_JOBID bash -c "$CMD"