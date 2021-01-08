#!/bin/bash
#SBATCH --job-name=cfq_cleanup_conda_envs
#SBATCH --output=/home/eecs/paras/slurm/cfq/%j.log
set -x
hostname; pwd
eval "$(conda shell.bash hook)"
conda env list | cut -d" " -f1 | grep "cfq_" | xargs -I {} conda env remove -n {}
