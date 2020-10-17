#!/bin/bash
#SBATCH --job-name=cfq_train
#SBATCH --output=/home/eecs/paras/slurm/cfq/%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=50000
#SBATCH --time=125:00:00

set -x

# print host statistics
set -x
date;hostname;pwd
free -mh
df -h
gpustat -cup
nvidia-smi
free -m | awk 'NR==2{printf "Memory Usage: %s/%sMB (%.2f%%)\n", $3,$2,$3*100/$2 }'
df -h | awk '$NF=="/"{printf "Disk Usage: %d/%dGB (%s)\n", $3,$2,$5}'
top -bn1 | grep load | awk '{printf "CPU Load: %.2f\n", $(NF-2)}' 
chmod 755 -R ~/slurm
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "CUDA_DEVICE_ORDER = $CUDA_DEVICE_ORDER"
echo "SLURM_JOB_ID = $SLURM_JOB_ID"

eval "$(conda shell.bash hook)"

conda env remove -n cfq
conda create -y -n cfq python=3.8
conda install -y -n cfq python=3.8 mkl tensorflow-gpu
conda install -y -n cfq pytorch torchvision cudatoolkit=10.1 -c pytorch
conda activate cfq
pip install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip install -e .
