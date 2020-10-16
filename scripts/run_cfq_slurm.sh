#!/bin/bash
#SBATCH --job-name=cfq_train
#SBATCH --output=/home/eecs/paras/slurm/cfq/%j.log
#SBATCH --ntasks=1
#SBATCH --mem=50000
#SBATCH --gres="gpu:1"
#SBATCH --time=125:00:00
#SBATCH --exclude=atlas,blaze,r16,freddie,como

set -x
export ROOT_DIR="$(dirname "$0")/.."
cd $ROOT_DIR

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

conda create -y -n cfq python=3.8
conda install -y mamba -c conda-forge
mamba install -y -n cfq mkl tensorflow-gpu
mamba install -y -n cfq pytorch torchvision cudatoolkit=10.1 -c pytorch
conda activate cfq
pip install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip install -e .

# load dataset to data dir
export CFQ_DIR="data/cfq"
[[ -d $CFQ_DIR ]] || (python scripts/download_cfq.py --data_dir $CFQ_DIR)

[ -z "$SWEEPID" ] && { echo "Need to set SWEEPID"; exit 1; }
[ -z "$CFQSPLIT" ] && { echo "Need to set CFQSPLIT"; exit 1; }
echo "SWEEPID = $SWEEPID"
echo "CFQSPLIT = $CFQSPLIT"

# load arguments for training
wandb agent cfq/cfq_sweep_$CFQSPLIT/$SWEEPID
