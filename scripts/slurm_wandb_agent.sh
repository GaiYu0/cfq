#!/bin/bash
#SBATCH --job-name=cfq_train
#SBATCH --output=/home/eecs/paras/slurm/cfq/%j.log
#SBATCH --ntasks=1
#SBATCH --mem=50000
#SBATCH --gres="gpu:1"
#SBATCH --time=125:00:00
#SBATCH --exclude=r[3,5-6,8-9,10-12,16],atlas,blaze,freddie,havoc

# run on steropes, pavia, como
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

export ENV_NAME="cfq_$CUDA_VISIBLE_DEVICES"
echo "Conda env name = $ENV_NAME"
conda create -y -n $ENV_NAME python=3.8
conda install -y -n $ENV_NAME python=3.8 mkl tensorflow-gpu
conda install -y -n $ENV_NAME pytorch torchvision cudatoolkit=10.1 -c pytorch
conda activate $ENV_NAME
pip install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip install -e .

# load dataset to data dir
export CFQ_DIR="data/cfq"
[[ -d $CFQ_DIR ]] || (python scripts/download_cfq.py --data_dir $CFQ_DIR)

# load data to cache
export DATA_CACHE="/data/paras/data_cache/cfq"
mkdir -p $DATA_CACHE
chmod 755 $DATA_CACHE
rsync -avhW --no-compress --progress $CFQ_DIR $DATA_CACHE

# load wandb parameters
export NJOBS=${NJOBS:-16}
[ -z "$SWEEPID" ] && { echo "Need to set SWEEPID"; exit 1; }
echo "SWEEPID = $SWEEPID"

# load arguments for training
wandb agent --count $NJOBS $SWEEPID
