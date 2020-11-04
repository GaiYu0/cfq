#!/bin/bash
#SBATCH --job-name=cfq_train_bert
#SBATCH --output=/home/eecs/paras/slurm/cfq/%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=400000
#SBATCH --gres="gpu:8"
#SBATCH --time=125:00:00
#SBATCH --exclude=atlas,blaze,r16,freddie

set -x

# check arguments
[ -z "$FLAGFILE" ] && { echo "Need to set FLAGFILE"; exit 1; }

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

if ! command -v mamba &> /dev/null; then
    conda install -y mamba -c conda-forge
fi

export ENV_NAME="cfq_`(echo $CUDA_VISIBLE_DEVICES | cut -d, -f1)`"
echo "Conda env name = $ENV_NAME"
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Using previously created cfq environment $ENV_NAME"
    mamba env update -n $ENV_NAME --file environment.yml
else
    echo "Creating new cfq environment $ENV_NAME"
    mamba env create -n $ENV_NAME --file environment.yml
fi
conda activate $ENV_NAME
pip install --no-cache torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip install -e .

# load dataset to data dir
export CFQ_DIR="data/cfq"
export DATA_CACHE="/data/paras/data_cache/cfq"
export RUN_CACHE="/data/paras/data_cache/cfq_runs"
[[ -d $CFQ_DIR ]] || (python scripts/download_cfq.py --data_dir $CFQ_DIR)
mkdir -p $DATA_CACHE
chmod 755 $DATA_CACHE
rsync -avhW --no-compress --progress $CFQ_DIR $DATA_CACHE

# train!
python cfq/train.py \
    --run_dir_root "$RUN_CACHE" \
    --data_root_path "$DATA_CACHE" \
    --flagfile "$FLAGFILE"
