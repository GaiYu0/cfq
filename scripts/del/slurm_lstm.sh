#!/bin/bash
#SBATCH --job-name=cfq_train
#SBATCH --output=/home/eecs/paras/slurm/cfq/%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=50000
#SBATCH --gres="gpu:1"
#SBATCH --time=125:00:00
#SBATCH --exclude=atlas,blaze,r16

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

if conda env list | grep -q '^cfq'; then
   echo "CFQ environment already installed"
else
    conda create -y -n cfq python=3.8
    conda install -y -n cfq python=3.8 mkl tensorflow-gpu
    conda install -y -n cfq pytorch torchvision cudatoolkit=10.1 -c pytorch
fi
conda activate cfq
pip install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip install -e .

# load dataset to data dir
[ -z "$RUNNAME" ] && { echo "Need to set RUNNAME"; exit 1; }
export RUNDIR_NAME="$RUNNAME_`date +%F-%T`"
export CFQ_DIR="data/cfq"
[[ -d $CFQ_DIR ]] || (python scripts/download_cfq.py --data_dir $CFQ_DIR)

# load arguments for training
export SEED=${SEED:-2}
export PRECISION=${PRECISION:-32}
export BATCHSIZE=${BATCHSIZE:-256}
export GRADIENT_CLIP_VALUE=${GRADIENT_CLIP_VALUE:-"0.0"}
export OPTIMIZER=${OPTIMIZER:-"Adam"}
export LR=${LR:-"4e-4"}
export N_EPOCHS=${N_EPOCHS:-150}
export WARMUP_EPOCHS=${WARMUP_EPOCHS:-10}
export COSINE_LR_PERIOD=${COSINE_LR_PERIOD:-"0.5"}
export CFQ_SPLIT=${CFQ_SPLIT:-"random_split"}

# train
python cfq/train.py \
    --run_name "$RUNNAME" \
    --run_dir_name "$RUNDIR_NAME" \
    --seed $SEED \
    --precision $PRECISION \
    --batch_size $BATCHSIZE \
    --gradient_clip_val $GRADIENT_CLIP_VALUE \
    --optimizer_name $OPTIMIZER \
    --lr "$LR" \
    --num_epochs $N_EPOCHS \
    --warmup_epochs $WARMUP_EPOCHS \
    --cosine_lr_period $COSINE_LR_PERIOD \
    --cfq_split "$CFQ_SPLIT"
