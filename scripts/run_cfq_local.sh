#!/bin/bash
set -x
export ROOT_DIR="$(dirname "$0")/.."
cd $ROOT_DIR

eval "$(conda shell.bash hook)"

conda create -y -n cfq
conda activate cfq
conda install -y -n cfq python=3.8 mkl tensorflow-gpu
conda install -y -n cfq pytorch torchvision cudatoolkit=10.1 -c pytorch
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
