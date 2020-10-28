#!/bin/bash
set -x
export ROOT_DIR="$(dirname "$0")/.."
cd $ROOT_DIR

eval "$(conda shell.bash hook)"

if conda env list | grep -q "^cfq "; then
    echo "Using previously created cfq environment"
    conda install -y mamba -c conda-forge
    mamba env update -n cfq --file environment.yml
else
    echo "Creating new cfq environment"
    conda install -y mamba -c conda-forge
    mamba env create -n cfq --file environment.yml
fi
conda activate cfq
pip install --no-cache torch-scatter==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
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
    --seq_model lstm \
    --cfq_split "$CFQ_SPLIT"
