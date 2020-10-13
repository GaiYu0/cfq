#!/bin/bash

set -x
export ROOT_DIR="$(dirname "$0")/.."
cd $ROOT_DIR

eval "$(conda shell.bash hook)"
conda create -y -n cfq
conda activate cfq
conda install -y -n cfq python=3.8
conda install -y -n cfq -c conda-forge nvidia-apex mkl
pip install -e .

[ -z "$RUNNAME" ] && { echo "Need to set RUNNAME"; exit 1; }
export BATCHSIZE=${BATCHSIZE:-64}
export LR=${LR:-"4e-4"}
export N_EPOCHS=${N_EPOCHS:-100}
export SEED=${SEED:-2}
export DROPOUT=${DROPOUT:-"0.0"}
export OPTIMIZER=${OPTIMIZER:-"Adam"}

export CFQSPLIT=${CFQSPLIT:-"1.0"}
export WPOS=${WPOS:-"1.0"}
export SEQ_MODEL=${SEQ_MODEL:-"lstm"}
export SEQ_NINP=${SEQ_NINP:-64}
export SEQ_NHEAD=${SEQ_NHEAD:-16}
export SEQ_HIDDENDIM=${SEQ_HIDDENDIM:-256}
export SEQ_NLAYERS=${SEQ_NLAYERS:-2}
export GRAPH_MODEL=${GRAPH_MODEL:-"rgcn"}
export GRAPH_NINP=${GRAPH_NINP:-64}
export GRAPH_HIDDENDIM=${GRAPH_HIDDENDIM:-64}
export GRAPH_NLAYERS=${GRAPH_NLAYERS:-4}
export GAMMA=${GAMMA:-1}
export NTL_NINP=${NTL_NINP:-64}
export NTL_HIDDENDIM=${NTL_HIDDENDIM:-64}

export RUNDIR="data/runs/$RUNNAME_`date +%F-%T`"
export OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}
export CFQ_DIR="data/cfq"
[[ -d $CFQ_DIR ]] || (python scripts/download_cfq.py --data_dir $CFQ_DIR)

mkdir -p "$OUTPUT_DIR"

python3 diffdnn/run_glue_pl.py --gpus 8 \
    --data_dir $DATA_DIR \
    --task $TASK \
    --model_name_or_path $BERT_MODEL \
    --output_dir $OUTPUT_DIR \
    --max_seq_length  $MAX_LENGTH \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $NUM_EPOCHS \
    --train_batch_size $BATCH_SIZE \
    --seed $SEED \
    --do_train \
    --do_predict