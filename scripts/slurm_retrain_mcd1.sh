#!/bin/bash
#SBATCH --job-name=cfq_mcd1
#SBATCH --output=/home/eecs/paras/slurm/%j
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=400000
#SBATCH --gres="gpu:1"
#SBATCH --time=125:00:00
#SBATCH --exclude=atlas,blaze,r16,freddie

# load dataset to data dir
export CFQ_DIR="data/cfq"
export DATA_CACHE="/data/$USER/data_cache/cfq"
[[ -d $CFQ_DIR ]] || (python scripts/download_cfq.py --data_dir $CFQ_DIR)
mkdir -p $DATA_CACHE
chmod 755 $DATA_CACHE
rsync -avhW --no-compress --progress $CFQ_DIR $DATA_CACHE

# from https://github.com/moby/moby/issues/2838#issuecomment-385145030
function docker() {
    case "$1" in
        run)
            shift
            if [ -t 1 ]; then # have tty
                command docker run --init -it "$@"
            else
                id=`command docker run -d --init "$@"`
                trap "command docker kill $id" INT TERM SIGINT SIGTERM
                command docker logs --follow $id
            fi
            ;;
        *)
            command docker "$@"
    esac
}

set -x
docker build -t cfq .
docker run --rm -t --init \
  --gpus="device=$CUDA_VISIBLE_DEVICES" \
  --ipc=host \
  --user="$(id -u):$(id -g)" \
  --volume="/dev/hugepages:/dev/hugepages" \
  --volume="$DATA_CACHE:/app/data" \
  --volume="$HOME/.netrc:/home/user/.netrc" \
  --env="PYTHONPATH=/app" \
  cfq python cfq/train.py --batch_size=1024 --cfq_split=mcd1 --cosine_lr_period=0.5 --dropout=0.5 --gradient_clip_val=5 --lr=0.005 --num_epochs=300 --optimizer_name=AdamW --seed=42 --seq_hidden_dim=128 --seq_inp=16 --seq_nlayers=3 --sweep_mode=True --wandb_project=gold_tuned_sweep_seeds --warmup_epochs=20
