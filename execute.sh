#!/bin/bash
# the SBATCH directives must appear before any executable
# line in this script
#SBATCH -p rise # partition (queue)
#SBATCH -N 1 # number of nodes requested
#SBATCH -n 9 # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=1 # number of cores per task
# I think gpu:4 will request 4 of any kind of gpu per node,
# and gpu:v100_32:8 should request 8 v100_32 per node
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:v100_32:1
##SBATCH --gres=gpu:v100_32_maxq:1
#SBATCH --nodelist=steropes # if you need specific nodes
##SBATCH --exclude=atlas,blaze,freddie # nodes not yet on SLURM-only
#SBATCH -t 1-0:00 # time requested (D-HH:MM)
# slurm will cd to this directory before running the script
# you can also just run sbatch submit.sh from the directory
# you want to be in
##SBATCH -D /home/eecs/drothchild/slurm
# use these two lines to control the output file. Default is
# slurm-<jobid>.out. By default stdout and stderr go to the same
# place, but if you use both commands below they'll be split up
# filename patterns here: https://slurm.schedmd.com/sbatch.html
# %N is the hostname (if used, will create output(s) per node)
# %j is jobid
##SBATCH -o slurm.%N.%j.out # STDOUT
##SBATCH -e slurm.%N.%j.err # STDERR
# if you want to get emails as your jobs run/fail
##SBATCH --mail-type=NONE # Mail events (NONE, BEGIN, END, FAIL, ALL)
##SBATCH --mail-user=<your_email> # Where to send mail 
# print some info for context
pwd
hostname
date
echo starting job...
# activate your virtualenv
# source /data/drothchild/virtualenvs/pytorch/bin/activate
# or do your conda magic, etc.
source ~/.bashrc
conda activate ./env
# python will buffer output of your script unless you set this
# if you're not using python, figure out how to turn off output
# buffering when stdout is a file, or else when watching your output
# script you'll only get updated every several lines printed
export PYTHONUNBUFFERED=1
# do ALL the research
split=mcd1
# split=mcd2
# split=mcd3
# split=question_pattern_split
# split=question_complexity_split
# split=query_pattern_split
# split=query_complexity_split
# split=random_split
# seq_model=lstm
seq_model=transformer
seq_ninp=256
nhead=8
seq_nhid=1024
seq_nlayer=6
lr=1e-3
dropout=0.0
w_pos=1.0
num_warmup_steps=0
#python3 train.py --input-dir /work/wendi/cfq --output-dir /data/wendi/cfq --split $split --train-batch-size 64 --eval-batch-size 64 --num-workers 8 --seq-model $seq_model --seq-ninp $seq_ninp --nhead $nhead --seq-nhid $seq_nhid --seq-nlayer $seq_nlayer --gr-model rgcn --gr-ninp 64 --gr-nhid 64 --gr-nlayer 4 --w-pos $w_pos --gamma 1 --ntl-ninp 64 --ntl-nhid 64 --dropout $dropout --bilinear --optim Adam --lr $lr --num-epochs 100 --num-warmup-steps $num_warmup_steps
python3 train.py --input-dir /work/yu_gai/cfq --output-dir /data/yu_gai/cfq --split $split --train-batch-size 64 --eval-batch-size 64 --num-workers 8 --seq-model lstm --seq-ninp 128 --nhead 4 --seq-nhid 512 --seq-nlayer 2 --gr-model rgcn --gr-ninp 64 --gr-nhid 64 --gr-nlayer 4 --w-pos 1 --gamma 1 --ntl-ninp 64 --ntl-nhid 64 --dropout 0.0 --optim Adam --lr 1e-3 --num-epochs 100 --num-warmup-steps 0 --seq2seq
echo exit status: $?
# print completion time
date
