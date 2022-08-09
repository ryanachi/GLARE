#!/bin/sh

# arg 1: job name
# arg 2: num examples
# arg 3: start index

#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --open-mode=truncate
#SBATCH --partition=jag-standard
#SBATCH --output=sl/slurm-%j.out
#SBATCH --time=10-0
#SBATCH --exclude=jagupard[4-8],jagupard17

NAME=$1
NUM_EXAMPLES=$2
START_INDEX=$3

source /juice/scr/$PWD/infilling-adversarial/venv/bin/activate
python /juice/scr/$PWD/infilling-adversarial/attack_sampler.py \
    --train \
    --out $NAME \
    --num_examples $NUM_EXAMPLES \
    --start_index $START_INDEX \
    --max_char_length 300 \
    --wandb
