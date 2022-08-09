# arg 1: experiment name
# arg 2: dataset
NAME=$1
DATASET=$2

source ~/infilling-adversarial/venv/bin/activate
python ~/infilling-adversarial/attack_sampler.py \
    --wandb \
    --out $NAME \
    --num_examples 1000 \
    --start_index 0 \
    --max_seq_length 100 \
    --recipe bert_attack \
    $DATASET