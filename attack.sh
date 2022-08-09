 # arg 1: job name
# arg 2: dataset
# arg 3: num examples
# arg 4: start index
NAME=$1
NUM_EXAMPLES=$2
START_INDEX=$3
DATASET=$4

source ~/infilling-adversarial/venv/bin/activate
python ~/infilling-adversarial/run_attack.py \
    --wandb \
    # --train \
    --out $NAME \
    --num_examples $NUM_EXAMPLES \
    --start_index $START_INDEX \
    --max_seq_length 100 \
    --checkpoint_interval 100 \
    --max_candidates 30 \
    --threshold 0.7 \
    $DATASET

bash evaluate.sh $NAME $DATASET
