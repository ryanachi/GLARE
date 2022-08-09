NAME=$1
DATASET=$2

for METRIC in use languagetool perplexity
do
    python eval_${METRIC}.py $NAME $DATASET --filepath ${NAME}_other_metrics.txt
done