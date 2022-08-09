# arg 1: dataset name
DATASET=$1
MODEL=$2

if [ "$MODEL" == "ilm" ]
then
	pip install -e ../${MODEL}
	python train_ported.py \
		experiment_${DATASET} \
		${DATASET} \
		data/char_masks/${DATASET} \
		--seed 0 \
		--train_examples_tag train \
		--eval_examples_tag valid \
		--eval_max_num_examples 512 \
		--train_num_epochs 2 \
		--wandb
else
	python train_blank_lm_ported.py \
		--train raw_data/${DATASET}_blank_lm/train.txt \
		--valid raw_data/${DATASET}_blank_lm/valid.txt \
		--root_dir ${DATASET}/${MODEL} \
		--vocab_size 10000 \
		--model_type blm \
		--share_emb_prj_weight \
		--add_eos \
		--max_steps 5000 \
		--max_len 100
fi