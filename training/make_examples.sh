# arg 1: name of dataset
DATASET=$1

for SPLIT in train valid
do
	python ../ilm/create_ilm_examples.py \
		${SPLIT} \
        data/char_masks/${DATASET} \
		--seed 0 \
		--data_name custom \
        --data_dir raw_data/$DATASET \
		--data_split ${SPLIT}
done