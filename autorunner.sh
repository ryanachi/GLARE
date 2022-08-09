NUM_EXAMPLES=1000
SUFFIX=final3
OLD_SUFFIX=ood
START_INDEX=1000

source attack_old.sh ag_news_$OLD_SUFFIX $NUM_EXAMPLES $START_INDEX ag_news
source attack_old.sh qnli_$OLD_SUFFIX $NUM_EXAMPLES $START_INDEX qnli
source attack_old.sh mnli_$OLD_SUFFIX $NUM_EXAMPLES $START_INDEX mnli
# source attack.sh mnli_$SUFFIX $NUM_EXAMPLES $START_INDEX mnli
source attack_old.sh yelp_$OLD_SUFFIX $NUM_EXAMPLES $START_INDEX yelp_polarity
# source attack.sh yelp_$SUFFIX $NUM_EXAMPLES $START_INDEX yelp_polarity