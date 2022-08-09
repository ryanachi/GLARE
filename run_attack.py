import transformers
import textattack
from textattack.attack_results import SuccessfulAttackResult, FailedAttackResult, MaximizedAttackResult, SkippedAttackResult
from file_logger_slim import FileLoggerSlim
from collections import deque
import tqdm
import time
import argparse
from dotenv import load_dotenv
import os
from os.path import realpath, dirname
import re

from attack_ILM import ILM
from old_recipe import OldILM
from textattack.attack_recipes.clare_li_2020 import CLARE2020
from textattack.attack_recipes.bert_attack_li_2020 import BERTAttackLi2020
from textattack.attack_recipes.textfooler_jin_2019 import TextFoolerJin2019
from textattack.attack_recipes.textbugger_li_2018 import TextBuggerLi2018

def get_text(ex):
    if 'premise' in ex.keys():
        return ex['premise'] + ' ' + ex['hypothesis']
    elif 'question' in ex.keys():
        return ex['question'] + ' ' + ex['sentence']
    else:
        return ex['text']

load_dotenv(os.path.join(dirname(realpath(__file__)), '.env'))

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str)
parser.add_argument('--victim', type=str, default='bert-base-uncased')
parser.add_argument('--recipe', type=str, default='glare')

parser.add_argument('--train', action='store_const', const=True)
parser.add_argument('--out', type=str, default='temp')

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--num_examples', type=int, default=-1)
parser.add_argument('--start_index', type=int, default=0)
parser.add_argument('--prune', action='store_const', const=True, default=False)
parser.add_argument('--max_char_length', type=int)
parser.add_argument('--max_seq_length', type=int)

parser.add_argument('--max_candidates', type=int, default=10)
parser.add_argument('--max_extension', type=int, default=5)
parser.add_argument('--threshold', type=float, default=0.8)

parser.add_argument('--checkpoint_interval', type=int)#, default=50)
parser.add_argument('--attack_n', type=int)
parser.add_argument('--checkpoint_dir', type=str, default=os.path.join(dirname(realpath(__file__)), 'checkpoints'))
parser.add_argument('--wandb', action='store_const', const=True, default=False)
args = parser.parse_args()

args.train = f'training/{args.dataset}' if args.train else os.environ['MODEL_DIR']
MODEL = args.victim
DATASET = args.dataset

if 'nli' in args.dataset:
    DATASET = DATASET.upper()

if os.path.exists(os.path.join(dirname(realpath(__file__)), f'results/{args.out}/{args.out}')):
    prompt = input(f'This will overwrite experiment results in results/{args.out}/{args.out}. Continue? [y]/n')

    if prompt != '\n' and prompt != 'y':
        quit()

model = transformers.AutoModelForSequenceClassification.from_pretrained(f"textattack/{MODEL}-{DATASET.replace('_', '-')}")
tokenizer = textattack.models.tokenizers.AutoTokenizer(f"textattack/{MODEL}-{DATASET.replace('_', '-')}")
model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

# Create the goal function using the model
goal_function = textattack.goal_functions.classification.UntargetedClassification(model_wrapper)

# Import the dataset

if args.dataset == 'mnli':
    dataset = textattack.datasets.HuggingFaceDataset('glue', 'mnli', 'validation_matched', [1, 2, 0])
elif args.dataset == 'qnli':
    dataset = textattack.datasets.HuggingFaceDataset('glue', 'qnli', 'validation')
else:
    dataset = textattack.datasets.HuggingFaceDataset(DATASET, None, "test")

start_time = time.time()

# Filter dataset.
dataset._dataset = dataset._dataset.shuffle(seed=args.seed)

if args.prune:  
    dataset._dataset = dataset._dataset.filter(lambda x: re.search(r"\W{2}|\\", get_text(x)) == None)

if args.max_char_length:
    dataset._dataset = dataset._dataset.filter(lambda x: len(get_text(x)) < args.max_char_length)

if args.max_seq_length:
    dataset._dataset = dataset._dataset.filter(lambda x: len(get_text(x).split(' ')) < args.max_seq_length)

dataset.examples = list(dataset._dataset)
args.num_examples = len(dataset.examples) - args.start_index if args.num_examples == -1 else args.num_examples
worklist = deque(range(args.start_index, min(args.start_index + args.num_examples, len(dataset.examples))))
worklist_tail = worklist[-1]

# Build attack on victim model
if args.recipe == 'glare':
    attack = ILM.build(model_wrapper, args.train, args.max_candidates, args.max_extension, args.threshold, args.dataset)
elif args.recipe == 'glare-old':
    attack = OldILM.build(model_wrapper, args.train, args.max_candidates, args.threshold, dataset=args.dataset)
elif args.recipe == 'clare':
    attack = CLARE2020.build(model_wrapper)
elif args.recipe == 'bert_attack':
    attack = BERTAttackLi2020.build(model_wrapper)
elif args.recipe == 'textfooler':
    attack = TextFoolerJin2019.build(model_wrapper)
elif args.recipe == 'textbugger':
    attack = TextBuggerLi2018.build(model_wrapper)

build_time = time.time()
print('Attack built on', MODEL, f'in {build_time - start_time}')

# Logging + Checkpoints
log_manager = textattack.loggers.AttackLogManager()
log_manager.enable_stdout()

if args.wandb:
    log_manager.enable_wandb()

file_head = os.path.join(dirname(realpath(__file__)), f'results/{args.out}/{args.out}')
log_manager.add_output_csv(file_head + '.csv', color_method='html')
log_manager.add_output_file(file_head + '.txt')
log_manager.loggers.append(FileLoggerSlim(file_head + '_slim.txt'))
args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.out)

with open(f'{file_head}_args.txt', 'w') as f:
    for k, v in vars(args).items():
        f.write(f'{k}: {v}\n')

f.close()

pbar = tqdm.tqdm(total=args.num_examples, smoothing=0)
num_results = 0
num_failures = 0
num_successes = 0
results_iterable = attack.attack_dataset(dataset, indices=worklist)
i = 0

while i < args.num_examples:
    # print(f'current example: {get_text(dataset.examples[i + args.start_index])}')
    result = next(results_iterable)
    log_manager.log_result(result)

    if (
        type(result) == SuccessfulAttackResult
        or type(result) == MaximizedAttackResult
    ):
        num_successes += 1
    if type(result) == FailedAttackResult:
        num_failures += 1

    if not isinstance(result, SkippedAttackResult):
        num_results += 1
    
    pbar.set_description(
        "({}) [Succeeded / Failed / Total / Success Rate] {} / {} / {} / {}".format(
            args.out, num_successes, num_failures, num_results, round(num_successes / num_results, 2)
        )
    )

    if (
        args.checkpoint_interval
        and len(log_manager.results) % args.checkpoint_interval == 0
    ):
        new_checkpoint = textattack.shared.Checkpoint(
            args, log_manager, worklist, worklist_tail
        )
        new_checkpoint.save()
        log_manager.flush()

    pbar.update(1)
    i += 1

pbar.close()
print()
# Enable summary stdout
log_manager.log_summary()
log_manager.flush()
print()
finish_time = time.time()
textattack.shared.logger.info(f"Attack time: {finish_time - build_time}s")

with open(file_head + '.txt', 'a') as f:
    f.write(f"\nAttack time: {finish_time - build_time}s\n")