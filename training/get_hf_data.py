from textattack.datasets.huggingface_dataset import HuggingFaceDataset
from textattack.shared import AttackedText
import datasets
import shutil
import os
from tqdm import tqdm
import argparse
from spacy.lang.en import English
import random

nlp = English()
tokenizer = nlp.tokenizer

parser = argparse.ArgumentParser()

parser.add_argument('dataset_name', type=str)
parser.add_argument('--format', type=str, default='ilm')
parser.add_argument('--blank_lm_mask_ratio', type=float, default=0.2)

args = parser.parse_args()

def filter(ex):
    return '\\' not in ex['text']

if 'nli' in args.dataset_name:
    train_dataset = HuggingFaceDataset('glue', args.dataset_name, 'train')
    valid_dataset = HuggingFaceDataset('glue', args.dataset_name, 'validation_matched' if args.dataset_name == 'mnli' else 'validation')
else:
    train_dataset = datasets.load_dataset(args.dataset_name, split='train').filter(filter)
    valid_dataset = datasets.load_dataset(args.dataset_name, split='test').filter(filter)

DATA_DIR=os.path.join(os.getcwd(), f'raw_data/{args.dataset_name}_{args.format}')

if os.path.exists(DATA_DIR):
    shutil.rmtree(DATA_DIR)
    
os.makedirs(DATA_DIR)

def write(split, split_dataset):
    with open(f'{DATA_DIR}/{split}.txt', 'w') as f:
        for ex in tqdm(split_dataset):
            text = AttackedText(ex[0]).text if 'nli' in args.dataset_name else ex['text']

            if args.format == 'ilm':
                f.write('\n\n\n' + text)
            elif args.format == 'blank_lm':
                tokens = []
                blank = False

                for t in tokenizer(text):
                    if random.random() <= 0.2 or t.text == '\n':
                        blank = True
                    else:
                        if blank:
                            tokens.append('<blank>')

                        tokens.append(t.text)
                        blank = False
                
                f.write(' '.join(tokens) + '\n')

write('train', train_dataset)
write('valid', valid_dataset)