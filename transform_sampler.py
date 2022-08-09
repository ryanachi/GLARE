# TODO: custom args for text and indices

from textattack.shared import AttackedText
from transform_ilm import TransformILM
import os
from dotenv import load_dotenv
import argparse
import time
from os.path import realpath, dirname
from textattack.datasets.huggingface_dataset import HuggingFaceDataset
from nli_constraint import NLIConstraint

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str)
parser.add_argument('--train', action='store_const', const=os.path.join(dirname(realpath(__file__)), f'training/train'))
args = parser.parse_args()

load_dotenv(os.path.join(dirname(realpath(__file__)), '.env'))

if args.train:
    model_dir = os.path.join(dirname(realpath(__file__)), f'training/{args.dataset}')
else:
    model_dir = os.environ['MODEL_DIR']

if 'nli' in args.dataset:
    dataset = HuggingFaceDataset('glue', args.dataset, 'test_matched' if args.dataset == 'mnli' else 'test')
    od, _ = dataset[0]
    at = AttackedText(od)
    text = at.text
    indices = NLIConstraint()._get_modifiable_indices(at)
    split1 = at.text.split('\n')
    split = split1[0].split() + split1[1].split()
else:
    text = "This booth is totally secluded from the rest of the restaurant and we got to watch the chefs prep the meals. Pretty amazing if you're a foodie." 
    at = AttackedText(text)
    indices = [3, 4, 5, 18, 19, len(text.strip().split(' ')) - 1]
    split = text.split(' ')

for i in indices:
    split[i] = '[' + split[i] + ']'

print("Input text:", ' '.join(split))
start_time = time.time()
transform = TransformILM(model_dir, max_extension=5, max_candidates=5, debug=True)
load_time = time.time()
print(f"Transform loaded in {load_time - start_time}s")
candidates = transform._get_transformations(at, indices)
print(f"Candidates generated in {time.time() - load_time}s")

for c in candidates:
    print(c.text + '\n')