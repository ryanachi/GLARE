import argparse
import os
from textattack.constraints.grammaticality.language_tool import LanguageTool
from textattack.shared import AttackedText
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('job', type=str)
parser.add_argument('dataset', type=str)
parser.add_argument('--num_examples', type=int, default=1000)
parser.add_argument('--filepath', type=str)
args = parser.parse_args()

if args.filepath == None:
    args.filepath = f'{args.job}_languagetool.txt'

lt = LanguageTool()

total = 0
num = 0

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'results/{args.job}/{args.job}_slim.txt'), 'r') as f:
    pbar = tqdm.tqdm(total=args.num_examples)
    index = f.readline().strip()

    while index.isnumeric():
        status = f.readline().strip()
        original = f.readline().strip()
        perturbed = f.readline().strip()

        if 'nli' in args.dataset:
            f.readline()
            original = perturbed[12 if args.dataset == 'mnli' else 10:]
            perturbed = f.readline().strip()[12 if args.dataset == 'mnli' else 10:]

        f.readline()

        if 'FAILED' not in status and 'SKIPPED' not in status:
            total += lt.get_errors(AttackedText(perturbed)) - lt.get_errors(AttackedText(original))
            num += 1
        
        pbar.update(1)
        index = f.readline()[:-1]

    pbar.close()

print(f'Number of docs evaluated: {num}')
print(f'Average change in LanguageTool errors: {total/num}\n')

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'results/{args.job}/{args.filepath}'), 'a') as g:
    g.write(f'Average change in LanguageTool errors: {total/num}\n\n')

g.close()
