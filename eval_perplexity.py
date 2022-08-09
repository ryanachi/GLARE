import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import argparse
import os
import tqdm

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

model_id = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

parser = argparse.ArgumentParser()
parser.add_argument('job', type=str)
parser.add_argument('dataset', type=str)
parser.add_argument('--num_examples', type=int, default=1000)
parser.add_argument('--filepath', type=str)
args = parser.parse_args()

if args.filepath == None:
    args.filepath = f'{args.job}_ppl.txt'

original_corpus = ""
perturbed_corpus = ""
num = 0
print('Reading data:')

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
            original_corpus += '\n\n' + original
            perturbed_corpus += '\n\n' + perturbed
            num += 1

        pbar.update(1)
        index = f.readline().strip()

    pbar.close()

# OPTIONAL: print original & perturbed text as a sanity check        
print(f"Original text: {original_corpus[:1000]}\n")
print(f"Perturbed text: {perturbed_corpus[:1000]}")

pbar.close()
f = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'results/{args.job}/{args.filepath}'), 'a') 

def calculate_perplexity(text, split):
    encodings = tokenizer(text, return_tensors='pt')
    max_length = model.config.n_positions

    lls = []
    stride = 512

    for i in tqdm.tqdm(range(0, encodings.input_ids.size(1), stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i    # may be different from stride on last loop
        input_ids = encodings.input_ids[:,begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:,:-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs[0] * trg_len

        lls.append(log_likelihood)

    ppl = torch.exp(torch.stack(lls).sum() / end_loc)
    f.write(f'{split} perplexity: {ppl}\n')
    return ppl

print(f'Original perplexity: {calculate_perplexity(original_corpus, "Original")}')
print(f'Perturbed perplexity: {calculate_perplexity(perturbed_corpus, "Perturbed")}')
f.write('\n')
f.close()
