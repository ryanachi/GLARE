import argparse
import os
from difflib import SequenceMatcher

'''
Parser arguments:
in_file: path to current results
out_file: location to save processed results
opt: extra options
    1: GCP textattack copypasta (tqdm and --Result XX--... may be on overlapping lines, multi-line sentences)
    2: Textattack textbugger MNLI log-to-txt
'''
parser = argparse.ArgumentParser()
parser.add_argument('in_file', type=str)
parser.add_argument('out_file', type=str)
parser.add_argument('opt', nargs='?', type=int, default=0)
args = parser.parse_args()


assert os.path.exists(args.in_file)

with open(args.in_file, 'r') as f:
    lines = f.readlines()

for i in range(len(lines)):
    lines[i] = lines[i][:-1]
    assert '\n' not in lines[i]

split_indices = []
if args.opt == 0:
    for i in range(len(lines)):
        if '-' * 45 in lines[i]:
            split_indices.append(i)
elif args.opt == 1:
    for i in range(1, len(lines)):
        if '-'*45 in lines[i] and ('s/it' in lines[i-1] or 'it/s' in lines[i-1]):
        #if '[Succeeded / Failed / Total]' in line and line.index('[Succeeded / Failed / Total]') == 0:
            split_indices.append(i)
if args.opt == 2:
    for i in range(len(lines)):
        if '-' * 45 in lines[i]:
            split_indices.append(i)

print(f"Found {len(split_indices)} examples")

if args.opt == 1:
    found = False
    for i in range(split_indices[-1], len(lines)):
        if '[Succeeded / Failed / Total]' in lines[i]:
            split_indices.append(i+1)
            found = True
            break
    if not found:
        split_indices.append(len(lines)+1)
else:
    split_indices.append(len(lines) + 1)

partitions = []
if args.opt == 1:
    for i in range(len(split_indices)-1):
        partitions.append(lines[split_indices[i]+1:split_indices[i+1]-1])
elif args.opt == 2:
    for i in range(len(split_indices)-1):
        partitions.append(lines[split_indices[i]+1:split_indices[i+1]])

output = []
if args.opt == 1:
    for i, part in enumerate(partitions):
        out_part = []
        states = part[0].split(' --> ')
        assert(len(states) == 2)
        pre, post = tuple(states)
        out_part.append(f"[[{pre}]] --> [[{post}]]")

        if post == '[SKIPPED]' or post == '[FAILED]':
            sentence = ''.join(part[1:])
            out_part.append(sentence)
            out_part.append(sentence)
        else:
            first_line = part[1]
            best_idx = 2
            best_ratio = -1.
            for j in range(1 + (len(part)-1)//2, len(part) - (len(part)-1)//2 + 1):
                r = SequenceMatcher(None, first_line, part[j]).ratio()
                if r > best_ratio:
                    best_ratio = r
                    best_idx = j
            out_part.append(''.join(part[1:best_idx]))
            out_part.append(''.join(part[best_idx:]))

        output.append(out_part)
elif args.opt == 2:
    for i, part in enumerate(partitions):
        out_part = []
        out_part.append(part[0]) # Class/Skip/Fail
        out_part.append(part[1]) # New line
        premise = part[2].replace('[[[[Premise]]]]: ', '').replace('[[', '').replace(']]', '')
        hypothesis = part[3].replace('[[[[Hypothesis]]]]: ', '').replace('[[', '').replace(']]', '')
        out_part.append('*'*12 + premise + ' ' + hypothesis)
        if 'SKIPPED' not in part[0] and 'FAILED' not in part[0]:
            if len(part) < 7:
                print(i)
            premise = part[5].replace('[[[[Premise]]]]: ', '').replace('[[', '').replace(']]', '')
            hypothesis = part[6].replace('[[[[Hypothesis]]]]: ', '').replace('[[', '').replace(']]', '')
            out_part.append('*' * 12 + premise + ' ' + hypothesis)
        output.append(out_part)

with open(args.out_file, 'w') as f:
    for idx, i in enumerate(output):
        f.write(str(idx + 1) + '\n')
        for j in i:
            f.write(j + '\n')
        f.write('\n')

result_start = 0
result_found = False
result_end = len(lines)
for i in range(split_indices[-2], len(lines)):
    if result_found:
        if '+-------------------------------+--------+' in lines[i]:
            result_end = i
            break
    elif '| Attack Results                |        |' in lines[i-1] \
            and '+-------------------------------+--------+' in lines[i]:
        result_start = i + 1
        result_found = True

if result_found:
    result_output = []
    for line in lines[result_start:result_end]:
        result_partition = line.split('|')
        result_partition = [r for r in result_partition if r != '']
        label = result_partition[0].strip()
        value = result_partition[1].strip()
        result_output.append(f'{label} {value}')

    print("Results parsed:")
    with open(args.out_file, 'a') as f:
        for i in result_output:
            print(i)
            f.write(i + '\n')