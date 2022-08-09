import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from ilm.ilm.tokenize_util import Tokenizer, decode
from torch.cuda import memory_allocated, memory_summary

"""
Returns a list of 1-token infills for the first masked sequence in a sentence, ordered by softmaxed logit values.
Returns a (topk, 1) tensor.
Params: 
- logits: the logit output of the model after reading in the sentence.
- topk: the number of infills to take.

TODO: add option to use nucleus instead of topk
"""
def ordered_tokens(logits, first=False):
    if first:
        _, indices = torch.topk(F.softmax(logits[0], dim=-1), logits.shape[0], dim=-1)
        return indices.unsqueeze(1)
    else:
        _, indices = torch.topk(F.softmax(logits, dim=-1), 1, dim=-1)
        return indices

def infill_word(
        model,
        special_tokens_to_ids,
        x,
        device,
        num_candidates=5,
        k=1,
        max_extension=10,
        nucleus=0.95):
    _sep_id = special_tokens_to_ids['<|startofinfill|>']
    _end_span_id = special_tokens_to_ids['<|endofinfill|>']
    _special_ids = special_tokens_to_ids.values()

      # Make sure example doesn't already ends with [sep]
    if x[-1] == _sep_id:
        x = x[:-1]
    
    # Count number of blanks
    blank_idxs = []

    for i, tok_id in enumerate(x):
        if tok_id in _special_ids:
            blank_idxs.append(i)

    k = len(blank_idxs)

    if k == 0:
        raise ValueError()

    with torch.no_grad():
        context = torch.tensor(x + [_sep_id], dtype=torch.long, device=device).unsqueeze(0).repeat(num_candidates, 1)
        terminated = []
        first = True

        while context.shape[0] > 0:
            #print(context.shape)
            logits = model(context)[0][:, -1]
            #print(logits.shape)
            next_tokens = ordered_tokens(logits, first)
            #print(next_tokens.shape)
            context = torch.cat((context, next_tokens), dim=1)
            #print(context.shape, '\n')
        
            num_predicted_spans = (context == _end_span_id).long().sum(dim=1)
            
            terminate_expected = num_predicted_spans >= k
            terminate_toolong = torch.ones_like(context).long().sum(dim=1) >= len(x) + max_extension
            terminate = terminate_expected | terminate_toolong
            
            if torch.any(terminate):
                terminated_seqs = context[terminate, len(x)+1:]
                terminated.extend([s.tolist() for s in terminated_seqs.cpu().numpy()])
                context = context[~terminate, :]

            first = False

    return [decode(infill[:-1], Tokenizer.GPT2) for infill in terminated]