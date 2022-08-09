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
def ordered_tokens(logits,
                   topk,
                   temp=1.,
                   nucleus=1.):
    if temp == 0:
        return torch.argmax(logits, dim=-1).unsqueeze(-1)
    elif temp != 1:
        logits /= temp

    _, indices = torch.topk(F.softmax(logits, dim=-1), topk, dim=-1)
    return indices


def infill_word(
        model,
        special_tokens_to_ids,
        x,
        device,
        num_candidates=5,
        max_sequence_length=256,
        nucleus=0.95):
    _sep_id = special_tokens_to_ids['<|startofinfill|>']
    _end_span_id = special_tokens_to_ids['<|endofinfill|>']
    _special_ids = special_tokens_to_ids.values()

    # TODO: handle no-blanks error and ends-with-[sep]-already error

    with torch.no_grad():
        tensors = [torch.tensor(line + [_sep_id], dtype=torch.long, device=device) for line in x]
        context = pad_sequence(tensors, batch_first=True)
        del tensors
        # lengths = [len(line) + 1 for line in x]
        # masks = torch.zeros_like(context)

        # for e_id, src_len in enumerate(lengths):
        #     masks[e_id, :src_len] = 1
        
        # masks = masks.float()
        # print(masks)
        logits = model(context)[0][:, -1]
        del context

        # TODO: better way of zeroing out non-full words (may be better solved in get_transformations?)
        next_tokens = ordered_tokens(logits, topk=num_candidates).cpu().numpy()
        del logits

    infills = []

    for i in range(next_tokens.shape[0]):
        infills.append([decode([idx], Tokenizer.GPT2) for idx in next_tokens[i]])

    return infills