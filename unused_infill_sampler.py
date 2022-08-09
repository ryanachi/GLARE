from ilm.ilm.tokenize_util import Tokenizer, encode, decode
from textattack.shared import AttackedText, utils
from infilling_model import InfillingModel
from infill import infill_word

"""
Convenience function which replaces a word at an index in a textattack.shared.AttackedText and returns the raw text only.
Params:
- oldtext: textattack.shared.AttackedText--the old text
- idx: int--the index to replace at
- word: str--the string to replace with. MUST be alphanumeric, no whitespace, non-empty
"""
def textonly_replace(oldtext, idx, word):
    perturbed_text = ""
    original_text = AttackedText.SPLIT_TOKEN.join(oldtext._text_input.values())
    new_words = oldtext.words.copy()
    new_words[idx] = word

    # Create the new attacked text by swapping out words from the original
    # text with a sequence of 0+ words in the new text.
    for input_word, adv_word_seq in zip(oldtext.words, new_words):
        word_start = original_text.index(input_word)
        word_end = word_start + len(input_word)
        perturbed_text += original_text[:word_start]
        original_text = original_text[word_end:]

        # Add substitute word(s) to new sentence.
        perturbed_text += adv_word_seq
    
    perturbed_text += original_text  # Add all of the ending punctuation.
    return perturbed_text

model = InfillingModel('./training/yelp_polarity')
model.to(utils.device)

text = "This booth is totally secluded from the rest of the restaurant and we got to watch the chefs prep the meals. Pretty amazing if you're a foodie." 
original_text = AttackedText(text)
current_text = AttackedText(text)

transformed_texts = []
masked_docs = []
_blank_id = encode(' _', Tokenizer.GPT2)[0]
idxs = [3, 4, 5]
cur_len = 0
blocked_texts = []

always_ngram = True

for i, idx in enumerate(idxs):
    if i > 0 and idx - idxs[i - 1] == 1:
        cur_len += 1

    if i == len(idxs) - 1 or idxs[i + 1] - idx > 1:
        words = current_text.words[:]

        for j in range(1, cur_len + 1):
            words[idx - j] = ''

        btext = current_text.generate_new_attacked_text(words)
        blocked_texts.append(btext)
        masked_text = textonly_replace(btext, idx - cur_len, ' _')
        print('Masked text:', masked_text)
        masked_ids = encode(masked_text)
        new_idx = masked_ids.index(_blank_id)               # the underscore may have shifted position, so we need to re-index and find it
        masked_ids[new_idx] = model.ilm_tokens_to_ids['<|infill_ngram|>' if cur_len > 0 or always_ngram else '<|infill_word|>']
        masked_docs.append(masked_ids)
        blocked_texts.append(btext)
        cur_len = 0

for i, doc in enumerate(masked_docs):
    infills = infill_word(model.model, model.ilm_tokens_to_ids, doc, utils.device, num_candidates=5)
    print(infills)

    for c in infills:
        new_text = blocked_texts[i].replace_word_at_index(idxs[i], c.strip())
        print(new_text.text)
        print('map:', new_text.attack_attrs['original_index_map'])
