from textattack.shared import utils, AttackedText
from textattack.transformations import Transformation, WordSwap
from ilm.ilm.tokenize_util import encode, decode, Tokenizer
from infill import infill_word
from infilling_model import InfillingModel

"""
Convenience function which replaces a word at an index in a textattack.shared.AttackedText and returns the raw text only.
Params:
- oldtext: textattack.shared.AttackedText--the old text
- new_words: a sequence of strings containing the sequence to replace with. Must follow the same conventions as TextAttack replacement lists.
"""
def textonly_replace(oldtext, new_words):
    perturbed_text = ""
    original_text = ' '.join(oldtext._text_input.values())

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

class TransformILM(WordSwap):
    def __init__(
        self,
        infill_dirpath,
        max_candidates=50,
        max_extension=5,
        allow_delete=False,
        debug=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_candidates = max_candidates
        self.max_extension = max_extension
        self._infilling_model = InfillingModel(infill_dirpath)
        self._infilling_model.to(utils.device)
        self.allow_delete = allow_delete
        self._debug = debug

    def _get_transformations(self, current_text, indices_to_modify):
        """
        Get a list of candidate transformations for a text which swap one or more words in a given list of indices.

        Params:
            current_text: an AttackedText representing the text to transform.
            indices_to_modify: a list of indices specifying which words to transform.

        Returns: a list of candidate transformations for the text. No word is guaranteed to have been swapped, but at least one word is changed in each candidate.
        """
        if len(indices_to_modify) == 0:
            return []

        transformed_texts = []
        kept = 0
        masked_docs = []
        collapsed_words = []
        _blank_id = encode(' _', Tokenizer.GPT2)[0]
        idxs = list(indices_to_modify)
        j = 0

        while j < len(idxs):
            cur = idxs[j]
            span_len = 1
            words = current_text.words[:]
            j += 1

            while j < len(idxs) and idxs[j] - cur == span_len:
                span_len += 1
                words[idxs[j]] = ''
                idxs.pop(j)

            words[cur] = ' _'
            collapsed_words.append(words)
            masked_text = textonly_replace(current_text, words)
            masked_ids = encode(masked_text)

            if len(masked_ids) > 1000:
                print('Too long!')
                return []

            new_idx = masked_ids.index(_blank_id)               # the underscore may have shifted position, so we need to re-index and find it
            masked_ids[new_idx] = self._infilling_model.ilm_tokens_to_ids['<|infill_ngram|>']
            masked_docs.append(masked_ids)

        for i, doc in enumerate(masked_docs):
            infills = infill_word(self._infilling_model.model, self._infilling_model.ilm_tokens_to_ids, doc, utils.device, num_candidates=self.max_candidates)

            for infill in infills:
                if infill == '':
                    if self.allow_delete:
                        collapsed_words[i][idxs[i]] = ''
                        transformed_texts.append(current_text.generate_new_attacked_text(collapsed_words[i]))
                        kept += 1

                    continue

                if idxs[i] > 0 and infill[0] != ' ' or '\n' in infill:
                    continue

                collapsed_words[i][idxs[i]] = infill.strip()
                new_text = current_text.generate_new_attacked_text(collapsed_words[i])
                shift = len(new_text.words) - len(current_text.words)

                if len(new_text.attack_attrs["newly_modified_indices"]) > 0 and abs(shift) <= self.max_extension:
                    transformed_texts.append(new_text)
                    kept += 1 

        if self._debug:
            print("Kept:", kept / (len(idxs) * self.max_candidates))
        
        return transformed_texts