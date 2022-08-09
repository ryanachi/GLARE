from textattack.shared import utils, AttackedText
from textattack.transformations import Transformation, WordSwap
from ilm.ilm.tokenize_util import encode, decode, Tokenizer
from old_infill import infill_word
from infilling_model import InfillingModel

"""
Convenience function which replaces a word at an index in a textattack.shared.AttackedText and returns the raw text only.
Params:
- oldtext: textattack.shared.AttackedText--the old text
- idx: int--the index to replace at
- word: str--the string to replace with. MUST be alphanumeric, no whitespace, non-empty
"""
def textonly_replace(oldtext, idx, word):
    perturbed_text = ""
    original_text = ' '.join(oldtext._text_input.values())
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

class TransformILM(WordSwap):
    def __init__(
        self,
        infill_dirpath,
        max_candidates=50,
        max_extension=10,
        debug=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_candidates = max_candidates
        self.max_extension = max_extension
        self._infilling_model = InfillingModel(infill_dirpath)
        self._infilling_model.to(utils.device)
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
        _blank_id = encode(' _', Tokenizer.GPT2)[0]
        idxs = list(indices_to_modify)

        for idx in idxs:
            masked_text = textonly_replace(current_text, idx, ' _')
            masked_ids = encode(masked_text)

            if len(masked_ids) > 1000:
                print('Too long!')
                return []

            new_idx = masked_ids.index(_blank_id)               # the underscore may have shifted position, so we need to re-index and find it
            masked_ids[new_idx] = self._infilling_model.ilm_tokens_to_ids['<|infill_word|>']
            masked_docs.append(masked_ids)
        
        infills = infill_word(self._infilling_model.model, self._infilling_model.ilm_tokens_to_ids, masked_docs, utils.device, num_candidates=self.max_candidates)

        for idx in range(len(infills)):
          infill_strings = set(['', current_text.words[idxs[idx]]])

          for infill_string in infills[idx]:
            # TODO: Insertions/Deletions
            if infill_string[0] != ' ':
                continue
            
            new_text = current_text.replace_word_at_index(idxs[idx], infill_string[1:])
            if len(new_text.attack_attrs["newly_modified_indices"]) > 0 and infill_string not in infill_strings:
                transformed_texts.append(new_text)
                infill_strings.add(infill_string)
                kept += 1

        if self._debug:
            print("Kept:", kept / (len(idxs) * self.max_candidates))
        
        return transformed_texts
