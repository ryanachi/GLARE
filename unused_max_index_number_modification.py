
"""
Max Word Index Modification
-----------------------------
"""
from textattack.constraints import PreTransformationConstraint
import random

class MaxIndexNumberModification(PreTransformationConstraint):
    """A constraint disallowing the modification of words which are past some
    maximum length limit."""

    def __init__(self, max_length, seed=42):
        self.max_length = max_length
        random.seed(seed)

    def _get_modifiable_indices(self, current_text):
        """Returns the word indices in current_text which are able to be
        modified."""
        idxs = list(range(len(current_text.words)))
        random.shuffle(idxs)
        return set(idxs[:self.max_length])

    def extra_repr_keys(self):
        return ["max_length"]
