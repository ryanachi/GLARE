from textattack.constraints import PreTransformationConstraint

class NLIConstraint(PreTransformationConstraint):
    def _get_modifiable_indices(self, current_text):
        text = current_text.text
        first_words = []
        indices = []
        second = False

        for i, w in enumerate(current_text.words):
            ind = text.index(w)

            if text[ind - 1] == '\n':
                second = True

            if not second:
                first_words.append(w)
            elif w not in first_words:
                indices.append(i)

            text = text[ind + len(w):]

        return set(indices)
