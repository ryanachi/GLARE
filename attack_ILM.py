from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedySearch, GreedyWordSwapWIR
from textattack.shared.attack import Attack
from textattack.attack_recipes.attack_recipe import AttackRecipe
from textattack.constraints.semantics.word_embedding_distance import WordEmbeddingDistance

from transform_ilm import TransformILM
from nli_constraint import NLIConstraint
from greedy_ilm_wir import GreedyILMWIR

import os
from os.path import dirname, realpath
from dotenv import load_dotenv

load_dotenv(os.path.join(dirname(realpath(__file__)), '.env'))

class ILM(AttackRecipe):
    @staticmethod
    def build(model, infill_dirpath=None, max_candidates=10, max_extension=5, threshold=0.8, dataset='yelp_polarity'):
        model_dir = os.environ['MODEL_DIR'] if infill_dirpath is None else infill_dirpath

        transformation = TransformILM(
            infill_dirpath=model_dir,
            max_candidates=max_candidates,
            max_extension=3
        )

        # Don't modify the same word twice or stopwords.
        constraints = [RepeatModification(), StopwordModification()]#, MaxIndexNumberModification(max_length=max_indices)]

        # Constrain word swaps that do not meet at least 0.9 cosine similarity in embedding space.
        # constraints.append(WordEmbeddingDistance(min_cos_sim=0.7))

        use_constraint = UniversalSentenceEncoder(
            threshold=threshold,
            metric="cosine",
            compare_against_original=True,
            window_size=None
        )
        constraints.append(use_constraint)

        # In the sentence-pair tasks (e.g. MNLI, QNLI), we attack the longer sentence excluding the tokens that appear in both.
        if 'nli' in dataset:
            constraints.append(NLIConstraint())

        # Goal is untargeted classification.
        # "The score is then the negative probability of predicting the gold label from f, using [x_{adv}] as the input"
        goal_function = UntargetedClassification(model)

        search_method = GreedyILMWIR(wir_method='unk')

        return Attack(goal_function, constraints, transformation, search_method)