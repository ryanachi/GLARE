# Wrapper Class for our infilling model.
import pickle
import os
import string
import torch
from transformers import GPT2LMHeadModel
from ilm.ilm.tokenize_util import update_tokenizer, Tokenizer, tokens_to_ids

class InfillingModel:
    def __init__(self, model_dir):
        with open(os.path.join(model_dir, 'additional_ids_to_tokens.pkl'), 'rb') as f:
            self.ilm_ids_to_tokens = pickle.load(f)

        self.ilm_tokens_to_ids = { v: k for k, v in self.ilm_ids_to_tokens.items() }

        try:
            update_tokenizer(self.ilm_ids_to_tokens, Tokenizer.GPT2)
        except ValueError:
            print("Already Updated")

        self.model = GPT2LMHeadModel.from_pretrained(model_dir)
        self.model.eval()

    def to(self, device):
        self.model.to(device=device)
