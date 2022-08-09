# Wrapper Class for our infilling model.
import pickle
import os
import string
import torch
import sys
import pytorch_lightning as pl
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, './blank_lm')

from vocab import Vocab
from utils import load_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class InfillingModelBlank:
    def __init__(self, model_path, seed=1111):
        pl.seed_everything(seed)
        self.model = load_model(model_path)
        self.model.eval()
        self.vocab = Vocab(os.path.join(self.model.hparams.root_dir, 'vocab.txt'))

    def to(self, device):
        self.model.to(device=device)
