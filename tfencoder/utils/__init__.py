import copy

import torch.nn as nn

from .embedding import Embeddings as TFEmbeddings
from .optimizer import TFOptimizer
from .positional_encoding import PositionalEncoding as TFPositionalEncoding


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
