import math

import torch
import torch.nn as nn


class TFEmbedding(nn.Module):
    def __init__(self, d_model: int, n_vocab: int) -> None:
        super(TFEmbedding, self).__init__()
        self.lut = nn.Embedding(n_vocab, d_model)
        self.d_model = d_model
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lut.weight)

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        """
        Args:
            x : shape (batch_size, max_len)

        Returns:
            shape (batch_size, max_len, d_model)
        """
        return self.lut(x) * math.sqrt(self.d_model)
