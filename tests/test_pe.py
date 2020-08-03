import torch

from transformer_encoder.utils import PositionalEncoding

d_model = 512
dropout = 0.1
max_len = 100
batch_size = 64


def test_pe():
    PE = PositionalEncoding(d_model, dropout=dropout, max_len=max_len)
    embeds = torch.randn(batch_size, max_len, d_model)  # (batch_size, max_len, d_model)
    out = PE(embeds)
    assert embeds.size() == out.size()
