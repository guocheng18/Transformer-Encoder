import torch

from tfencoder.utils import TFPositionalEncoding

d_model = 512
dropout = 0.1
max_len = 100
batch_size = 64

def test_pe():
    PE = TFPositionalEncoding(d_model, dropout, max_len)
    embeds = torch.randn(batch_size, max_len, d_model)  # (batch_size, max_len, d_model)
    out = PE(embeds)
    assert embeds.size() == out.size()
