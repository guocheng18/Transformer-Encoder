import torch

from transformer_encoder.encoder import TransformerEncoder

d_model = 512
n_heads = 8
batch_size = 64
max_len = 100
d_ff = 2048
dropout = 0.1
n_layers = 6


def test_encoder():
    enc = TransformerEncoder(n_layers, d_model, d_ff, n_heads, dropout)
    x = torch.randn(batch_size, max_len, d_model)
    mask = torch.randn(batch_size, max_len).ge(0)
    out = enc(x, mask)
    assert x.size() == out.size()
