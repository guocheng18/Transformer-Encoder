import torch

from transformer_encoder.layer_norm import LayerNorm

d_model = 512
eps = 1e-6
batch_size = 64
max_len = 100


def test_ln():
    LN = LayerNorm(d_model, eps)
    x = torch.randn(batch_size, max_len, d_model)
    out = LN(x)
    assert x.size() == out.size()
