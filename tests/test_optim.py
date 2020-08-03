import torch.optim as optim

from transformer_encoder.encoder import TransformerEncoder
from transformer_encoder.utils import WarmupOptimizer

d_model = 512
n_heads = 8
d_ff = 2048
dropout = 0.1
n_layers = 6

scale_factor = 1
warmup_steps = 20


def test_optim():
    enc = TransformerEncoder(d_model, d_ff, n_heads=n_heads, n_layers=n_layers, dropout=dropout)
    opt = WarmupOptimizer(optim.Adam(enc.parameters()), d_model, scale_factor, warmup_steps)
    assert type(opt.rate(step=1)) is float  # step starts from 1
    opt.step()
