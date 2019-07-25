import torch.optim as optim

from tfencoder.encoder import TFEncoder
from tfencoder.utils import TFOptimizer

d_model = 512
n_heads = 8
d_ff = 2048
dropout = 0.1
n_layers = 6

factor = 1
warmup = 20

def test_optim():
    enc = TFEncoder(n_layers, d_model, d_ff, n_heads, dropout)
    opt = TFOptimizer(d_model, factor, warmup, optim.Adam(enc.parameters()))
    assert type(opt.rate(step=1)) is float # step starts from 1
    opt.step()
