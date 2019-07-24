import torch
from tfencoder.feed_forward import FeedForward

batch_size = 64
max_len = 100
d_model = 512
d_ff = 2048
dropout = 0.1

def test_ff():
    ff = FeedForward(d_model, d_ff, dropout)
    x = torch.randn(batch_size, max_len, d_model)
    out = ff(x)
    assert x.size() == out.size()