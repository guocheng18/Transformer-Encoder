import torch

from transformer_encoder.encoder_layer import EncoderLayer
from transformer_encoder.feed_forward import FeedForward
from transformer_encoder.multi_head_attention import MultiHeadAttention

d_model = 512
n_heads = 8
batch_size = 64
max_len = 100
d_ff = 2048
dropout = 0.1


def test_enclayer():
    # Components
    mha = MultiHeadAttention(n_heads, d_model)
    ff = FeedForward(d_model, d_ff, dropout)
    enclayer = EncoderLayer(d_model, mha, ff, dropout)
    # Input
    x = torch.randn(batch_size, max_len, d_model)
    mask = torch.randn(batch_size, max_len).ge(0)
    out = enclayer(x, mask)
    assert x.size() == out.size()

