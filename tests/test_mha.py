import torch
from tfencoder.multi_head_attention import MultiHeadedAttention


d_model = 512
n_heads = 8
batch_size = 64
max_len = 100


def test_mha():
    mha = MultiHeadedAttention(n_heads, d_model)
    x = torch.randn(batch_size, max_len, d_model)
    out = mha(x, x, x)
    assert out.size() == x.size()


def test_masked_mha():
    mha = MultiHeadedAttention(n_heads, d_model)
    x = torch.randn(batch_size, max_len, d_model)
    mask = torch.randn(batch_size, max_len).ge(0)
    out = mha(x, x, x, mask)
    assert out.size() == x.size()
