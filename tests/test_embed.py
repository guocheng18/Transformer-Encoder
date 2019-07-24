import torch

from tfencoder.embedding import Embeddings

d_model = 512
n_vocab = 10000
seq = [[1, 2, 3, 4, 5], [6, 2, 1, 0, 0]]

def test_embedding():
    embed = Embeddings(d_model, n_vocab)
    src = torch.LongTensor(seq)
    out = embed(src)
    assert isinstance(out, torch.FloatTensor)
    assert out.size() == torch.Size([src.size(0), src.size(1), d_model])
