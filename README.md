# Transformer Encoder
<p>
    <img src="https://img.shields.io/travis/com/guocheng2018/transformer-encoder" />
    <img src="https://img.shields.io/badge/contributions-welcome-brightgreen" />
</p>

This repository provides a pytorch implementation of the encoder of [Transformer](http://papers.nips.cc/paper/7181-attention-is-all-you-need/).

<p>
    <img src="https://i.ibb.co/YhR6wWf/encoder.png" alt="encoder" border="0" />
</p>

## Getting started

Build a transformer encoder
```python
from transformer_encoder import TransformerEncoder

encoder = TransformerEncoder(d_model=512, d_ff=2048, n_heads=8, n_layers=6, dropout=0.1)

input_seqs = ...
mask = ...
out = encoder(input_seqs, mask)
```

Add positional encoding to input embeddings
```python
import torch.nn as nn
from transformer_encoder.utils import PositionalEncoding

input_layer = nn.Sequential(
    nn.Embedding(num_embeddings=10000, embedding_dim=512),
    PositionalEncoding(d_model=512, dropout=0.1, max_len=5000)
)
```

Optimize model with the warming up strategy 
```python
import torch.optim as optim
from transformer_encoder.utils import WarmupOptimizer

model = ...

base_optimizer = optim.Adam(model.parameters(), lr=1e-3)
optimizer = WarmupOptimizer(base_optimizer, d_model=512, scale_factor=1, warmup_steps=100)

optimizer.zero_grad()
loss = ...
loss.backward()
optimizer.step()
```

## API Reference

*transformer_encoder.TransformerEncoder(d_model, d_ff, n_heads=1, n_layers=1, dropout=0.1)*

- `d_model`: dimension of each word vector
- `d_ff`: hidden dimension of feed forward layer
- `n_heads`: number of heads in self-attention (defaults to 1)
- `n_layers`: number of stacked layers of encoder (defaults to 1)
- `dropout`: dropout rate (defaults to 0.1)

*transformer_encoder.TransformerEncoder.forward(x, mask)*

- `x (~torch.FloatTensor)`: shape *(batch_size, max_seq_len, d_model)*
- `mask (~torch.ByteTensor)`: shape *(batch_size, max_seq_len)*

*transformer_encoder.utils.PositionalEncoding(d_model, dropout=0.1, max_len=5000)*

- `d_model`: same as TransformerEncoder
- `dropout`: dropout rate (defaults to 0.1)
- `max_len`: max sequence length (defaults to 5000)

*transformer_encoder.utils.PositionalEncoding.forward(x)*

- `x (~torch.FloatTensor)`: shape *(batch_size, max_seq_len, d_model)*

*transformer_encoder.utils.WarmupOptimizer(base_optimizer, d_model, scale_factor, warmup_steps)*

- `base_optimizer (~torch.optim.Optimzier)`: e.g. adam optimzier
- `d_model`: equals d_model in TransformerEncoder
- `scale_factor`: scale factor of learning rate
- `warmup_steps`: warming up steps 


## Installation
Requires `python 3.5+`, `pytorch 1.0.0+`
```
pip install transformer_encoder
```
