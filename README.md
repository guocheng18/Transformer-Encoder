# Transformer Encoder
<p>
    <img src="https://img.shields.io/badge/python-3.5 | 3.6 | 3.7-blue" />
    <img src="https://img.shields.io/pypi/v/tfencoder?color=orange" />
    <img src="https://img.shields.io/badge/license-MIT-green" />
    <img src="https://img.shields.io/travis/com/guocheng2018/transformer-encoder" />
</p>
This package provides an easy-to-use interface of transformer encoder.

# Installation

Requirements: `python(>=3.5)`, `pytorch(>=1.0.0)`

Install from pypi:
```
pip install tfencoder
```
Or from Github for the latest version:
```
pip install git+https://github.com/guocheng2018/transformer-encoder.git
```

# Go through

**tfeccoder.TFEncoder(n_layers, d_model, d_ff, n_heads, dropout)**

- `n_layers`: number of stacked layers of encoder
- `d_model`: dimension of each word vector
- `d_ff`: hidden dimension of feed forward layer
- `n_heads`: number of heads in self-attention
- `dropout`: dropout rate, default 0.1

`forward(x, mask)`

- `x(~torch.FloatTensor)`: shape *(batch_size, max_seq_len, d_model)*
- `mask(~torch.ByteTensor)`: shape *(batch_size, max_seq_len)*

Example:
```python
import torch
import tfencoder

encoder = tfencoder.TFEncoder(6, 512, 2048, 8, dropout=0.1)

x = torch.randn(64, 100, 512)  # (batch_size, max_seq_len, d_model)
mask = torch.randn(64, 100).ge(0)  # a random mask

out = encoder(x, mask)
```

This package also provides the embedding, positional encoding and scheduled optimizer that are used in transformer as extra functionalities.

**tfencoder.utils.TFEmbedding(d_model, n_vocab)**

- `d_model`: same as TFEncoder
- `n_vocab`: vocabulary size

`forward(x)`

- `x(~torch.LongTensor)`: shape *(batch_size, max_seq_len)*

**tfencoder.utils.TFPositionalEncoding(d_model, dropout, max_len)**

- `d_model`: same as TFEncoder
- `dropout`: dropout rate
- `max_len`: max sequence length

`forward(x)`

- `x(~torch.FloatTensor)`: shape *(batch_size, max_seq_len, d_model)*

You can combine this two, for example:
```python
import torch
import torch.nn as nn

from tfencoder.utils import TFEmbedding, TFPositionalEncoding

tfembed = TFEmbedding(512, 6)
tfpe = TFPositionalEncoding(512, 0.1, max_len=5)
tfinput = nn.Sequential(tfembed, tfpe)

x = torch.LongTensor([[1,2,3,4,5], [1,2,3,0,0]])
out = tfinput(x)
```

**tfencoder.utils.TFOptimizer(d_model, factor, warmup, optimizer)**

- `d_model`: equals d_model in TFEncoder
- `factor`: scale factor of learning rate
- `warmup`: warmup steps 
- `optimizer(~torch.optim.Optimzier)`: e.g. Adam

Example:
```python
import torch.optim as optim

from tfencoder import TFEncoder
from tfencoder.utils import TFOptimizer

encoder = TFEncoder(6, 512, 2048, 8, dropout=0.1)
optimizer = TFOptimizer(512, 1, 1000, optim.Adam(encoder.parameters(), lr=0))

optimizer.zero_grad()

loss = ...
loss.backward()

optimizer.step()
```

# Contribution
Any contributions are welcome!