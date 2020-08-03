# Transformer Encoder
<p>
    <img src="https://img.shields.io/travis/com/guocheng2018/transformer-encoder" />
</p>
This repo provides an easy-to-use interface of transformer encoder. You can use it as a general sequence feature extractor and incorporate it in 
your model.<br><br>
<p>
    <img src="https://i.ibb.co/YhR6wWf/encoder.png" alt="encoder" border="0" />
</p>

## Examples

Quickstart
```python
import torch
import transformer_encoder
from transformer_encoder.utils import PositionalEncoding

# Model
encoder = transformer_encoder.TransformerEncoder(d_model=512, d_ff=2048, n_heads=8, n_layers=6, dropout=0.1)

# Input embeds
input_embeds = torch.nn.Embedding(num_embeddings=6, embedding_dim=512)
pe_embeds = PositionalEncoding(d_model=512, dropout=0.1, max_len=5)
encoder_input = torch.nn.Sequential(input_embeds, pe_embeds)

# Input data (zero-padding)
batch_seqs = torch.tensor([[1,2,3,4,5], [1,2,3,0,0]], dtype=torch.long)
mask = batch_seqs.ne(0)

# Run model
out = encoder(encoder_input(batch_seqs), mask)
```

Using the built-in warming up optimizer 
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

## Install from PyPI
Requires `python 3.5+`, `pytorch 1.0.0+`
```
pip install transformer_encoder
```

## API

**transformer_encoder.TransformerEncoder(d_model, d_ff, n_heads=1, n_layers=1, dropout=0.1)**

- `d_model`: dimension of each word vector
- `d_ff`: hidden dimension of feed forward layer
- `n_heads`: number of heads in self-attention (defaults to 1)
- `n_layers`: number of stacked layers of encoder (defaults to 1)
- `dropout`: dropout rate (defaults to 0.1)

**transformer_encoder.TransformerEncoder.forward(x, mask)**

- `x (~torch.FloatTensor)`: shape *(batch_size, max_seq_len, d_model)*
- `mask (~torch.ByteTensor)`: shape *(batch_size, max_seq_len)*

**transformer_encoder.utils.PositionalEncoding(d_model, dropout=0.1, max_len=5000)**

- `d_model`: same as TransformerEncoder
- `dropout`: dropout rate (defaults to 0.1)
- `max_len`: max sequence length (defaults to 5000)

**transformer_encoder.utils.PositionalEncoding.forward(x)**

- `x (~torch.FloatTensor)`: shape *(batch_size, max_seq_len, d_model)*

**transformer_encoder.utils.WarmupOptimizer(base_optimizer, d_model, scale_factor, warmup_steps)**

- `base_optimizer (~torch.optim.Optimzier)`: e.g. adam optimzier
- `d_model`: equals d_model in TransformerEncoder
- `scale_factor`: scale factor of learning rate
- `warmup_steps`: warming up steps 


## Contribution
Any contributions are welcome!