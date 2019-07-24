import torch
import torch.nn as nn

from .encoder_layer import EncoderLayer
from .feed_forward import FeedForward
from .layer_norm import LayerNorm
from .multi_head_attention import MultiHeadedAttention
from .utils import clones


class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(
        self, x: torch.FloatTensor, mask: torch.ByteTensor
    ) -> torch.FloatTensor:
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class TFEncoder(nn.Module):
    """The encoder of transformer

    Args:
        `n_layers`: number of stacked encoder layers
        `d_model`: model dimension
        `d_ff`: hidden dimension of feed forward layer
        `n_heads`: number of heads of self-attention
        `dropout`: dropout rate, default 0.1
    """

    def __init__(
        self, n_layers: int, d_model: int, d_ff: int, n_heads: int, dropout: float = 0.1
    ) -> None:
        super(TFEncoder, self).__init__()
        self.encoder = Encoder(
            EncoderLayer(
                d_model,
                MultiHeadedAttention(n_heads, d_model, dropout),
                FeedForward(d_model, d_ff, dropout),
                dropout,
            ),
            n_layers,
        )
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)

    def forward(
        self, x: torch.FloatTensor, mask: torch.ByteTensor
    ) -> torch.FloatTensor:
        return self.encoder(x, mask)
