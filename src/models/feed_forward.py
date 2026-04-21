"""Feed-Forward Network for ViT."""

import torch.nn as nn
from src.models.layers import Linear, Dropout
from src.models.activations import gelu


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.dropout = Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(gelu(self.linear1(x))))
