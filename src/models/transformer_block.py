"""ViT Transformer Block."""

import torch.nn as nn
from src.models.attention import MultiHeadSelfAttention
from src.models.feed_forward import FeedForward
from src.models.layers import LayerNorm, Dropout


class TransformerBlock(nn.Module):
    """Pre-LN Transformer Block (ViT style)."""

    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1):
        super().__init__()
        self.norm1 = LayerNorm(d_model)
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout = Dropout(dropout)

    def forward(self, x):
        # Pre-LN: LayerNorm → Sublayer → Add
        attn_out, attn_weights = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x, attn_weights
