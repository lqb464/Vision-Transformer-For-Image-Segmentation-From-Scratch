"""
ViT Encoder: Patch Embedding → Position Embedding → Transformer Blocks.
"""

import torch
import torch.nn as nn
from src.models.patch_embedding import PatchEmbedding
from src.models.transformer_block import TransformerBlock
from src.models.layers import LayerNorm, Dropout


class ViTEncoder(nn.Module):
    """
    Vision Transformer Encoder.
    
    Args:
        in_channels: Image channels (3 for RGB)
        patch_size: Patch size
        d_model: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of Transformer blocks
        d_ff: FFN hidden dimension
        image_size: Input image size
        dropout: Dropout rate
    """

    def __init__(self, in_channels=3, patch_size=16, d_model=256,
                 num_heads=8, num_layers=6, d_ff=None,
                 image_size=128, dropout=0.1):
        super().__init__()

        self.patch_size = patch_size
        self.d_model = d_model
        self.num_patches = (image_size // patch_size) ** 2

        # 1. Patch Embedding
        self.patch_embedding = PatchEmbedding(
            in_channels=in_channels,
            patch_size=patch_size,
            d_model=d_model,
            image_size=image_size,
        )

        # 2. Learnable Position Embeddings (not using sinusoidal for ViT)
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_patches, d_model) * 0.02
        )

        # 3. Dropout
        self.dropout = Dropout(dropout)

        # 4. Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # 5. Final LayerNorm
        self.norm = LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: (batch, C, H, W)
        Returns:
            output: (batch, num_patches, d_model)
            all_attn: list of attention weights
        """
        # Patch embedding
        x = self.patch_embedding(x)  # (batch, num_patches, d_model)

        # Add position embedding
        x = x + self.position_embedding

        x = self.dropout(x)

        # Transformer blocks
        all_attn = []
        for block in self.blocks:
            x, attn = block(x)
            all_attn.append(attn)

        x = self.norm(x)

        return x, all_attn
