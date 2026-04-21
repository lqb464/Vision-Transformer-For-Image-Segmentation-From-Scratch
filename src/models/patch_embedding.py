"""
Custom Patch Embedding for Vision Transformer.

Converts image (C, H, W) into sequence of patch tokens:
    Image (3, 224, 224) → patches (196, 768) if patch_size=16

Formula:
    1. Divide image into grid of patches (non-overlapping)
    2. Each patch = patch_size × patch_size × channels
    3. Flatten each patch into vector
    4. Project through Linear layer → d_model
"""

import torch
import torch.nn as nn
import math
from src.models.layers import Conv2d


class PatchEmbedding(nn.Module):
    """
    Converts image into sequence of patch embeddings.
    
    Uses Conv2d with kernel_size=patch_size, stride=patch_size
    to cut and project patches in one step.
    
    Args:
        in_channels: Number of image channels (3 for RGB)
        patch_size: Size of each patch (e.g., 16)
        d_model: Output embedding size
        image_size: Image size (e.g., 224)
    """

    def __init__(self, in_channels=3, patch_size=16, d_model=256, image_size=128):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.image_size = image_size

        # Number of patches = (H / patch_size) * (W / patch_size)
        self.num_patches = (image_size // patch_size) ** 2

        # Use Conv2d to cut + project patches in one step
        # kernel_size = stride = patch_size → non-overlapping patches
        self.projection = Conv2d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )

    def forward(self, x):
        """
        Args:
            x: (batch, C, H, W) — ảnh input
        Returns:
            patches: (batch, num_patches, d_model)
        """
        batch_size = x.shape[0]

        # Conv2d: (batch, C, H, W) → (batch, d_model, H/P, W/P)
        x = self.projection(x)

        # Reshape: (batch, d_model, H/P, W/P) → (batch, d_model, num_patches) → (batch, num_patches, d_model)
        x = x.flatten(2)  # (batch, d_model, num_patches)
        x = x.transpose(1, 2)  # (batch, num_patches, d_model)

        return x
