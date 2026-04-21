"""
Segmentation Head — Decoder for image segmentation.

Converts patch tokens from ViT Encoder into segmentation mask.

Architecture:
    1. Reshape patch tokens → 2D feature map
    2. Upsample (progressive) using ConvTranspose2d + Conv2d
    3. Output: pixel-wise class predictions
"""

import torch
import torch.nn as nn
import math
from src.models.layers import Linear, Conv2d, ConvTranspose2d, LayerNorm
from src.models.activations import relu


class SegmentationHead(nn.Module):
    """
    Decoder head: converts ViT encoder output into segmentation mask.
    
    Args:
        d_model: Embedding dimension from ViT
        num_classes: Number of segmentation classes
        patch_size: Patch size (for reshaping)
        image_size: Output image size
    """

    def __init__(self, d_model=256, num_classes=3, patch_size=16, image_size=128):
        super().__init__()

        self.d_model = d_model
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.image_size = image_size
        self.grid_size = image_size // patch_size

        # Project d_model → intermediate channels
        self.project = Linear(d_model, d_model)

        # Progressive upsampling
        # grid_size → grid_size*2 → grid_size*4 → ... → image_size
        upsample_layers = []
        current_size = self.grid_size
        in_ch = d_model

        while current_size < image_size:
            out_ch = max(in_ch // 2, num_classes * 4)
            upsample_layers.append(ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))
            in_ch = out_ch
            current_size *= 2

        self.upsample = nn.ModuleList(upsample_layers)

        # Final conv: channels → num_classes
        self.final_conv = Conv2d(in_ch, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        Args:
            x: (batch, num_patches, d_model) — ViT encoder output
        Returns:
            logits: (batch, num_classes, H, W)
        """
        B, N, D = x.shape

        # Project
        x = self.project(x)

        # Reshape: (batch, num_patches, d_model) → (batch, d_model, grid_h, grid_w)
        x = x.transpose(1, 2)  # (B, D, N)
        x = x.view(B, D, self.grid_size, self.grid_size)

        # Progressive upsample
        for up_layer in self.upsample:
            x = relu(up_layer(x))

        # Final: (batch, channels, H, W) → (batch, num_classes, H, W)
        logits = self.final_conv(x)

        # Ensure output size matches
        if logits.shape[2] != self.image_size or logits.shape[3] != self.image_size:
            logits = torch.nn.functional.interpolate(
                logits, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False
            )

        return logits


class ViTSegmentation(nn.Module):
    """
    Full ViT Segmentation Model = ViT Encoder + Segmentation Head.
    """

    def __init__(self, in_channels=3, patch_size=16, d_model=256,
                 num_heads=8, num_layers=6, d_ff=None,
                 image_size=128, num_classes=3, dropout=0.1):
        super().__init__()

        from src.models.vit_encoder import ViTEncoder

        self.encoder = ViTEncoder(
            in_channels=in_channels,
            patch_size=patch_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            image_size=image_size,
            dropout=dropout,
        )

        self.decoder = SegmentationHead(
            d_model=d_model,
            num_classes=num_classes,
            patch_size=patch_size,
            image_size=image_size,
        )

    def forward(self, x):
        """
        Args:
            x: (batch, C, H, W)
        Returns:
            logits: (batch, num_classes, H, W)
            attn_weights: list
        """
        encoder_output, attn_weights = self.encoder(x)
        logits = self.decoder(encoder_output)
        return logits, attn_weights

    def predict(self, x):
        """Return predicted mask."""
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x)
            return torch.argmax(logits, dim=1)


def build_vit_segmentation(in_channels=3, patch_size=16, d_model=256,
                            num_heads=8, num_layers=6, image_size=128,
                            num_classes=3, dropout=0.1):
    """Hàm tiện ích tạo ViT Segmentation model."""
    return ViTSegmentation(
        in_channels=in_channels, patch_size=patch_size, d_model=d_model,
        num_heads=num_heads, num_layers=num_layers, image_size=image_size,
        num_classes=num_classes, dropout=dropout,
    )
