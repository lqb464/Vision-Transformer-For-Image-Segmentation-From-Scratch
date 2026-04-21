# Vision Transformer for Image Segmentation — From Scratch

This project builds a **Vision Transformer (ViT)** completely from scratch for the **Image Segmentation** task — pixel-by-pixel image partitioning.

## Structure

```
├── data/                          # Segmentation data
├── scripts/
│   ├── train.py                   # Train ViT Segmentation
│   └── test.py                    # Evaluate + Visualize
├── src/
│   ├── data/
│   │   ├── transforms.py          # Image+Mask transforms (synchronized)
│   │   ├── dataset.py             # SegmentationDataset + Oxford Pets + Synthetic
│   │   └── dataloader.py          # DataLoader
│   ├── models/
│   │   ├── activations.py         # ReLU, GELU, Softmax
│   │   ├── layers.py              # Linear, LayerNorm, Conv2d, ConvTranspose2d
│   │   ├── patch_embedding.py     # Image → Patch Tokens (Conv stride)
│   │   ├── attention.py           # Multi-Head Self-Attention
│   │   ├── feed_forward.py        # MLP block
│   │   ├── transformer_block.py   # Pre-LN ViT Block
│   │   ├── vit_encoder.py         # ViT Encoder (patches + pos embed + blocks)
│   │   └── segmentation_head.py   # Decoder + ViTSegmentation model
│   └── training/
│       ├── losses.py              # CrossEntropy + Dice + Combined
│       ├── optimizers.py          # AdamW + Cosine Scheduler
│       ├── trainer.py             # Training loop + checkpoint
│       ├── evaluate.py            # mIoU, Dice, Pixel Accuracy
│       └── visualize.py           # Training curves, mask overlays
├── checkpoints/
├── requirements.txt
└── README.md
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Training

```bash
# Synthetic data (fast, test pipeline)
python scripts/train.py --dataset synthetic --epochs 20

# Oxford-IIIT Pets (real data)
python scripts/train.py --dataset pets --image_size 128 --patch_size 16 --epochs 30
```

## Evaluation

```bash
python scripts/test.py
```

## ViT Segmentation Architecture

```
Image (3, H, W) → Patch Embedding (Conv stride=P) → Patch Tokens (N, D)
    → + Learnable Position Embeddings
    → [Pre-LN → Multi-Head Self-Attention → Add → Pre-LN → FFN → Add] × L
    → LayerNorm
    → Reshape → 2D Feature Map (D, H/P, W/P)
    → Progressive Upsampling (ConvTranspose2d × log₂(P))
    → Conv 1×1 → Segmentation Mask (C, H, W)
```

## Custom Components

| Component | Description |
|-----------|-------------|
| Patch Embedding | Divide image into patches using Conv stride |
| Learnable Position Embedding | Trainable position embedding |
| Multi-Head Self-Attention | Scaled dot-product attention |
| Pre-LN Transformer Block | LayerNorm before sublayer (ViT style) |
| Segmentation Decoder | Progressive upsampling + final conv |
| CrossEntropy + Dice Loss | Combined loss for segmentation |
| AdamW | Adam + decoupled weight decay |
| Cosine Scheduler | Cosine annealing + warmup |
| mIoU, Dice | Per-class metrics |
