"""
Script for training ViT Segmentation.

How to run:
    python scripts/train.py                          # Synthetic data (fast)
    python scripts/train.py --dataset pets            # Oxford-IIIT Pets
    python scripts/train.py --epochs 30 --d_model 256
"""

import os
import argparse
import random
import torch

from src.data.transforms import ToTensor, Resize, Normalize, RandomHorizontalFlip, Compose
from src.data.dataset import SegmentationDataset, load_oxford_pets, create_synthetic_data
from src.data.dataloader import get_dataloader

from src.models.segmentation_head import build_vit_segmentation
from src.training.losses import get_loss_function
from src.training.optimizers import get_optimizer, CosineScheduler
from src.training.trainer import Trainer


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train ViT Segmentation")
    parser.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "pets"])
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_classes", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  🔲 VISION TRANSFORMER FOR IMAGE SEGMENTATION")
    print("=" * 60)

    # ===== LOAD DATA =====
    print(f"\n=== STEP 1: LOAD DATA ({args.dataset}) ===")

    if args.dataset == "pets":
        train_imgs, train_masks, test_imgs, test_masks, class_names = load_oxford_pets(
            "data", image_size=args.image_size
        )
    else:
        train_imgs, train_masks, test_imgs, test_masks, class_names = create_synthetic_data(
            num_train=300, num_test=50, image_size=args.image_size, num_classes=args.num_classes
        )
        print(f"[+] Synthetic data: {len(train_imgs)} train, {len(test_imgs)} test")

    # Transforms
    train_transform = Compose([
        ToTensor(),
        RandomHorizontalFlip(p=0.5),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    test_transform = Compose([
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_dataset = SegmentationDataset(train_imgs, train_masks, transform=train_transform, num_classes=args.num_classes)
    test_dataset = SegmentationDataset(test_imgs, test_masks, transform=test_transform, num_classes=args.num_classes)

    train_loader = get_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = get_dataloader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"[+] Classes: {class_names}")
    print(f"[+] Image size: {args.image_size}x{args.image_size}")

    # ===== BUILD MODEL =====
    print(f"\n=== STEP 2: INITIALIZE ViT SEGMENTATION ===")

    model = build_vit_segmentation(
        in_channels=3,
        patch_size=args.patch_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        image_size=args.image_size,
        num_classes=args.num_classes,
        dropout=0.1,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"[+] Parameters: {num_params:,}")
    print(f"[+] patch={args.patch_size}, d_model={args.d_model}, heads={args.num_heads}, layers={args.num_layers}")

    optimizer = get_optimizer(model, lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(train_loader)
    scheduler = CosineScheduler(optimizer, total_steps=total_steps, warmup_steps=total_steps // 10)
    criterion = get_loss_function('combined')

    # ===== TRAIN =====
    trainer = Trainer(
        model=model, optimizer=optimizer, criterion=criterion,
        train_loader=train_loader, val_loader=test_loader,
        scheduler=scheduler, num_classes=args.num_classes,
        class_names=class_names, epochs=args.epochs,
        device=device, save_dir="checkpoints", log_every=20,
    )

    trainer.train()

    print(f"\n[✓] Checkpoint saved in: checkpoints/")


if __name__ == "__main__":
    main()
