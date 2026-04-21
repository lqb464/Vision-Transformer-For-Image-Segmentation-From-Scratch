"""
Script for evaluating and visualizing segmentation results.

How to run:
    python scripts/test.py
"""

import os
import argparse
import random
import torch

from src.data.transforms import ToTensor, Normalize, Compose
from src.data.dataset import SegmentationDataset, create_synthetic_data
from src.data.dataloader import get_dataloader

from src.models.segmentation_head import build_vit_segmentation
from src.training.evaluate import evaluate_segmentation, print_segmentation_metrics
from src.training.visualize import plot_training_history, plot_segmentation_results


def main():
    random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  📊 EVALUATE ViT SEGMENTATION")
    print("=" * 60)

    # Load test data
    image_size = 64
    num_classes = 3
    _, _, test_imgs, test_masks, class_names = create_synthetic_data(
        num_train=10, num_test=50, image_size=image_size, num_classes=num_classes
    )

    test_transform = Compose([
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    test_dataset = SegmentationDataset(test_imgs, test_masks, transform=test_transform)
    test_loader = get_dataloader(test_dataset, batch_size=8, shuffle=False)

    # Load model
    model = build_vit_segmentation(
        patch_size=8, d_model=128, num_heads=4, num_layers=4,
        image_size=image_size, num_classes=num_classes,
    ).to(device)

    ckpt_path = "checkpoints/best_model_weights.pt"
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"[+] Loaded: {ckpt_path}")
    else:
        print(f"[!] Not found: {ckpt_path}")
        return

    # Evaluate
    metrics = evaluate_segmentation(model, test_loader, num_classes, device)
    print_segmentation_metrics(metrics, class_names)

    # Visualize
    os.makedirs("outputs", exist_ok=True)
    plot_training_history(save_path="outputs/training_history.png", show=False)

    # Predict samples
    model.eval()
    images, masks = next(iter(test_loader))
    images_gpu = images.to(device)
    with torch.no_grad():
        preds = model.predict(images_gpu)

    plot_segmentation_results(
        images[:4], masks[:4], preds[:4].cpu(),
        class_names=class_names, num_samples=4,
        save_path="outputs/segmentation_results.png", show=False,
    )

    print(f"\n[✓] Results saved in: outputs/")


if __name__ == "__main__":
    main()
