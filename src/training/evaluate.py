"""Evaluate Segmentation: mIoU, Dice coefficient."""

import torch
import numpy as np


def compute_iou(pred_mask, true_mask, num_classes):
    """
    Compute IoU (Intersection over Union) for each class.
    
    IoU = TP / (TP + FP + FN)
    """
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred_mask == cls)
        true_cls = (true_mask == cls)

        intersection = (pred_cls & true_cls).sum().item()
        union = (pred_cls | true_cls).sum().item()

        if union == 0:
            iou = float('nan')
        else:
            iou = intersection / union

        ious.append(iou)

    return ious


def compute_dice(pred_mask, true_mask, num_classes, smooth=1.0):
    """Tính Dice coefficient cho mỗi class."""
    dices = []
    for cls in range(num_classes):
        pred_cls = (pred_mask == cls).float()
        true_cls = (true_mask == cls).float()

        intersection = (pred_cls * true_cls).sum().item()
        total = pred_cls.sum().item() + true_cls.sum().item()

        dice = (2.0 * intersection + smooth) / (total + smooth)
        dices.append(dice)

    return dices


def evaluate_segmentation(model, dataloader, num_classes, device='cpu'):
    """
    Evaluate segmentation model on dataset.
    
    Returns:
        metrics: dict với mIoU, mDice, per-class metrics
    """
    model.eval()
    all_ious = [[] for _ in range(num_classes)]
    all_dices = [[] for _ in range(num_classes)]
    pixel_correct = 0
    pixel_total = 0

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            logits, _ = model(images)
            preds = torch.argmax(logits, dim=1)

            # Pixel accuracy
            pixel_correct += (preds == masks).sum().item()
            pixel_total += masks.numel()

            for i in range(images.size(0)):
                ious = compute_iou(preds[i].cpu(), masks[i].cpu(), num_classes)
                dices = compute_dice(preds[i].cpu(), masks[i].cpu(), num_classes)

                for cls in range(num_classes):
                    if not np.isnan(ious[cls]):
                        all_ious[cls].append(ious[cls])
                    all_dices[cls].append(dices[cls])

    # Average
    per_class_iou = [np.mean(ious) if ious else 0.0 for ious in all_ious]
    per_class_dice = [np.mean(dices) if dices else 0.0 for dices in all_dices]

    metrics = {
        "pixel_accuracy": pixel_correct / max(pixel_total, 1),
        "mIoU": np.mean(per_class_iou),
        "mDice": np.mean(per_class_dice),
        "per_class_iou": per_class_iou,
        "per_class_dice": per_class_dice,
    }

    return metrics


def print_segmentation_metrics(metrics, class_names=None):
    """In metrics đẹp."""
    print(f"\nPixel Accuracy: {metrics['pixel_accuracy']:.4f}")
    print(f"mIoU: {metrics['mIoU']:.4f}")
    print(f"mDice: {metrics['mDice']:.4f}")

    if class_names:
        print(f"\n{'Class':<15} {'IoU':>10} {'Dice':>10}")
        print("-" * 40)
        for i, name in enumerate(class_names):
            iou = metrics['per_class_iou'][i] if i < len(metrics['per_class_iou']) else 0
            dice = metrics['per_class_dice'][i] if i < len(metrics['per_class_dice']) else 0
            print(f"{name:<15} {iou:>10.4f} {dice:>10.4f}")
