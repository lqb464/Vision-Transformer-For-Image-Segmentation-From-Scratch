"""Visualization for segmentation: masks overlay, training curves."""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_training_history(history_path="checkpoints/train_history.json",
                          save_path=None, show=True):
    if not os.path.exists(history_path):
        return None

    with open(history_path) as f:
        history = json.load(f)

    epochs = [h["epoch"] for h in history]
    losses = [h["loss"] for h in history]
    mious = [h.get("miou") for h in history]
    has_miou = any(m is not None for m in mious)

    sns.set_theme(style="whitegrid")

    if has_miou:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(epochs, losses, 'o-', color='#e74c3c', linewidth=2)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('📉 Training Loss', fontweight='bold')

    if has_miou:
        valid_e = [e for e, m in zip(epochs, mious) if m is not None]
        valid_m = [m for m in mious if m is not None]
        ax2.plot(valid_e, valid_m, 's-', color='#2ecc71', linewidth=2)
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('mIoU')
        ax2.set_title('📈 Mean IoU', fontweight='bold')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    return fig


def plot_segmentation_results(images, true_masks, pred_masks, class_names=None,
                               num_samples=4, save_path=None, show=True):
    """Hiển thị ảnh gốc, mask thật, mask dự đoán."""
    import torch

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    cmap = plt.cm.get_cmap('tab10', max(3, len(class_names) if class_names else 3))

    for i in range(min(num_samples, len(images))):
        img = images[i]
        if isinstance(img, torch.Tensor):
            img = img.cpu()
            if img.shape[0] == 3:
                img = img.permute(1, 2, 0)
                img = img * 0.25 + 0.5
                img = img.clamp(0, 1)
            img = img.numpy()

        true_m = true_masks[i]
        if isinstance(true_m, torch.Tensor):
            true_m = true_m.cpu().numpy()

        pred_m = pred_masks[i]
        if isinstance(pred_m, torch.Tensor):
            pred_m = pred_m.cpu().numpy()

        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Input Image')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(true_m, cmap=cmap, vmin=0, vmax=cmap.N - 1)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(pred_m, cmap=cmap, vmin=0, vmax=cmap.N - 1)
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')

    fig.suptitle('🔍 Segmentation Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    return fig
