"""Training loop for ViT Segmentation."""

import os
import json
import torch
from src.training.evaluate import evaluate_segmentation, print_segmentation_metrics


def clip_grad_norm(parameters, max_norm):
    params = list(parameters)
    total_norm_sq = sum((p.grad ** 2).sum().item() for p in params if p.grad is not None)
    total_norm = total_norm_sq ** 0.5
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for p in params:
            if p.grad is not None:
                p.grad.mul_(scale)
    return total_norm


class Trainer:
    def __init__(self, model, optimizer, criterion, train_loader,
                 val_loader=None, scheduler=None, num_classes=3,
                 class_names=None, epochs=20, device='cpu',
                 save_dir='checkpoints', log_every=50):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.num_classes = num_classes
        self.class_names = class_names
        self.epochs = epochs
        self.device = device
        self.save_dir = save_dir
        self.log_every = log_every

        self.history = []
        self.best_miou = 0.0
        os.makedirs(save_dir, exist_ok=True)

    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0.0

        for batch_idx, (images, masks) in enumerate(self.train_loader):
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()
            logits, _ = self.model(images)
            loss = self.criterion(logits, masks)
            loss.backward()

            clip_grad_norm(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

            epoch_loss += loss.item()

            if self.log_every and (batch_idx + 1) % self.log_every == 0:
                print(f"  Epoch {epoch:02d} | Batch {batch_idx + 1}/{len(self.train_loader)} | Loss: {loss.item():.4f}")

        return epoch_loss / max(len(self.train_loader), 1)

    def save_checkpoint(self, epoch, loss, miou=None, is_best=False):
        ckpt = {
            "epoch": epoch, "loss": float(loss),
            "miou": float(miou) if miou else None,
            "best_miou": float(self.best_miou),
            "model_state_dict": self.model.state_dict(),
            "history": self.history,
        }
        torch.save(ckpt, os.path.join(self.save_dir, "last_checkpoint.pt"))
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, "last_model_weights.pt"))
        if is_best:
            torch.save(ckpt, os.path.join(self.save_dir, "best_checkpoint.pt"))
            torch.save(self.model.state_dict(), os.path.join(self.save_dir, "best_model_weights.pt"))

        with open(os.path.join(self.save_dir, "train_history.json"), "w") as f:
            json.dump(self.history, f, indent=2)

    def train(self, start_epoch=1):
        print("\n=== START TRAINING ViT SEGMENTATION ===")

        for epoch in range(start_epoch, self.epochs + 1):
            avg_loss = self.train_epoch(epoch)

            miou = None
            if self.val_loader is not None:
                metrics = evaluate_segmentation(
                    self.model, self.val_loader, self.num_classes, self.device
                )
                miou = metrics["mIoU"]
                if epoch % 5 == 0 or epoch == self.epochs:
                    print_segmentation_metrics(metrics, self.class_names)

            is_best = miou is not None and miou > self.best_miou
            if is_best:
                self.best_miou = miou

            log = {"epoch": epoch, "loss": float(avg_loss), "miou": miou, "is_best": bool(is_best)}
            self.history.append(log)
            self.save_checkpoint(epoch, avg_loss, miou, is_best)

            miou_str = f" | mIoU: {miou:.4f}" if miou is not None else ""
            best_str = " ★" if is_best else ""
            print(f"Epoch {epoch:02d}/{self.epochs} | Loss: {avg_loss:.4f}{miou_str}{best_str}")

        print(f"\n[✓] COMPLETED! Best mIoU: {self.best_miou:.4f}")
        return self.history
