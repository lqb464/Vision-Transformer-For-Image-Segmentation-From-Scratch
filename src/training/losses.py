"""Loss functions for Segmentation: CrossEntropy + Dice Loss."""

import torch


class CrossEntropyLoss:
    """CrossEntropy cho pixel-wise classification."""

    def __init__(self, ignore_index=-100):
        self.ignore_index = ignore_index

    def __call__(self, logits, targets):
        """
        Args:
            logits: (batch, num_classes, H, W)
            targets: (batch, H, W) — class indices
        """
        B, C, H, W = logits.shape
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)
        targets_flat = targets.reshape(-1)

        mask = (targets_flat != self.ignore_index)
        valid_count = mask.sum().item()

        if valid_count == 0:
            return torch.tensor(0.0, requires_grad=True, device=logits.device)

        max_val = logits_flat.max(dim=1, keepdim=True)[0]
        shifted = logits_flat - max_val
        log_sum_exp = torch.log(torch.exp(shifted).sum(dim=1))
        log_softmax = shifted - log_sum_exp.unsqueeze(1)

        nll = 0.0
        for i in range(logits_flat.size(0)):
            if mask[i]:
                nll = nll + (-log_softmax[i, targets_flat[i].item()])

        return nll / valid_count


class DiceLoss:
    """
    Dice Loss cho segmentation.
    
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    DiceLoss = 1 - Dice
    """

    def __init__(self, smooth=1.0):
        self.smooth = smooth

    def __call__(self, logits, targets):
        """
        Args:
            logits: (batch, num_classes, H, W)
            targets: (batch, H, W)
        """
        num_classes = logits.shape[1]

        # Softmax
        max_val = logits.max(dim=1, keepdim=True)[0]
        shifted = logits - max_val
        probs = torch.exp(shifted) / torch.exp(shifted).sum(dim=1, keepdim=True)

        # One-hot encode targets: (B, H, W) → (B, C, H, W)
        targets_onehot = torch.zeros_like(logits)
        targets_onehot.scatter_(1, targets.unsqueeze(1), 1.0)

        # Dice per class
        dice_loss = 0.0
        for c in range(num_classes):
            pred_c = probs[:, c].reshape(-1)
            target_c = targets_onehot[:, c].reshape(-1)

            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()

            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_loss += (1.0 - dice)

        return dice_loss / num_classes


class CombinedLoss:
    """CrossEntropy + Dice Loss."""

    def __init__(self, ce_weight=0.5, dice_weight=0.5):
        self.ce_loss = CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def __call__(self, logits, targets):
        return self.ce_weight * self.ce_loss(logits, targets) + \
               self.dice_weight * self.dice_loss(logits, targets)


def get_loss_function(loss_type='combined'):
    if loss_type == 'ce':
        return CrossEntropyLoss()
    elif loss_type == 'dice':
        return DiceLoss()
    else:
        return CombinedLoss()
