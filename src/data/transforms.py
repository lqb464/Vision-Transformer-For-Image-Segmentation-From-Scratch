"""
Transforms for image segmentation.
Need to transform image AND mask simultaneously.
"""

import torch
import numpy as np
import random


class ToTensor:
    """Convert image and mask to tensor."""

    def __call__(self, image, mask):
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = image[:, :, np.newaxis]
            image = torch.from_numpy(image.transpose(2, 0, 1).copy()).float() / 255.0
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask.copy()).long()
        return image, mask


class Resize:
    """Resize image và mask về cùng kích thước."""

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, image, mask):
        import torch.nn.functional as F

        # Image: (C, H, W) → resize
        if isinstance(image, torch.Tensor):
            image = F.interpolate(
                image.unsqueeze(0), size=self.size, mode='bilinear', align_corners=False
            ).squeeze(0)
        # Mask: (H, W) → resize (nearest neighbor để giữ label nguyên)
        if isinstance(mask, torch.Tensor):
            mask = F.interpolate(
                mask.float().unsqueeze(0).unsqueeze(0), size=self.size, mode='nearest'
            ).squeeze(0).squeeze(0).long()

        return image, mask


class Normalize:
    """Normalize image theo mean/std."""

    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).float().view(-1, 1, 1)
        self.std = torch.tensor(std).float().view(-1, 1, 1)

    def __call__(self, image, mask):
        image = (image - self.mean) / self.std
        return image, mask


class RandomHorizontalFlip:
    """Lật ngang cả image và mask."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = image.flip(-1)
            mask = mask.flip(-1)
        return image, mask


class Compose:
    """Ghép nhiều transforms."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask
