"""
Dataset for Image Segmentation.
Supports Oxford-IIIT Pet dataset (loaded via torchvision).
"""

import os
import torch
import numpy as np
from PIL import Image


class SegmentationDataset:
    """
    Dataset for image segmentation.
    Each sample consists of: (image, segmentation_mask)
    """

    def __init__(self, images, masks, transform=None, num_classes=3):
        """
        Args:
            images: list of images (numpy array or PIL Image)
            masks: list of masks (numpy array)
            transform: Compose transforms for both image+mask
            num_classes: Number of segmentation classes
        """
        self.images = images
        self.masks = masks
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        mask = self.masks[index]

        if isinstance(image, Image.Image):
            image = np.array(image)
        if isinstance(mask, Image.Image):
            mask = np.array(mask)

        if self.transform is not None:
            image, mask = self.transform(image, mask)
        else:
            if isinstance(image, np.ndarray):
                if image.ndim == 2:
                    image = image[:, :, np.newaxis]
                image = torch.from_numpy(image.transpose(2, 0, 1).copy()).float() / 255.0
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask.copy()).long()

        return image, mask


def load_oxford_pets(data_dir="data", image_size=128):
    """
    Load Oxford-IIIT Pet dataset using torchvision.
    3 classes: background(0), foreground/pet(1), border(2)
    
    Returns:
        train_images, train_masks, test_images, test_masks, class_names
    """
    import torchvision
    from PIL import Image

    # Download dataset
    trainval = torchvision.datasets.OxfordIIITPet(
        root=data_dir, split='trainval', target_types='segmentation', download=True
    )
    test = torchvision.datasets.OxfordIIITPet(
        root=data_dir, split='test', target_types='segmentation', download=True
    )

    def process_set(dataset, max_samples=None):
        images = []
        masks = []
        n = len(dataset) if max_samples is None else min(len(dataset), max_samples)
        for i in range(n):
            img, seg = dataset[i]

            # Resize
            img = img.resize((image_size, image_size), Image.BILINEAR)
            seg = seg.resize((image_size, image_size), Image.NEAREST)

            img_np = np.array(img)
            mask_np = np.array(seg)

            # Oxford Pet: 1=foreground, 2=background, 3=border
            # Convert to 0=background, 1=foreground, 2=border
            mask_np = mask_np - 1
            mask_np = np.clip(mask_np, 0, 2)

            images.append(img_np)
            masks.append(mask_np)

        return images, masks

    print("[*] Loading Oxford-IIIT Pet dataset...")
    train_images, train_masks = process_set(trainval)
    test_images, test_masks = process_set(test)

    class_names = ['background', 'pet', 'border']

    print(f"[+] Train: {len(train_images)} images")
    print(f"[+] Test:  {len(test_images)} images")

    return train_images, train_masks, test_images, test_masks, class_names


def create_synthetic_data(num_train=200, num_test=50, image_size=64, num_classes=3):
    """
    Create simple synthetic data to test pipeline.
    Each image contains 1 random circle/rectangle.
    """
    import random

    def generate_sample(size, n_classes):
        img = np.random.randint(50, 200, (size, size, 3), dtype=np.uint8)
        mask = np.zeros((size, size), dtype=np.int64)

        # Draw random shape
        shape_type = random.choice(['circle', 'rectangle'])
        cx, cy = random.randint(size // 4, 3 * size // 4), random.randint(size // 4, 3 * size // 4)
        r = random.randint(size // 8, size // 4)

        if shape_type == 'circle':
            for y in range(size):
                for x in range(size):
                    if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                        mask[y, x] = 1
                        img[y, x] = [255, 100, 100]
                    elif (x - cx) ** 2 + (y - cy) ** 2 <= (r + 3) ** 2:
                        mask[y, x] = 2
                        img[y, x] = [100, 255, 100]
        else:
            x1, y1 = max(0, cx - r), max(0, cy - r)
            x2, y2 = min(size, cx + r), min(size, cy + r)
            mask[y1:y2, x1:x2] = 1
            img[y1:y2, x1:x2] = [100, 100, 255]
            # Border
            for bw in range(2):
                if y1 - bw >= 0:
                    mask[y1 - bw, x1:x2] = 2
                if y2 + bw < size:
                    mask[y2 + bw, x1:x2] = 2

        return img, mask

    train_images = []
    train_masks = []
    for _ in range(num_train):
        img, m = generate_sample(image_size, num_classes)
        train_images.append(img)
        train_masks.append(m)

    test_images = []
    test_masks = []
    for _ in range(num_test):
        img, m = generate_sample(image_size, num_classes)
        test_images.append(img)
        test_masks.append(m)

    class_names = ['background', 'foreground', 'border']
    return train_images, train_masks, test_images, test_masks, class_names