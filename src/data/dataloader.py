"""DataLoader for Segmentation."""

import torch
import random


class SegmentationDataLoader:
    def __init__(self, dataset, batch_size=8, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start:start + self.batch_size]
            images, masks = [], []
            for idx in batch_idx:
                img, msk = self.dataset[idx]
                images.append(img)
                masks.append(msk)
            yield torch.stack(images), torch.stack(masks)


def get_dataloader(dataset, batch_size=8, shuffle=True):
    return SegmentationDataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
