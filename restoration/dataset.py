"""Paired LR/HR dataset. Images are 256x256 RGB, LR2/ and HR2/ share filenames."""

import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def to_tensor(img_hwc_uint8):
    """HWC uint8 -> CHW float in 0..1 (same scaling the NPU calibration uses)."""
    return torch.from_numpy(np.ascontiguousarray(img_hwc_uint8)).permute(2, 0, 1).float() / 255.0


class PairedDataset(Dataset):
    def __init__(self, root, crop=None, augment=False, cache=True):
        self.lr_dir = os.path.join(root, "LR2")
        self.hr_dir = os.path.join(root, "HR2")
        self.names = sorted(f for f in os.listdir(self.hr_dir) if not f.startswith("."))
        self.crop = crop
        self.augment = augment
        # ~0.9 GB for the full training set; decoding JPEGs every epoch is the bottleneck otherwise.
        self.cache = [self._load(i) for i in range(len(self.names))] if cache else None

    def _load(self, i):
        n = self.names[i]
        return load_rgb(os.path.join(self.lr_dir, n)), load_rgb(os.path.join(self.hr_dir, n))

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        lr, hr = self.cache[i] if self.cache is not None else self._load(i)

        if self.crop and self.crop < lr.shape[0]:
            y = random.randint(0, lr.shape[0] - self.crop)
            x = random.randint(0, lr.shape[1] - self.crop)
            lr = lr[y:y + self.crop, x:x + self.crop]
            hr = hr[y:y + self.crop, x:x + self.crop]

        if self.augment:
            # Same random flip / rotation for LR and HR (8 dihedral variants).
            if random.random() < 0.5:
                lr, hr = lr[:, ::-1], hr[:, ::-1]
            if random.random() < 0.5:
                lr, hr = lr[::-1], hr[::-1]
            k = random.randint(0, 3)
            lr, hr = np.rot90(lr, k), np.rot90(hr, k)

        return to_tensor(lr), to_tensor(hr)
