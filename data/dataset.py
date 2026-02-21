"""
Training dataset: loads distorted/corrected image pairs from the Kaggle dataset.

Expected directory layout (each subfolder is one sample):
    TRAIN_DIR/
        {uuid}_g0/
            original.jpg    <- distorted input
            generated.jpg   <- corrected ground truth
"""

import os
from pathlib import Path
from typing import Optional

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import config


def build_photometric_aug() -> A.Compose:
    """Photometric-only augmentations. Never geometric — that corrupts the
    distortion ↔ correction correspondence."""
    return A.Compose(
        [
            A.RandomBrightnessContrast(
                brightness_limit=config.AUG_BRIGHTNESS_LIMIT,
                contrast_limit=config.AUG_CONTRAST_LIMIT,
                p=0.5,
            ),
            A.ImageCompression(
                quality_lower=config.AUG_JPEG_QUALITY_RANGE[0],
                quality_upper=config.AUG_JPEG_QUALITY_RANGE[1],
                p=0.3,
            ),
            A.HueSaturationValue(
                hue_shift_limit=config.AUG_HUE_SHIFT,
                sat_shift_limit=config.AUG_SAT_SHIFT,
                val_shift_limit=0,
                p=0.3,
            ),
            A.GaussNoise(
                var_limit=config.AUG_NOISE_VAR_LIMIT,
                p=0.2,
            ),
        ],
        additional_targets={"target": "image"},  # apply same photometric aug to both
    )


class LensTrainDataset(Dataset):
    """Loads (distorted, corrected) image pairs and resizes to `resolution`."""

    def __init__(
        self,
        root: Path = config.TRAIN_DIR,
        resolution: int = 512,
        split: str = "train",
        val_fraction: float = config.VAL_SPLIT,
        seed: int = config.SEED,
        augment: bool = True,
    ):
        self.resolution = resolution
        self.augment = augment and split == "train"
        self.aug = build_photometric_aug() if self.augment else None

        # Collect sample directories
        all_dirs = sorted(
            [d for d in Path(root).iterdir() if d.is_dir()],
            key=lambda p: p.name,
        )

        # Deterministic train/val split
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(all_dirs))
        n_val = int(len(all_dirs) * val_fraction)

        if split == "val":
            selected = indices[:n_val]
        else:
            selected = indices[n_val:]

        self.sample_dirs = [all_dirs[i] for i in selected]

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, idx: int):
        d = self.sample_dirs[idx]
        orig_path = d / "original.jpg"
        gen_path = d / "generated.jpg"

        # Read as RGB
        distorted = cv2.imread(str(orig_path))
        corrected = cv2.imread(str(gen_path))
        distorted = cv2.cvtColor(distorted, cv2.COLOR_BGR2RGB)
        corrected = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)

        # Resize
        distorted = cv2.resize(distorted, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA)
        corrected = cv2.resize(corrected, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA)

        # Photometric augmentation (same transform applied to both)
        if self.aug is not None:
            augmented = self.aug(image=distorted, target=corrected)
            distorted = augmented["image"]
            corrected = augmented["target"]

        # To tensor: HWC uint8 → CHW float32 [0, 1]
        distorted = torch.from_numpy(distorted).permute(2, 0, 1).float() / 255.0
        corrected = torch.from_numpy(corrected).permute(2, 0, 1).float() / 255.0

        return distorted, corrected
