"""
Training dataset: loads distorted/corrected image pairs from the Kaggle dataset.

Actual flat directory layout:
    TRAIN_DIR/
        {uuid}_{g}_original.jpg   <- distorted input
        {uuid}_{g}_generated.jpg  <- corrected ground truth
        ...
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


def apply_synthetic_distortion(
    image: np.ndarray,
    k1: float,
    k2: float,
    cx_offset: float = 0.0,
    cy_offset: float = 0.0,
) -> np.ndarray:
    """Apply Brown-Conrady radial distortion to a clean GT image.

    For each pixel (xd, yd) in the output (distorted) image, we compute the
    corresponding clean-image coordinate (xu, yu) via the first-order inverse
    of the forward distortion model:

        forward: xd = xu * (1 + k1*ru² + k2*ru⁴)
        inverse approx: xu ≈ xd / (1 + k1*rd² + k2*rd⁴)

    Then use cv2.remap to sample from the clean image at those locations.

    Args:
        image:     (H, W, 3) uint8 RGB clean GT image
        k1:        primary radial coefficient  (negative=barrel, positive=pincushion)
        k2:        secondary radial coefficient
        cx_offset: principal point x offset in [-1, 1] (fraction of half-width)
        cy_offset: principal point y offset in [-1, 1] (fraction of half-height)

    Returns:
        (H, W, 3) uint8 synthetically distorted image
    """
    H, W = image.shape[:2]

    # Principal point in pixel space
    cx_px = W / 2.0 + cx_offset * W / 2.0
    cy_px = H / 2.0 + cy_offset * H / 2.0

    # Pixel grid for the distorted output
    y_pix, x_pix = np.mgrid[0:H, 0:W].astype(np.float32)

    # Normalize to [-1, 1] relative to principal point
    xd_n = (x_pix - cx_px) / (W / 2.0)
    yd_n = (y_pix - cy_px) / (H / 2.0)

    rd2 = xd_n ** 2 + yd_n ** 2
    rd4 = rd2 ** 2

    # Inverse of forward Brown-Conrady (first-order approx)
    scale = 1.0 + k1 * rd2 + k2 * rd4
    # Clamp to avoid divide-by-zero / sign flip at extreme corners
    scale = np.clip(scale, 0.1, 10.0)
    inv_scale = 1.0 / scale

    xu_n = xd_n * inv_scale
    yu_n = yd_n * inv_scale

    # Back to pixel coords in the clean image
    map_x = (xu_n * (W / 2.0) + cx_px).astype(np.float32)
    map_y = (yu_n * (H / 2.0) + cy_px).astype(np.float32)

    return cv2.remap(
        image, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


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
                quality_range=config.AUG_JPEG_QUALITY_RANGE,
                p=0.3,
            ),
            A.HueSaturationValue(
                hue_shift_limit=config.AUG_HUE_SHIFT,
                sat_shift_limit=config.AUG_SAT_SHIFT,
                val_shift_limit=0,
                p=0.3,
            ),
            A.GaussNoise(
                std_range=config.AUG_NOISE_STD_RANGE,
                p=0.2,
            ),
        ],
        additional_targets={"target": "image"},  # apply same photometric aug to both
    )


class LensTrainDataset(Dataset):
    """Loads (distorted, corrected) image pairs and resizes to `resolution`.

    When augment=True (training split only):
      - Photometric augmentation is applied to both images identically.
      - With probability SYNTH_AUG_PROB, the real distorted image is replaced
        by a synthetically distorted version of the GT, expanding the training
        distribution to lens types not seen in the dataset.
    """

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
        self.synth_aug = self.augment  # synthetic distortion only during training

        # Collect all (original, generated) path pairs from the flat directory.
        # Files are named {uuid}_{g}_original.jpg / {uuid}_{g}_generated.jpg.
        root = Path(root)
        all_pairs = sorted([
            (f, root / f.name.replace("_original.jpg", "_generated.jpg"))
            for f in root.glob("*_original.jpg")
        ], key=lambda p: p[0].name)

        # Deterministic train/val split
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(all_pairs))
        n_val = int(len(all_pairs) * val_fraction)

        if split == "val":
            selected = indices[:n_val]
        else:
            selected = indices[n_val:]

        self.pairs = [all_pairs[i] for i in selected]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        orig_path, gen_path = self.pairs[idx]

        # Read as RGB
        distorted = cv2.imread(str(orig_path))
        corrected = cv2.imread(str(gen_path))
        distorted = cv2.cvtColor(distorted, cv2.COLOR_BGR2RGB)
        corrected = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)

        # Resize
        distorted = cv2.resize(distorted, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA)
        corrected = cv2.resize(corrected, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA)

        # Synthetic distortion augmentation: replace the real distorted image
        # with a synthetically distorted version of the GT. The model then sees
        # a wider range of lens types beyond those in the training set.
        if self.synth_aug and np.random.random() < config.SYNTH_AUG_PROB:
            k1 = np.random.uniform(*config.SYNTH_K1_RANGE)
            k2 = np.random.uniform(*config.SYNTH_K2_RANGE)
            cx_off = np.random.uniform(*config.SYNTH_CENTER_RANGE)
            cy_off = np.random.uniform(*config.SYNTH_CENTER_RANGE)
            distorted = apply_synthetic_distortion(corrected, k1, k2, cx_off, cy_off)

        # Photometric augmentation (same transform applied to both)
        if self.aug is not None:
            augmented = self.aug(image=distorted, target=corrected)
            distorted = augmented["image"]
            corrected = augmented["target"]

        # To tensor: HWC uint8 → CHW float32 [0, 1]
        distorted = torch.from_numpy(distorted).permute(2, 0, 1).float() / 255.0
        corrected = torch.from_numpy(corrected).permute(2, 0, 1).float() / 255.0

        return distorted, corrected
