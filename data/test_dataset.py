"""
Test dataset: loads the 1,000 distorted test images for inference.

Expected layout:
    TEST_DIR/
        {image_id}.jpg
"""

from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset

import config


class LensTestDataset(Dataset):
    """Loads distorted test images, resizes, and returns (tensor, image_id, original_size)."""

    def __init__(
        self,
        root: Path = config.TEST_DIR,
        resolution: int = 768,
    ):
        self.resolution = resolution
        self.image_paths = sorted(Path(root).glob("*.jpg"))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        image_id = path.stem

        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        img = cv2.resize(img, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return tensor, image_id, (orig_h, orig_w)
