"""
SSIM loss wrapper using Kornia.

Returns 1 - SSIM so that minimizing the loss maximizes structural similarity.
"""

import torch
import torch.nn as nn
import kornia


class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 11):
        super().__init__()
        self.window_size = window_size

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   (B, 3, H, W)
            target: (B, 3, H, W)

        Returns:
            Scalar loss in [0, 1] (lower = more similar)
        """
        ssim_val = kornia.metrics.ssim(pred, target, window_size=self.window_size)
        # ssim returns (B, C, H, W) map — average to scalar
        return 1.0 - ssim_val.mean()
