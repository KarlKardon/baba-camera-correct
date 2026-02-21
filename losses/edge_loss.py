"""
Multi-scale Sobel edge magnitude L1 loss.

Computes Sobel edge maps at 3 scales (1x, 0.5x, 0.25x) and averages the
L1 distance between predicted and target edge maps. This aligns with the
competition's 40%-weighted edge similarity metric.

Uses Kornia's differentiable Sobel operator.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia


class MultiScaleEdgeLoss(nn.Module):
    def __init__(self, scales=(1.0, 0.5, 0.25)):
        super().__init__()
        self.scales = scales

    def _edge_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """Sobel edge magnitude on grayscale input."""
        gray = kornia.color.rgb_to_grayscale(x)
        edges = kornia.filters.sobel(gray)
        return edges

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   (B, 3, H, W) predicted corrected image
            target: (B, 3, H, W) ground truth corrected image

        Returns:
            Scalar loss
        """
        total = 0.0
        for scale in self.scales:
            if scale < 1.0:
                size = (int(pred.shape[2] * scale), int(pred.shape[3] * scale))
                p = F.interpolate(pred, size=size, mode="bilinear", align_corners=False)
                t = F.interpolate(target, size=size, mode="bilinear", align_corners=False)
            else:
                p, t = pred, target

            edge_p = self._edge_magnitude(p)
            edge_t = self._edge_magnitude(t)
            total = total + F.l1_loss(edge_p, edge_t)

        return total / len(self.scales)
