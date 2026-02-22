"""
Gradient orientation losses:

1. Line straightness proxy — cosine similarity of gradient directions on
   strong-edge pixels. Differentiable stand-in for the Hough-based line
   straightness metric (22% of competition score).

2. Gradient orientation histogram KL divergence — compares the distribution
   of gradient angles between predicted and target (18% of competition score).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia


class GradientOrientationLoss(nn.Module):
    """Combined line-straightness proxy + gradient histogram loss."""

    def __init__(self, num_bins: int = 36, edge_threshold: float = 0.05):
        super().__init__()
        self.num_bins = num_bins
        self.edge_threshold = edge_threshold

    def _gradient_components(self, x: torch.Tensor):
        """Compute spatial gradients (dx, dy) on grayscale."""
        gray = kornia.color.rgb_to_grayscale(x)
        dx = kornia.filters.filter2d(
            gray,
            torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=x.dtype, device=x.device) / 4.0,
        )
        dy = kornia.filters.filter2d(
            gray,
            torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=x.dtype, device=x.device) / 4.0,
        )
        return dx, dy

    def _cosine_similarity_on_edges(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Cosine similarity of gradient directions, masked to strong edges.

        This is the line-straightness proxy: if the predicted image has straight
        lines in the same orientation as the target, gradient directions will
        align on edge pixels.
        """
        dx_p, dy_p = self._gradient_components(pred)
        dx_t, dy_t = self._gradient_components(target)

        # Edge magnitude for masking
        mag_t = torch.sqrt(dx_t ** 2 + dy_t ** 2 + 1e-8)
        mask = (mag_t > self.edge_threshold).float()

        # Cosine similarity per pixel
        dot = dx_p * dx_t + dy_p * dy_t
        mag_p = torch.sqrt(dx_p ** 2 + dy_p ** 2 + 1e-8)
        cos_sim = dot / (mag_p * mag_t + 1e-8)

        # Average over edge pixels only
        masked_cos = (cos_sim * mask).sum() / (mask.sum() + 1e-8)
        return 1.0 - masked_cos  # loss: 0 = perfect alignment

    def _angle_histogram_kl(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """KL divergence between gradient angle histograms.

        Uses soft binning for differentiability.
        """
        dx_p, dy_p = self._gradient_components(pred)
        dx_t, dy_t = self._gradient_components(target)

        # Gradient magnitudes as weights
        mag_p = torch.sqrt(dx_p ** 2 + dy_p ** 2 + 1e-8)
        mag_t = torch.sqrt(dx_t ** 2 + dy_t ** 2 + 1e-8)

        # Angles in [0, pi) — we use atan2 and fold to [0, pi)
        angle_p = torch.atan2(dy_p, dx_p + 1e-8)  # [-pi, pi]
        angle_t = torch.atan2(dy_t, dx_t + 1e-8)
        angle_p = angle_p % torch.pi  # [0, pi)
        angle_t = angle_t % torch.pi

        # Soft histogram via Gaussian binning
        bin_centers = torch.linspace(0, torch.pi, self.num_bins, device=pred.device)
        bin_centers = bin_centers.view(1, 1, self.num_bins, 1, 1)
        sigma = torch.pi / self.num_bins

        # Reshape angles for broadcasting: (B, 1, 1, H, W)
        angle_p = angle_p.unsqueeze(2)
        angle_t = angle_t.unsqueeze(2)
        mag_p = mag_p.unsqueeze(2)
        mag_t = mag_t.unsqueeze(2)

        # Soft assignment weighted by edge magnitude
        weights_p = torch.exp(-0.5 * ((angle_p - bin_centers) / sigma) ** 2) * mag_p
        weights_t = torch.exp(-0.5 * ((angle_t - bin_centers) / sigma) ** 2) * mag_t

        # Sum over spatial dims to get histogram per image: (B, 1, num_bins)
        hist_p = weights_p.sum(dim=(-2, -1))
        hist_t = weights_t.sum(dim=(-2, -1))

        # Normalize to distributions
        hist_p = hist_p / (hist_p.sum(dim=-1, keepdim=True) + 1e-8)
        hist_t = hist_t / (hist_t.sum(dim=-1, keepdim=True) + 1e-8)

        # KL divergence: KL(target || pred)
        kl = (hist_t * torch.log((hist_t + 1e-8) / (hist_p + 1e-8))).sum(dim=-1)
        return kl.mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Returns:
            line_proxy_loss: cosine similarity loss on edge pixels
            gradient_hist_loss: angle histogram KL divergence
        """
        line_proxy = self._cosine_similarity_on_edges(pred, target)
        hist_kl = self._angle_histogram_kl(pred, target)
        return line_proxy, hist_kl
