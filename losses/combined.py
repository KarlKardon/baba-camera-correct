"""
Combined loss: weighted sum of all loss components, configurable per training phase.

Uses the LossWeights dataclass from config to set per-phase weights.
Components with weight=0 are skipped entirely (no computation cost).
"""

import torch
import torch.nn as nn

from config import LossWeights
from losses.edge_loss import MultiScaleEdgeLoss
from losses.ssim_loss import SSIMLoss
from losses.gradient_loss import GradientOrientationLoss
from losses.flow_reg import FlowRegularization


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Smooth L1 variant — less sensitive to outliers than MSE."""
    return torch.sqrt((pred - target) ** 2 + eps ** 2).mean()


class CombinedLoss(nn.Module):
    def __init__(self, weights: LossWeights):
        super().__init__()
        self.weights = weights

        self.edge_loss = MultiScaleEdgeLoss() if weights.edge > 0 else None
        self.ssim_loss = SSIMLoss() if weights.ssim > 0 else None
        self.gradient_loss = GradientOrientationLoss() if (weights.line_proxy > 0 or weights.gradient_orientation > 0) else None
        self.flow_reg = FlowRegularization() if weights.flow_reg > 0 else None

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        flow: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            pred:   (B, 3, H, W) predicted corrected image
            target: (B, 3, H, W) ground truth
            flow:   (B, 2, Hf, Wf) residual flow, or None

        Returns:
            total_loss: scalar
            loss_dict: individual components for logging
        """
        loss_dict = {}
        total = torch.tensor(0.0, device=pred.device)

        # Edge alignment (40% of competition score)
        if self.edge_loss is not None:
            l = self.edge_loss(pred, target)
            loss_dict["edge"] = l.item()
            total = total + self.weights.edge * l

        # Line straightness proxy + gradient orientation
        if self.gradient_loss is not None:
            line_proxy, grad_hist = self.gradient_loss(pred, target)
            if self.weights.line_proxy > 0:
                loss_dict["line_proxy"] = line_proxy.item()
                total = total + self.weights.line_proxy * line_proxy
            if self.weights.gradient_orientation > 0:
                loss_dict["grad_orient"] = grad_hist.item()
                total = total + self.weights.gradient_orientation * grad_hist

        # SSIM
        if self.ssim_loss is not None:
            l = self.ssim_loss(pred, target)
            loss_dict["ssim"] = l.item()
            total = total + self.weights.ssim * l

        # Photometric (Charbonnier)
        if self.weights.photometric > 0:
            l = charbonnier_loss(pred, target)
            loss_dict["photometric"] = l.item()
            total = total + self.weights.photometric * l

        # Flow regularization
        if self.flow_reg is not None and flow is not None:
            l = self.flow_reg(flow)
            loss_dict["flow_reg"] = l.item()
            total = total + self.weights.flow_reg * l

        loss_dict["total"] = total.item()
        return total, loss_dict
