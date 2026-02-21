"""
Residual flow regularization:

1. Total Variation (TV) — encourages spatial smoothness, prevents local
   artifacts that destroy line straightness.
2. L2 magnitude penalty — keeps residual flow small so the parametric model
   does the heavy lifting.
"""

import torch
import torch.nn as nn


class FlowRegularization(nn.Module):
    def __init__(self, tv_weight: float = 0.5, l2_weight: float = 0.5):
        super().__init__()
        self.tv_weight = tv_weight
        self.l2_weight = l2_weight

    def forward(self, flow: torch.Tensor | None) -> torch.Tensor:
        """
        Args:
            flow: (B, 2, H, W) residual displacement field, or None

        Returns:
            Scalar regularization loss (0.0 if flow is None)
        """
        if flow is None:
            return torch.tensor(0.0)

        # Total variation: sum of absolute spatial differences
        tv_h = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :]).mean()
        tv_w = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1]).mean()
        tv_loss = tv_h + tv_w

        # L2 magnitude penalty
        l2_loss = (flow ** 2).mean()

        return self.tv_weight * tv_loss + self.l2_weight * l2_loss
