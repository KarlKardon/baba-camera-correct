"""
Head 2 — Residual Flow Field Decoder.

Takes the spatial feature map from the backbone and predicts a 2-channel
(dx, dy) displacement field at FLOW_RESOLUTION×FLOW_RESOLUTION.
This is bilinearly upsampled to full image resolution during warping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class FlowHead(nn.Module):
    def __init__(self, in_channels: int, flow_res: int = config.FLOW_RESOLUTION):
        super().__init__()
        self.flow_res = flow_res

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 2, 3, padding=1),  # dx, dy
        )

        # Initialize to zero so initial flow = no displacement
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, spatial_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spatial_features: (B, C, H', W') from backbone

        Returns:
            flow: (B, 2, flow_res, flow_res) displacement field
        """
        x = self.decoder(spatial_features)
        # Resize to fixed flow resolution
        flow = F.interpolate(x, size=(self.flow_res, self.flow_res), mode="bilinear", align_corners=True)
        # Scale output to small range to keep residuals gentle
        flow = flow * 0.1
        return flow
