"""
Combined lens correction model: Backbone + ParamHead + FlowHead + Warp.
"""

import torch
import torch.nn as nn

from model.backbone import Backbone
from model.param_head import ParamHead
from model.flow_head import FlowHead
from model.warp import warp_image


class LensCorrectionModel(nn.Module):
    def __init__(self, flow_head_active: bool = False):
        super().__init__()
        self.backbone = Backbone()
        feat_dim = self.backbone.feat_dim

        self.param_head = ParamHead(in_features=feat_dim)
        self.flow_head = FlowHead(in_channels=feat_dim)
        self.flow_head_active = flow_head_active

    def set_flow_head_active(self, active: bool):
        """Toggle residual flow head on/off between training phases."""
        self.flow_head_active = active
        for p in self.flow_head.parameters():
            p.requires_grad = active

    def forward(self, distorted: torch.Tensor):
        """
        Args:
            distorted: (B, 3, H, W) distorted input image

        Returns:
            corrected: (B, 3, H, W) warped corrected output
            params: dict of predicted distortion parameters
            flow: (B, 2, Hf, Wf) residual flow (or None)
        """
        spatial, pooled = self.backbone(distorted)
        params = self.param_head(pooled)

        flow = None
        if self.flow_head_active:
            flow = self.flow_head(spatial)

        corrected = warp_image(distorted, params, residual_flow=flow)

        return corrected, params, flow
