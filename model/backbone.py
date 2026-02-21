"""
Shared backbone: ConvNeXt-Small from timm, pretrained on ImageNet.

Returns:
    features: (B, C, H', W') spatial feature map for the flow head
    pooled:   (B, C) global average-pooled vector for the parametric head
"""

import timm
import torch
import torch.nn as nn

import config


class Backbone(nn.Module):
    def __init__(self, model_name: str = config.BACKBONE, pretrained: bool = True):
        super().__init__()
        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=False,
            num_classes=0,       # remove classifier, keep pooling
        )
        # Get feature dim from the model
        self.feat_dim = self.encoder.num_features

        # We also need spatial features (before pooling) for the flow head.
        # timm ConvNeXt with num_classes=0 returns pooled features.
        # Create a features_only version for spatial maps.
        self.encoder_spatial = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(-1,),  # last stage only
        )

    def forward(self, x: torch.Tensor):
        # Spatial features from last stage: (B, C, H/32, W/32)
        spatial = self.encoder_spatial(x)[-1]
        # Global average pool for parametric head
        pooled = spatial.mean(dim=(-2, -1))
        return spatial, pooled
