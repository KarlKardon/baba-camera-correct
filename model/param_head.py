"""
Head 1 — Parametric Distortion Regression.

Takes the global-average-pooled backbone vector and predicts 9 distortion
parameters:
    k1, k2, k3, k4  — radial distortion coefficients
    p1, p2           — tangential distortion coefficients
    cx, cy           — principal point offset (normalized [-1, 1])
    s                — scale/crop factor

Initialized to predict near-zero values (identity warp).
"""

import torch
import torch.nn as nn

import config


class ParamHead(nn.Module):
    def __init__(self, in_features: int, num_params: int = config.NUM_DISTORTION_PARAMS):
        super().__init__()
        self.num_params = num_params
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_params),
        )
        # Initialize final layer to near-zero so initial prediction ≈ identity warp
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        # Small positive bias for scale factor (last param) so we start at s≈1
        self.mlp[-1].bias.data[-1] = 1.0

    def forward(self, pooled: torch.Tensor) -> dict:
        raw = self.mlp(pooled)  # (B, 9)

        # Unpack with appropriate activation ranges
        k1 = raw[:, 0]  # unconstrained — typically [-1, 1] for real lenses
        k2 = raw[:, 1]
        k3 = raw[:, 2]
        k4 = raw[:, 3]
        p1 = raw[:, 4] * 0.1   # tangential is small
        p2 = raw[:, 5] * 0.1
        cx = torch.tanh(raw[:, 6])  # principal point in [-1, 1]
        cy = torch.tanh(raw[:, 7])
        s = torch.sigmoid(raw[:, 8]) * 0.5 + 0.75  # scale in [0.75, 1.25]

        return {
            "k1": k1, "k2": k2, "k3": k3, "k4": k4,
            "p1": p1, "p2": p2,
            "cx": cx, "cy": cy,
            "s": s,
        }
