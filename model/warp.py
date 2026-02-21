"""
Differentiable warp pipeline.

Constructs an inverse sampling grid from the predicted distortion parameters,
adds the residual flow field, and warps the input image via grid_sample.

Uses Brown-Conrady distortion model (same as OpenCV undistort):
    r² = x² + y²
    x' = x(1 + k1*r² + k2*r⁴ + k3*r⁶ + k4*r⁸) + 2*p1*x*y + p2*(r² + 2*x²)
    y' = y(1 + k1*r² + k2*r⁴ + k3*r⁶ + k4*r⁸) + p1*(r² + 2*y²) + 2*p2*x*y

Since we're *undistorting*, we compute the inverse map: for each output pixel,
where to sample in the distorted input.
"""

import torch
import torch.nn.functional as F


def make_base_grid(H: int, W: int, device: torch.device) -> torch.Tensor:
    """Create a normalized coordinate grid in [-1, 1] range.

    Returns: (1, H, W, 2)
    """
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)


def parametric_grid(params: dict, H: int, W: int) -> torch.Tensor:
    """Build inverse sampling grid from Brown-Conrady distortion parameters.

    For undistortion: given an output (corrected) pixel, compute where to
    sample in the distorted input. We apply the *distortion* model to output
    coordinates to find input coordinates.

    Args:
        params: dict with keys k1..k4, p1, p2, cx, cy, s — each (B,)
        H, W: output spatial dimensions

    Returns:
        grid: (B, H, W, 2) sampling grid in [-1, 1] for grid_sample
    """
    B = params["k1"].shape[0]
    device = params["k1"].device

    base = make_base_grid(H, W, device).expand(B, -1, -1, -1)  # (B, H, W, 2)

    # Shift by principal point
    cx = params["cx"].view(B, 1, 1, 1)
    cy = params["cy"].view(B, 1, 1, 1)
    x = base[..., 0:1] - cx * 0.1  # scale cx/cy effect
    y = base[..., 1:2] - cy * 0.1

    r2 = x ** 2 + y ** 2
    r4 = r2 ** 2
    r6 = r2 ** 3
    r8 = r2 ** 4

    k1 = params["k1"].view(B, 1, 1, 1)
    k2 = params["k2"].view(B, 1, 1, 1)
    k3 = params["k3"].view(B, 1, 1, 1)
    k4 = params["k4"].view(B, 1, 1, 1)
    p1 = params["p1"].view(B, 1, 1, 1)
    p2 = params["p2"].view(B, 1, 1, 1)
    s = params["s"].view(B, 1, 1, 1)

    # Radial distortion
    radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8

    # Tangential distortion
    x_tang = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x ** 2)
    y_tang = p1 * (r2 + 2.0 * y ** 2) + 2.0 * p2 * x * y

    # Distorted coordinates (where to sample in input)
    x_dist = x * radial + x_tang
    y_dist = y * radial + y_tang

    # Apply scale and re-add principal point offset
    x_out = x_dist / s + cx * 0.1
    y_out = y_dist / s + cy * 0.1

    return torch.cat([x_out, y_out], dim=-1)  # (B, H, W, 2)


def warp_image(
    image: torch.Tensor,
    params: dict,
    residual_flow: torch.Tensor | None = None,
) -> torch.Tensor:
    """Warp a distorted image to produce the corrected output.

    Args:
        image: (B, 3, H, W) distorted input
        params: distortion parameters from ParamHead
        residual_flow: (B, 2, Hf, Wf) optional residual displacement field

    Returns:
        warped: (B, 3, H, W) corrected output
    """
    B, C, H, W = image.shape

    # Build parametric sampling grid
    grid = parametric_grid(params, H, W)  # (B, H, W, 2)

    # Add residual flow if provided
    if residual_flow is not None:
        # Upsample flow to image resolution
        flow_up = F.interpolate(
            residual_flow, size=(H, W), mode="bilinear", align_corners=True
        )  # (B, 2, H, W)
        # Convert from (B, 2, H, W) to (B, H, W, 2)
        flow_up = flow_up.permute(0, 2, 3, 1)
        grid = grid + flow_up

    # Clamp grid to avoid extreme sampling
    grid = grid.clamp(-2.0, 2.0)

    # Warp with reflection padding to avoid black borders
    warped = F.grid_sample(
        image, grid, mode="bilinear", padding_mode="reflection", align_corners=True
    )

    return warped
