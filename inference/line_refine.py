"""
Post-inference line refinement via Levenberg-Marquardt optimization.

For images with strong linear structure (>=15 detected line segments),
adjusts the parametric distortion coefficients (k1, k2, cx, cy, s) to
minimize line curvature. Accepts the refinement only if the proxy score
improves.

Uses scipy.optimize.least_squares for LM optimization and OpenCV for
Hough line detection.
"""

import cv2
import numpy as np
from scipy.optimize import least_squares

import config

# Bounds aligned with training-time activations in ParamHead
K1_BOUNDS = (-0.8, 0.8)
K2_BOUNDS = (-0.4, 0.4)
CX_BOUNDS = (-1.0, 1.0)
CY_BOUNDS = (-1.0, 1.0)
S_BOUNDS = (0.75, 1.25)


def detect_lines(image_uint8: np.ndarray) -> np.ndarray | None:
    """Detect line segments using Canny + probabilistic Hough transform.

    Args:
        image_uint8: (H, W, 3) RGB uint8 image

    Returns:
        lines: (N, 4) array of (x1, y1, x2, y2) or None if too few
    """
    gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    raw = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=50, minLineLength=30, maxLineGap=10,
    )
    if raw is None:
        return None
    return raw.reshape(-1, 4)


def _line_curvature_residuals(
    params_vec: np.ndarray,
    sample_points: np.ndarray,
    img_h: int,
    img_w: int,
) -> np.ndarray:
    """Compute residuals measuring how curved lines become after warping.

    We take points sampled along detected line segments, apply the inverse
    distortion model, and measure deviation from collinearity.

    Args:
        params_vec: [k1, k2, cx, cy, s] — 5 params to optimize
        sample_points: (M, 2) points in normalized coords from detected lines
        img_h, img_w: image dimensions for normalization

    Returns:
        residuals: (M,) collinearity error per point
    """
    k1, k2, cx, cy, s = params_vec

    # Normalized coords [-1, 1]
    x = sample_points[:, 0]
    y = sample_points[:, 1]

    # Shift by principal point
    xc = x - cx * 0.1
    yc = y - cy * 0.1

    r2 = xc ** 2 + yc ** 2
    r4 = r2 ** 2

    radial = 1.0 + k1 * r2 + k2 * r4
    x_undist = xc * radial / s + cx * 0.1
    y_undist = yc * radial / s + cy * 0.1

    # Measure collinearity: for every 3 consecutive points on a line,
    # the cross product should be zero if they're collinear
    residuals = []
    # Points are grouped by line (triplets)
    n_triplets = len(x_undist) // 3
    for i in range(n_triplets):
        idx = i * 3
        ax, ay = x_undist[idx], y_undist[idx]
        bx, by = x_undist[idx + 1], y_undist[idx + 1]
        cx_, cy_ = x_undist[idx + 2], y_undist[idx + 2]
        # Cross product of (B-A) x (C-A) — zero means collinear
        cross = (bx - ax) * (cy_ - ay) - (by - ay) * (cx_ - ax)
        residuals.append(cross)

    return np.array(residuals) if residuals else np.zeros(1)


def _sample_line_points(lines: np.ndarray, img_h: int, img_w: int) -> np.ndarray:
    """Sample 3 points per line segment in normalized [-1, 1] coordinates.

    Returns: (N*3, 2) array of points grouped in triplets per line.
    """
    points = []
    for x1, y1, x2, y2 in lines:
        for t in [0.0, 0.5, 1.0]:
            px = x1 + t * (x2 - x1)
            py = y1 + t * (y2 - y1)
            # Normalize to [-1, 1]
            nx = (px / img_w) * 2.0 - 1.0
            ny = (py / img_h) * 2.0 - 1.0
            points.append([nx, ny])
    return np.array(points)


def refine_distortion(
    warped_image: np.ndarray,
    initial_params: dict,
    max_iters: int = config.LINE_REFINE_LM_ITERS,
    min_segments: int = config.LINE_REFINE_MIN_SEGMENTS,
) -> tuple[dict, float, float] | None:
    """Attempt to refine distortion parameters using line straightness.

    Args:
        warped_image: (H, W, 3) RGB uint8 — the already-warped output
        initial_params: dict with k1, k2, k3, k4, p1, p2, cx, cy, s (scalars)
        max_iters: LM iteration limit
        min_segments: minimum detected lines to attempt refinement

    Returns:
        Refined params dict if refinement succeeded, None if skipped.
    """
    lines = detect_lines(warped_image)
    if lines is None or len(lines) < min_segments:
        return None

    img_h, img_w = warped_image.shape[:2]

    # Filter to longer lines (more reliable)
    lengths = np.sqrt((lines[:, 2] - lines[:, 0]) ** 2 + (lines[:, 3] - lines[:, 1]) ** 2)
    mean_length = np.mean(lengths)
    if mean_length < 20:
        return None

    # Use top lines by length
    top_idx = np.argsort(lengths)[::-1][:50]
    top_lines = lines[top_idx]

    sample_pts = _sample_line_points(top_lines, img_h, img_w)

    # Initial param vector: [k1, k2, cx, cy, s]
    x0 = np.array([
        initial_params["k1"],
        initial_params["k2"],
        initial_params["cx"],
        initial_params["cy"],
        initial_params["s"],
    ], dtype=np.float64)

    lower = np.array([K1_BOUNDS[0], K2_BOUNDS[0], CX_BOUNDS[0], CY_BOUNDS[0], S_BOUNDS[0]])
    upper = np.array([K1_BOUNDS[1], K2_BOUNDS[1], CX_BOUNDS[1], CY_BOUNDS[1], S_BOUNDS[1]])

    # Baseline curvature cost
    initial_cost = np.linalg.norm(_line_curvature_residuals(x0, sample_pts, img_h, img_w))

    result = least_squares(
        _line_curvature_residuals,
        x0,
        bounds=(lower, upper),
        args=(sample_pts, img_h, img_w),
        method="trf",
        max_nfev=max_iters,
    )

    if not result.success:
        return None

    refined_cost = np.linalg.norm(_line_curvature_residuals(result.x, sample_pts, img_h, img_w))

    # Require meaningful improvement
    if refined_cost >= initial_cost * 0.99:
        return None

    # Build refined params (keep k3, k4, p1, p2 from original)
    refined = dict(initial_params)
    refined["k1"] = float(np.clip(result.x[0], *K1_BOUNDS))
    refined["k2"] = float(np.clip(result.x[1], *K2_BOUNDS))
    refined["cx"] = float(np.clip(result.x[2], *CX_BOUNDS))
    refined["cy"] = float(np.clip(result.x[3], *CY_BOUNDS))
    refined["s"] = float(np.clip(result.x[4], *S_BOUNDS))

    return refined, initial_cost, refined_cost
