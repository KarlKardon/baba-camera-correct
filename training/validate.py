"""
Validation: runs model on val set and computes both the training loss
and competition-proxy metrics.

The proxy scorer mirrors competition scoring:
  - Edge F1 (40%) — multi-scale Canny edge precision/recall/F1
  - Line angle distribution (22%) — Hough angle histogram similarity
  - Gradient orientation (18%) — gradient direction histogram correlation
  - SSIM (15%) — structural similarity
  - Pixel MAE (5%) — mean absolute pixel error
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from losses.combined import CombinedLoss


# ── Competition-proxy metrics (non-differentiable, NumPy/OpenCV) ──────────────


def _to_uint8(tensor: torch.Tensor) -> np.ndarray:
    """(C, H, W) float [0,1] tensor → (H, W, C) uint8 numpy."""
    img = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def edge_f1(pred_np: np.ndarray, target_np: np.ndarray) -> float:
    """Multi-scale Canny edge F1 score."""
    scores = []
    for scale in [1.0, 0.5, 0.25]:
        h, w = int(pred_np.shape[0] * scale), int(pred_np.shape[1] * scale)
        p = cv2.resize(pred_np, (w, h))
        t = cv2.resize(target_np, (w, h))

        p_gray = cv2.cvtColor(p, cv2.COLOR_RGB2GRAY)
        t_gray = cv2.cvtColor(t, cv2.COLOR_RGB2GRAY)

        edges_p = cv2.Canny(p_gray, 50, 150)
        edges_t = cv2.Canny(t_gray, 50, 150)

        # Dilate target edges for tolerance
        kernel = np.ones((3, 3), np.uint8)
        edges_t_dilated = cv2.dilate(edges_t, kernel, iterations=1)
        edges_p_dilated = cv2.dilate(edges_p, kernel, iterations=1)

        # Precision: fraction of predicted edges near target edges
        tp_p = np.sum((edges_p > 0) & (edges_t_dilated > 0))
        fp_p = np.sum((edges_p > 0) & (edges_t_dilated == 0))
        precision = tp_p / (tp_p + fp_p + 1e-8)

        # Recall: fraction of target edges near predicted edges
        tp_r = np.sum((edges_t > 0) & (edges_p_dilated > 0))
        fn_r = np.sum((edges_t > 0) & (edges_p_dilated == 0))
        recall = tp_r / (tp_r + fn_r + 1e-8)

        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        scores.append(f1)

    return float(np.mean(scores))


def line_angle_similarity(pred_np: np.ndarray, target_np: np.ndarray, num_bins: int = 36) -> float:
    """Hough line angle distribution similarity."""
    def _angle_hist(img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                                minLineLength=30, maxLineGap=10)
        if lines is None or len(lines) == 0:
            return np.ones(num_bins) / num_bins  # uniform if no lines

        angles = []
        lengths = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) % np.pi
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            angles.append(angle)
            lengths.append(length)

        hist, _ = np.histogram(angles, bins=num_bins, range=(0, np.pi), weights=lengths)
        hist = hist / (hist.sum() + 1e-8)
        return hist

    hist_p = _angle_hist(pred_np)
    hist_t = _angle_hist(target_np)

    # Bhattacharyya coefficient (1 = identical distributions)
    bc = np.sum(np.sqrt(hist_p * hist_t))
    return float(bc)


def gradient_orientation_similarity(pred_np: np.ndarray, target_np: np.ndarray, num_bins: int = 36) -> float:
    """Gradient direction histogram correlation."""
    def _grad_hist(img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(dx ** 2 + dy ** 2)
        angle = np.arctan2(dy, dx) % np.pi

        hist, _ = np.histogram(angle.ravel(), bins=num_bins, range=(0, np.pi),
                               weights=mag.ravel())
        hist = hist / (hist.sum() + 1e-8)
        return hist

    hist_p = _grad_hist(pred_np)
    hist_t = _grad_hist(target_np)

    # Pearson correlation
    corr = np.corrcoef(hist_p, hist_t)[0, 1]
    return float(max(corr, 0.0))  # clamp negative correlations to 0


def compute_ssim_cv(pred_np: np.ndarray, target_np: np.ndarray) -> float:
    """SSIM via OpenCV/numpy (for validation, independent of Kornia)."""
    from skimage.metrics import structural_similarity
    gray_p = cv2.cvtColor(pred_np, cv2.COLOR_RGB2GRAY)
    gray_t = cv2.cvtColor(target_np, cv2.COLOR_RGB2GRAY)
    return float(structural_similarity(gray_p, gray_t))


def pixel_mae(pred_np: np.ndarray, target_np: np.ndarray) -> float:
    """Mean absolute pixel difference, normalized to [0, 1]."""
    return float(np.mean(np.abs(pred_np.astype(np.float32) - target_np.astype(np.float32))) / 255.0)


def proxy_score(pred_np: np.ndarray, target_np: np.ndarray) -> dict:
    """Compute all competition-proxy metrics for a single image pair.

    Returns dict with individual metrics and weighted total (higher = better).
    """
    ef1 = edge_f1(pred_np, target_np)
    las = line_angle_similarity(pred_np, target_np)
    gos = gradient_orientation_similarity(pred_np, target_np)

    try:
        ssim_val = compute_ssim_cv(pred_np, target_np)
    except ImportError:
        # Fallback if scikit-image not available
        ssim_val = 0.0

    mae = pixel_mae(pred_np, target_np)
    pixel_score = max(0.0, 1.0 - mae * 10)  # rough mapping: 0.1 MAE → 0 score

    weighted = (
        0.40 * ef1
        + 0.22 * las
        + 0.18 * gos
        + 0.15 * ssim_val
        + 0.05 * pixel_score
    )

    return {
        "edge_f1": ef1,
        "line_angle_sim": las,
        "grad_orient_sim": gos,
        "ssim": ssim_val,
        "pixel_mae": mae,
        "proxy_score": weighted,
    }


# ── Validation loop ───────────────────────────────────────────────────────────


@torch.no_grad()
def run_validation(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: CombinedLoss,
    device: torch.device,
    compute_proxy: bool = False,
    max_proxy_samples: int = 100,
) -> dict:
    """Run validation and return average loss dict.

    Args:
        compute_proxy: If True, also compute competition-proxy metrics on a
            subset of samples. This is slower (OpenCV Hough etc.) so only
            enable periodically.
        max_proxy_samples: Max images to compute proxy metrics on.
    """
    model.eval()
    running = {}
    n_batches = 0
    proxy_metrics = []
    proxy_count = 0

    for distorted, corrected in val_loader:
        distorted = distorted.to(device)
        corrected = corrected.to(device)

        with autocast(device_type="cuda"):
            pred, params, flow = model(distorted)
            _, loss_dict = criterion(pred, corrected, flow)

        for k, v in loss_dict.items():
            running[k] = running.get(k, 0.0) + v
        n_batches += 1

        # Proxy metrics on subset
        if compute_proxy and proxy_count < max_proxy_samples:
            for i in range(pred.shape[0]):
                if proxy_count >= max_proxy_samples:
                    break
                pred_np = _to_uint8(pred[i])
                target_np = _to_uint8(corrected[i])
                proxy_metrics.append(proxy_score(pred_np, target_np))
                proxy_count += 1

    avg = {k: v / n_batches for k, v in running.items()}

    if proxy_metrics:
        # Average proxy metrics
        keys = proxy_metrics[0].keys()
        for k in keys:
            avg[f"proxy_{k}"] = np.mean([m[k] for m in proxy_metrics])
        print(f"  Proxy score ({proxy_count} samples): {avg['proxy_proxy_score']:.4f}")

    return avg
