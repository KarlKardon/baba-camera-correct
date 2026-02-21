"""
Visualization utilities:
- Side-by-side comparisons (distorted | predicted | ground truth)
- Distortion parameter overlay
- Residual flow field plots
"""

import cv2
import numpy as np
from pathlib import Path


def side_by_side(
    distorted: np.ndarray,
    predicted: np.ndarray,
    ground_truth: np.ndarray | None = None,
    save_path: str | None = None,
    title: str = "",
) -> np.ndarray:
    """Create a side-by-side comparison image.

    Args:
        distorted: (H, W, 3) RGB uint8
        predicted: (H, W, 3) RGB uint8
        ground_truth: (H, W, 3) RGB uint8, optional
        save_path: if set, saves the result
        title: optional title text

    Returns:
        combined: (H, W_total, 3) RGB uint8
    """
    h = distorted.shape[0]
    images = [distorted, predicted]
    labels = ["Distorted", "Predicted"]

    if ground_truth is not None:
        images.append(ground_truth)
        labels.append("Ground Truth")

    # Resize all to same height
    resized = []
    for img in images:
        if img.shape[0] != h:
            scale = h / img.shape[0]
            img = cv2.resize(img, (int(img.shape[1] * scale), h))
        resized.append(img)

    # Add labels
    labeled = []
    for img, label in zip(resized, labels):
        img_copy = img.copy()
        cv2.putText(img_copy, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 0), 2)
        labeled.append(img_copy)

    # Separator
    sep = np.ones((h, 3, 3), dtype=np.uint8) * 128

    parts = []
    for i, img in enumerate(labeled):
        parts.append(img)
        if i < len(labeled) - 1:
            parts.append(sep)

    combined = np.concatenate(parts, axis=1)

    if title:
        cv2.putText(combined, title, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 1)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    return combined


def visualize_flow_field(
    flow: np.ndarray,
    save_path: str | None = None,
) -> np.ndarray:
    """Visualize a 2-channel flow field as an HSV color wheel image.

    Args:
        flow: (2, H, W) or (H, W, 2) displacement field

    Returns:
        vis: (H, W, 3) RGB uint8
    """
    if flow.ndim == 3 and flow.shape[0] == 2:
        flow = np.transpose(flow, (1, 2, 0))  # (H, W, 2)

    dx, dy = flow[..., 0], flow[..., 1]
    mag = np.sqrt(dx ** 2 + dy ** 2)
    angle = np.arctan2(dy, dx)

    # Normalize magnitude for visualization
    mag_norm = mag / (mag.max() + 1e-8)

    # HSV: hue = direction, saturation = 1, value = magnitude
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = ((angle + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = (mag_norm * 255).astype(np.uint8)

    vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    return vis


def batch_visualize(
    pred_dir: str,
    distorted_dir: str,
    gt_dir: str | None = None,
    output_dir: str = "outputs/visualizations",
    max_images: int = 20,
):
    """Generate side-by-side comparisons for a batch of images.

    Args:
        pred_dir: directory with predicted {image_id}.jpg
        distorted_dir: directory with distorted test images
        gt_dir: optional directory with ground truth
        output_dir: where to save comparison images
        max_images: how many comparisons to generate
    """
    pred_path = Path(pred_dir)
    pred_files = sorted(pred_path.glob("*.jpg"))[:max_images]

    for pf in pred_files:
        image_id = pf.stem

        pred_img = cv2.imread(str(pf))
        pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)

        dist_file = Path(distorted_dir) / f"{image_id}.jpg"
        if not dist_file.exists():
            continue
        dist_img = cv2.imread(str(dist_file))
        dist_img = cv2.cvtColor(dist_img, cv2.COLOR_BGR2RGB)

        gt_img = None
        if gt_dir:
            gt_candidates = list(Path(gt_dir).glob(f"{image_id}*/generated.jpg"))
            if gt_candidates:
                gt_img = cv2.imread(str(gt_candidates[0]))
                gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

        # Resize to common height
        target_h = 512
        for arr_name in ["dist_img", "pred_img", "gt_img"]:
            arr = locals()[arr_name]
            if arr is not None:
                scale = target_h / arr.shape[0]
                locals()[arr_name] = cv2.resize(arr, (int(arr.shape[1] * scale), target_h))

        side_by_side(
            dist_img, pred_img, gt_img,
            save_path=f"{output_dir}/{image_id}_compare.jpg",
            title=image_id,
        )

    print(f"Saved {len(pred_files)} comparisons to {output_dir}/")
