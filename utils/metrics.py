"""
Utility to run proxy scoring on saved image files (e.g., after inference).

Usage:
    python utils/metrics.py --pred_dir outputs/submissions/run1 --gt_dir data_raw/lens-correction-train-clea
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.validate import proxy_score


def score_directory(pred_dir: Path, gt_dir: Path, max_samples: int = 0) -> dict:
    """Score all predicted images against ground truth.

    Expects pred_dir to contain {image_id}.jpg files, and gt_dir to contain
    subfolders {image_id}_g0/generated.jpg with the ground truth.

    If gt_dir contains direct .jpg files, matches by filename instead.
    """
    pred_files = sorted(pred_dir.glob("*.jpg"))
    if max_samples > 0:
        pred_files = pred_files[:max_samples]

    all_metrics = []
    for pf in pred_files:
        image_id = pf.stem

        # Try subfolder structure first (training data layout)
        gt_candidates = list(gt_dir.glob(f"{image_id}*/generated.jpg"))
        if gt_candidates:
            gt_path = gt_candidates[0]
        else:
            # Try direct file match
            gt_path = gt_dir / f"{image_id}.jpg"

        if not gt_path.exists():
            print(f"  SKIP {image_id}: no ground truth found")
            continue

        pred_img = cv2.imread(str(pf))
        gt_img = cv2.imread(str(gt_path))
        pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

        # Resize to match if needed
        if pred_img.shape != gt_img.shape:
            gt_img = cv2.resize(gt_img, (pred_img.shape[1], pred_img.shape[0]))

        metrics = proxy_score(pred_img, gt_img)
        all_metrics.append(metrics)

        if len(all_metrics) % 50 == 0:
            print(f"  Scored {len(all_metrics)}/{len(pred_files)} images...")

    if not all_metrics:
        print("No images scored.")
        return {}

    # Average
    keys = all_metrics[0].keys()
    avg = {k: np.mean([m[k] for m in all_metrics]) for k in keys}

    print(f"\n{'='*40}")
    print(f"Results over {len(all_metrics)} images:")
    for k, v in sorted(avg.items()):
        print(f"  {k:25s}: {v:.4f}")
    print(f"{'='*40}")

    return avg


def main():
    parser = argparse.ArgumentParser(description="Score predictions against ground truth")
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=0, help="0 = all")
    args = parser.parse_args()

    score_directory(Path(args.pred_dir), Path(args.gt_dir), args.max_samples)


if __name__ == "__main__":
    main()
