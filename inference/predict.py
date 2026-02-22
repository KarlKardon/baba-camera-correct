"""
Full inference pipeline for the 1,000 test images.

For each image:
  1. Forward pass → parametric params + residual flow
  2. Warp via grid_sample (reflection padding)
  3. Optional line refinement (Hough + LM optimization)
  4. Resize back to original dimensions and save as {image_id}.jpg

Usage:
    python inference/predict.py \
        --checkpoint outputs/checkpoints/phase2_full_best.pt \
        --output_dir outputs/submissions/run1 \
        --resolution 768 \
        --refine
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from data.test_dataset import LensTestDataset
from model.lens_model import LensCorrectionModel
from model.warp import parametric_grid, warp_image
from inference.line_refine import refine_distortion


def _tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """(3, H, W) float [0,1] → (H, W, 3) uint8 RGB."""
    return (t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def _params_to_scalars(params: dict, idx: int) -> dict:
    """Extract scalar params for a single image from batched tensors."""
    return {k: float(v[idx].cpu()) for k, v in params.items()}


def _rewarp_with_params(
    distorted: torch.Tensor,
    new_params: dict,
    device: torch.device,
    residual_flow: torch.Tensor | None = None,
) -> torch.Tensor:
    """Re-warp a single image with adjusted scalar params, preserving flow."""
    batch_params = {k: torch.tensor([v], device=device) for k, v in new_params.items()}
    return warp_image(distorted, batch_params, residual_flow=residual_flow)


@torch.no_grad()
def run_inference(
    checkpoint_path: str,
    output_dir: str,
    resolution: int = 768,
    do_refine: bool = False,
    batch_size: int = 4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = LensCorrectionModel(flow_head_active=True)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    print(f"Loaded checkpoint: {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")

    # Dataset
    test_ds = LensTestDataset(resolution=resolution)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    print(f"Test images: {len(test_ds)}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    total = 0
    refined_count = 0
    t0 = time.time()

    for batch_tensors, batch_ids, batch_orig_sizes in test_loader:
        batch_tensors = batch_tensors.to(device)

        with autocast(device_type="cuda"):
            pred, params, flow = model(batch_tensors)

        for i in range(pred.shape[0]):
            image_id = batch_ids[i]
            orig_h, orig_w = batch_orig_sizes[0][i].item(), batch_orig_sizes[1][i].item()

            result = pred[i]

            # Optional line refinement with acceptance gate
            if do_refine:
                result_np = _tensor_to_uint8(result)
                scalar_params = _params_to_scalars(params, i)

                refined_params = refine_distortion(result_np, scalar_params)
                if refined_params is not None:
                    single_input = batch_tensors[i:i+1]
                    flow_single = flow[i:i+1] if flow is not None else None
                    rewarp = _rewarp_with_params(single_input, refined_params, device, flow_single)
                    # Accept because refine_distortion already enforced improvement and bounds
                    refined_count += 1
                    result = rewarp[0]

            # Convert to uint8 and resize to original dimensions
            out_img = _tensor_to_uint8(result)
            out_img = cv2.resize(out_img, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)

            # Save as JPEG (RGB → BGR for OpenCV)
            save_path = out_path / f"{image_id}.jpg"
            cv2.imwrite(str(save_path), cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])

            total += 1

        if total % 100 == 0:
            elapsed = time.time() - t0
            print(f"  Processed {total}/{len(test_ds)} ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    print(f"\nDone: {total} images in {elapsed:.1f}s ({total/elapsed:.1f} img/s)")
    if do_refine:
        print(f"  Line-refined: {refined_count}/{total}")
    print(f"  Output: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Run inference on test images")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint .pt file")
    parser.add_argument("--output_dir", type=str, default="outputs/submissions/run1",
                        help="Directory to save corrected images")
    parser.add_argument("--resolution", type=int, default=768,
                        help="Inference resolution (images resized to this)")
    parser.add_argument("--refine", action="store_true",
                        help="Enable line refinement post-processing")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    run_inference(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        resolution=args.resolution,
        do_refine=args.refine,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
