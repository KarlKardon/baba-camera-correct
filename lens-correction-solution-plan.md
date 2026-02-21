# Lens Distortion Correction — Full Solution Plan

## Competition Context

Kaggle competition: Automatic Lens Correction. Given ~23,000 training pairs (distorted → corrected) and 1,000 distorted test images, produce geometrically corrected outputs. Scoring is MAE against per-image scores (0.0–1.0) with weighted multi-metric evaluation:

| Metric | Weight | What It Measures |
|---|---|---|
| Edge Similarity | 40% | Multi-scale Canny edge F1 score |
| Line Straightness | 22% | Hough line angle distribution match |
| Gradient Orientation | 18% | Gradient direction histogram similarity |
| SSIM | 15% | Structural similarity index |
| Pixel Accuracy | 5% | Mean absolute pixel difference |

Lower MAE = better. Perfect correction = 0.0 MAE. Hard fail conditions: catastrophic local error or structurally broken output → score 0.0 for that image.

**Core insight:** This is a geometric rectification problem, not image enhancement. 62% of the score is edge/line geometry. Warp pixels, don't generate them.

---

## Tech Stack

- **Python 3.10+**
- **PyTorch 2.x** — core framework, `torch.compile` for inference, AMP (mixed precision) throughout
- **timm** — pretrained backbones (ConvNeXt-Small or Swin-T)
- **Kornia** — differentiable Sobel, SSIM, spatial transforms, grid utilities
- **OpenCV** — Hough line detection, Canny edges (for line refinement and validation metric)
- **Albumentations** — photometric-only augmentation (brightness, contrast, JPEG compression, color jitter — **never geometric**)
- **NumPy / SciPy** — numerical optimization in line refinement (`scipy.optimize.least_squares`)
- **Weights & Biases** — experiment tracking (optional but recommended)
- **Pillow / OpenCV** — image I/O
- **Execution:** local development, Kaggle GPU (T4/P100) for training and submission

---

## Architecture

### Shared Backbone

ConvNeXt-Small from `timm`, pretrained on ImageNet, fine-tuned. Input: distorted image at 512×512 (Phase 1) or 768–1024 (Phase 2). Outputs both a spatial feature map and a global average-pooled feature vector.

### Head 1 — Parametric Distortion Regression

MLP on the pooled feature vector. Predicts **9 parameters**:

- **Radial coefficients:** k1, k2, k3, k4 (4 terms covers both standard rectilinear and mild fisheye regimes)
- **Tangential coefficients:** p1, p2
- **Principal point offset:** cx, cy (normalized to [-1, 1])
- **Scale/crop:** s (scalar zoom factor to eliminate black borders)

Initialize this head to predict near-zero values (identity warp). This is critical for stable early training.

### Head 2 — Residual Flow Field

Lightweight decoder (3–4 conv layers with bilinear upsampling) on the spatial feature map. Predicts a 2-channel (dx, dy) displacement field at **64×64 resolution**, bilinearly upsampled to full image resolution. This captures per-lens deviations the parametric model cannot represent.

### Distortion Type Handling

Use 4 radial coefficients (k1–k4) which can approximate both Brown-Conrady rectilinear and mild fisheye. If ablation shows this is insufficient, add a learned soft gate (sigmoid scalar) that blends between Brown-Conrady and equidistant fisheye warp grids. **Start without the gate — add only if needed.**

### Warp Pipeline

1. From predicted parameters, construct a normalized inverse sampling grid (OpenCV-style undistort math, implemented differentiably in PyTorch/Kornia)
2. Add the upsampled residual flow field to the parametric grid
3. `F.grid_sample(input_image, combined_grid, mode='bilinear', padding_mode='reflection')` — **reflection padding is critical** to avoid black-border edge artifacts that directly hurt the 40%-weighted edge similarity metric
4. Apply learned scale/crop factor to ensure output framing matches training targets

---

## Loss Function

Weighted to approximate competition metric proportions:

| Component | Weight | Implementation |
|---|---|---|
| Edge alignment | 0.40 | Multi-scale Sobel magnitude L1 (3 scales via Kornia) |
| Line straightness proxy | 0.20 | Gradient orientation cosine similarity on strong-edge regions |
| Gradient orientation | 0.15 | Gradient direction histogram KL divergence |
| SSIM | 0.15 | `1 - SSIM` via Kornia (`window_size=11`) |
| Photometric | 0.05 | Charbonnier loss (smooth L1 variant, ε=1e-3) |
| Flow regularization | 0.05 | Total variation on residual flow + L2 magnitude penalty |

**Notes:**
- The line straightness proxy (gradient orientation on edge pixels) is differentiable and correlates with Hough-based scoring. The actual Hough metric is used in validation only (non-differentiable).
- Flow regularization prevents the residual head from introducing local artifacts that harm line straightness.
- Photometric loss is intentionally low-weighted — pixel accuracy is only 5% of the competition score.

---

## Training Strategy

### Phase 1 — Parametric Only (50–80 epochs)

- **Resolution:** 512×512
- **Active heads:** Head 1 only. Residual flow head frozen, outputting zero.
- **Losses:** Photometric + SSIM + edge alignment (simplified weighting)
- **Optimizer:** AdamW, LR 1e-4 with cosine annealing
- **Batch size:** 8–16 (tune to GPU memory)
- **Goal:** Learn stable global distortion correction before introducing residual refinement

### Phase 2 — Full Pipeline (30–50 epochs)

- **Resolution:** 768 or 1024 (as high as GPU memory allows)
- **Active heads:** Both. Residual flow head unfrozen with **0.1× LR multiplier**
- **Losses:** Full loss suite with competition-aligned weights (table above)
- **Flow regularization:** Start strong, gradually relax over epochs
- **Batch size:** 4–8 (reduced due to higher resolution)
- **Goal:** Refine edges, handle per-lens deviations, optimize for competition metric

### Phase 3 (optional) — Hard Case Fine-tuning

- Identify highest-loss images from validation set
- Additional 10–20 epochs with upweighted edge loss
- Can use sample weighting to focus on failure cases

### Data Augmentation (all phases)

**Photometric only** via Albumentations:
- Random brightness/contrast
- JPEG compression simulation
- Color jitter / hue-saturation shifts
- Gaussian noise

**Never apply geometric augmentations** (flips, rotations, crops) — these would corrupt the distortion ↔ correction correspondence.

### Mixed Precision

Use `torch.cuda.amp` (autocast + GradScaler) in all phases. Essential for fitting higher resolutions on Kaggle GPUs.

---

## Validation

### Local Proxy Scorer

Build a validation metric that mirrors competition scoring. Compute on a 90/10 train/val split from the ~23k training pairs:

1. **Edge F1 (weight 0.40):** Multi-scale Canny edge detection on output and target, compute precision/recall/F1
2. **Line angle distribution (weight 0.22):** Hough transform on both images, compare angle histograms via KL divergence or Earth Mover's Distance
3. **Gradient orientation similarity (weight 0.18):** Gradient direction histogram comparison
4. **SSIM (weight 0.15):** Standard SSIM computation
5. **Pixel MAE (weight 0.05):** Mean absolute pixel difference

Combine with competition weights to get a single proxy score. **This is your primary optimization target** — don't rely solely on leaderboard submissions.

### Calibration

Submit early to calibrate the gap between proxy score and leaderboard score. Adjust proxy weights if they diverge significantly.

---

## Inference Pipeline

For each of the 1,000 test images:

1. **Forward pass** → parametric parameters (9 values) + residual flow field (H×W×2)
2. **Build combined sampling grid** from parametric warp + upsampled residual flow
3. **Warp** via `F.grid_sample` with `padding_mode='reflection'`
4. **Line refinement gate:**
   - Run Canny + Hough line detection on the warped output
   - If ≥15 detected line segments with mean length above threshold:
     - Run 5–10 iterations of Levenberg-Marquardt optimization on k1, k2, cx, cy, s to minimize line curvature (`scipy.optimize.least_squares`)
     - Re-warp with adjusted parameters
     - **Accept only if proxy score improves** (compare before/after SSIM + edge F1)
   - If fewer lines detected, skip refinement (avoid degrading images with weak geometric cues)
5. **Save** as `{image_id}.jpg`

**No TTA (test-time augmentation)** — geometric transforms change the distortion field, making TTA counterproductive.

---

## Project Structure

```
lens-correction/
├── config.py              # Hyperparams, loss weights, paths, phase configs
├── data/
│   ├── dataset.py         # Training pair loader + Albumentations pipeline
│   └── test_dataset.py    # Test image loader
├── model/
│   ├── backbone.py        # timm ConvNeXt/Swin wrapper
│   ├── param_head.py      # Parametric distortion regression (9 params)
│   ├── flow_head.py       # Residual flow decoder (64×64 → full res)
│   ├── lens_model.py      # Combined model (backbone + both heads)
│   └── warp.py            # Differentiable grid construction + grid_sample
├── losses/
│   ├── edge_loss.py       # Multi-scale Sobel magnitude L1
│   ├── ssim_loss.py       # Kornia SSIM wrapper
│   ├── gradient_loss.py   # Gradient orientation cosine sim + histogram KL
│   ├── flow_reg.py        # TV smoothness + L2 magnitude regularization
│   └── combined.py        # Weighted combination with per-phase configs
├── training/
│   ├── train.py           # Phase 1 & 2 training loops with AMP
│   └── validate.py        # Proxy scorer (edge F1, Hough, SSIM, etc.)
├── inference/
│   ├── predict.py         # Full inference pipeline (batch processing)
│   └── line_refine.py     # Hough detection + LM optimization refinement
├── utils/
│   ├── metrics.py         # Local proxy metric computation
│   └── visualization.py   # Side-by-side comparisons, distortion field plots
└── notebooks/
    └── kaggle_submit.ipynb  # Kaggle GPU execution + submission packaging
```

---

## Risk Mitigations

### Residual Flow Overfitting
If the flow head learns to compensate for backbone errors rather than lens-specific deviations, it produces local artifacts that destroy line straightness.
- **Mitigation:** Strong TV regularization, low-resolution bottleneck (64×64), 0.1× LR multiplier. Ablate with and without — if validation proxy score doesn't improve, drop the residual head entirely.

### Border Artifacts
Black borders from warping create strong artificial edges that directly hurt the 40%-weighted edge similarity.
- **Mitigation:** `padding_mode='reflection'` in `grid_sample` + learned scale factor. Visually inspect output edges on 50+ images before any submission.

### Kaggle GPU Memory Constraints
ConvNeXt-Small fits comfortably. If memory is tight at high resolution:
- **Mitigation:** Drop to ConvNeXt-Tiny or reduce batch size. **Prioritize resolution over batch size** — resolution matters more for edge metrics.

### Metric Mismatch
The local proxy scorer won't perfectly match the competition scorer.
- **Mitigation:** Submit early and often to calibrate. Adjust proxy weights based on observed divergence.

### Line Refinement Instability
LM optimization on line curvature can degrade images with few or ambiguous lines.
- **Mitigation:** Confidence gate (≥15 line segments), before/after comparison, accept only if proxy score improves.

---

## Ablation Checklist

Run these experiments to validate each component contributes positively:

- [ ] Parametric-only (no residual flow) — this is the baseline
- [ ] Parametric + residual flow — should improve on edge F1
- [ ] With vs without line refinement at inference
- [ ] ConvNeXt-Small vs Swin-T backbone
- [ ] 4 vs 3 vs 2 radial coefficients
- [ ] Loss weight sensitivity: double/halve edge loss weight
- [ ] Resolution impact: 512 vs 768 vs 1024 at inference
- [ ] Reflection vs zero padding in grid_sample
- [ ] With vs without learned scale factor (s)

---

## Submission Workflow

1. Train on Kaggle GPU using `kaggle_submit.ipynb`
2. Run inference on all 1,000 test images
3. Zip corrected images (named `{image_id}.jpg`)
4. Upload zip to `bounty.autohdr.com` for scoring
5. Download `submission.csv` from scoring service
6. Submit CSV to Kaggle
