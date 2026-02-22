"""
Hyperparameters, paths, and per-phase training configs for lens distortion correction.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Paths ──────────────────────────────────────────────────────────────────────

# Allow overriding dataset locations via environment variables so the 40GB
# bucket/mount path can be pasted without code changes. Defaults remain local.
DATA_ROOT = Path(os.getenv("DATA_ROOT", "data_raw"))
TRAIN_DIR = Path(os.getenv("TRAIN_DIR", DATA_ROOT / "lens-correction-train-cleaned"))
TEST_DIR = Path(os.getenv("TEST_DIR", DATA_ROOT / "test-originals"))
OUTPUT_DIR = Path("outputs")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
SUBMISSION_DIR = OUTPUT_DIR / "submissions"

# ── Model ──────────────────────────────────────────────────────────────────────

BACKBONE = "convnext_small.fb_in22k_ft_in1k"  # timm model name
NUM_DISTORTION_PARAMS = 9  # k1 k2 k3 k4 p1 p2 cx cy s
FLOW_RESOLUTION = 64       # residual flow field spatial size

# ── Loss weights — per-phase ───────────────────────────────────────────────────

@dataclass
class LossWeights:
    edge: float = 0.40
    line_proxy: float = 0.20
    gradient_orientation: float = 0.15
    ssim: float = 0.15
    photometric: float = 0.05
    flow_reg: float = 0.05


PHASE1_LOSS = LossWeights(
    edge=0.0,             # no edge loss until backbone has learned stable warp
    line_proxy=0.0,
    gradient_orientation=0.0,
    ssim=0.50,
    photometric=0.50,
    flow_reg=0.0,         # no flow head active
)

PHASE2_LOSS = LossWeights()  # full competition-aligned weights

# ── Training ───────────────────────────────────────────────────────────────────

@dataclass
class PhaseConfig:
    name: str
    resolution: int
    epochs: int
    batch_size: int
    lr: float
    flow_head_active: bool
    flow_lr_mult: float
    loss_weights: LossWeights
    resume_from: Optional[str] = None

PHASE1 = PhaseConfig(
    name="phase1_parametric",
    resolution=512,
    epochs=60,
    batch_size=8,
    lr=1e-4,
    flow_head_active=False,
    flow_lr_mult=0.0,
    loss_weights=PHASE1_LOSS,
)

PHASE2 = PhaseConfig(
    name="phase2_full",
    resolution=768,
    epochs=40,
    batch_size=4,
    lr=3e-5,
    flow_head_active=True,
    flow_lr_mult=0.1,
    loss_weights=PHASE2_LOSS,
)

# ── Validation ─────────────────────────────────────────────────────────────────

VAL_SPLIT = 0.10  # fraction of training set held out
SEED = 42

# ── Augmentation ───────────────────────────────────────────────────────────────

AUG_BRIGHTNESS_LIMIT = 0.2
AUG_CONTRAST_LIMIT = 0.2
AUG_JPEG_QUALITY_RANGE = (60, 100)
AUG_HUE_SHIFT = 10
AUG_SAT_SHIFT = 20
AUG_NOISE_VAR_LIMIT = (5.0, 25.0)

# Synthetic distortion augmentation: apply random Brown-Conrady warp to GT
# images to generate additional (distorted, GT) training pairs on-the-fly.
SYNTH_AUG_PROB = 0.5               # fraction of training samples to augment
SYNTH_K1_RANGE = (-0.4, 0.4)       # primary radial coefficient
SYNTH_K2_RANGE = (-0.15, 0.15)     # secondary radial coefficient
SYNTH_CENTER_RANGE = (-0.05, 0.05) # principal point offset (normalized)

# ── Inference ──────────────────────────────────────────────────────────────────

LINE_REFINE_MIN_SEGMENTS = 15
LINE_REFINE_LM_ITERS = 10
