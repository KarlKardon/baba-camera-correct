#!/bin/bash
# Training entrypoint for Akash deployment.
#
# Expected environment variables (set in SDL):
#   DATA_URL            — direct download URL for the training data zip
#                         (upload the Kaggle zip you already have locally to
#                          Google Drive / S3 / Dropbox and paste the link here)
#   PHASE               — "1", "2", or "both" (default: both)
#   PHASE1_EPOCHS       — override Phase 1 epochs (optional)
#   PHASE2_EPOCHS       — override Phase 2 epochs (optional)
#   PHASE1_BATCH        — override Phase 1 batch size (optional)
#   PHASE2_BATCH        — override Phase 2 batch size (optional)
#   PHASE2_RESOLUTION   — override Phase 2 resolution (optional)
#
# Checkpoints and submission zip are written to /app/outputs/.
# A lightweight HTTP server serves /app/outputs/ on port 8080 for download.

set -euo pipefail

echo "========================================"
echo " Lens Correction Training — Akash Node"
echo "========================================"
date

# ── 1. Download data ───────────────────────────────────────────────────────────
DATA_DIR=/app/data_raw
mkdir -p "$DATA_DIR"

if [ ! -d "$DATA_DIR/lens-correction-train-clea" ]; then
    if [ -z "${DATA_URL:-}" ]; then
        echo "ERROR: DATA_URL is not set. Set it to a direct download link for the dataset zip."
        exit 1
    fi
    echo "[1/5] Downloading dataset from DATA_URL (~40 GB)..."
    curl -L "$DATA_URL" -o "$DATA_DIR/dataset.zip"
    echo "Unzipping..."
    unzip -q "$DATA_DIR/dataset.zip" -d "$DATA_DIR"
    rm "$DATA_DIR/dataset.zip"
    echo "[1/5] Data downloaded."
else
    echo "[1/5] Data already present, skipping download."
fi

# ── 3. Apply config overrides ──────────────────────────────────────────────────
OVERRIDE_SCRIPT=/tmp/apply_overrides.py
cat > "$OVERRIDE_SCRIPT" <<'PYEOF'
import config, sys

if "PHASE1_EPOCHS" in __import__("os").environ:
    config.PHASE1.epochs = int(__import__("os").environ["PHASE1_EPOCHS"])
if "PHASE2_EPOCHS" in __import__("os").environ:
    config.PHASE2.epochs = int(__import__("os").environ["PHASE2_EPOCHS"])
if "PHASE1_BATCH" in __import__("os").environ:
    config.PHASE1.batch_size = int(__import__("os").environ["PHASE1_BATCH"])
if "PHASE2_BATCH" in __import__("os").environ:
    config.PHASE2.batch_size = int(__import__("os").environ["PHASE2_BATCH"])
if "PHASE2_RESOLUTION" in __import__("os").environ:
    config.PHASE2.resolution = int(__import__("os").environ["PHASE2_RESOLUTION"])

print(f"Phase 1: {config.PHASE1.epochs} epochs, batch={config.PHASE1.batch_size}, res={config.PHASE1.resolution}")
print(f"Phase 2: {config.PHASE2.epochs} epochs, batch={config.PHASE2.batch_size}, res={config.PHASE2.resolution}")
PYEOF
echo "[2/4] Config:"
python "$OVERRIDE_SCRIPT"

# ── 3. Train ───────────────────────────────────────────────────────────────────
PHASE="${PHASE:-both}"
echo "[3/4] Starting training (PHASE=$PHASE)..."
python - <<'PYEOF'
import os, sys, torch
sys.path.insert(0, "/app")
import config
from training.train import train_phase

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

phase_arg = os.environ.get("PHASE", "both")

model = None
if phase_arg in ("1", "both"):
    model = train_phase(config.PHASE1, device)

if phase_arg in ("2", "both"):
    ckpt = str(config.CHECKPOINT_DIR / "phase1_parametric_best.pt")
    if phase_arg == "2" and os.path.exists(ckpt):
        config.PHASE2.resume_from = ckpt
    train_phase(config.PHASE2, device, model=model)

print("Training complete.")
PYEOF

# ── 4. Run inference & package submission ─────────────────────────────────────
echo "[4/4] Running inference on test images..."
python inference/predict.py \
    --checkpoint outputs/checkpoints/phase2_full_best.pt \
    --output_dir outputs/submissions/final \
    --resolution "${PHASE2_RESOLUTION:-768}" \
    --refine \
    --batch_size 4

echo "Packaging submission zip..."
python - <<'PYEOF'
import zipfile
from pathlib import Path

sub_dir = Path("outputs/submissions/final")
zip_path = Path("outputs/submissions/submission.zip")
files = sorted(sub_dir.glob("*.jpg"))
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
    for f in files:
        zf.write(f, f.name)
print(f"Created {zip_path} ({zip_path.stat().st_size/1e6:.1f} MB, {len(files)} images)")
PYEOF

echo ""
echo "========================================"
echo " Training Done! Serving outputs on :8080"
echo " Download: http://<akash-host>:8080/"
echo "========================================"

# Serve outputs directory so you can download checkpoints + submission zip
cd /app/outputs
python -m http.server 8080
