"""
Training loop for Phase 1 (parametric-only) and Phase 2 (full pipeline).

Supports:
- Mixed precision (AMP) throughout
- Cosine annealing LR schedule
- Differential LR for flow head (0.1× multiplier in Phase 2)
- Checkpoint saving/resuming
- Per-epoch validation with proxy scorer
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from config import PhaseConfig
from data.dataset import LensTrainDataset
from model.lens_model import LensCorrectionModel
from losses.combined import CombinedLoss
from training.validate import run_validation


def build_optimizer(model: LensCorrectionModel, phase: PhaseConfig) -> AdamW:
    """Build optimizer with differential LR for flow head."""
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters()
                       if "flow_head" not in n and p.requires_grad],
            "lr": phase.lr,
        },
    ]
    if phase.flow_head_active:
        param_groups.append({
            "params": model.flow_head.parameters(),
            "lr": phase.lr * phase.flow_lr_mult,
        })
    return AdamW(param_groups, weight_decay=1e-4)


def save_checkpoint(
    model: LensCorrectionModel,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    scaler: GradScaler,
    epoch: int,
    best_val_loss: float,
    phase_name: str,
    path: Path,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val_loss": best_val_loss,
        "phase": phase_name,
    }, path)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(
    path: str,
    model: LensCorrectionModel,
    optimizer: AdamW | None = None,
    scheduler: CosineAnnealingLR | None = None,
    scaler: GradScaler | None = None,
) -> tuple[int, float]:
    """Load checkpoint, returns (start_epoch, best_val_loss)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    print(f"  Resumed from {path} (epoch {ckpt['epoch']}, best_val={ckpt['best_val_loss']:.4f})")
    return ckpt["epoch"] + 1, ckpt["best_val_loss"]


def train_one_epoch(
    model: LensCorrectionModel,
    loader: DataLoader,
    criterion: CombinedLoss,
    optimizer: AdamW,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
) -> dict:
    """Train for one epoch. Returns average loss dict."""
    model.train()
    running = {}
    n_batches = 0

    for batch_idx, (distorted, corrected) in enumerate(loader):
        distorted = distorted.to(device)
        corrected = corrected.to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda"):
            pred, params, flow = model(distorted)
            loss, loss_dict = criterion(pred, corrected, flow)

        scaler.scale(loss).backward()

        # Gradient clipping to prevent instability
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        # Accumulate losses
        for k, v in loss_dict.items():
            running[k] = running.get(k, 0.0) + v
        n_batches += 1

        if batch_idx % 50 == 0:
            print(f"  Epoch {epoch} [{batch_idx}/{len(loader)}] loss={loss_dict['total']:.4f}")

    return {k: v / n_batches for k, v in running.items()}


def train_phase(phase: PhaseConfig, device: torch.device, model: LensCorrectionModel | None = None):
    """Run a full training phase."""
    print(f"\n{'='*60}")
    print(f"Starting {phase.name}")
    print(f"  Resolution: {phase.resolution}, Epochs: {phase.epochs}")
    print(f"  Batch size: {phase.batch_size}, LR: {phase.lr}")
    print(f"  Flow head: {'ON' if phase.flow_head_active else 'OFF'}")
    print(f"{'='*60}\n")

    # Model
    if model is None:
        model = LensCorrectionModel(flow_head_active=phase.flow_head_active)
    else:
        model.set_flow_head_active(phase.flow_head_active)
    model = model.to(device)

    # Data
    train_ds = LensTrainDataset(
        resolution=phase.resolution, split="train", augment=True,
    )
    val_ds = LensTrainDataset(
        resolution=phase.resolution, split="val", augment=False,
    )
    train_loader = DataLoader(
        train_ds, batch_size=phase.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=phase.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    print(f"  Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # Loss, optimizer, scheduler, scaler
    criterion = CombinedLoss(phase.loss_weights)
    optimizer = build_optimizer(model, phase)
    scheduler = CosineAnnealingLR(optimizer, T_max=phase.epochs, eta_min=1e-6)
    scaler = GradScaler()

    # Resume if specified
    start_epoch = 0
    best_val_loss = float("inf")
    if phase.resume_from:
        start_epoch, best_val_loss = load_checkpoint(
            phase.resume_from, model, optimizer, scheduler, scaler
        )

    # Training loop
    for epoch in range(start_epoch, phase.epochs):
        t0 = time.time()

        train_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch,
        )
        scheduler.step()

        # Validation
        val_losses = run_validation(model, val_loader, criterion, device)

        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]

        print(
            f"  Epoch {epoch:3d} | "
            f"train_loss={train_losses['total']:.4f} | "
            f"val_loss={val_losses['total']:.4f} | "
            f"lr={lr_now:.2e} | "
            f"{elapsed:.0f}s"
        )

        # Log individual components
        for k in sorted(val_losses):
            if k != "total":
                print(f"    val_{k}: {val_losses[k]:.4f}")

        # Checkpoint best model
        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, best_val_loss,
                phase.name,
                config.CHECKPOINT_DIR / f"{phase.name}_best.pt",
            )

        # Periodic checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, best_val_loss,
                phase.name,
                config.CHECKPOINT_DIR / f"{phase.name}_epoch{epoch}.pt",
            )

    print(f"\n{phase.name} complete. Best val loss: {best_val_loss:.4f}")
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Phase 1: parametric only
    model = train_phase(config.PHASE1, device)

    # Phase 2: full pipeline (reuses model from Phase 1)
    train_phase(config.PHASE2, device, model=model)


if __name__ == "__main__":
    main()
