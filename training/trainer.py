"""
Training Loop

Full training + evaluation pipeline with:
  - Gradient clipping and LR warmup
  - KL annealing (passed to HybridLoss each epoch)
  - TensorBoard logging (loss curves, AUROC, confidence histograms)
  - Checkpoint management (save best-k by val AUROC, resume support)
  - Apple Silicon (MPS) compatible device handling

Run:
    python -m training.trainer
"""

import os
import time
import warnings
import json
import dataclasses
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    _SKLEARN = True
except ImportError:
    warnings.warn("sklearn not installed. AUROC will not be computed. pip install scikit-learn")
    _SKLEARN = False

import numpy as np

from configs.config import Config, get_config
from models.model import CardioODEFlow
from training.loss import HybridLoss, compute_class_weights
from data.dataset import build_dataloaders, SUPER_CLASSES


# ── Device selection ──────────────────────────────────────────────────────────

def get_device(cfg: Config) -> torch.device:
    if cfg.device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    return torch.device(cfg.device)


# ── Checkpoint helpers ────────────────────────────────────────────────────────

class CheckpointManager:
    """Saves the best-k checkpoints by validation AUROC."""

    def __init__(self, ckpt_dir: str, keep_k: int = 3):
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.keep_k = keep_k
        self.history: list = []   # list of (auroc, path)

    def save(
        self,
        model: nn.Module,
        optimizer,
        scheduler,
        epoch: int,
        val_auroc: float,
        cfg: Config,
    ) -> Path:
        path = self.ckpt_dir / f"epoch_{epoch:04d}_auroc_{val_auroc:.4f}.pt"
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "val_auroc": val_auroc,
        }, path)

        # Save config snapshot next to checkpoint for full reproducibility
        config_path = path.with_suffix(".config.json")
        with open(config_path, "w") as f:
            json.dump(dataclasses.asdict(cfg), f, indent=2)

        self.history.append((val_auroc, path))
        self.history.sort(key=lambda x: x[0], reverse=True)

        # Remove excess checkpoints and their companion config JSONs
        while len(self.history) > self.keep_k:
            _, old_path = self.history.pop()
            if old_path.exists():
                old_path.unlink()
            old_config = old_path.with_suffix(".config.json")
            if old_config.exists():
                old_config.unlink()
            print(f"  Removed old checkpoint: {old_path.name}")

        print(f"  Saved checkpoint: {path.name}  (AUROC={val_auroc:.4f})")
        return path

    def best_path(self) -> Optional[Path]:
        if self.history:
            return self.history[0][1]
        return None

    @staticmethod
    def load(path: str, model: nn.Module, optimizer=None, scheduler=None, device=None):
        ckpt = torch.load(path, map_location=device or "cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        if optimizer and "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if scheduler and ckpt.get("scheduler_state"):
            scheduler.load_state_dict(ckpt["scheduler_state"])
        print(f"Loaded checkpoint from epoch {ckpt['epoch']} (AUROC={ckpt.get('val_auroc', '?'):.4f})")
        return ckpt["epoch"]


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: CardioODEFlow,
    loader,
    loss_fn: HybridLoss,
    device: torch.device,
) -> Dict[str, float]:
    """
    Full evaluation pass.

    Returns dict with: loss/*, auroc/*, auprc/*, confidence_mean
    """
    model.eval()
    all_probs, all_labels = [], []
    all_confidences = []
    total_loss, total_bce, total_kl, n_batches = 0.0, 0.0, 0.0, 0

    for batch in loader:
        ecg = batch["ecg"].to(device)
        labels = batch["label"].to(device)
        age = batch["age"].to(device)
        sex = batch["sex"].to(device)

        out = model(ecg, age, sex)
        loss, metrics = loss_fn(out["logits"], labels, out["log_prob"])

        total_loss += metrics["loss/total"]
        total_bce += metrics["loss/bce"]
        total_kl += metrics["loss/kl"]
        n_batches += 1

        all_probs.append(out["probs"].cpu().float().numpy())
        all_labels.append(labels.cpu().float().numpy())
        all_confidences.append(out["confidence"].cpu().float().numpy())

    all_probs = np.concatenate(all_probs, axis=0)      # (N, C)
    all_labels = np.concatenate(all_labels, axis=0)    # (N, C)
    all_conf = np.concatenate(all_confidences, axis=0) # (N,)

    results: Dict[str, float] = {
        "loss/total": total_loss / max(n_batches, 1),
        "loss/bce": total_bce / max(n_batches, 1),
        "loss/kl": total_kl / max(n_batches, 1),
        "confidence_mean": float(all_conf.mean()),
    }

    # Per-class and macro AUROC / AUPRC
    if _SKLEARN:
        aurocs, auprcs = [], []
        for c, cls_name in enumerate(SUPER_CLASSES):
            n_pos = all_labels[:, c].sum()
            if n_pos > 0 and n_pos < len(all_labels):   # guard all-zero AND all-positive
                try:
                    auc = roc_auc_score(all_labels[:, c], all_probs[:, c])
                    ap = average_precision_score(all_labels[:, c], all_probs[:, c])
                    results[f"auroc/{cls_name}"] = auc
                    results[f"auprc/{cls_name}"] = ap
                    aurocs.append(auc)
                    auprcs.append(ap)
                except Exception:
                    pass
        if aurocs:
            results["auroc/macro"] = float(np.mean(aurocs))
            results["auprc/macro"] = float(np.mean(auprcs))

    model.train()   # restore training mode after evaluation
    return results


# ── Training step ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model: CardioODEFlow,
    loader,
    optimizer: torch.optim.Optimizer,
    loss_fn: HybridLoss,
    device: torch.device,
    cfg: Config,
    writer: Optional[SummaryWriter],
    global_step: int,
) -> Tuple[Dict[str, float], int]:
    """
    Single epoch training pass.

    Returns (epoch_metrics, updated_global_step).
    """
    model.train()
    epoch_loss, epoch_bce, epoch_kl, n_batches = 0.0, 0.0, 0.0, 0

    for batch_idx, batch in enumerate(loader):
        ecg = batch["ecg"].to(device)
        labels = batch["label"].to(device)
        age = batch["age"].to(device)
        sex = batch["sex"].to(device)

        optimizer.zero_grad(set_to_none=True)

        out = model(ecg, age, sex)
        loss, metrics = loss_fn(out["logits"], labels, out["log_prob"])

        loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=cfg.training.grad_clip_norm,
        )

        optimizer.step()
        global_step += 1

        epoch_loss += metrics["loss/total"]
        epoch_bce += metrics["loss/bce"]
        epoch_kl += metrics["loss/kl"]
        n_batches += 1

        # Log every N steps
        if writer and global_step % cfg.training.log_every_n_steps == 0:
            for k, v in metrics.items():
                writer.add_scalar(f"train/{k}", v, global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

        if batch_idx % 50 == 0:
            print(
                f"    step {batch_idx:4d}/{len(loader)}  "
                f"loss={metrics['loss/total']:.4f}  "
                f"bce={metrics['loss/bce']:.4f}  "
                f"kl={metrics['loss/kl']:.4f}"
            )

    return {
        "loss/total": epoch_loss / max(n_batches, 1),
        "loss/bce": epoch_bce / max(n_batches, 1),
        "loss/kl": epoch_kl / max(n_batches, 1),
    }, global_step


# ── Main training entry point ─────────────────────────────────────────────────

def train(cfg: Optional[Config] = None, resume_from: Optional[str] = None):
    if cfg is None:
        cfg = get_config()

    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)

    device = get_device(cfg)
    print(f"\nDevice: {device}")
    print(f"ODE solver: CPU (torchdiffeq MPS workaround)\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Loading datasets...")
    loaders = build_dataloaders(cfg.data, seed=cfg.training.seed)
    train_loader = loaders.get("train")
    val_loader = loaders.get("val")

    if train_loader is None:
        raise RuntimeError("No training data loaded. Check your dataset paths in DataConfig.")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = CardioODEFlow(cfg).to(device)
    param_counts = model.count_parameters()
    print(f"\nModel parameters:")
    for k, v in param_counts.items():
        print(f"  {k:25s}: {v:>10,}")

    # ── Loss ──────────────────────────────────────────────────────────────────
    loss_fn = HybridLoss(
        bce_weight=cfg.training.bce_weight,
        kl_weight=cfg.training.kl_weight,
        kl_anneal_epochs=cfg.training.kl_anneal_epochs,
    )

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    # ── LR Scheduler: linear warmup → cosine decay ────────────────────────────
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=cfg.training.warmup_epochs,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.training.epochs - cfg.training.warmup_epochs,
        eta_min=cfg.training.min_lr,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[cfg.training.warmup_epochs],
    )

    # ── Checkpoint manager ────────────────────────────────────────────────────
    ckpt_manager = CheckpointManager(
        cfg.training.checkpoint_dir,
        keep_k=cfg.training.keep_best_k,
    )

    start_epoch = 0
    if resume_from:
        start_epoch = CheckpointManager.load(
            resume_from, model, optimizer, scheduler, device
        )

    # ── TensorBoard ───────────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=cfg.training.log_dir)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_auroc = 0.0
    global_step = start_epoch * (len(train_loader) if train_loader else 1)

    print(f"\nStarting training for {cfg.training.epochs} epochs...\n")

    for epoch in range(start_epoch, cfg.training.epochs):
        t0 = time.time()
        loss_fn.set_epoch(epoch)

        print(f"Epoch {epoch + 1}/{cfg.training.epochs}  (KL weight={loss_fn.kl_weight:.4f})")

        # Train
        train_metrics, global_step = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, cfg, writer, global_step
        )

        # Scheduler step
        scheduler.step()

        # Log training metrics
        for k, v in train_metrics.items():
            writer.add_scalar(f"train_epoch/{k}", v, epoch)

        # Validate
        val_auroc = 0.0
        if val_loader:
            val_metrics = evaluate(model, val_loader, loss_fn, device)
            val_auroc = val_metrics.get("auroc/macro", 0.0)

            for k, v in val_metrics.items():
                writer.add_scalar(f"val/{k}", v, epoch)

            elapsed = time.time() - t0
            print(
                f"  → val loss={val_metrics['loss/total']:.4f}  "
                f"AUROC={val_auroc:.4f}  "
                f"conf={val_metrics.get('confidence_mean', 0):.3f}  "
                f"time={elapsed:.1f}s"
            )

        # Checkpoint
        if (epoch + 1) % cfg.training.save_every_n_epochs == 0 or val_auroc > best_val_auroc:
            ckpt_manager.save(model, optimizer, scheduler, epoch + 1, val_auroc, cfg)
            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                print(f"  ✓ New best AUROC: {best_val_auroc:.4f}")

        print()

    writer.close()
    print(f"\nTraining complete. Best val AUROC: {best_val_auroc:.4f}")
    print(f"Best checkpoint: {ckpt_manager.best_path()}")
    return model, ckpt_manager.best_path()


if __name__ == "__main__":
    train()
