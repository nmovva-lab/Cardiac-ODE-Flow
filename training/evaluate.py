"""
Evaluation Script

Comprehensive evaluation of a trained Cardio-ODE-Flow model on the test set.

Produces:
  1. Per-class and macro AUROC / AUPRC
  2. Calibration curve (ECE — Expected Calibration Error)
  3. Confidence score analysis (high-conf vs low-conf stratification)
  4. ODE trajectory visualisation (optional)
  5. JSON results file for inclusion in paper tables

Run:
    python -m training.evaluate --checkpoint checkpoints/epoch_0050_auroc_0.8900.pt
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL = True
except ImportError:
    _MPL = False
    warnings.warn("matplotlib not installed. Plots will not be generated.")

try:
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        roc_curve, precision_recall_curve,
        brier_score_loss,
    )
    from sklearn.calibration import calibration_curve
    _SKLEARN = True
except ImportError:
    _SKLEARN = False

from configs.config import get_config
from models.model import CardioODEFlow
from training.trainer import get_device, CheckpointManager, evaluate
from data.dataset import build_dataloaders, SUPER_CLASSES


# ── Expected Calibration Error ────────────────────────────────────────────────

def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    ECE over all classes (flattened).
    Lower is better (0 = perfect calibration).
    """
    probs_flat = probs.flatten()
    labels_flat = labels.flatten()
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs_flat)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (probs_flat >= lo) & (probs_flat < hi)
        if mask.sum() == 0:
            continue
        conf = probs_flat[mask].mean()
        acc = labels_flat[mask].mean()
        ece += (mask.sum() / n) * abs(conf - acc)

    return float(ece)


# ── Confidence stratification ─────────────────────────────────────────────────

def confidence_stratification(
    probs: np.ndarray,
    labels: np.ndarray,
    confidences: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compare AUROC on high-confidence vs low-confidence samples.
    Demonstrates that the model's uncertainty is informative.
    """
    high_mask = confidences >= threshold
    low_mask = ~high_mask

    results: Dict[str, float] = {
        "n_high_conf": int(high_mask.sum()),
        "n_low_conf": int(low_mask.sum()),
        "confidence_threshold": threshold,
    }

    if _SKLEARN:
        for name, mask in [("high_conf", high_mask), ("low_conf", low_mask)]:
            if mask.sum() > 10:
                try:
                    auroc = roc_auc_score(labels[mask], probs[mask], average="macro")
                    results[f"auroc_{name}"] = float(auroc)
                except Exception:
                    pass

    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_roc_curves(
    probs: np.ndarray,
    labels: np.ndarray,
    save_dir: Path,
):
    if not _MPL or not _SKLEARN:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    for c, cls_name in enumerate(SUPER_CLASSES):
        if labels[:, c].sum() > 0:
            fpr, tpr, _ = roc_curve(labels[:, c], probs[:, c])
            auc = roc_auc_score(labels[:, c], probs[:, c])
            ax.plot(fpr, tpr, label=f"{cls_name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Cardio-ODE-Flow")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = save_dir / "roc_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_calibration(
    probs: np.ndarray,
    labels: np.ndarray,
    save_dir: Path,
):
    if not _MPL or not _SKLEARN:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    for c, cls_name in enumerate(SUPER_CLASSES):
        if labels[:, c].sum() > 0:
            try:
                frac_pos, mean_pred = calibration_curve(
                    labels[:, c], probs[:, c], n_bins=10, strategy="quantile"
                )
                ax.plot(mean_pred, frac_pos, marker="o", markersize=4, label=cls_name)
            except Exception:
                pass
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curves — Cardio-ODE-Flow")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = save_dir / "calibration.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_confidence_histogram(
    confidences: np.ndarray,
    labels: np.ndarray,
    save_dir: Path,
):
    if not _MPL:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(confidences, bins=40, alpha=0.7, color="steelblue", label="All samples")
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Score Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = save_dir / "confidence_histogram.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def _compute_metrics(probs: np.ndarray, labels: np.ndarray) -> Dict:
    """Compute AUROC/AUPRC/Brier per class and macro."""
    if not _SKLEARN:
        return {}
    results: Dict = {}
    aurocs, auprcs, briers = [], [], []
    for c, cls_name in enumerate(SUPER_CLASSES):
        if labels[:, c].sum() > 0 and labels[:, c].sum() < len(labels):
            try:
                auc = roc_auc_score(labels[:, c], probs[:, c])
                ap = average_precision_score(labels[:, c], probs[:, c])
                brier = brier_score_loss(labels[:, c], probs[:, c])
                results[cls_name] = {"auroc": auc, "auprc": ap, "brier": brier}
                aurocs.append(auc)
                auprcs.append(ap)
                briers.append(brier)
            except Exception:
                pass
    if aurocs:
        results["macro"] = {
            "auroc": float(np.mean(aurocs)),
            "auprc": float(np.mean(auprcs)),
            "brier": float(np.mean(briers)),
        }
    return results


# ── Main evaluation ───────────────────────────────────────────────────────────

@torch.no_grad()
def full_evaluation(
    checkpoint_path: str,
    output_dir: str = "eval_results",
    cfg=None,
) -> Dict:
    if cfg is None:
        cfg = get_config()

    device = get_device(cfg)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = CardioODEFlow(cfg).to(device)
    CheckpointManager.load(checkpoint_path, model, device=device)
    model.eval()

    print("\nLoading test set...")
    loaders = build_dataloaders(cfg.data)
    test_loader = loaders.get("test")
    if test_loader is None:
        raise RuntimeError("No test data available.")

    # Collect outputs, tracking source dataset per sample for per-dataset breakdown
    all_probs, all_labels, all_confidences, all_sources = [], [], [], []

    for batch in test_loader:
        ecg = batch["ecg"].to(device)
        labels = batch["label"].to(device)
        age = batch["age"].to(device)
        sex = batch["sex"].to(device)
        sources = batch["source"]   # list of strings

        out = model(ecg, age, sex)
        all_probs.append(out["probs"].cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_confidences.append(out["confidence"].cpu().numpy())
        all_sources.extend(sources)

    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    confidences = np.concatenate(all_confidences, axis=0)
    sources = np.array(all_sources)

    print(f"\nTest set: {len(probs)} samples")

    results: Dict = {"n_test": int(len(probs)), "classes": SUPER_CLASSES}

    # ── Pooled metrics ────────────────────────────────────────────────────────
    pooled = _compute_metrics(probs, labels)
    results["pooled"] = pooled
    if "macro" in pooled:
        print(f"\n  [Pooled]  Macro AUROC={pooled['macro']['auroc']:.4f}  "
              f"AUPRC={pooled['macro']['auprc']:.4f}")
    for cls_name in SUPER_CLASSES:
        if cls_name in pooled:
            m = pooled[cls_name]
            print(f"    {cls_name:6s}  AUROC={m['auroc']:.4f}  "
                  f"AUPRC={m['auprc']:.4f}  Brier={m['brier']:.4f}")

    # ── Per-dataset breakdown ─────────────────────────────────────────────────
    results["per_dataset"] = {}
    for ds_name in sorted(set(all_sources)):
        mask = sources == ds_name
        if mask.sum() < 10:
            continue
        ds_metrics = _compute_metrics(probs[mask], labels[mask])
        results["per_dataset"][ds_name] = ds_metrics
        macro = ds_metrics.get("macro", {})
        print(f"\n  [{ds_name}]  N={mask.sum()}  "
              f"Macro AUROC={macro.get('auroc', float('nan')):.4f}  "
              f"AUPRC={macro.get('auprc', float('nan')):.4f}")

    # ── Calibration ───────────────────────────────────────────────────────────
    results["ece"] = expected_calibration_error(probs, labels)
    results["confidence_stats"] = {
        "mean": float(confidences.mean()),
        "std": float(confidences.std()),
        "min": float(confidences.min()),
        "max": float(confidences.max()),
    }
    results["confidence_stratification"] = confidence_stratification(
        probs, labels, confidences, threshold=0.5
    )

    print(f"\n  ECE:         {results['ece']:.4f}")
    print(f"  Conf mean:   {results['confidence_stats']['mean']:.4f}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    json_path = out_dir / "test_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {json_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_roc_curves(probs, labels, out_dir)
    plot_calibration(probs, labels, out_dir)
    plot_confidence_histogram(confidences, labels, out_dir)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="eval_results")
    args = parser.parse_args()
    full_evaluation(args.checkpoint, args.output_dir)
