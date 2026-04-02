"""
Hybrid Loss Function

L_total = w_bce * L_BCE + w_kl(t) * L_KL

  L_BCE: Binary Cross-Entropy over multi-label diagnostic classes
         with optional class frequency re-weighting.

  L_KL:  KL divergence between the flow's latent distribution q(z|x)
         and the standard Normal prior p(z) = N(0, I).
         Estimated via the flow's log_prob output:
             KL ≈ -log p(z|x) = -log_prob   (ELBO lower-bound)

  w_kl(t): Linear warmup from 0 → w_kl over kl_anneal_epochs.
           This prevents posterior collapse during early training
           (a common failure mode in VAE-style models).

Additionally we log:
  - Per-class AUROC (deferred to evaluation; this module only computes losses)
  - Flow negative log-likelihood as a standalone diagnostic metric
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Optional, Tuple


class HybridLoss(nn.Module):
    """
    Args:
        bce_weight:       scalar multiplier on BCE loss
        kl_weight:        target scalar multiplier on KL loss (after warmup)
        kl_anneal_epochs: number of epochs for linear KL warmup
        class_weights:    (num_classes,) optional frequency-based re-weighting
        label_smoothing:  smoothing ε for BCE targets (reduces over-confidence)
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        kl_weight: float = 0.1,
        kl_anneal_epochs: int = 20,
        class_weights: Optional[Tensor] = None,
        label_smoothing: float = 0.01,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.kl_weight_target = kl_weight
        self.kl_anneal_epochs = kl_anneal_epochs
        self.label_smoothing = label_smoothing

        # Register class weights as buffer so they move with .to(device)
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

        # Track current epoch for KL annealing (updated externally)
        self._current_epoch: int = 0

    def set_epoch(self, epoch: int):
        """Call at the start of each epoch to update KL weight."""
        self._current_epoch = epoch

    @property
    def kl_weight(self) -> float:
        """Current annealed KL weight."""
        if self.kl_anneal_epochs <= 0:
            return self.kl_weight_target
        progress = min(1.0, self._current_epoch / self.kl_anneal_epochs)
        return self.kl_weight_target * progress

    def bce_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Multi-label BCE with optional label smoothing and class weighting.

        Args:
            logits:  (B, C) raw logits
            targets: (B, C) binary labels in {0, 1}

        Returns:
            scalar loss
        """
        # Label smoothing: push labels toward (ε, 1-ε)
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        if self.class_weights is not None:
            loss = F.binary_cross_entropy_with_logits(
                logits, targets,
                pos_weight=self.class_weights,
                reduction="none",
            )  # (B, C)
            return loss.mean()
        else:
            return F.binary_cross_entropy_with_logits(logits, targets)

    def kl_loss(self, log_prob: Tensor) -> Tensor:
        """
        KL divergence approximated via flow negative log-likelihood.

        Under the flow:
            KL[q(z|x) || p(z)] ≈ E_q[-log p(z|x)]
                                 = -E[log_prob]

        Args:
            log_prob: (B,) — flow log p(h) for each sample

        Returns:
            scalar KL estimate
        """
        return -log_prob.mean()

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        log_prob: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Args:
            logits:   (B, num_classes)
            targets:  (B, num_classes) float binary labels
            log_prob: (B,) from normalizing flow

        Returns:
            total_loss: scalar
            metrics:    dict of named loss components for logging
        """
        l_bce = self.bce_loss(logits, targets)
        l_kl = self.kl_loss(log_prob)

        total = self.bce_weight * l_bce + self.kl_weight * l_kl

        metrics = {
            "loss/total": total.item(),
            "loss/bce": l_bce.item(),
            "loss/kl": l_kl.item(),
            "loss/kl_weight": self.kl_weight,
        }
        return total, metrics


# ── Class weight computation ──────────────────────────────────────────────────

def compute_class_weights(labels: Tensor, beta: float = 0.999) -> Tensor:
    """
    Effective number of samples re-weighting (Cui et al., 2019).
    Reduces the penalty for over-represented classes.

    Args:
        labels: (N, C) binary label matrix over the full training set
        beta:   smoothing factor (0.9, 0.99, 0.999, or 1.0 = inverse freq)

    Returns:
        weights: (C,) one weight per class
    """
    class_counts = labels.sum(dim=0).clamp(min=1)     # (C,)
    effective_num = 1.0 - beta ** class_counts
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum() * labels.shape[1]  # normalise to C
    return weights
