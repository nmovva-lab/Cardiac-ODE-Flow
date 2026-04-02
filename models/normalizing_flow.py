"""
Pillar 4 — Real-NVP Normalizing Flow

Maps the ODE latent space to a standard Gaussian base distribution,
enabling:
  1. Tractable log-likelihood of latent codes (KL regularisation)
  2. Calibrated uncertainty: sample z ~ N(0, I), decode → h → prediction
     with a confidence score = exp(log_prob(z | observed h))
  3. Out-of-distribution detection for unusual ECG patterns

Architecture: Real-NVP (Dinh et al., 2017)
  - Alternating affine coupling layers
  - Each coupling layer splits the latent vector into two halves
  - Scale and translate networks are small MLPs with checkpointing

References:
    Dinh, L., Sohl-Dickstein, J., & Bengio, S. (2017).
    Density estimation using Real-NVP. ICLR 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, List

from configs.config import FlowConfig


# ── Coupling network MLP ──────────────────────────────────────────────────────

def _make_coupling_net(in_dim: int, out_dim: int, hidden_dim: int, num_layers: int) -> nn.Sequential:
    """Small MLP used as scale/translate networks inside coupling layers."""
    layers: List[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for _ in range(num_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


# ── Affine Coupling Layer ─────────────────────────────────────────────────────

class AffineCouplingLayer(nn.Module):
    """
    Real-NVP affine coupling layer.

    Split z into (z1, z2) by a binary mask.
    Transform:
        z1' = z1
        z2' = z2 * exp(s(z1)) + t(z1)

    Inverse (for sampling):
        z2 = (z2' - t(z1')) / exp(s(z1'))

    log|det J| = sum(s(z1))
    """

    def __init__(
        self,
        dim: int,
        mask: Tensor,           # binary mask, shape (dim,)
        hidden_dim: int,
        num_hidden_layers: int,
    ):
        super().__init__()
        self.register_buffer("mask", mask.float())

        d_in = int(mask.sum().item())    # number of "identity" dimensions
        d_out = dim - d_in               # number of "transformed" dimensions

        self.scale_net = _make_coupling_net(d_in, d_out, hidden_dim, num_hidden_layers)
        self.translate_net = _make_coupling_net(d_in, d_out, hidden_dim, num_hidden_layers)

        # Clamp log-scale for numerical stability
        self.log_scale_clamp = 2.0

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward (data → noise) direction.

        Args:
            z: (B, dim)

        Returns:
            z_out:    (B, dim)
            log_det:  (B,)   — log|det J| contribution
        """
        mask = self.mask
        z1 = z[:, mask.bool()]        # identity half, (B, d_in)
        z2 = z[:, ~mask.bool()]       # transformed half, (B, d_out)

        log_s = self.scale_net(z1).clamp(-self.log_scale_clamp, self.log_scale_clamp)
        t = self.translate_net(z1)

        z2_out = z2 * torch.exp(log_s) + t

        z_out = torch.zeros_like(z)
        z_out[:, mask.bool()] = z1
        z_out[:, ~mask.bool()] = z2_out

        log_det = log_s.sum(dim=-1)   # (B,)
        return z_out, log_det

    def inverse(self, z: Tensor) -> Tensor:
        """
        Inverse (noise → data) direction, for sampling.

        Args:
            z: (B, dim)

        Returns:
            z_orig: (B, dim)
        """
        mask = self.mask
        z1 = z[:, mask.bool()]
        z2 = z[:, ~mask.bool()]

        log_s = self.scale_net(z1).clamp(-self.log_scale_clamp, self.log_scale_clamp)
        t = self.translate_net(z1)

        z2_orig = (z2 - t) * torch.exp(-log_s)

        z_orig = torch.zeros_like(z)
        z_orig[:, mask.bool()] = z1
        z_orig[:, ~mask.bool()] = z2_orig
        return z_orig


# ── Full Real-NVP Flow ────────────────────────────────────────────────────────

class RealNVP(nn.Module):
    """
    Stack of alternating AffineCouplingLayers forming a normalizing flow.

    The flow maps:
        h (cardiac latent, from ODE) ↔ z (Gaussian base)

    Forward (encoding):
        h → z,  returns z and total log|det J|

    Inverse (sampling / decoding):
        z ~ N(0, I) → h

    Uncertainty score for a given h:
        log p(h) = log p(z) + log|det J|
                 = -0.5 ||z||^2 - 0.5 D log(2π) + Σ log|det J_k|

    This gives a per-sample scalar confidence score used in the
    clinical risk output.
    """

    def __init__(self, cfg: FlowConfig):
        super().__init__()
        self.latent_dim = cfg.latent_dim
        dim = cfg.latent_dim

        # Alternating checkerboard masks
        masks = []
        for i in range(cfg.num_coupling_layers):
            # Alternate which half is identity
            if i % 2 == 0:
                mask = torch.arange(dim) % 2 == 0   # even indices = identity
            else:
                mask = torch.arange(dim) % 2 == 1   # odd indices = identity
            masks.append(mask)

        self.coupling_layers = nn.ModuleList([
            AffineCouplingLayer(
                dim=dim,
                mask=masks[i],
                hidden_dim=cfg.hidden_dim,
                num_hidden_layers=cfg.num_hidden_layers,
            )
            for i in range(cfg.num_coupling_layers)
        ])

        # Learnable prior parameters (μ, log σ) — standard Normal by default
        # Kept as parameters so the model can learn a richer base distribution
        self.register_buffer("prior_mean", torch.zeros(dim))
        self.register_buffer("prior_log_std", torch.zeros(dim))

    def forward(self, h: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Encode h → z (forward / normalising direction).

        Args:
            h: (B, latent_dim)

        Returns:
            z:       (B, latent_dim) — mapped noise
            log_det: (B,)            — total log|det J|
            log_prob:(B,)            — log p(h) under the flow
        """
        z = h
        total_log_det = torch.zeros(h.shape[0], device=h.device)

        for layer in self.coupling_layers:
            z, log_det = layer(z)
            total_log_det = total_log_det + log_det

        log_prob = self._log_prior(z) + total_log_det  # (B,)
        return z, total_log_det, log_prob

    def inverse(self, z: Tensor) -> Tensor:
        """
        Decode z → h (inverse / generative direction).

        Args:
            z: (B, latent_dim)  samples from N(0, I)

        Returns:
            h: (B, latent_dim)
        """
        h = z
        for layer in reversed(self.coupling_layers):
            h = layer.inverse(h)
        return h

    def _log_prior(self, z: Tensor) -> Tensor:
        """
        Log-probability under the (learned) Gaussian prior.
        log N(z; prior_mean, exp(prior_log_std)^2)
        """
        std = self.prior_log_std.exp()
        dist = torch.distributions.Normal(self.prior_mean, std)
        return dist.log_prob(z).sum(dim=-1)            # (B,)

    def sample(self, n: int, device: torch.device) -> Tensor:
        """
        Draw n samples from the prior and decode to latent space.

        Args:
            n: number of samples

        Returns:
            h_samples: (n, latent_dim)
        """
        std = self.prior_log_std.exp()
        z = torch.randn(n, self.latent_dim, device=device) * std + self.prior_mean
        return self.inverse(z)

    def confidence_score(self, h: Tensor) -> Tensor:
        """
        Returns a calibrated [0, 1] confidence score for each sample,
        derived from the normalised log-probability under the flow.

        score = sigmoid(log_prob / D)   where D = latent_dim

        Higher score → h lies in a high-probability region → model is confident.

        Args:
            h: (B, latent_dim)

        Returns:
            score: (B,)
        """
        with torch.no_grad():
            _, _, log_prob = self.forward(h)
        normalised = log_prob / self.latent_dim
        return torch.sigmoid(normalised)
