"""
Pillar 3 — Neural Ordinary Differential Equation (Neural ODE)

Evolves the latent cardiac state h(t) continuously over time:

    dh/dt = f_θ(h(t), t, context)

where:
  h(t)    ∈ R^{latent_dim}  — cardiac state at time t
  context ∈ R^{context_dim} — patient-specific embedding (age, sex)
  f_θ                       — learned MLP (the ODE function)

Solved with torchdiffeq's `odeint` (or `odeint_adjoint` for
memory-efficient backprop via the adjoint method).

Key benefits over a standard RNN:
  ✓ Handles irregular / missing timesteps naturally
  ✓ Adjoint method: O(1) memory in sequence length
  ✓ Adaptive step-size solver (dopri5) adjusts to signal complexity
  ✓ Continuous-time interpolation of latent state
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional

try:
    from torchdiffeq import odeint_adjoint as odeint
    _ADJOINT = True
except ImportError:
    try:
        from torchdiffeq import odeint
        _ADJOINT = False
    except ImportError:
        raise ImportError(
            "torchdiffeq is required. Install with:\n"
            "  pip install torchdiffeq"
        )

from configs.config import ODEConfig


# ── Context encoder (age + sex → embedding) ───────────────────────────────────

class ContextEncoder(nn.Module):
    """
    Encodes scalar patient covariates into a fixed-size context vector
    injected into every ODE function evaluation.

    Input:
        age  (B,)  — normalised to [0, 1]
        sex  (B,)  — 0 = female, 1 = male (float)

    Output:
        context (B, context_dim)
    """

    def __init__(self, context_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.SiLU(),
            nn.Linear(32, context_dim),
            nn.LayerNorm(context_dim),
        )

    def forward(self, age: Tensor, sex: Tensor) -> Tensor:
        covariates = torch.stack([age, sex], dim=-1).float()  # (B, 2)
        return self.net(covariates)                           # (B, context_dim)


# ── ODE dynamics function f_θ ─────────────────────────────────────────────────

class ODEFunc(nn.Module):
    """
    The learnable right-hand side of the ODE: dh/dt = f_θ(h, t, context).

    Architecture: Augmented Neural ODE with time and context conditioning.
      1. Concatenate [h, t_embed, context] → augmented state
      2. Pass through MLP with residual connections and layer norm
      3. Output has same dim as h (no dimensionality change)

    Concats time as a scalar feature (sinusoidal embedding for
    better representation of the temporal position).
    """

    def __init__(self, cfg: ODEConfig):
        super().__init__()
        self.latent_dim = cfg.latent_dim
        self.context_dim = cfg.context_dim

        # Time embedding: scalar t → 16-dim sinusoidal
        self.time_embed_dim = 16
        in_dim = cfg.latent_dim + self.time_embed_dim + cfg.context_dim

        # MLP layers with gated linear units for smooth dynamics
        layers = []
        prev_dim = in_dim
        for _ in range(3):                         # depth 3 by default
            layers += [
                nn.Linear(prev_dim, cfg.ode_hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(cfg.ode_hidden_dim),
            ]
            prev_dim = cfg.ode_hidden_dim
        layers.append(nn.Linear(prev_dim, cfg.latent_dim))

        self.net = nn.Sequential(*layers)

        # Residual gate for stability
        self.gate = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
            nn.Sigmoid(),
        )

        # Context is set before each odeint call; None until then
        self._context: Optional[Tensor] = None

    def set_context(self, context: Tensor):
        """Call before odeint to inject patient context."""
        self._context = context

    def _time_embedding(self, t: Tensor, B: int) -> Tensor:
        """Sinusoidal time embedding, broadcast to batch."""
        freqs = torch.exp(
            torch.arange(self.time_embed_dim // 2, device=t.device)
            * -(torch.log(torch.tensor(10000.0)) / (self.time_embed_dim // 2 - 1))
        )                                              # (dim/2,)
        t_val = t.squeeze() if t.dim() > 0 else t     # scalar
        args = t_val * freqs                           # (dim/2,)
        embed = torch.cat([args.sin(), args.cos()])    # (dim,)
        return embed.unsqueeze(0).expand(B, -1)        # (B, dim)

    def forward(self, t: Tensor, h: Tensor) -> Tensor:
        """
        Called by the ODE solver at each evaluation point.

        Args:
            t: scalar time tensor
            h: (B, latent_dim)

        Returns:
            dh/dt: (B, latent_dim)
        """
        B = h.shape[0]
        t_emb = self._time_embedding(t, B)             # (B, time_embed_dim)

        if self._context is None:
            context = torch.zeros(B, self.context_dim, device=h.device, dtype=h.dtype)
        else:
            context = self._context.to(h.device, h.dtype)

        inp = torch.cat([h, t_emb, context], dim=-1)  # (B, in_dim)
        dh = self.net(inp)                             # (B, latent_dim)
        gate = self.gate(h)                            # (B, latent_dim)
        return gate * dh                               # gated residual dynamics


# ── Neural ODE solver wrapper ─────────────────────────────────────────────────

class NeuralODE(nn.Module):
    """
    Wraps ODEFunc with torchdiffeq's odeint.

    Accepts:
        h0:      (B, latent_dim) — initial state from BiGRU
        t_span:  (T_out,)       — query times (e.g. torch.linspace(0, 1, 50))
        context: (B, context_dim) — patient covariates from ContextEncoder

    Returns:
        h_traj: (T_out, B, latent_dim) — latent state at each query time
        h_final:(B, latent_dim)        — state at t_span[-1]

    DEVICE NOTE (Apple Silicon):
        torchdiffeq's dopri5 solver is not MPS-compatible. We automatically
        move h0 and the ODE function to CPU for integration, then move the
        result back to the original device. This is handled transparently.
    """

    def __init__(self, cfg: ODEConfig):
        super().__init__()
        self.cfg = cfg
        self.odefunc = ODEFunc(cfg)
        self.context_encoder = ContextEncoder(cfg.context_dim)

    def forward(
        self,
        h0: Tensor,
        t_span: Tensor,
        age: Tensor,
        sex: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            h0:     (B, latent_dim)
            t_span: (T_out,) — must be sorted ascending, t[0] = 0
            age:    (B,)
            sex:    (B,)

        Returns:
            h_final: (B, latent_dim)
            h_traj:  (T_out, B, latent_dim)
        """
        original_device = h0.device

        # Encode patient context
        context = self.context_encoder(age, sex)       # (B, context_dim)
        self.odefunc.set_context(context)

        # ── MPS/device fallback: move ODE computation to CPU ──────────────────
        # torchdiffeq dopri5 does not support MPS backend.
        ode_device = torch.device("cpu")
        h0_cpu = h0.to(ode_device)
        t_span_cpu = t_span.to(ode_device)

        # Move odefunc parameters to CPU for integration
        self.odefunc = self.odefunc.to(ode_device)
        if self.odefunc._context is not None:
            self.odefunc._context = self.odefunc._context.to(ode_device)

        # ── Solve ODE ─────────────────────────────────────────────────────────
        ode_kwargs = dict(
            rtol=self.cfg.rtol,
            atol=self.cfg.atol,
            method=self.cfg.solver,
        )
        if self.cfg.adjoint and _ADJOINT:
            h_traj = odeint(self.odefunc, h0_cpu, t_span_cpu, **ode_kwargs)
        else:
            from torchdiffeq import odeint as odeint_plain
            h_traj = odeint_plain(self.odefunc, h0_cpu, t_span_cpu, **ode_kwargs)
        # h_traj: (T_out, B, latent_dim)

        # Move odefunc back to original device for parameter updates
        self.odefunc = self.odefunc.to(original_device)

        # Move trajectory back to original device
        h_traj = h_traj.to(original_device)
        h_final = h_traj[-1]                           # (B, latent_dim)

        return h_final, h_traj
