"""
Pillar 2 — Bidirectional GRU Temporal Encoder

Takes the GCN's graph-level embedding (already spatially enriched)
and processes it as a sequence to extract:

  h0 : initial hidden state for the Neural ODE
       shape (B, latent_dim) — the "where we are in cardiac state-space"

The GRU sees the ECG as a sequence of overlapping windows, each
window first passed through the Lead Temporal Encoder + GCN.
For the full model we treat the time-axis at the GCN output level:
the sequence dimension is reconstructed by splitting the raw ECG into
W non-overlapping windows before the GCN, giving shape (B, W, gcn_out_dim).
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

from configs.config import GRUConfig


class BidirectionalGRUEncoder(nn.Module):
    """
    Bidirectional GRU over a sequence of per-window GCN graph embeddings.

    Input:  x  (B, W, input_dim)   W = number of temporal windows
    Output:
        h0         (B, latent_dim)  — concatenated final hidden states, projected
        all_hidden (B, W, latent_dim) — full sequence of hidden states (for aux losses)
    """

    def __init__(self, cfg: GRUConfig, latent_dim: int):
        super().__init__()
        self.cfg = cfg
        self.latent_dim = latent_dim

        self.gru = nn.GRU(
            input_size=cfg.input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            bidirectional=cfg.bidirectional,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )

        # Bidirectional: final hidden dim = 2 * hidden_dim
        gru_out_dim = cfg.hidden_dim * (2 if cfg.bidirectional else 1)

        # Project to latent_dim (= ODE state dimension)
        self.h0_proj = nn.Sequential(
            nn.Linear(gru_out_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Tanh(),   # bound the initial ODE state
        )

        # Project full sequence output for auxiliary use
        self.seq_proj = nn.Linear(gru_out_dim, latent_dim)

        self.input_norm = nn.LayerNorm(cfg.input_dim)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: (B, W, input_dim)

        Returns:
            h0:         (B, latent_dim)
            all_hidden: (B, W, latent_dim)
        """
        x = self.input_norm(x)
        # out: (B, W, 2*hidden_dim),  hn: (num_layers*2, B, hidden_dim)
        out, hn = self.gru(x)

        # Extract final hidden state from both directions, last layer
        # hn shape: (num_layers * num_directions, B, hidden_dim)
        # For bidirectional: index [-2] = forward last layer, [-1] = backward last layer
        if self.cfg.bidirectional:
            h_fwd = hn[-2]   # (B, hidden_dim) — forward direction, last layer
            h_bwd = hn[-1]   # (B, hidden_dim) — backward direction, last layer
            h_cat = torch.cat([h_fwd, h_bwd], dim=-1)   # (B, 2*hidden_dim)
        else:
            h_cat = hn[-1]

        h0 = self.h0_proj(h_cat)                        # (B, latent_dim)
        all_hidden = self.seq_proj(self.dropout(out))    # (B, W, latent_dim)

        return h0, all_hidden


# ── Window splitter utility ───────────────────────────────────────────────────

class WindowSplitter(nn.Module):
    """
    Splits a raw ECG (B, 12, T) into W non-overlapping windows
    along the time axis, producing (B, W, 12, window_size).

    This allows the GCN + GRU pipeline to see the ECG as a
    temporal sequence of short spatial snapshots.
    """

    def __init__(self, sequence_length: int, num_windows: int):
        super().__init__()
        assert sequence_length % num_windows == 0, (
            f"sequence_length {sequence_length} must be divisible by num_windows {num_windows}"
        )
        self.window_size = sequence_length // num_windows
        self.num_windows = num_windows

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, 12, T)

        Returns:
            (B, W, 12, window_size)
        """
        B, N, T = x.shape
        # Reshape T into (W, window_size)
        x = x.reshape(B, N, self.num_windows, self.window_size)
        # Swap to (B, W, N, window_size)
        return x.permute(0, 2, 1, 3).contiguous()
