"""
Pillar 1 — Anatomical Graph Convolutional Network (GCN)

Models the 12 ECG leads as nodes in a graph where edges reflect:
  - Anatomical electrode proximity in 3D thoracic space
  - Shared electrical axis projection (e.g. I ↔ aVL, II ↔ aVF)
  - Einthoven triangle relationships (I, II, III)

Lead index mapping (fixed throughout):
  0=I, 1=II, 2=III, 3=aVR, 4=aVL, 5=aVF, 6=V1, 7=V2, 8=V3, 9=V4, 10=V5, 11=V6
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple

from configs.config import GraphConfig


# ── Anatomical adjacency matrix ───────────────────────────────────────────────

def build_anatomical_adjacency() -> Tensor:
    """
    Returns a (12, 12) symmetric adjacency matrix with edge weights ∈ (0, 1].
    Self-loops are included (diagonal = 1.0).

    Weights are heuristic anatomical similarities; they are learnable
    via the GCN's attention mechanism on top of this prior.
    """
    # Lead order: I=0, II=1, III=2, aVR=3, aVL=4, aVF=5, V1=6..V6=11
    edges = [
        # Einthoven triangle (limb leads)
        (0, 1, 0.9),   # I ↔ II
        (0, 2, 0.8),   # I ↔ III
        (1, 2, 0.9),   # II ↔ III
        # Augmented ↔ standard limb
        (0, 4, 0.95),  # I ↔ aVL (same axis)
        (1, 5, 0.95),  # II ↔ aVF
        (3, 1, 0.7),   # aVR ↔ II (inverse)
        (3, 0, 0.6),   # aVR ↔ I
        (4, 2, 0.7),   # aVL ↔ III
        (5, 2, 0.7),   # aVF ↔ III
        # Precordial chain (adjacent leads share myocardial territory)
        (6, 7, 0.95),  # V1 ↔ V2
        (7, 8, 0.95),  # V2 ↔ V3
        (8, 9, 0.95),  # V3 ↔ V4
        (9, 10, 0.95), # V4 ↔ V5
        (10, 11, 0.95),# V5 ↔ V6
        (6, 8, 0.7),   # V1 ↔ V3 (skip)
        (7, 9, 0.7),   # V2 ↔ V4
        (8, 10, 0.7),  # V3 ↔ V5
        (9, 11, 0.7),  # V4 ↔ V6
        # Septal / right-sided cross-coupling
        (6, 3, 0.5),   # V1 ↔ aVR (right-sided)
        (11, 4, 0.4),  # V6 ↔ aVL (lateral)
        (11, 5, 0.5),  # V6 ↔ aVF (inferior-lateral)
    ]

    A = torch.zeros(12, 12)
    for i, j, w in edges:
        A[i, j] = w
        A[j, i] = w  # symmetric
    A.fill_diagonal_(1.0)  # self-loops
    return A


def normalise_adjacency(A: Tensor) -> Tensor:
    """Symmetric normalisation: D^{-1/2} A D^{-1/2}"""
    deg = A.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
    D = torch.diag(deg_inv_sqrt)
    return D @ A @ D


# ── Graph Attention layer ─────────────────────────────────────────────────────

class GraphAttentionLayer(nn.Module):
    """
    Single GATv2-style attention layer.
    Learns to modulate the anatomical prior A_hat with data-driven attention.

    Input:  X  (B, N, in_dim)   node features
            A  (N, N)            normalised adjacency (fixed prior)
    Output: X' (B, N, out_dim)
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, X: Tensor, A: Tensor) -> Tensor:
        # X: (B, N, in_dim)
        B, N, _ = X.shape
        Wh = self.W(X)                                       # (B, N, out_dim)

        # Pairwise attention logits
        Wh_i = Wh.unsqueeze(2).expand(-1, -1, N, -1)        # (B, N, N, out)
        Wh_j = Wh.unsqueeze(1).expand(-1, N, -1, -1)        # (B, N, N, out)
        e = self.leaky(self.a(torch.cat([Wh_i, Wh_j], dim=-1)).squeeze(-1))  # (B, N, N)

        # Mask with adjacency (zero out non-edges)
        mask = (A == 0).unsqueeze(0)
        e = e.masked_fill(mask, float("-inf"))
        alpha = F.softmax(e, dim=-1)                         # (B, N, N)
        # Replace NaN rows (fully-masked nodes) with uniform attention — safe fallback
        alpha = torch.nan_to_num(alpha, nan=0.0)
        alpha = self.dropout(alpha)

        # Modulate by anatomical prior
        alpha = alpha * A.unsqueeze(0)

        out = torch.bmm(alpha, Wh)                           # (B, N, out_dim)
        return F.elu(out)


# ── Full GCN ──────────────────────────────────────────────────────────────────

class AnatomicalGCN(nn.Module):
    """
    Stack of GraphAttentionLayers operating over all 12 leads simultaneously.

    Input:  X  (B, N=12, T, C) — raw or windowed per-lead features
            After a 1-D temporal encoder the shape becomes (B, N, temporal_dim)
            before entering the GCN.

    Output: node_embeddings (B, N, gcn_output_dim)
            graph_embedding  (B, gcn_output_dim)  ← mean-pooled across nodes
    """

    def __init__(self, cfg: GraphConfig, temporal_dim: int):
        """
        Args:
            cfg:          GraphConfig
            temporal_dim: dimensionality of per-lead temporal features fed in
        """
        super().__init__()
        self.cfg = cfg

        # Register adjacency as a buffer (moves with .to(device) calls)
        A_raw = build_anatomical_adjacency()
        A_norm = normalise_adjacency(A_raw)
        self.register_buffer("A", A_norm)

        # Per-lead temporal feature projection
        self.input_proj = nn.Sequential(
            nn.Linear(temporal_dim, cfg.gcn_hidden_dim),
            nn.LayerNorm(cfg.gcn_hidden_dim),
            nn.GELU(),
        )

        # Stack of GAT layers with residual connections
        dims = [cfg.gcn_hidden_dim] + [cfg.gcn_hidden_dim] * (cfg.gcn_layers - 1) + [cfg.gcn_output_dim]
        self.layers = nn.ModuleList([
            GraphAttentionLayer(dims[i], dims[i + 1], cfg.gcn_dropout)
            for i in range(cfg.gcn_layers)
        ])

        # Residual projections where dims differ
        self.residuals = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1], bias=False) if dims[i] != dims[i + 1] else nn.Identity()
            for i in range(cfg.gcn_layers)
        ])

        self.norm = nn.LayerNorm(cfg.gcn_output_dim)
        self.dropout = nn.Dropout(cfg.gcn_dropout)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: (B, N=12, temporal_dim)  per-lead temporal representations

        Returns:
            node_emb:  (B, 12, gcn_output_dim)
            graph_emb: (B, gcn_output_dim)
        """
        h = self.input_proj(x)           # (B, 12, hidden_dim)

        for layer, res_proj in zip(self.layers, self.residuals):
            h_new = layer(h, self.A)
            h = self.dropout(h_new) + res_proj(h)   # residual

        node_emb = self.norm(h)
        graph_emb = node_emb.mean(dim=1)             # global mean-pool
        return node_emb, graph_emb


# ── Per-lead temporal encoder (1-D CNN) ───────────────────────────────────────

class LeadTemporalEncoder(nn.Module):
    """
    Lightweight 1-D CNN that compresses each lead's raw signal
    (T,) → a fixed-size embedding fed into the GCN.

    Applied identically and independently to all 12 leads.
    """

    def __init__(self, sequence_length: int, out_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            # Stage 1: local waveform features (P, QRS, T)
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.GELU(),
            # Stage 2: rhythm / interval features
            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            # Stage 3: global context
            nn.Conv1d(128, out_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),   # → (B*12, out_dim, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, 12, T)  raw multi-lead ECG

        Returns:
            (B, 12, out_dim)
        """
        B, N, T = x.shape
        x_flat = x.reshape(B * N, 1, T)          # treat each lead independently
        enc = self.encoder(x_flat).squeeze(-1)   # (B*N, out_dim)
        return enc.reshape(B, N, -1)             # (B, N, out_dim)
