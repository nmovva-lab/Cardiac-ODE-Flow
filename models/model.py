"""
Cardio-ODE-Flow: Full integrated model

Combines all four pillars into a single nn.Module:

  Raw ECG (B, 12, T)
      ↓
  [Window Splitter]          → (B, W, 12, window_size)
      ↓
  [Lead Temporal Encoder]    → (B, W, 12, gcn_input_dim)    per window, per lead
      ↓
  [Anatomical GCN]           → (B, W, gcn_output_dim)       per window, graph-pooled
      ↓
  [Bidirectional GRU]        → h0 (B, latent_dim)
      ↓
  [Neural ODE]  ← context(age, sex)
                             → h_final (B, latent_dim)
      ↓
  [Real-NVP Flow]            → z, log_prob
      ↓
  [Classifier MLP]           → logits (B, num_classes)
                               confidence (B,)

Forward returns a named dict so downstream code is explicit.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Optional

from configs.config import Config, ModelConfig
from models.gcn import AnatomicalGCN, LeadTemporalEncoder
from models.gru_encoder import BidirectionalGRUEncoder, WindowSplitter
from models.neural_ode import NeuralODE
from models.normalizing_flow import RealNVP


class CardioODEFlow(nn.Module):
    """
    Full Cardio-ODE-Flow model.

    Args:
        cfg: full Config object

    Inputs (all tensors on same device except ODE which auto-moves):
        ecg:     (B, 12, T)   raw multi-lead ECG signal (normalised)
        age:     (B,)         patient age normalised to [0, 1]
        sex:     (B,)         0.0 = female, 1.0 = male

    Outputs (dict):
        logits:     (B, num_classes)   raw classification scores
        probs:      (B, num_classes)   sigmoid probabilities per class
        confidence: (B,)               calibrated uncertainty score [0, 1]
        log_prob:   (B,)               flow log-likelihood of latent code
        z:          (B, latent_dim)    flow-mapped noise code
        h_final:    (B, latent_dim)    ODE terminal latent state
        h_traj:     (T_ode, B, latent_dim) full ODE trajectory (if return_traj=True)
    """

    # Number of temporal windows to split the ECG into before GCN
    NUM_WINDOWS = 10

    def __init__(self, cfg: Config):
        super().__init__()
        mc = cfg.model

        # ── Pillar 0: Window splitter
        self.window_splitter = WindowSplitter(
            sequence_length=cfg.data.sequence_length,
            num_windows=self.NUM_WINDOWS,
        )
        window_size = cfg.data.sequence_length // self.NUM_WINDOWS

        # ── Pillar 1: Temporal encoder + GCN
        gcn_input_dim = mc.graph.gcn_hidden_dim   # temporal encoder output dim
        self.temporal_encoder = LeadTemporalEncoder(
            sequence_length=window_size,
            out_dim=gcn_input_dim,
        )
        self.gcn = AnatomicalGCN(
            cfg=mc.graph,
            temporal_dim=gcn_input_dim,
        )

        # ── Pillar 2: Bidirectional GRU
        latent_dim = mc.ode.latent_dim
        self.gru_encoder = BidirectionalGRUEncoder(
            cfg=mc.gru,
            latent_dim=latent_dim,
        )

        # ── Pillar 3: Neural ODE
        self.neural_ode = NeuralODE(cfg=mc.ode)

        # ── Pillar 4: Normalizing flow
        self.flow = RealNVP(cfg=mc.flow)

        # ── Classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, mc.classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(mc.classifier_dropout),
            nn.Linear(mc.classifier_hidden_dim, mc.num_classes),
        )

        # ODE time grid — fixed, [0, 1], 50 query points
        self.register_buffer(
            "t_span",
            torch.linspace(0.0, 1.0, 50),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialise weights conservatively for stable early training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, p in m.named_parameters():
                    if "weight" in name:
                        nn.init.orthogonal_(p)
                    elif "bias" in name:
                        nn.init.zeros_(p)

    def encode(self, ecg: Tensor) -> Tensor:
        """
        Run the GCN + GRU pipeline to get the initial ODE state h0.
        Useful for pre-computing features without the ODE.

        Args:
            ecg: (B, 12, T)

        Returns:
            h0: (B, latent_dim)
        """
        B = ecg.shape[0]

        # Split into windows: (B, W, 12, window_size)
        windows = self.window_splitter(ecg)
        W = windows.shape[1]

        # Encode each window's leads: flatten B*W for batch efficiency
        windows_flat = windows.reshape(B * W, 12, -1)        # (B*W, 12, ws)
        lead_feats = self.temporal_encoder(windows_flat)      # (B*W, 12, gcn_in)

        # GCN over all windows: (B*W, gcn_out_dim)
        _, graph_emb = self.gcn(lead_feats)                   # (B*W, gcn_out)

        # Reshape back to sequence: (B, W, gcn_out_dim)
        graph_emb = graph_emb.reshape(B, W, -1)

        # BiGRU: h0 (B, latent_dim)
        h0, _ = self.gru_encoder(graph_emb)
        return h0

    def forward(
        self,
        ecg: Tensor,
        age: Tensor,
        sex: Tensor,
        return_traj: bool = False,
    ) -> Dict[str, Tensor]:
        """
        Full forward pass.

        Args:
            ecg:         (B, 12, T)
            age:         (B,)   — normalised [0, 1]
            sex:         (B,)   — 0.0 or 1.0
            return_traj: if True, include h_traj in output

        Returns:
            dict with keys: logits, probs, confidence, log_prob, z, h_final
            (and h_traj if return_traj=True)
        """
        # ── Pillars 1 + 2: Spatial encoding → h0
        h0 = self.encode(ecg)                               # (B, latent_dim)

        # ── Pillar 3: Neural ODE evolution
        h_final, h_traj = self.neural_ode(
            h0=h0,
            t_span=self.t_span,
            age=age,
            sex=sex,
        )                                                   # (B, L), (T, B, L)

        # ── Pillar 4: Normalizing flow → latent distribution
        z, log_det, log_prob = self.flow(h_final)          # each (B,)

        # ── Classifier
        logits = self.classifier(h_final)                  # (B, num_classes)
        probs = torch.sigmoid(logits)                      # multi-label

        # ── Confidence score
        confidence = self.flow.confidence_score(h_final)   # (B,)

        output = {
            "logits": logits,
            "probs": probs,
            "confidence": confidence,
            "log_prob": log_prob,
            "log_det": log_det,
            "z": z,
            "h0": h0,
            "h_final": h_final,
        }
        if return_traj:
            output["h_traj"] = h_traj

        return output

    @torch.no_grad()
    def predict(
        self,
        ecg: Tensor,
        age: Tensor,
        sex: Tensor,
        threshold: float = 0.5,
    ) -> Dict[str, Tensor]:
        """
        Inference-mode forward. Returns predictions + confidence.

        Args:
            ecg:       (B, 12, T)
            age:       (B,)
            sex:       (B,)
            threshold: classification threshold for binary decisions

        Returns:
            dict: probs, predictions (binary), confidence
        """
        self.eval()
        out = self.forward(ecg, age, sex, return_traj=False)
        return {
            "probs": out["probs"],
            "predictions": (out["probs"] > threshold).float(),
            "confidence": out["confidence"],
        }

    def count_parameters(self) -> Dict[str, int]:
        """Returns parameter count per module for model card reporting."""
        counts = {}
        for name, module in [
            ("temporal_encoder", self.temporal_encoder),
            ("gcn", self.gcn),
            ("gru_encoder", self.gru_encoder),
            ("neural_ode", self.neural_ode),
            ("flow", self.flow),
            ("classifier", self.classifier),
        ]:
            counts[name] = sum(p.numel() for p in module.parameters() if p.requires_grad)
        counts["total"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return counts
