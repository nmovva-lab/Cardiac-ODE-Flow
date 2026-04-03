"""
Cardio-ODE-Flow: Central configuration.
All hyperparameters live here — edit this file to run ablations.
"""
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    # Paths to each dataset root (set these before running)
    ptbxl_path: str = "data/raw/ptbxl"
    georgia_path: str = "data/raw/georgia"
    chapman_path: str = "data/raw/chapman"

    sampling_rate: int = 500          # Hz — PTB-XL supports 100 or 500
    sequence_length: int = 5000       # samples at 500 Hz = 10 s
    num_leads: int = 12
    val_fraction: float = 0.10
    test_fraction: float = 0.10
    num_workers: int = 4
    pin_memory: bool = True

    # Normalisation
    normalize: bool = True            # per-lead z-score

    # Augmentation (training only)
    augment: bool = True
    aug_noise_std: float = 0.01
    aug_scale_range: tuple = (0.9, 1.1)
    aug_lead_dropout_p: float = 0.05  # randomly zero one lead


@dataclass
class GraphConfig:
    """
    Anatomical adjacency for the 12-lead graph.
    Lead order: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
    Edges encode anatomical proximity + electrical axis similarity.
    """
    num_leads: int = 12
    gcn_hidden_dim: int = 128
    gcn_output_dim: int = 256
    gcn_layers: int = 3
    gcn_dropout: float = 0.1


@dataclass
class GRUConfig:
    input_dim: int = 256              # = gcn_output_dim
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    bidirectional: bool = True


@dataclass
class ODEConfig:
    latent_dim: int = 256             # = gru hidden_dim (after projection)
    ode_hidden_dim: int = 512
    context_dim: int = 32             # age + sex embedding size
    # torchdiffeq solver settings
    solver: str = "dopri5"            # dopri5 = adaptive RK45
    rtol: float = 1e-4
    atol: float = 1e-5
    adjoint: bool = True              # memory-efficient adjoint method


@dataclass
class FlowConfig:
    """Real-NVP normalizing flow for latent uncertainty quantification."""
    latent_dim: int = 256
    num_coupling_layers: int = 6
    hidden_dim: int = 512
    num_hidden_layers: int = 2        # layers inside each coupling MLP


@dataclass
class ModelConfig:
    graph: GraphConfig = field(default_factory=GraphConfig)
    gru: GRUConfig = field(default_factory=GRUConfig)
    ode: ODEConfig = field(default_factory=ODEConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)

    num_classes: int = 5              # diagnostic risk classes
    classifier_hidden_dim: int = 256
    classifier_dropout: float = 0.2

    def __post_init__(self):
        """Enforce that dependent dimensions are consistent across sub-configs."""
        # GRU input must equal GCN output
        self.gru.input_dim = self.graph.gcn_output_dim
        # Flow latent dim must equal ODE latent dim (they share the same space)
        self.flow.latent_dim = self.ode.latent_dim


@dataclass
class TrainingConfig:
    seed: int = 42
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0

    # Hybrid loss weights
    bce_weight: float = 1.0
    kl_weight: float = 0.1            # annealed from 0 → kl_weight
    kl_anneal_epochs: int = 20        # linear warmup

    # LR schedule
    scheduler: str = "cosine"         # "cosine" | "plateau" | "none"
    warmup_epochs: int = 5
    min_lr: float = 1e-6

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 5
    keep_best_k: int = 3

    # Logging
    log_dir: str = "runs"
    log_every_n_steps: int = 50


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Device — auto-detected in main, override here if needed
    # "mps" | "cpu" | "cuda"
    device: str = "auto"
    ode_device: str = "cpu"           # ODE solver: keep on CPU (MPS incompatible)
    mixed_precision: bool = False     # AMP — CPU/MPS only: keep False


def get_config() -> Config:
    return Config()
