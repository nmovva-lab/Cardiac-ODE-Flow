# Cardio-ODE-Flow

A deep learning framework for ECG risk stratification treating cardiac electrical activity as a continuous-time dynamical system.

## Architecture

```
Raw ECG (B, 12, T)
    ↓
Window Splitter           → (B, W, 12, window_size)
    ↓
Lead Temporal Encoder     → (B, W, 12, gcn_in_dim)   1-D CNN per lead
    ↓
Anatomical GCN            → (B, W, gcn_out_dim)       graph attention over 12 leads
    ↓
Bidirectional GRU         → h0 (B, latent_dim)        initial ODE state
    ↓
Neural ODE  ← context(age, sex)
              dh/dt = f_θ(h, t, context)
    ↓
Real-NVP Flow             → z, log_prob               latent distribution
    ↓
Classifier MLP            → logits (B, 5)             diagnostic risk classes
                            confidence (B,)            calibrated uncertainty
```

## Setup

```bash
# 1. Install dependencies
pip install torch torchvision torchaudio          # from pytorch.org (MPS support)
pip install -r requirements.txt

# 2. Verify installation
python main.py sanity
```

## Dataset Structure

```
data/raw/
    ptbxl/
        ptbxl_database.csv
        records500/
            00000/00001_hr.hea
            00000/00001_hr.dat
            ...
    georgia/
        A0001.hea
        A0001.dat
        ...
    chapman/
        JS00001.hea
        JS00001.dat
        ...
```

Edit `configs/config.py` → `DataConfig` to point at your dataset paths.

## Training

```bash
# Train with default config
python main.py train

# Specify dataset paths from CLI
python main.py train \
    --ptbxl  /path/to/ptbxl \
    --georgia /path/to/georgia \
    --chapman /path/to/chapman

# Resume from checkpoint
python main.py train --resume checkpoints/epoch_0050_auroc_0.8900.pt
```

Training logs appear in `runs/` (TensorBoard):
```bash
tensorboard --logdir runs/
```

## Evaluation

```bash
python main.py eval --checkpoint checkpoints/epoch_0050_auroc_0.8900.pt
```

Outputs in `eval_results/`:
- `test_results.json` — AUROC, AUPRC, Brier score, ECE per class
- `roc_curves.png` — ROC curves
- `calibration.png` — calibration curves
- `confidence_histogram.png` — confidence score distribution

## Key Design Decisions

### MPS / Apple Silicon
`torchdiffeq` does not support the MPS backend. The Neural ODE integration runs on CPU automatically, while all other operations use MPS. This is handled transparently in `models/neural_ode.py`.

### KL Annealing
KL weight linearly warms up from 0 → `kl_weight` over `kl_anneal_epochs` epochs. This prevents posterior collapse in the normalizing flow during early training.

### Irregular Sampling
Because the ODE is solved at arbitrary query times, the model natively handles ECGs with missing segments or non-uniform sampling. Pass custom `t_span` tensors to `NeuralODE.forward()` to query specific time points.

### Confidence Score
`RealNVP.confidence_score(h)` returns a scalar in [0, 1] per sample:
```
score = sigmoid(log_prob(h) / latent_dim)
```
High score → h lies in a high-density region of the learned latent distribution → model is confident. This can be thresholded to flag uncertain predictions for clinical review.

## File Structure

```
cardio_ode_flow/
├── main.py                     ← entry point (train / eval / sanity)
├── requirements.txt
├── configs/
│   └── config.py               ← all hyperparameters
├── models/
│   ├── gcn.py                  ← Pillar 1: Anatomical GCN
│   ├── gru_encoder.py          ← Pillar 2: Bidirectional GRU
│   ├── neural_ode.py           ← Pillar 3: Neural ODE
│   ├── normalizing_flow.py     ← Pillar 4: Real-NVP
│   └── model.py                ← Full integrated model
├── training/
│   ├── loss.py                 ← Hybrid BCE + KL loss
│   ├── trainer.py              ← Training loop + checkpointing
│   └── evaluate.py             ← Test set evaluation + plots
└── data/
    └── dataset.py              ← PTB-XL / Georgia / Chapman pipeline
```

## Hyperparameter Ablations

Key hyperparameters to sweep for the paper:

| Parameter | Default | Ablation range |
|-----------|---------|----------------|
| `gcn_layers` | 3 | 1, 2, 3, 4 |
| `ode.solver` | dopri5 | dopri5, rk4, euler |
| `flow.num_coupling_layers` | 6 | 2, 4, 6, 8 |
| `kl_weight` | 0.1 | 0.0, 0.01, 0.1, 1.0 |
| `ode.adjoint` | True | True, False |
| `latent_dim` | 256 | 64, 128, 256, 512 |

Set `kl_weight=0.0` to ablate the normalizing flow entirely.
Set `gcn_layers=0` and bypass the GCN to ablate the graph component.
