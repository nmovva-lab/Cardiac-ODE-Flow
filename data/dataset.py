"""
Data Pipeline — PTB-XL + Georgia + Chapman-Shaoxing

Unified ECG dataset that:
  1. Loads waveforms from all three sources with a common interface
  2. Resamples to a target rate (500 Hz by default)
  3. Applies per-lead z-score normalisation
  4. Exposes a shared multi-label diagnostic schema (5 super-classes)
  5. Applies training-time augmentation (noise, scaling, lead dropout)

Diagnostic super-class mapping (following PTB-XL convention):
  0: NORM  — Normal ECG
  1: MI    — Myocardial Infarction
  2: STTC  — ST/T-wave Change
  3: CD    — Conduction Disturbance
  4: HYP   — Hypertrophy

Requirements (install separately):
  pip install wfdb neurokit2 pandas numpy scipy

Dataset directory structure expected:
  data/raw/ptbxl/
      ptbxl_database.csv
      records500/        ← 500 Hz WFDB records
  data/raw/georgia/
      *.mat or *.hea + *.dat
  data/raw/chapman/
      *.mat or *.hea + *.dat
"""

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler

try:
    import wfdb
except ImportError:
    warnings.warn("wfdb not installed. PTB-XL loading will fail. pip install wfdb")

try:
    from scipy.signal import resample_poly
    from math import gcd
except ImportError:
    warnings.warn("scipy not installed. Resampling will be unavailable.")

from configs.config import DataConfig


# ── Shared label taxonomy ─────────────────────────────────────────────────────

SUPER_CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

# PTB-XL diagnostic superclass column → our index
PTBXL_SUPERCLASS_MAP = {
    "NORM": 0, "MI": 1, "STTC": 2, "CD": 3, "HYP": 4
}

# Georgia / Chapman rhythm/diagnostic codes → super-class index
# (partial mapping; extend as needed for your label files)
GEORGIA_CODE_MAP: Dict[str, int] = {
    "Normal": 0,
    "Myocardial Infarction": 1,
    "ST Depression": 2,
    "ST Elevation": 2,
    "T Wave Abnormal": 2,
    "Left Bundle Branch Block": 3,
    "Right Bundle Branch Block": 3,
    "1st Degree AV Block": 3,
    "Left Ventricular Hypertrophy": 4,
    "Right Ventricular Hypertrophy": 4,
}


# ── Signal utilities ──────────────────────────────────────────────────────────

def resample_signal(signal: np.ndarray, from_hz: int, to_hz: int) -> np.ndarray:
    """
    Resample a (T, C) or (C, T) signal using polyphase filtering.
    Assumes shape (T, num_leads).
    """
    if from_hz == to_hz:
        return signal
    g = gcd(from_hz, to_hz)
    up, down = to_hz // g, from_hz // g
    return resample_poly(signal, up, down, axis=0)


def normalize_leads(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-lead z-score normalisation. signal: (T, 12)"""
    mean = signal.mean(axis=0, keepdims=True)
    std = signal.std(axis=0, keepdims=True).clip(min=eps)
    return (signal - mean) / std


def pad_or_crop(signal: np.ndarray, target_len: int) -> np.ndarray:
    """Ensure signal has exactly target_len samples (T axis)."""
    T = signal.shape[0]
    if T >= target_len:
        # Centre crop
        start = (T - target_len) // 2
        return signal[start: start + target_len]
    else:
        # Zero-pad at end
        pad = np.zeros((target_len - T, signal.shape[1]), dtype=signal.dtype)
        return np.concatenate([signal, pad], axis=0)


# ── Augmentation ──────────────────────────────────────────────────────────────

def augment_ecg(signal: Tensor, cfg: DataConfig) -> Tensor:
    """
    In-place-safe augmentation applied to (12, T) tensor.

    1. Gaussian noise: σ ~ U(0, noise_std)
    2. Amplitude scaling: s ~ U(scale_min, scale_max)
    3. Lead dropout: one lead randomly zeroed with probability p
    """
    signal = signal.clone()

    # Gaussian noise
    if cfg.aug_noise_std > 0:
        noise_level = torch.rand(1).item() * cfg.aug_noise_std
        signal = signal + torch.randn_like(signal) * noise_level

    # Amplitude scale
    lo, hi = cfg.aug_scale_range
    scale = lo + (hi - lo) * torch.rand(1).item()
    signal = signal * scale

    # Lead dropout
    if cfg.aug_lead_dropout_p > 0 and torch.rand(1).item() < cfg.aug_lead_dropout_p:
        lead_idx = torch.randint(0, 12, (1,)).item()
        signal[lead_idx] = 0.0

    return signal


# ── PTB-XL Dataset ───────────────────────────────────────────────────────────

class PTBXLDataset(Dataset):
    """
    Loads PTB-XL at 500 Hz using the WFDB library.
    Labels: 5 super-classes from the 'diagnostic_superclass' column.

    Expected CSV columns: filename_hr, patient_id, age, sex,
                          diagnostic_superclass (JSON list), split
    """

    def __init__(
        self,
        root: str,
        split: str,                  # "train" | "val" | "test"
        cfg: DataConfig,
        augment: bool = False,
    ):
        super().__init__()
        self.root = Path(root)
        self.cfg = cfg
        self.augment = augment

        csv_path = self.root / "ptbxl_database.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"PTB-XL metadata CSV not found: {csv_path}")

        df = pd.read_csv(csv_path, index_col="ecg_id")
        df["diagnostic_superclass"] = df["diagnostic_superclass"].apply(
            lambda x: eval(x) if isinstance(x, str) else []
        )

        # PTB-XL splits: fold 9 = val, fold 10 = test, rest = train
        if split == "train":
            df = df[df["strat_fold"] <= 8]
        elif split == "val":
            df = df[df["strat_fold"] == 9]
        elif split == "test":
            df = df[df["strat_fold"] == 10]
        else:
            raise ValueError(f"Unknown split: {split}")

        self.metadata = df.reset_index()
        self.target_len = cfg.sequence_length

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        row = self.metadata.iloc[idx]

        # Load WFDB record (500 Hz)
        record_path = str(self.root / row["filename_hr"])
        try:
            record = wfdb.rdrecord(record_path)
            signal_np = record.p_signal.astype(np.float32)  # (T, 12)
            source_hz = record.fs
        except Exception as e:
            warnings.warn(f"Failed to load {record_path}: {e}. Returning zeros.")
            signal_np = np.zeros((self.target_len, 12), dtype=np.float32)
            source_hz = self.cfg.sampling_rate

        # Resample if needed
        if source_hz != self.cfg.sampling_rate:
            signal_np = resample_signal(signal_np, source_hz, self.cfg.sampling_rate)

        signal_np = pad_or_crop(signal_np, self.target_len)  # (T, 12)

        if self.cfg.normalize:
            signal_np = normalize_leads(signal_np)

        # Convert to tensor: (12, T)
        ecg = torch.from_numpy(signal_np.T).float()

        if self.augment and self.cfg.augment:
            ecg = augment_ecg(ecg, self.cfg)

        # Build multi-hot label vector
        label = torch.zeros(len(SUPER_CLASSES), dtype=torch.float32)
        for sc in row["diagnostic_superclass"]:
            if sc in PTBXL_SUPERCLASS_MAP:
                label[PTBXL_SUPERCLASS_MAP[sc]] = 1.0

        # Patient covariates
        age = float(row.get("age", 50)) / 100.0   # normalise 0–1
        age = min(max(age, 0.0), 1.0)
        sex_raw = str(row.get("sex", "0")).strip().lower()
        sex = 1.0 if sex_raw in {"1", "male", "m"} else 0.0

        return {
            "ecg": ecg,
            "label": label,
            "age": torch.tensor(age, dtype=torch.float32),
            "sex": torch.tensor(sex, dtype=torch.float32),
            "record_id": str(row.get("ecg_id", idx)),
            "source": "ptbxl",
        }


# ── Generic WFDB Dataset (Georgia + Chapman) ──────────────────────────────────

class GenericWFDBDataset(Dataset):
    """
    Loads any WFDB-compatible dataset where each record's labels
    are stored in the WFDB header comments (PhysioNet Challenge format).

    Expects a directory of .hea / .dat pairs.
    Labels are extracted from the 'Dx:' comment line in .hea files.
    """

    def __init__(
        self,
        root: str,
        split: str,
        cfg: DataConfig,
        augment: bool = False,
        source_name: str = "generic",
        val_fraction: float = 0.10,
        test_fraction: float = 0.10,
        seed: int = 42,
    ):
        super().__init__()
        self.root = Path(root)
        self.cfg = cfg
        self.augment = augment
        self.source_name = source_name
        self.target_len = cfg.sequence_length

        # Collect all record stubs
        all_records = sorted([
            p.stem for p in self.root.glob("*.hea")
        ])
        if not all_records:
            warnings.warn(f"No .hea files found in {root}. Dataset will be empty.")

        # Deterministic split
        rng = np.random.RandomState(seed)
        indices = np.arange(len(all_records))
        rng.shuffle(indices)
        n = len(all_records)
        n_test = int(n * test_fraction)
        n_val = int(n * val_fraction)

        if split == "test":
            selected = indices[:n_test]
        elif split == "val":
            selected = indices[n_test: n_test + n_val]
        else:
            selected = indices[n_test + n_val:]

        self.records = [all_records[i] for i in selected]

    def __len__(self) -> int:
        return len(self.records)

    def _parse_labels(self, header: "wfdb.Record") -> Tensor:
        """Extract Dx codes from header comments and map to super-classes."""
        label = torch.zeros(len(SUPER_CLASSES), dtype=torch.float32)
        comments = header.comments if header.comments else []
        for comment in comments:
            if comment.startswith("Dx:"):
                codes = [c.strip() for c in comment[3:].split(",")]
                for code in codes:
                    if code in GEORGIA_CODE_MAP:
                        label[GEORGIA_CODE_MAP[code]] = 1.0
        return label

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        record_name = self.records[idx]
        record_path = str(self.root / record_name)

        try:
            record = wfdb.rdrecord(record_path)
            signal_np = record.p_signal.astype(np.float32)  # (T, leads)
            source_hz = record.fs
        except Exception as e:
            warnings.warn(f"Failed to load {record_path}: {e}. Returning zeros.")
            signal_np = np.zeros((self.target_len, 12), dtype=np.float32)
            source_hz = self.cfg.sampling_rate

        # Ensure 12-lead signal (some records may have fewer)
        if signal_np.shape[1] < 12:
            pad = np.zeros((signal_np.shape[0], 12 - signal_np.shape[1]), dtype=np.float32)
            signal_np = np.concatenate([signal_np, pad], axis=1)
        elif signal_np.shape[1] > 12:
            signal_np = signal_np[:, :12]

        if source_hz != self.cfg.sampling_rate:
            signal_np = resample_signal(signal_np, source_hz, self.cfg.sampling_rate)

        signal_np = pad_or_crop(signal_np, self.target_len)
        if self.cfg.normalize:
            signal_np = normalize_leads(signal_np)

        ecg = torch.from_numpy(signal_np.T).float()   # (12, T)
        if self.augment and self.cfg.augment:
            ecg = augment_ecg(ecg, self.cfg)

        # Labels
        try:
            header = wfdb.rdheader(record_path)
            label = self._parse_labels(header)
        except Exception:
            label = torch.zeros(len(SUPER_CLASSES), dtype=torch.float32)

        # Covariates — try to extract from header
        age, sex = 0.5, 0.0
        try:
            for comment in (record.comments or []):
                if comment.startswith("Age:"):
                    age_raw = comment[4:].strip()
                    if age_raw.isdigit():
                        age = int(age_raw) / 100.0
                elif comment.startswith("Sex:"):
                    sex = 1.0 if comment[4:].strip().upper() in {"M", "MALE"} else 0.0
        except Exception:
            pass

        return {
            "ecg": ecg,
            "label": label,
            "age": torch.tensor(age, dtype=torch.float32),
            "sex": torch.tensor(sex, dtype=torch.float32),
            "record_id": record_name,
            "source": self.source_name,
        }


# ── Combined multi-center DataLoader factory ──────────────────────────────────

def build_dataloaders(cfg: DataConfig, seed: int = 42) -> Dict[str, DataLoader]:
    """
    Builds train / val / test DataLoaders combining PTB-XL, Georgia,
    and Chapman-Shaoxing into a single shuffled dataset.

    Class imbalance is handled via WeightedRandomSampler on the train set.

    Returns:
        dict with keys "train", "val", "test"
    """
    loaders = {}

    for split in ["train", "val", "test"]:
        is_train = split == "train"
        datasets = []

        # PTB-XL
        if cfg.ptbxl_path and Path(cfg.ptbxl_path).exists():
            try:
                ds = PTBXLDataset(cfg.ptbxl_path, split, cfg, augment=is_train)
                datasets.append(ds)
                print(f"  PTB-XL {split}: {len(ds)} records")
            except Exception as e:
                warnings.warn(f"Could not load PTB-XL ({split}): {e}")

        # Georgia
        if cfg.georgia_path and Path(cfg.georgia_path).exists():
            try:
                ds = GenericWFDBDataset(
                    cfg.georgia_path, split, cfg,
                    augment=is_train, source_name="georgia",
                )
                datasets.append(ds)
                print(f"  Georgia {split}: {len(ds)} records")
            except Exception as e:
                warnings.warn(f"Could not load Georgia ({split}): {e}")

        # Chapman-Shaoxing
        if cfg.chapman_path and Path(cfg.chapman_path).exists():
            try:
                ds = GenericWFDBDataset(
                    cfg.chapman_path, split, cfg,
                    augment=is_train, source_name="chapman",
                )
                datasets.append(ds)
                print(f"  Chapman {split}: {len(ds)} records")
            except Exception as e:
                warnings.warn(f"Could not load Chapman ({split}): {e}")

        if not datasets:
            warnings.warn(f"No datasets loaded for split '{split}'.")
            continue

        combined = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

        # Weighted sampler for training to address class imbalance
        sampler = None
        if is_train:
            all_labels = torch.stack([
                combined[i]["label"] for i in range(len(combined))
            ])                                              # (N, C)
            # Weight each sample by inverse frequency of its positive classes
            class_freq = all_labels.mean(dim=0).clamp(min=1e-5)  # (C,)
            sample_weights = (all_labels / class_freq).max(dim=1).values
            sample_weights = sample_weights / sample_weights.sum()
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(combined),
                replacement=True,
                generator=torch.Generator().manual_seed(seed),
            )

        loaders[split] = DataLoader(
            combined,
            batch_size=cfg.batch_size if is_train else cfg.batch_size * 2,
            sampler=sampler,
            shuffle=(is_train and sampler is None),
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=is_train,
            persistent_workers=cfg.num_workers > 0,
        )

    return loaders
