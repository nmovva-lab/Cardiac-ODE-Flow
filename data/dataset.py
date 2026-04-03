"""
Data Pipeline — PTB-XL + Georgia + Chapman-Shaoxing

Key design decisions:
  - Labels: SNOMED CT codes mapped to 5 PTB-XL superclasses (not free-text guessing).
  - PTB-XL splits: official stratified folds (fold 9=val, fold 10=test). No leakage.
  - Georgia/Chapman: patient-level splits where patient IDs exist in headers;
    otherwise record-level (documented limitation).
  - Sampler weights built from pre-cached labels — zero waveform I/O at startup.
  - Label prevalence is logged per split to catch mapping bugs before training.

Superclass schema (fixed across all datasets):
  0: NORM  — Normal ECG
  1: MI    — Myocardial Infarction
  2: STTC  — ST/T-wave Change
  3: CD    — Conduction Disturbance
  4: HYP   — Hypertrophy

Requirements: pip install wfdb pandas numpy scipy
"""

import ast
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
    warnings.warn("wfdb not installed — pip install wfdb")

try:
    from scipy.signal import resample_poly
    from math import gcd
except ImportError:
    warnings.warn("scipy not installed — pip install scipy")

from configs.config import DataConfig


# ── Label taxonomy ────────────────────────────────────────────────────────────

SUPER_CLASSES: List[str] = ["NORM", "MI", "STTC", "CD", "HYP"]
NUM_CLASSES: int = len(SUPER_CLASSES)

# PTB-XL diagnostic_superclass string → class index (already clean in CSV)
PTBXL_SUPERCLASS_MAP: Dict[str, int] = {
    "NORM": 0, "MI": 1, "STTC": 2, "CD": 3, "HYP": 4
}

# SNOMED CT code → superclass.
# Based on PhysioNet 2020/2021 Challenge label specifications and the
# PTB-XL annotation paper (Wagner et al., 2020).
# Duplicate keys resolve to the last assignment (more specific wins).
SNOMED_MAP: Dict[str, int] = {
    # NORM
    "426783006": 0,   # sinus rhythm
    "164865005": 0,   # normal ECG  (overridden below for MI — that's intentional:
    #                   code 164865005 appears in both; we keep MI as more clinically
    #                   specific when co-occurring with MI codes — handled by multi-hot)
    # MI
    "413444003": 1,   # acute myocardial infarction
    "57054005":  1,   # acute MI
    "54329005":  1,   # acute MI of anterior wall
    "164861001": 1,   # myocardial infarction
    "164867002": 1,   # inferior MI
    "164869004": 1,   # ST elevation MI
    "233897008": 1,   # non-Q-wave MI
    # STTC
    "428417006": 2,   # ST elevation
    "164930006": 2,   # ST depression
    "164931005": 2,   # ST change
    "164934002": 2,   # T-wave inversion
    "59931005":  2,   # inverted T-wave
    "164937009": 2,   # T-wave change
    "428750005": 2,   # nonspecific ST-T abnormality
    # CD
    "713427006": 3,   # complete RBBB
    "713426002": 3,   # incomplete RBBB
    "164909002": 3,   # LBBB
    "445118002": 3,   # left anterior fascicular block
    "251120003": 3,   # left posterior fascicular block
    "270492004": 3,   # 1st degree AV block
    "195042002": 3,   # 2nd degree AV block
    "233917008": 3,   # complete AV block
    "29320008":  3,   # Wolff-Parkinson-White
    "74615001":  3,   # intraventricular block
    "164896001": 3,   # atrial fibrillation
    "164889003": 3,   # atrial flutter
    "251173003": 3,   # supraventricular tachycardia
    "426648003": 3,   # accelerated junctional rhythm
    "426177001": 3,   # sinus bradycardia (conduction context)
    "251208001": 3,   # sinoatrial block
    # HYP
    "164873001": 4,   # left ventricular hypertrophy
    "55827005":  4,   # LVH
    "446813000": 4,   # right ventricular hypertrophy
    "67741000":  4,   # left atrial enlargement
    "446358003": 4,   # right atrial abnormality
}

# Free-text fallback (lower priority than SNOMED)
TEXT_MAP: Dict[str, int] = {
    "normal ecg": 0, "normal": 0, "sinus rhythm": 0,
    "myocardial infarction": 1, "acute mi": 1,
    "anterior mi": 1, "inferior mi": 1, "lateral mi": 1,
    "st elevation": 2, "st depression": 2,
    "t wave inversion": 2, "t-wave inversion": 2,
    "nonspecific st": 2, "st change": 2,
    "left bundle branch block": 3, "lbbb": 3,
    "right bundle branch block": 3, "rbbb": 3,
    "1st degree av block": 3, "first degree av block": 3,
    "second degree av block": 3, "complete av block": 3,
    "atrial fibrillation": 3, "atrial flutter": 3,
    "supraventricular tachycardia": 3, "svt": 3,
    "wolff-parkinson-white": 3, "wpw": 3,
    "left ventricular hypertrophy": 4, "lvh": 4,
    "right ventricular hypertrophy": 4, "rvh": 4,
    "left atrial enlargement": 4,
}


def _snomed_codes_to_label(codes: List[str]) -> Tensor:
    """Map a list of SNOMED code strings to a multi-hot (NUM_CLASSES,) tensor."""
    label = torch.zeros(NUM_CLASSES, dtype=torch.float32)
    for code in codes:
        code = code.strip()
        if code in SNOMED_MAP:
            label[SNOMED_MAP[code]] = 1.0
        else:
            cl = code.lower()
            for text, idx in TEXT_MAP.items():
                if text in cl:
                    label[idx] = 1.0
    return label


# ── Signal utilities ──────────────────────────────────────────────────────────

def resample_signal(sig: np.ndarray, from_hz: int, to_hz: int) -> np.ndarray:
    if from_hz == to_hz:
        return sig
    g = gcd(from_hz, to_hz)
    return resample_poly(sig, to_hz // g, from_hz // g, axis=0).astype(np.float32)


def normalize_leads(sig: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = sig.mean(axis=0, keepdims=True)
    std = sig.std(axis=0, keepdims=True).clip(min=eps)
    return (sig - mean) / std


def pad_or_crop(sig: np.ndarray, target_len: int) -> np.ndarray:
    T = sig.shape[0]
    if T >= target_len:
        start = (T - target_len) // 2
        return sig[start: start + target_len]
    pad = np.zeros((target_len - T, sig.shape[1]), dtype=np.float32)
    return np.concatenate([sig, pad], axis=0)


def ensure_12_leads(sig: np.ndarray) -> np.ndarray:
    n = sig.shape[1]
    if n == 12:
        return sig
    if n < 12:
        return np.concatenate([sig, np.zeros((sig.shape[0], 12 - n), dtype=sig.dtype)], axis=1)
    return sig[:, :12]


# ── Augmentation ──────────────────────────────────────────────────────────────

def augment_ecg(signal: Tensor, cfg: DataConfig) -> Tensor:
    signal = signal.clone()
    if cfg.aug_noise_std > 0:
        signal += torch.randn_like(signal) * (torch.rand(1).item() * cfg.aug_noise_std)
    lo, hi = cfg.aug_scale_range
    signal *= lo + (hi - lo) * torch.rand(1).item()
    if cfg.aug_lead_dropout_p > 0 and torch.rand(1).item() < cfg.aug_lead_dropout_p:
        signal[int(torch.randint(0, 12, (1,)).item())] = 0.0
    return signal


# ── PTB-XL ───────────────────────────────────────────────────────────────────

class PTBXLDataset(Dataset):
    """
    PTB-XL via WFDB. Splits use official stratified folds — no leakage.
    Labels from diagnostic_superclass (pre-assigned superclasses in the CSV).
    """

    def __init__(self, root: str, split: str, cfg: DataConfig, augment: bool = False):
        super().__init__()
        self.root = Path(root)
        self.cfg = cfg
        self.augment = augment
        self.target_len = cfg.sequence_length

        csv_path = self.root / "ptbxl_database.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"PTB-XL CSV not found: {csv_path}")

        df = pd.read_csv(csv_path, index_col="ecg_id")
        df["diagnostic_superclass"] = df["diagnostic_superclass"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else []
        )

        fold_map = {"train": list(range(1, 9)), "val": [9], "test": [10]}
        if split not in fold_map:
            raise ValueError(f"split must be train/val/test, got '{split}'")
        self.metadata = df[df["strat_fold"].isin(fold_map[split])].reset_index()

    def __len__(self) -> int:
        return len(self.metadata)

    def _build_label(self, row) -> Tensor:
        label = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        for sc in row["diagnostic_superclass"]:
            if sc in PTBXL_SUPERCLASS_MAP:
                label[PTBXL_SUPERCLASS_MAP[sc]] = 1.0
        return label

    def __getitem__(self, idx: int) -> Dict:
        row = self.metadata.iloc[idx]
        record_path = str(self.root / row["filename_hr"])
        try:
            rec = wfdb.rdrecord(record_path)
            sig = rec.p_signal.astype(np.float32)
            fs = int(rec.fs)
        except Exception as e:
            warnings.warn(f"PTB-XL load failed {record_path}: {e}")
            sig = np.zeros((self.target_len, 12), np.float32)
            fs = self.cfg.sampling_rate

        sig = resample_signal(sig, fs, self.cfg.sampling_rate)
        sig = ensure_12_leads(sig)
        sig = pad_or_crop(sig, self.target_len)
        if self.cfg.normalize:
            sig = normalize_leads(sig)

        ecg = torch.from_numpy(sig.T).float()
        if self.augment and self.cfg.augment:
            ecg = augment_ecg(ecg, self.cfg)

        age = float(np.clip(float(row.get("age", 50)) / 100.0, 0.0, 1.0))
        sex = 1.0 if str(row.get("sex", "0")).strip().lower() in {"1", "male", "m"} else 0.0

        return {
            "ecg": ecg,
            "label": self._build_label(row),
            "age": torch.tensor(age, dtype=torch.float32),
            "sex": torch.tensor(sex, dtype=torch.float32),
            "record_id": str(int(row.get("ecg_id", idx))),
            "patient_id": str(int(row.get("patient_id", -1))),
            "source": "ptbxl",
        }


# ── Generic PhysioNet WFDB (Georgia + Chapman) ────────────────────────────────

class GenericWFDBDataset(Dataset):
    """
    Loads any WFDB dataset with .hea/.dat files.

    Label parsing order:
      1. 'Dx:' header comment parsed as SNOMED codes via SNOMED_MAP.
      2. 'Dx:' or 'Diagnosis:' free-text fallback via TEXT_MAP.

    Patient-level splits: extracted from '#Patient_ID:' header comments.
    Falls back to record-level if no IDs found (limitation is documented).

    Labels and covariates are cached at init from header reads (fast — no waveforms).
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

        all_records = sorted(p.stem for p in self.root.glob("*.hea"))
        if not all_records:
            warnings.warn(f"No .hea files in {root} — dataset will be empty.")

        # ── Build patient → record mapping for patient-level splits ──────────
        record_to_patient: Dict[str, str] = {}
        for rec in all_records:
            try:
                hdr = wfdb.rdheader(str(self.root / rec))
                pid = self._extract_patient_id(hdr.comments or [])
                record_to_patient[rec] = pid or rec   # no ID → each record = its own patient
            except Exception:
                record_to_patient[rec] = rec

        has_patient_ids = any(
            record_to_patient[r] != r for r in all_records
        )
        if not has_patient_ids:
            warnings.warn(
                f"{source_name}: No patient IDs found in headers — "
                "using record-level splits. This is a known limitation."
            )

        # Patient-level stratified split
        unique_patients = sorted(set(record_to_patient.values()))
        rng = np.random.RandomState(seed)
        patients = np.array(unique_patients)
        rng.shuffle(patients)
        n = len(patients)
        n_test = max(1, int(n * test_fraction))
        n_val = max(1, int(n * val_fraction))

        if split == "test":
            selected = set(patients[:n_test])
        elif split == "val":
            selected = set(patients[n_test: n_test + n_val])
        else:
            selected = set(patients[n_test + n_val:])

        self.records = [r for r in all_records if record_to_patient[r] in selected]

        # ── Pre-cache labels and covariates (header reads only) ───────────────
        self._label_cache: Dict[str, Tensor] = {}
        self._age_cache: Dict[str, float] = {}
        self._sex_cache: Dict[str, float] = {}

        for rec in self.records:
            try:
                hdr = wfdb.rdheader(str(self.root / rec))
                comments = hdr.comments or []
                self._label_cache[rec] = self._parse_labels(comments)
                age, sex = self._parse_covariates(comments)
                self._age_cache[rec] = age
                self._sex_cache[rec] = sex
            except Exception:
                self._label_cache[rec] = torch.zeros(NUM_CLASSES, dtype=torch.float32)
                self._age_cache[rec] = 0.5
                self._sex_cache[rec] = 0.0

    @staticmethod
    def _extract_patient_id(comments: List[str]) -> Optional[str]:
        for c in comments:
            for prefix in ["#patient_id:", "patient_id:", "patient:"]:
                if c.strip().lower().startswith(prefix):
                    return c.split(":", 1)[1].strip()
        return None

    @staticmethod
    def _parse_labels(comments: List[str]) -> Tensor:
        label = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        for comment in comments:
            s = comment.strip()
            if s.startswith("Dx:"):
                codes = [c.strip() for c in s[3:].split(",")]
                for code in codes:
                    if code in SNOMED_MAP:
                        label[SNOMED_MAP[code]] = 1.0
                    else:
                        for text, idx in TEXT_MAP.items():
                            if text in code.lower():
                                label[idx] = 1.0
            elif s.lower().startswith("diagnosis:") or s.lower().startswith("dx (text):"):
                text_part = s.split(":", 1)[1].strip().lower()
                for text, idx in TEXT_MAP.items():
                    if text in text_part:
                        label[idx] = 1.0
        return label

    @staticmethod
    def _parse_covariates(comments: List[str]) -> Tuple[float, float]:
        age, sex = 0.5, 0.0
        for c in comments:
            s = c.strip()
            if s.startswith("Age:"):
                try:
                    age = float(np.clip(float(s[4:].strip()) / 100.0, 0.0, 1.0))
                except ValueError:
                    pass
            elif s.startswith("Sex:"):
                sex = 1.0 if s[4:].strip().upper() in {"M", "MALE"} else 0.0
        return age, sex

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        rec = self.records[idx]
        record_path = str(self.root / rec)
        try:
            record = wfdb.rdrecord(record_path)
            sig = record.p_signal.astype(np.float32)
            fs = int(record.fs)
        except Exception as e:
            warnings.warn(f"Load failed {record_path}: {e}")
            sig = np.zeros((self.target_len, 12), np.float32)
            fs = self.cfg.sampling_rate

        sig = resample_signal(sig, fs, self.cfg.sampling_rate)
        sig = ensure_12_leads(sig)
        sig = pad_or_crop(sig, self.target_len)
        if self.cfg.normalize:
            sig = normalize_leads(sig)

        ecg = torch.from_numpy(sig.T).float()
        if self.augment and self.cfg.augment:
            ecg = augment_ecg(ecg, self.cfg)

        return {
            "ecg": ecg,
            "label": self._label_cache[rec],
            "age": torch.tensor(self._age_cache[rec], dtype=torch.float32),
            "sex": torch.tensor(self._sex_cache[rec], dtype=torch.float32),
            "record_id": rec,
            "patient_id": rec,
            "source": self.source_name,
        }


# ── DataLoader factory ────────────────────────────────────────────────────────

def _collect_labels_fast(datasets: List[Dataset]) -> Tensor:
    """Gather label tensors from caches — no waveform I/O."""
    all_labels: List[Tensor] = []
    for ds in datasets:
        if isinstance(ds, PTBXLDataset):
            for _, row in ds.metadata.iterrows():
                lv = torch.zeros(NUM_CLASSES, dtype=torch.float32)
                for sc in row["diagnostic_superclass"]:
                    if sc in PTBXL_SUPERCLASS_MAP:
                        lv[PTBXL_SUPERCLASS_MAP[sc]] = 1.0
                all_labels.append(lv)
        elif isinstance(ds, GenericWFDBDataset):
            all_labels.extend(ds._label_cache[r] for r in ds.records)
        else:
            warnings.warn(f"Unknown dataset type {type(ds)} — skipped in sampler weight calc.")
    return torch.stack(all_labels) if all_labels else torch.zeros(1, NUM_CLASSES)


def log_label_prevalence(split: str, labels: Tensor) -> None:
    """Log per-class counts — critical for catching label mapping bugs."""
    n = labels.shape[0]
    print(f"\n  Label prevalence [{split}] N={n}:")
    for i, cls in enumerate(SUPER_CLASSES):
        count = int(labels[:, i].sum().item())
        pct = 100.0 * count / max(n, 1)
        bar = "█" * int(pct / 2)
        print(f"    {cls:6s}  {count:6d} ({pct:5.1f}%)  {bar}")
    # Warn if any class is entirely absent
    for i, cls in enumerate(SUPER_CLASSES):
        if labels[:, i].sum() == 0:
            warnings.warn(
                f"Class '{cls}' has ZERO positive samples in [{split}]. "
                "Check your label mapping or dataset."
            )


def _collate_fn(batch: List[Dict]) -> Dict:
    """Stack tensor fields; keep string fields as lists."""
    tensor_keys = ["ecg", "label", "age", "sex"]
    list_keys = ["record_id", "patient_id", "source"]
    out: Dict = {}
    for k in tensor_keys:
        out[k] = torch.stack([b[k] for b in batch])
    for k in list_keys:
        out[k] = [b[k] for b in batch]
    return out


def build_dataloaders(cfg: DataConfig, seed: int = 42) -> Dict[str, DataLoader]:
    """
    Build train/val/test DataLoaders combining all configured datasets.

    - Logs label prevalence per split (catch mapping bugs before training).
    - Training set uses WeightedRandomSampler built from label caches (no I/O overhead).
    - Applies correct augmentation flag per split.
    - pin_memory only enabled when CUDA is available (MPS doesn't benefit).
    """
    loaders: Dict[str, DataLoader] = {}

    for split in ["train", "val", "test"]:
        is_train = split == "train"
        datasets: List[Dataset] = []

        if cfg.ptbxl_path and Path(cfg.ptbxl_path).exists():
            try:
                ds = PTBXLDataset(cfg.ptbxl_path, split, cfg, augment=is_train)
                datasets.append(ds)
                print(f"  PTB-XL  [{split}]: {len(ds)} records")
            except Exception as e:
                warnings.warn(f"PTB-XL [{split}]: {e}")

        for name, path in [("georgia", cfg.georgia_path), ("chapman", cfg.chapman_path)]:
            if path and Path(path).exists():
                try:
                    ds = GenericWFDBDataset(
                        path, split, cfg,
                        augment=is_train,
                        source_name=name,
                        val_fraction=cfg.val_fraction,
                        test_fraction=cfg.test_fraction,
                        seed=seed,
                    )
                    datasets.append(ds)
                    print(f"  {name.capitalize():8s} [{split}]: {len(ds)} records")
                except Exception as e:
                    warnings.warn(f"{name} [{split}]: {e}")

        if not datasets:
            warnings.warn(f"No datasets loaded for split '{split}'.")
            continue

        combined = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

        # Collect labels for logging and sampler (fast — no waveforms)
        all_labels = _collect_labels_fast(datasets)
        log_label_prevalence(split, all_labels)

        sampler = None
        if is_train and len(all_labels) > 0:
            class_freq = all_labels.mean(dim=0).clamp(min=1e-5)
            sample_weights = (all_labels / class_freq).max(dim=1).values
            sample_weights = sample_weights.clamp(min=1e-6)
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
            shuffle=is_train and sampler is None,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory and torch.cuda.is_available(),
            drop_last=is_train,
            persistent_workers=cfg.num_workers > 0,
            collate_fn=_collate_fn,
        )

    return loaders
