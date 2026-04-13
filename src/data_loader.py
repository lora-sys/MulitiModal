from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from src.config import FEATURE_BASELINES, Paths, TrainConfig, WESAD_LABEL_MAP


try:
    from scipy.io import loadmat
    from scipy.signal import find_peaks
except Exception:  # pragma: no cover - handled at runtime
    loadmat = None
    find_peaks = None

try:
    import h5py
except Exception:  # pragma: no cover
    h5py = None


def map_wesad_label(raw_label: int) -> float | None:
    """Required strict mapping: {1:1.0, 2:0.0, 3:0.6}; drop 0/others."""
    return WESAD_LABEL_MAP.get(int(raw_label), None)


def compute_stride(window_size: int, overlap: float) -> int:
    stride = int(window_size * (1.0 - overlap))
    return max(stride, 1)


def sliding_windows(arr: np.ndarray, window_size: int, stride: int) -> List[np.ndarray]:
    if arr.shape[-1] < window_size:
        return []
    out = []
    for start in range(0, arr.shape[-1] - window_size + 1, stride):
        out.append(arr[..., start : start + window_size])
    return out


def _safe_zscore(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    std_safe = np.where(std == 0, 1.0, std)
    return (x - mean) / std_safe


def load_scaler_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if path.exists():
        data = np.load(path)
        if "mean" in data and "std" in data:
            return data["mean"].astype(np.float32), data["std"].astype(np.float32)
        warnings.warn(f"Scaler file found but invalid keys at {path}, fallback to identity scaler.")
    return np.zeros((8,), dtype=np.float32), np.ones((8,), dtype=np.float32)


def estimate_hr_from_ecg(ecg: np.ndarray, sampling_rate: float = 700.0) -> float:
    if ecg.size < 5:
        return 70.0
    if find_peaks is None:
        return 70.0
    peaks, _ = find_peaks(ecg, distance=max(1, int(0.3 * sampling_rate)))
    if len(peaks) < 2:
        return 70.0
    rr = np.diff(peaks) / sampling_rate
    mean_rr = float(np.mean(rr)) if rr.size > 0 else 0.0
    if mean_rr <= 0:
        return 70.0
    return float(60.0 / mean_rr)


def _extract_nested(mat_obj, keys: List[str]):
    cur = mat_obj
    for key in keys:
        if isinstance(cur, np.ndarray) and cur.dtype.names is not None:
            cur = cur[key]
        elif isinstance(cur, np.ndarray) and cur.shape == (1, 1):
            cur = cur[0, 0]
            if hasattr(cur, key):
                cur = getattr(cur, key)
            else:
                cur = cur[key]
        elif hasattr(cur, "__getitem__"):
            cur = cur[key]
        else:
            raise KeyError(f"Cannot descend key={key}")
    return cur


def _to_1d(x) -> np.ndarray:
    arr = np.asarray(x).squeeze()
    return arr.astype(np.float32)


def _field(obj, name: str):
    """Robust field access for MATLAB structs loaded by scipy."""
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, np.ndarray) and obj.dtype.names and name in obj.dtype.names:
        return obj[name]
    if hasattr(obj, "__getitem__"):
        try:
            return obj[name]
        except Exception:
            pass
    raise KeyError(f"Field '{name}' not found")


@dataclass
class Sample:
    dynamic: np.ndarray  # [2, 1000]
    static: np.ndarray  # [8]
    target: float


class WESADDataset(Dataset):
    """WESAD windows with strict label reconstruction and 50% overlap."""

    def __init__(
        self,
        wesad_dir: Path,
        scaler_path: Path,
        window_size: int = 1000,
        overlap: float = 0.5,
    ):
        if loadmat is None:
            raise ImportError("scipy is required for WESAD .mat loading")
        self.samples: List[Sample] = []
        self.window_size = window_size
        self.stride = compute_stride(window_size, overlap)
        self.mean, self.std = load_scaler_npz(scaler_path)
        self._build(wesad_dir)

    def _build(self, wesad_dir: Path) -> None:
        mat_files = sorted(wesad_dir.rglob("*.mat"))
        if not mat_files:
            raise FileNotFoundError(f"No .mat files found under {wesad_dir}")

        for mat_path in mat_files:
            data = loadmat(mat_path, squeeze_me=False, struct_as_record=False)
            if "signal" not in data or "label" not in data:
                continue

            chest = data["signal"]["chest"][0, 0]
            ecg = _to_1d(_field(chest, "ECG"))
            eda = _to_1d(_field(chest, "EDA"))
            labels = _to_1d(data["label"]).astype(np.int32)

            try:
                temp = _to_1d(_field(chest, "Temp"))
            except Exception:
                temp = np.full_like(eda, 36.5)

            n = min(len(ecg), len(eda), len(labels), len(temp))
            ecg = ecg[:n]
            eda = eda[:n]
            labels = labels[:n]
            temp = temp[:n]

            for start in range(0, n - self.window_size + 1, self.stride):
                end = start + self.window_size
                window_labels = labels[start:end]
                # Strictly remove transition windows if any raw label is 0.
                if np.any(window_labels == 0):
                    continue

                window_label = int(np.round(np.median(window_labels)))
                target = map_wesad_label(window_label)
                if target is None:
                    continue

                ecg_w = ecg[start:end]
                eda_w = eda[start:end]
                temp_w = temp[start:end]

                hr = estimate_hr_from_ecg(ecg_w)
                static = np.array(
                    [
                        35.0,  # age (WESAD not provided in this loader)
                        0.0,  # gender (unknown -> neutral default)
                        FEATURE_BASELINES["bmi"],
                        hr,
                        FEATURE_BASELINES["sbp"],
                        FEATURE_BASELINES["dbp"],
                        FEATURE_BASELINES["spo2"],
                        float(np.mean(temp_w) if len(temp_w) else 36.5),
                    ],
                    dtype=np.float32,
                )
                static = _safe_zscore(static, self.mean, self.std)

                dyn = np.stack([ecg_w, eda_w], axis=0).astype(np.float32)
                self.samples.append(Sample(dynamic=dyn, static=static, target=float(target)))

        if not self.samples:
            raise RuntimeError("WESAD samples are empty after filtering. Check label mapping and source files.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return (
            torch.tensor(s.dynamic, dtype=torch.float32),
            torch.tensor(s.static, dtype=torch.float32),
            torch.tensor([s.target], dtype=torch.float32),
        )


class BUTPPGDataset(Dataset):
    """BUT PPG v2.0.0 loader for SBP regression."""

    def __init__(self, but_dir: Path, window_size: int = 1000):
        if h5py is None:
            raise ImportError("h5py is required for BUT PPG loading")
        self.window_size = window_size
        self.samples: List[Sample] = []
        self.proj = torch.nn.Conv1d(3, 2, kernel_size=1, bias=False)
        for p in self.proj.parameters():
            p.requires_grad = False
        self._build(but_dir)

    def _scan_datasets(self, h5file):
        found = {}

        def _visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                lname = name.lower()
                if "sbp" in lname:
                    found["sbp"] = np.asarray(obj)
                elif any(k in lname for k in ["ppg", "signal", "wave"]):
                    found.setdefault("signals", []).append(np.asarray(obj))

        h5file.visititems(_visit)
        return found

    def _build(self, but_dir: Path) -> None:
        h5_files = sorted(but_dir.rglob("*.h5"))
        if not h5_files:
            raise FileNotFoundError(f"No .h5 files found under {but_dir}")

        for h5_path in h5_files:
            with h5py.File(h5_path, "r") as f:
                found = self._scan_datasets(f)
                if "sbp" not in found or "signals" not in found:
                    continue
                sbp = np.asarray(found["sbp"]).reshape(-1)
                signal = np.asarray(found["signals"][0])

                # expected shape -> [N, C, L]
                if signal.ndim == 2:
                    signal = signal[:, None, :]
                if signal.ndim == 3 and signal.shape[1] not in (2, 3):
                    signal = np.transpose(signal, (0, 2, 1))
                if signal.shape[1] not in (2, 3):
                    continue

                n = min(len(sbp), signal.shape[0])
                for i in range(n):
                    x = signal[i].astype(np.float32)
                    # force to length 1000
                    if x.shape[-1] < self.window_size:
                        continue
                    x = x[..., : self.window_size]
                    if x.shape[0] == 3:
                        with torch.no_grad():
                            x = self.proj(torch.tensor(x[None, ...], dtype=torch.float32)).squeeze(0).numpy()
                    self.samples.append(
                        Sample(
                            dynamic=x,
                            static=np.zeros((8,), dtype=np.float32),
                            target=float(sbp[i]),
                        )
                    )

        if not self.samples:
            raise RuntimeError("BUT PPG samples are empty; check v2.0.0 file layout.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return (
            torch.tensor(s.dynamic, dtype=torch.float32),
            torch.tensor(s.static, dtype=torch.float32),
            torch.tensor([s.target], dtype=torch.float32),
        )


class MIMICStaticDataset(Dataset):
    """Static-only dataset for mechanism-level validation."""

    def __init__(self, csv_path: Path, scaler_path: Path):
        if not csv_path.exists():
            raise FileNotFoundError(f"MIMIC csv not found: {csv_path}")
        self.df = pd.read_csv(csv_path)
        self.mean, self.std = load_scaler_npz(scaler_path)
        self.features, self.sbp = self._build_arrays()

    def _col(self, *names: str, default: float = 0.0) -> np.ndarray:
        low = {c.lower(): c for c in self.df.columns}
        for name in names:
            if name.lower() in low:
                return self.df[low[name.lower()]].to_numpy(dtype=np.float32)
        return np.full((len(self.df),), default, dtype=np.float32)

    def _build_arrays(self):
        age = self._col("age", default=35.0)
        gender_raw = self._col("gender", "sex", default=0.0)
        gender = np.where(gender_raw > 0.5, 1.0, 0.0).astype(np.float32)
        bmi = self._col("bmi", default=FEATURE_BASELINES["bmi"])
        hr = self._col("hr", "heartrate", default=70.0)
        sbp = self._col("sbp", "systolic_bp", default=FEATURE_BASELINES["sbp"])
        dbp = self._col("dbp", "diastolic_bp", default=FEATURE_BASELINES["dbp"])
        spo2 = self._col("spo2", default=FEATURE_BASELINES["spo2"])
        temp = self._col("temp", "temperature", default=36.5)
        x = np.stack([age, gender, bmi, hr, sbp, dbp, spo2, temp], axis=1).astype(np.float32)
        x = _safe_zscore(x, self.mean, self.std)
        return x, sbp.astype(np.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor([self.sbp[idx]], dtype=torch.float32),
        )


def make_train_val_loaders(dataset: Dataset, batch_size: int, val_ratio: float = 0.2, seed: int = 42):
    n_total = len(dataset)
    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
