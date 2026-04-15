from __future__ import annotations

import csv
import pickle
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from src.config import FEATURE_BASELINES, WESAD_LABEL_MAP


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


def _safe_zscore(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    std_safe = np.where(std == 0, 1.0, std)
    return (x - mean) / std_safe


def load_scaler_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if path.exists():
        try:
            data = np.load(path)
            if "mean" in data and "std" in data:
                return data["mean"].astype(np.float32), data["std"].astype(np.float32)
            warnings.warn(f"Scaler file found but invalid keys at {path}, fallback to identity scaler.")
        except Exception:
            # This loader is npz-only; tolerate other formats (e.g. pkl for TCM model scaler).
            warnings.warn(f"Scaler path is not a readable npz ({path}); fallback to identity scaler.")
    return np.zeros((8,), dtype=np.float32), np.ones((8,), dtype=np.float32)


def _safe_float(v, default: float) -> float:
    try:
        if v is None:
            return default
        s = str(v).strip()
        if s == "" or s.lower() in {"nan", "none"}:
            return default
        return float(s)
    except Exception:
        return default


def _gender_to_numeric(v) -> float:
    s = str(v or "").strip().lower()
    if s.startswith("m"):
        return 1.0
    if s.startswith("f"):
        return 0.0
    return 0.0


def _compute_bmi(weight_kg: float, height_cm: float) -> float:
    if np.isnan(weight_kg) or np.isnan(height_cm) or height_cm <= 0:
        return np.nan
    h_m = height_cm / 100.0
    if h_m <= 0:
        return np.nan
    return float(weight_kg / (h_m * h_m))


def estimate_hr_from_ecg(ecg: np.ndarray, sampling_rate: float = 700.0) -> float:
    if ecg.size < 5 or find_peaks is None:
        return 70.0
    peaks, _ = find_peaks(ecg, distance=max(1, int(0.3 * sampling_rate)))
    if len(peaks) < 2:
        return 70.0
    rr = np.diff(peaks) / sampling_rate
    mean_rr = float(np.mean(rr)) if rr.size > 0 else 0.0
    if mean_rr <= 0:
        return 70.0
    return float(60.0 / mean_rr)


def _to_1d(x) -> np.ndarray:
    return np.asarray(x).squeeze().astype(np.float32)


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


def _parse_wesad_readme_demographics(readme_path: Path) -> Tuple[float, float, float]:
    """Return (age, gender_numeric[male=1/female=0], bmi)."""
    age = 35.0
    gender = 1.0  # default male=1
    bmi = FEATURE_BASELINES["bmi"]
    if not readme_path.exists():
        return age, gender, bmi

    text = readme_path.read_text(encoding="utf-8", errors="ignore")
    age_match = re.search(r"Age\s*:\s*(\d+)", text, flags=re.IGNORECASE)
    if age_match:
        age = float(age_match.group(1))

    gender_match = re.search(r"Gender\s*:\s*([A-Za-z]+)", text, flags=re.IGNORECASE)
    if gender_match:
        g = gender_match.group(1).strip().lower()
        if g.startswith("m"):
            gender = 1.0
        elif g.startswith("f"):
            gender = 0.0

    h_match = re.search(r"Height\s*\(cm\)\s*:\s*([0-9]+(?:\.[0-9]+)?)", text, flags=re.IGNORECASE)
    w_match = re.search(r"Weight\s*\(kg\)\s*:\s*([0-9]+(?:\.[0-9]+)?)", text, flags=re.IGNORECASE)
    if h_match and w_match:
        height_cm = float(h_match.group(1))
        weight_kg = float(w_match.group(1))
        bmi_val = _compute_bmi(weight_kg, height_cm)
        if not np.isnan(bmi_val):
            bmi = float(bmi_val)

    return age, gender, bmi


def _load_wesad_subject_info_csv(wesad_dir: Path) -> Dict[str, Dict[str, float]]:
    """Load optional subject_info.csv with keys: Subject_ID, Age, Gender, Height, Weight."""
    # Search both current wesad_dir and its parent to tolerate layouts:
    # data/wesad/WESAD/... and data/wesad/subject_info.csv
    candidates = list(wesad_dir.rglob("subject_info.csv"))
    parent = wesad_dir.parent
    if parent.exists():
        candidates.extend(parent.rglob("subject_info.csv"))
    # de-dup while preserving order
    seen = set()
    deduped = []
    for c in candidates:
        s = str(c.resolve())
        if s not in seen:
            seen.add(s)
            deduped.append(c)
    candidates = deduped
    if not candidates:
        return {}
    info_path = candidates[0]
    subject_map: Dict[str, Dict[str, float]] = {}
    with open(info_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid_raw = str(row.get("Subject_ID") or row.get("subject_id") or "").strip()
            if not sid_raw:
                continue
            sid = sid_raw if sid_raw.startswith("S") else f"S{sid_raw}"
            age = _safe_float(row.get("Age") or row.get("age"), 35.0)
            gender = _gender_to_numeric(row.get("Gender") or row.get("gender"))
            height_cm = _safe_float(row.get("Height") or row.get("height"), np.nan)
            weight_kg = _safe_float(row.get("Weight") or row.get("weight"), np.nan)
            bmi = _compute_bmi(weight_kg, height_cm)
            if np.isnan(bmi):
                bmi = FEATURE_BASELINES["bmi"]
            subject_map[sid] = {
                "age": float(age),
                "gender": float(gender),
                "bmi": float(bmi),
            }
    return subject_map


def estimate_hr_from_bvp_baseline(
    bvp: np.ndarray,
    labels: np.ndarray | None = None,
    sampling_rate: float = 64.0,
) -> float:
    """Estimate baseline HR from wrist BVP peaks (S2/baseline priority).

    Uses scipy.signal.find_peaks with distance=int(64/1.5) as requested.
    """
    if bvp.size < 8 or find_peaks is None:
        return 70.0

    baseline = bvp
    if labels is not None and labels.size == bvp.size:
        # WESAD baseline label is 1.0
        mask = labels.astype(np.int32) == 1
        if np.any(mask):
            baseline = bvp[mask]

    if baseline.size < 8:
        baseline = bvp

    distance = max(1, int(sampling_rate / 1.5))
    peaks, _ = find_peaks(baseline, distance=distance)
    if len(peaks) < 2:
        return 70.0

    rr = np.diff(peaks) / sampling_rate
    mean_rr = float(np.mean(rr)) if rr.size > 0 else 0.0
    if mean_rr <= 0:
        return 70.0
    return float(60.0 / mean_rr)


@dataclass
class Sample:
    dynamic: np.ndarray  # [2, 1000]
    static: np.ndarray  # [4] -> [Age, Gender, BMI, HeartRate]
    target: float


class WESADDataset(Dataset):
    """WESAD windows with strict label reconstruction and 50% overlap.

    Primary loader: .pkl (official released format).
    Fallback loader: .mat (kept for compatibility).
    """

    def __init__(
        self,
        wesad_dir: Path,
        scaler_path: Path,
        window_size: int = 1000,
        overlap: float = 0.5,
        use_tcm: bool = True,
    ):
        self.samples: List[Sample] = []
        self.window_size = window_size
        self.stride = compute_stride(window_size, overlap)
        self.use_tcm = bool(use_tcm)
        self.mean, self.std = load_scaler_npz(scaler_path)
        self.subject_static_4d: Dict[str, np.ndarray] = {}
        self._build(wesad_dir)

    def _build(self, wesad_dir: Path) -> None:
        pkl_files = sorted(wesad_dir.rglob("S*.pkl"))
        if pkl_files:
            self._build_from_pkl(pkl_files)
        else:
            self._build_from_mat(wesad_dir)

        if not self.samples:
            raise RuntimeError("WESAD samples are empty after filtering. Check source files and label mapping.")

    def _build_from_pkl(self, pkl_files: List[Path]) -> None:
        csv_subject_info = _load_wesad_subject_info_csv(pkl_files[0].parents[1] if pkl_files else Path("."))
        for pkl_path in pkl_files:
            subj_dir = pkl_path.parent
            subject_name = subj_dir.name

            try:
                with open(pkl_path, "rb") as f:
                    data = pickle.load(f, encoding="latin1")
            except Exception as exc:
                warnings.warn(f"Skip corrupted WESAD pickle {pkl_path}: {exc}")
                continue

            if "signal" not in data or "label" not in data or "chest" not in data["signal"]:
                warnings.warn(f"Skip unexpected WESAD pickle format: {pkl_path}")
                continue

            chest = data["signal"]["chest"]
            if "ECG" not in chest or "EDA" not in chest:
                warnings.warn(f"Skip WESAD subject missing ECG/EDA: {pkl_path}")
                continue

            ecg = _to_1d(chest["ECG"])
            eda = _to_1d(chest["EDA"])
            labels = _to_1d(data["label"]).astype(np.int32)
            temp = _to_1d(chest["Temp"]) if "Temp" in chest else np.full_like(eda, 36.5)
            wrist = data["signal"].get("wrist", {}) if isinstance(data["signal"], dict) else {}
            bvp = _to_1d(wrist.get("BVP", np.array([], dtype=np.float32)))

            n = min(len(ecg), len(eda), len(labels), len(temp))
            ecg = ecg[:n]
            eda = eda[:n]
            labels = labels[:n]
            temp = temp[:n]
            bvp = bvp[:n] if bvp.size >= n else bvp

            # Subject-level static 4D features: [Age, Gender, BMI, Calculated_HR]
            info_from_csv = csv_subject_info.get(subject_name)
            if info_from_csv is not None:
                age = info_from_csv["age"]
                gender = info_from_csv["gender"]
                bmi = info_from_csv["bmi"]
            else:
                age, gender, bmi = _parse_wesad_readme_demographics(subj_dir / f"{subject_name}_readme.txt")

            hr_baseline = estimate_hr_from_bvp_baseline(bvp, labels=labels if bvp.size == labels.size else None, sampling_rate=64.0)
            static_4d = np.array([age, gender, bmi, hr_baseline], dtype=np.float32)
            self.subject_static_4d[subject_name] = static_4d
            print(
                f"Subject {subject_name}: Age={age:.1f}, Gender={gender:.0f}, BMI={bmi:.1f}, HR={hr_baseline:.1f}",
                flush=True,
            )

            for start in range(0, n - self.window_size + 1, self.stride):
                end = start + self.window_size
                window_labels = labels[start:end]

                # Strictly remove transition windows.
                if np.any(window_labels == 0):
                    continue

                window_label = int(np.round(np.median(window_labels)))
                target = map_wesad_label(window_label)
                if target is None:
                    continue

                ecg_w = ecg[start:end]
                eda_w = eda[start:end]
                static = static_4d.copy()
                dyn = np.stack([ecg_w, eda_w], axis=0).astype(np.float32)
                self.samples.append(Sample(dynamic=dyn, static=static, target=float(target)))

    def _build_from_mat(self, wesad_dir: Path) -> None:
        if loadmat is None:
            raise ImportError("scipy is required for WESAD .mat loading")

        mat_files = sorted(wesad_dir.rglob("*.mat"))
        if not mat_files:
            raise FileNotFoundError(
                f"No WESAD .pkl or .mat files found under {wesad_dir}. Current code expects WESAD release files."
            )

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
                if np.any(window_labels == 0):
                    continue
                window_label = int(np.round(np.median(window_labels)))
                target = map_wesad_label(window_label)
                if target is None:
                    continue

                ecg_w = ecg[start:end]
                eda_w = eda[start:end]
                hr = estimate_hr_from_ecg(ecg_w)
                static = np.array([35.0, 1.0, FEATURE_BASELINES["bmi"], hr], dtype=np.float32)
                dyn = np.stack([ecg_w, eda_w], axis=0).astype(np.float32)
                self.samples.append(Sample(dynamic=dyn, static=static, target=float(target)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        static = s.static if self.use_tcm else np.zeros((4,), dtype=np.float32)
        return (
            torch.tensor(s.dynamic, dtype=torch.float32),
            torch.tensor([s.target], dtype=torch.float32),
            torch.tensor(static, dtype=torch.float32),
        )


class BUTPPGDataset(Dataset):
    """BUT PPG v2.0.0 loader for HR regression (quality-filtered).

    Data contract for cross-domain A:
    - Read `quality-hr-ann.csv`
    - Keep only rows with Quality == 1
    - Use HR column directly as BPM target
    - Read each `*_PPG.dat` and duplicate first channel to [2, L]
    """

    def __init__(
        self,
        but_dir: Path,
        window_size: int = 1000,
        scaler_path: Optional[Path] = None,
    ):
        self.window_size = window_size
        self.samples: List[Sample] = []
        self._build(but_dir)

    def _build(self, but_dir: Path) -> None:
        self._build_from_wfdb(but_dir)

        if not self.samples:
            raise RuntimeError("BUT PPG samples are empty; check data format and HR extraction.")

    def _load_quality_hr_ann(self, but_dir: Path) -> Dict[str, float]:
        ann_path = next(iter(sorted(but_dir.rglob("quality-hr-ann.csv"))), None)
        if ann_path is None:
            raise FileNotFoundError(
                f"quality-hr-ann.csv not found under {but_dir}. This file is required for BUT HR validation."
            )
        hr_labels: Dict[str, float] = {}
        with open(ann_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rid = (row.get("ID") or "").strip()
                if not rid:
                    continue
                quality = _safe_float(row.get("Quality"), np.nan)
                hr = _safe_float(row.get("HR"), np.nan)
                # Strict cleaning: only keep high-quality labeled records.
                if not np.isnan(quality) and int(quality) == 1 and (not np.isnan(hr)) and hr > 0:
                    hr_labels[rid] = float(hr)
        if not hr_labels:
            raise RuntimeError("quality-hr-ann.csv loaded but no Quality=1 HR labels were found.")
        return hr_labels

    def _load_subject_info(self, but_dir: Path) -> Dict[str, Dict]:
        info_path = next(iter(sorted(but_dir.rglob("subject-info.csv"))), None)
        if info_path is None:
            warnings.warn("subject-info.csv not found; fallback static features will be used.")
            return {}

        info: Dict[str, Dict] = {}
        with open(info_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rid = (row.get("ID") or "").strip()
                if not rid:
                    continue

                bp_str = (row.get("Blood pressure [mmHg]") or "").strip()
                sbp = None
                dbp = None
                if "/" in bp_str:
                    parts = bp_str.split("/")
                    try:
                        sbp = float(parts[0])
                        dbp = float(parts[1])
                    except Exception:
                        sbp = None
                        dbp = None

                age = _safe_float(row.get("Age [years]"), 35.0)
                gender = _gender_to_numeric(row.get("Gender"))
                height_cm = _safe_float(row.get("Height [cm]"), np.nan)
                weight_kg = _safe_float(row.get("Weight [kg]"), np.nan)
                spo2 = _safe_float(row.get("SpO2 [%]"), FEATURE_BASELINES["spo2"])
                bmi = _compute_bmi(weight_kg, height_cm)
                if np.isnan(bmi):
                    bmi = FEATURE_BASELINES["bmi"]

                static = np.array(
                    [
                        age,
                        gender,
                        bmi,
                        70.0,  # placeholder HR at metadata level (no leakage from target HR)
                    ],
                    dtype=np.float32,
                )

                info[rid] = {
                    "static": static,
                }

        return info

    def _parse_hea(self, hea_path: Path) -> Tuple[int, float, Optional[Path]]:
        if not hea_path.exists():
            fallback = hea_path.with_suffix(".dat")
            return 1, 30.0, fallback if fallback.exists() else None
        lines = hea_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if not lines:
            return 0, 30.0, None

        header_parts = lines[0].split()
        if len(header_parts) < 2:
            return 0, 30.0, None

        # Handle both standard WFDB and BUT custom-like header variants:
        # e.g. "100001_PPG 300 30 1" where the last token is channel count.
        n_sig = 0
        fs = 30.0
        if len(header_parts) >= 4:
            try:
                cand1 = int(float(header_parts[1]))
                cand2 = int(float(header_parts[3]))
                fs = float(header_parts[2])
                if cand1 > 64 and 1 <= cand2 <= 16:
                    n_sig = cand2
                else:
                    n_sig = cand1
            except Exception:
                pass
        if n_sig <= 0:
            try:
                n_sig = int(float(header_parts[1]))
            except Exception:
                n_sig = 1
            try:
                fs = float(header_parts[2]) if len(header_parts) >= 3 else 30.0
            except Exception:
                fs = 30.0

        dat_path = None
        for line in lines[1:]:
            parts = line.split()
            if not parts:
                continue
            if parts[0].endswith(".dat"):
                dat_path = hea_path.parent / parts[0]
                break

        if dat_path is None:
            fallback = hea_path.with_suffix(".dat")
            if fallback.exists():
                dat_path = fallback

        return n_sig, fs, dat_path

    def _fit_to_window(self, x: np.ndarray) -> np.ndarray:
        """Resize temporal length to window_size using linear interpolation."""
        if x.shape[-1] == self.window_size:
            return x.astype(np.float32)
        old_len = x.shape[-1]
        if old_len <= 1:
            return np.repeat(x, self.window_size, axis=-1).astype(np.float32)
        old_idx = np.linspace(0.0, 1.0, old_len)
        new_idx = np.linspace(0.0, 1.0, self.window_size)
        out = np.stack([np.interp(new_idx, old_idx, ch) for ch in x], axis=0)
        return out.astype(np.float32)

    def _build_from_wfdb(self, but_dir: Path) -> None:
        hr_labels = self._load_quality_hr_ann(but_dir)
        subject_info = self._load_subject_info(but_dir)
        dat_files = sorted(but_dir.rglob("*_PPG.dat"))
        if not dat_files:
            raise FileNotFoundError(f"No *_PPG.dat files found under {but_dir}")

        for dat_path in dat_files:
            rid = dat_path.stem.replace("_PPG", "").strip()
            if rid not in hr_labels:
                # Strictly skip Quality=0 or missing annotation IDs.
                continue
            meta = subject_info.get(rid)
            if meta is None:
                fallback_static = np.array(
                    [
                        35.0,
                        1.0,
                        FEATURE_BASELINES["bmi"],
                        70.0,
                    ],
                    dtype=np.float32,
                )
                meta = {"static": fallback_static}

            hea_path = dat_path.with_suffix(".hea")
            n_sig, fs, dat_path = self._parse_hea(hea_path)
            if n_sig <= 0 or dat_path is None or not dat_path.exists():
                continue

            raw = np.fromfile(dat_path, dtype=np.int16)
            if raw.size == 0:
                continue
            if n_sig > 1 and raw.size % n_sig == 0:
                signal = raw.reshape(-1, n_sig).astype(np.float32)  # [L, C]
                ch0 = signal[:, 0]
            else:
                ch0 = raw.astype(np.float32)
            ch0_t = torch.tensor(ch0, dtype=torch.float32)
            x = torch.stack([ch0_t, ch0_t], dim=0).numpy()
            x = self._fit_to_window(x)

            self.samples.append(
                Sample(
                    dynamic=x.astype(np.float32),
                    static=meta["static"].copy(),
                    target=float(hr_labels[rid]),
                )
            )

    def _scan_h5_datasets(self, h5file):
        found = {}

        def _visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                lname = name.lower()
                if "sbp" in lname:
                    found["sbp"] = np.asarray(obj)
                elif any(k in lname for k in ["hr", "heart_rate", "heart rate", "bpm", "pulse"]):
                    found.setdefault("hr", np.asarray(obj))
                elif any(k in lname for k in ["ppg", "signal", "wave"]):
                    found.setdefault("signals", []).append(np.asarray(obj))

        h5file.visititems(_visit)
        return found

    def _build_from_h5(self, but_dir: Path) -> None:
        raise RuntimeError(
            "BUT cross-domain validation now requires WFDB-style .dat + quality-hr-ann.csv and does not use .h5."
        )

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
