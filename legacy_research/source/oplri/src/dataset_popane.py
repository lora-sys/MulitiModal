from __future__ import annotations

import csv
import pickle
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from scipy.signal import butter, filtfilt, find_peaks, resample
except Exception as exc:  # pragma: no cover
    raise ImportError("scipy is required for POPANE dataset loader") from exc


# ---------------------------------------------------------------------------
# POPANE hard constraints and defaults
# ---------------------------------------------------------------------------
RAW_SR = 1000
TARGET_SR = 64
HEADER_LINES = 11
WINDOW_SEC = 10
STRIDE_SEC = 5
LABEL_MIN = 1.0
LABEL_MAX = 10.0
LABEL_LP_CUTOFF_HZ = 0.2
DEFAULT_BMI = 22.0
DEFAULT_HR = 70.0


@dataclass
class PopaneSample:
    dynamic: np.ndarray  # [2, window_len]
    static_raw: np.ndarray  # [4] -> [Age, Gender, BMI, Baseline_HR]
    label: float
    subject_id: str
    source_file: str


def _safe_float(v, default: float = np.nan) -> float:
    try:
        if v is None:
            return default
        s = str(v).strip()
        if s == "" or s.lower() in {"nan", "none"}:
            return default
        return float(s)
    except Exception:
        return default


def _normalize_gender(v) -> float:
    """
    Keep POPANE numeric coding unchanged.
    Official convention: man=0, woman=1.
    """
    if v is None:
        return 0.0
    num = _safe_float(v, default=np.nan)
    if not np.isnan(num):
        return float(num)
    s = str(v).strip().lower()
    if s in {"m", "male", "man"}:
        return 0.0
    if s in {"f", "female", "woman"}:
        return 1.0
    return 0.0


def _compute_bmi(weight_kg: float, height_cm: float) -> float:
    if np.isnan(weight_kg) or np.isnan(height_cm) or height_cm <= 0:
        return np.nan
    h_m = height_cm / 100.0
    if h_m <= 0:
        return np.nan
    return float(weight_kg / (h_m * h_m))


def _parse_sample_rate_from_header(header_map: Dict[str, str], default_sr: int = RAW_SR) -> int:
    for key in ("data_sample_rate", "sample_rate", "sampling_rate"):
        if key in header_map:
            m = re.search(r"([0-9]+)\s*hz", header_map[key], flags=re.IGNORECASE)
            if m:
                return int(m.group(1))
    return int(default_sr)


def _find_data_header_index(csv_path: Path) -> int:
    """
    Robustly locate the true column-header line.
    Typical POPANE files have:
      line 1-9 comment metadata
      line 10: timestamp,affect,ECG,...
      line 11: first sample
    But specification says first 11 lines metadata; we support both by scanning.
    """
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [f.readline() for _ in range(max(HEADER_LINES + 5, 32))]
    for idx, line in enumerate(lines):
        lower = line.strip().lower().replace(" ", "")
        if lower.startswith("timestamp,") and "ecg" in lower and "eda" in lower:
            return idx
    # fallback for strict spec
    return HEADER_LINES


def _parse_header_metadata(csv_path: Path, max_lines: int = HEADER_LINES) -> Dict[str, str]:
    """
    Parse first N header lines into normalized key->value map.
    Supports keys like:
      Subject_Age / Participant_Age
      Subject_Sex / Participant_Sex
      Participants_Height / Subject_Height
      Participants_Weight / Subject_Weight
    """
    out: Dict[str, str] = {}
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        for _ in range(max_lines):
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line.startswith("#"):
                continue
            payload = line[1:]
            parts = payload.split(",", 1)
            if len(parts) != 2:
                continue
            key = parts[0].strip().lower()
            val = parts[1].strip()
            key = key.replace(" ", "_")
            out[key] = val
    return out


def _extract_subject_id(csv_path: Path, header_map: Dict[str, str]) -> str:
    sid = header_map.get("subject_id")
    if sid:
        return str(sid).strip()
    m = re.match(r"(\d+)_", csv_path.name)
    if m:
        return m.group(1)
    return csv_path.stem.split("_")[0]


def _extract_static_from_header(header_map: Dict[str, str]) -> Tuple[float, float, float, float]:
    age = _safe_float(
        header_map.get("participant_age", header_map.get("subject_age")),
        default=np.nan,
    )
    sex_raw = header_map.get("subject_sex", header_map.get("participant_sex"))
    sex = _normalize_gender(sex_raw)
    height_cm = _safe_float(
        header_map.get("participants_height", header_map.get("subject_height")),
        default=np.nan,
    )
    weight_kg = _safe_float(
        header_map.get("participants_weight", header_map.get("subject_weight")),
        default=np.nan,
    )

    bmi = _compute_bmi(weight_kg, height_cm)
    if np.isnan(bmi):
        bmi = DEFAULT_BMI
    if np.isnan(age):
        age = 30.0

    return float(age), float(sex), float(bmi), DEFAULT_HR


def _read_ecg_eda_affect(csv_path: Path, usecols: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read only required columns to save memory.
    Returns timestamp, ECG, EDA and affect as 1-D float arrays.
    """
    data_header_idx = _find_data_header_index(csv_path)

    # Use csv.DictReader to avoid pandas dependency and keep memory low.
    timestamps: List[float] = []
    ecg_vals: List[float] = []
    eda_vals: List[float] = []
    affect_vals: List[float] = []

    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        for _ in range(data_header_idx):
            f.readline()
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"Failed to locate data header in: {csv_path}")

        label_col = "affect" if "affect" in reader.fieldnames else ("Valence" if "Valence" in reader.fieldnames else None)
        missing = [c for c in ("timestamp", "ECG", "EDA") if c not in reader.fieldnames]
        if label_col is None:
            missing.append("affect/Valence")
        if missing:
            raise RuntimeError(f"{csv_path} missing required columns: {missing}")

        for row in reader:
            t = _safe_float(row.get("timestamp"), default=np.nan)
            ecg = _safe_float(row.get("ECG"), default=np.nan)
            eda = _safe_float(row.get("EDA"), default=np.nan)
            aff = _safe_float(row.get(label_col), default=np.nan)
            if np.isnan(t) or np.isnan(ecg) or np.isnan(eda):
                continue
            if np.isnan(aff):
                aff = affect_vals[-1] if affect_vals else 5.5
            timestamps.append(t)
            ecg_vals.append(ecg)
            eda_vals.append(eda)
            affect_vals.append(aff)

    if not timestamps:
        raise RuntimeError(f"No valid signal rows parsed from {csv_path}")

    return (
        np.asarray(timestamps, dtype=np.float32),
        np.asarray(ecg_vals, dtype=np.float32),
        np.asarray(eda_vals, dtype=np.float32),
        np.asarray(affect_vals, dtype=np.float32),
    )


def _downsample_signal(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return x.astype(np.float32)
    if x.size < 4:
        return x.astype(np.float32)
    target_n = max(1, int(round(x.size * float(target_sr) / float(orig_sr))))
    return resample(x, target_n).astype(np.float32)


def _resample_timestamps(ts: np.ndarray, target_len: int) -> np.ndarray:
    if ts.size == target_len:
        return ts.astype(np.float32)
    old_idx = np.linspace(0.0, 1.0, ts.size)
    new_idx = np.linspace(0.0, 1.0, target_len)
    return np.interp(new_idx, old_idx, ts).astype(np.float32)


def _smooth_and_normalize_affect(
    timestamps: np.ndarray,
    affect: np.ndarray,
    target_sr: int,
) -> np.ndarray:
    """
    Handle piecewise-constant/discrete affect:
      1) detect change-points
      2) linear interpolation to full timeline
      3) low-pass smoothing
      4) normalize from [1,10] -> [0,1]
    """
    if affect.size < 3:
        y = np.clip((affect - LABEL_MIN) / (LABEL_MAX - LABEL_MIN), 0.0, 1.0)
        return y.astype(np.float32)

    # Change points for discrete segments.
    cp_idx = np.where(np.diff(affect) != 0)[0] + 1
    cp_idx = np.concatenate(([0], cp_idx, [affect.size - 1]))
    cp_t = timestamps[cp_idx]
    cp_v = affect[cp_idx]
    interp = np.interp(timestamps, cp_t, cp_v).astype(np.float32)

    # Low-pass smoothing.
    nyq = 0.5 * float(target_sr)
    wn = min(0.99, LABEL_LP_CUTOFF_HZ / max(nyq, 1e-6))
    if wn > 0 and interp.size > 10:
        b, a = butter(N=2, Wn=wn, btype="low")
        try:
            smooth = filtfilt(b, a, interp).astype(np.float32)
        except Exception:
            smooth = interp
    else:
        smooth = interp

    norm = (smooth - LABEL_MIN) / (LABEL_MAX - LABEL_MIN)
    return np.clip(norm, 0.0, 1.0).astype(np.float32)


def _estimate_baseline_hr(ecg: np.ndarray, sr: int) -> float:
    if ecg.size < int(sr * 2):
        return DEFAULT_HR
    # Distance for plausible 40-150 BPM.
    min_distance = max(1, int(sr / 2.5))
    peaks, _ = find_peaks(ecg, distance=min_distance)
    if peaks.size < 2:
        return DEFAULT_HR
    rr = np.diff(peaks) / float(sr)
    mean_rr = float(np.mean(rr)) if rr.size > 0 else 0.0
    if mean_rr <= 0:
        return DEFAULT_HR
    bpm = 60.0 / mean_rr
    if not np.isfinite(bpm):
        return DEFAULT_HR
    return float(np.clip(bpm, 35.0, 210.0))


class POPANEDataset(Dataset):
    """
    POPANE dataset loader with strict TCM-static and memory-safe signal processing.

    Output tuple:
      (dynamic_window[2,T], static_4d_scaled[4], label_scalar)
    """

    def __init__(
        self,
        root_dir: Path | str,
        tcm_scaler_path: Path | str,
        target_sr: int = TARGET_SR,
        window_sec: int = WINDOW_SEC,
        stride_sec: int = STRIDE_SEC,
        include_baseline_segments: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.target_sr = int(target_sr)
        self.window_len = int(window_sec * self.target_sr)
        self.stride_len = int(stride_sec * self.target_sr)
        self.include_baseline_segments = include_baseline_segments
        self.samples: List[PopaneSample] = []
        self.subject_meta: Dict[str, Dict[str, float]] = {}

        if not self.root_dir.exists():
            raise FileNotFoundError(f"POPANE root directory not found: {self.root_dir}")

        self.tcm_scaler = self._load_tcm_scaler(Path(tcm_scaler_path))
        self._build_subject_metadata()
        self._build_windows()

        if not self.samples:
            raise RuntimeError("POPANE dataset windows are empty after preprocessing.")

    def _load_tcm_scaler(self, scaler_path: Path):
        if not scaler_path.exists():
            raise FileNotFoundError(f"tcm_scaler not found: {scaler_path}")
        try:
            import joblib  # type: ignore

            scaler = joblib.load(scaler_path)
            if hasattr(scaler, "transform"):
                return scaler
        except Exception:
            pass
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        if not hasattr(scaler, "transform"):
            raise TypeError(f"Loaded scaler has no transform(): {scaler_path}")
        return scaler

    def _iter_csv_files(self) -> List[Path]:
        files = sorted(self.root_dir.rglob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV files found under {self.root_dir}")
        return files

    def _build_subject_metadata(self) -> None:
        baseline_files = [p for p in self._iter_csv_files() if "baseline.csv" in p.name.lower()]
        if not baseline_files:
            raise RuntimeError("No baseline files found (expected filenames containing 'Baseline.csv').")

        for csv_path in baseline_files:
            header_map = _parse_header_metadata(csv_path, max_lines=HEADER_LINES)
            subject_id = _extract_subject_id(csv_path, header_map)
            age, sex, bmi, _ = _extract_static_from_header(header_map)
            orig_sr = _parse_sample_rate_from_header(header_map, default_sr=RAW_SR)

            try:
                _, ecg, _, _ = _read_ecg_eda_affect(csv_path)
                baseline_hr = _estimate_baseline_hr(ecg, sr=orig_sr)
            except Exception as exc:
                warnings.warn(f"Baseline HR extraction failed for {csv_path.name}: {exc}")
                baseline_hr = DEFAULT_HR

            self.subject_meta[subject_id] = {
                "age": float(age),
                "sex": float(sex),
                "bmi": float(bmi),
                "baseline_hr": float(baseline_hr),
            }

    def _build_windows(self) -> None:
        csv_files = self._iter_csv_files()
        for csv_path in csv_files:
            is_baseline = "baseline.csv" in csv_path.name.lower()
            if is_baseline and not self.include_baseline_segments:
                continue

            header_map = _parse_header_metadata(csv_path, max_lines=HEADER_LINES)
            subject_id = _extract_subject_id(csv_path, header_map)
            if subject_id not in self.subject_meta:
                # If a subject has no baseline for any reason, fallback to current file metadata.
                age, sex, bmi, _ = _extract_static_from_header(header_map)
                self.subject_meta[subject_id] = {
                    "age": float(age),
                    "sex": float(sex),
                    "bmi": float(bmi),
                    "baseline_hr": DEFAULT_HR,
                }

            orig_sr = _parse_sample_rate_from_header(header_map, default_sr=RAW_SR)
            timestamps, ecg, eda, affect = _read_ecg_eda_affect(csv_path)

            # 1) Label interpolation/smoothing in original timeline.
            affect_cont = _smooth_and_normalize_affect(timestamps, affect, target_sr=orig_sr)

            # 2) Downsample ECG/EDA and label to target SR immediately (OOM-safe).
            ecg_ds = _downsample_signal(ecg, orig_sr=orig_sr, target_sr=self.target_sr)
            eda_ds = _downsample_signal(eda, orig_sr=orig_sr, target_sr=self.target_sr)
            y_ds = _downsample_signal(affect_cont, orig_sr=orig_sr, target_sr=self.target_sr)
            ts_ds = _resample_timestamps(timestamps, ecg_ds.size)

            n = min(ecg_ds.size, eda_ds.size, y_ds.size, ts_ds.size)
            if n < self.window_len:
                continue
            ecg_ds = ecg_ds[:n]
            eda_ds = eda_ds[:n]
            y_ds = y_ds[:n]

            dyn_2ch = np.stack([ecg_ds, eda_ds], axis=0).astype(np.float32)
            meta = self.subject_meta[subject_id]
            static_4 = np.array(
                [meta["age"], meta["sex"], meta["bmi"], meta["baseline_hr"]],
                dtype=np.float32,
            )

            for start in range(0, n - self.window_len + 1, self.stride_len):
                end = start + self.window_len
                dynamic_win = dyn_2ch[:, start:end]
                label_win = y_ds[start:end]
                label_scalar = float(np.mean(label_win))
                self.samples.append(
                    PopaneSample(
                        dynamic=dynamic_win,
                        static_raw=static_4,
                        label=label_scalar,
                        subject_id=subject_id,
                        source_file=csv_path.name,
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        static_scaled = self.tcm_scaler.transform(sample.static_raw.reshape(1, -1)).astype(np.float32).squeeze(0)
        return (
            torch.tensor(sample.dynamic, dtype=torch.float32),
            torch.tensor(static_scaled, dtype=torch.float32),
            torch.tensor([sample.label], dtype=torch.float32),
        )


__all__ = [
    "POPANEDataset",
    "TARGET_SR",
    "RAW_SR",
    "HEADER_LINES",
    "WINDOW_SEC",
    "STRIDE_SEC",
]
