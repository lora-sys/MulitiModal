from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

# Training control
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10  # 连续 10 轮不掉分就停

# Default pre-trained TCM assets provided by user.
TCM_CHECKPOINT_PATH = Path("/root/work/MulitiModal/checkpoints/best_tcm_model.pth")
TCM_SCALER_PATH = Path("/root/work/MulitiModal/checkpoints/tcm_scaler.pkl")
TCM_TRAINING_HISTORY_PATH = Path("/home/lora/repos/MulitiModal/tcm_ft_transformer/training_history.png")


@dataclass
class Paths:
    wesad_dir: Path = ROOT / "data" / "wesad"
    but_ppg_dir: Path = ROOT / "data" / "but_ppg"
    mimic_csv: Path = ROOT / "data" / "mimic" / "mimic_demo.csv"
    checkpoints: Path = ROOT / "checkpoints"
    results: Path = ROOT / "results"
    logs: Path = ROOT / "logs"


@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "cuda"
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = MAX_EPOCHS
    window_size: int = 1000
    window_overlap: float = 0.5


WESAD_LABEL_MAP = {1: 1.0, 2: 0.0, 3: 0.6}
FEATURE_BASELINES = {
    "bmi": 22.0,
    "sbp": 120.0,
    "dbp": 80.0,
    "spo2": 98.0,
}

CONSTITUTION_NAMES = [
    "平和质",
    "气虚质",
    "阳虚质",
    "阴虚质",
    "痰湿质",
    "湿热质",
    "血瘀质",
    "气郁质",
    "特禀质",
]


def ensure_dirs(paths: Paths) -> None:
    for p in [paths.checkpoints, paths.results, paths.logs, paths.wesad_dir, paths.but_ppg_dir, paths.mimic_csv.parent]:
        p.mkdir(parents=True, exist_ok=True)


def resolve_device(requested: str) -> str:
    import torch

    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def override_from_env(paths: Paths) -> Paths:
    return Paths(
        wesad_dir=Path(os.getenv("WESAD_DIR", str(paths.wesad_dir))),
        but_ppg_dir=Path(os.getenv("BUT_PPG_DIR", str(paths.but_ppg_dir))),
        mimic_csv=Path(os.getenv("MIMIC_CSV", str(paths.mimic_csv))),
        checkpoints=Path(os.getenv("CHECKPOINT_DIR", str(paths.checkpoints))),
        results=Path(os.getenv("RESULT_DIR", str(paths.results))),
        logs=Path(os.getenv("LOG_DIR", str(paths.logs))),
    )
