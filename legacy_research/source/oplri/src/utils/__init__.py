from __future__ import annotations

import json
import math
import random
from pathlib import Path
from datetime import datetime
from typing import Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _json_default(o):
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "__dict__"):
        return o.__dict__
    return str(o)


def save_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)


def load_json(path: Path, default: Dict | None = None) -> Dict:
    if not path.exists():
        return {} if default is None else default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(math.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        pearson = 0.0
    else:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    return {"mse": mse, "rmse": rmse, "mae": mae, "pearson": pearson}


def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()
