from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils import regression_metrics, to_numpy


def run_epoch(model: nn.Module, loader: DataLoader, optimizer, device: str) -> float:
    model.train()
    loss_fn = nn.MSELoss()
    total = 0.0
    count = 0
    for dynamic, static, target in loader:
        dynamic = dynamic.to(device)
        static = static.to(device)
        target = target.to(device)

        pred = model(dynamic, static)
        loss = loss_fn(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += float(loss.item()) * len(dynamic)
        count += len(dynamic)
    return total / max(count, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    ys, ps = [], []
    for dynamic, static, target in loader:
        dynamic = dynamic.to(device)
        static = static.to(device)
        target = target.to(device)
        pred = model(dynamic, static)
        ys.append(to_numpy(target))
        ps.append(to_numpy(pred))

    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(ps, axis=0)
    return regression_metrics(y_true, y_pred)


def fit_regression_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: str,
) -> Tuple[nn.Module, Dict[str, float], Dict[str, float]]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val_mse = float("inf")
    best_metrics: Dict[str, float] = {}

    for _ in range(epochs):
        run_epoch(model, train_loader, optimizer, device)
        metrics = evaluate(model, val_loader, device)
        if metrics["mse"] < best_val_mse:
            best_val_mse = metrics["mse"]
            best_metrics = metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    train_metrics = evaluate(model, train_loader, device)
    val_metrics = evaluate(model, val_loader, device)
    return model, train_metrics, val_metrics


def save_checkpoint(model: nn.Module, path: Path, extra: Dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)
