from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import EARLY_STOPPING_PATIENCE, MAX_EPOCHS
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
    model, train_metrics, val_metrics, _, _ = fit_regression_model_with_history(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
    )
    return model, train_metrics, val_metrics


def fit_regression_model_with_history(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = MAX_EPOCHS,
    lr: float,
    weight_decay: float,
    device: str,
    patience: int = EARLY_STOPPING_PATIENCE,
) -> Tuple[nn.Module, Dict[str, float], Dict[str, float], list[float], int]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    max_epochs = int(epochs) if epochs is not None else MAX_EPOCHS
    max_epochs = max(1, max_epochs)
    patience = max(1, int(patience))

    best_val_loss = float("inf")
    best_weights = None
    val_loss_history: list[float] = []
    counter = 0
    best_epoch = 1

    for epoch in range(1, max_epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device)
        metrics = evaluate(model, val_loader, device)
        val_loss = float(metrics["mse"])
        val_loss_history.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            counter = 0
            best_epoch = epoch
        else:
            counter += 1

        if counter >= patience:
            if best_weights is not None:
                model.load_state_dict(best_weights)
            print(f"Early stopping triggered at epoch {epoch}. Best epoch was {epoch - counter}.", flush=True)
            break

        print(
            f"[Epoch {epoch}/{max_epochs}] train_loss={train_loss:.6f} "
            f"val_mse={val_loss:.6f} best_val={best_val_loss:.6f} "
            f"patience_counter={counter}/{patience}",
            flush=True,
        )

    if best_weights is not None:
        model.load_state_dict(best_weights)

    train_metrics = evaluate(model, train_loader, device)
    val_metrics = evaluate(model, val_loader, device)
    return model, train_metrics, val_metrics, val_loss_history, best_epoch


def save_checkpoint(model: nn.Module, path: Path, extra: Dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)
