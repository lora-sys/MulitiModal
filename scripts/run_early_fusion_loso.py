#!/usr/bin/env python3
"""
快速 LOSO：只跑 Early Fusion 基线（15 折）
不跑完整 10-step 矩阵，大幅节省时间。

运行方式：
  cd /path/to/MulitiModal
  python scripts/run_early_fusion_loso.py --device cuda

输出：
  - paper/results/early_fusion_loso.json
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from src.config import TrainConfig, WESAD_LABEL_MAP, EARLY_STOPPING_PATIENCE
from src.data_loader import WESADDataset
from model import OPLRIRegressor
from run_experiments import (
    FrozenTCMPrior, HyperParams, make_loaders_from_indices,
    _unpack_batch, _run_forward, regression_metrics,
    _strip_for_json, timestamp
)


def train_early_fusion_fold(
    dataset: WESADDataset,
    train_indices: list,
    val_indices: list,
    hparams: HyperParams,
    epochs: int,
    patience: int,
    device: str,
) -> dict:
    """训练一个 Early Fusion fold"""
    train_loader, val_loader = make_loaders_from_indices(
        dataset, train_indices, val_indices,
        batch_size=hparams.batch_size, seed=42,
    )
    model = OPLRIRegressor.create_early_fusion(encoder_name="inceptiontime").to(device)

    # 只训练 regressor head
    for p in model.dynamic_encoder.parameters():
        p.requires_grad = False
    optimizer = torch.optim.AdamW(model.reg_head.parameters(), lr=hparams.lr, weight_decay=hparams.weight_decay)
    loss_fn = nn.MSELoss()

    best_val_mse = float("inf")
    best_epoch = 1
    best_state = None
    counter = 0
    val_mse_history = []

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        total_loss = 0.0
        n_items = 0
        for batch in train_loader:
            dynamic_x, static_x, target = _unpack_batch(batch, device)
            # Early Fusion forward: skip TCM/gates
            z_raw = model.dynamic_encoder(dynamic_x)
            final_input = torch.cat([z_raw.detach(), static_x], dim=-1)  # [B, 132]
            pred = model.forward_from_final_input(final_input)
            loss = loss_fn(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(dynamic_x)
            n_items += len(dynamic_x)
        train_loss = total_loss / max(n_items, 1)

        # Validate
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for batch in val_loader:
                dynamic_x, static_x, target = _unpack_batch(batch, device)
                z_raw = model.dynamic_encoder(dynamic_x)
                final_input = torch.cat([z_raw.detach(), static_x], dim=-1)
                pred = model.forward_from_final_input(final_input)
                ys.append(target.cpu().numpy())
                ps.append(pred.cpu().numpy())
        val_metrics = regression_metrics(np.concatenate(ys), np.concatenate(ps))
        val_mse = val_metrics["mse"]
        val_mse_history.append(val_mse)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    # Final eval with best state
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in val_loader:
            dynamic_x, static_x, target = _unpack_batch(batch, device)
            z_raw = model.dynamic_encoder(dynamic_x)
            final_input = torch.cat([z_raw.detach(), static_x], dim=-1)
            pred = model.forward_from_final_input(final_input)
            ys.append(target.cpu().numpy())
            ps.append(pred.cpu().numpy())
    final_metrics = regression_metrics(np.concatenate(ys), np.concatenate(ps))

    return {
        "best_val_mse": best_val_mse,
        "best_epoch": best_epoch,
        "val_mse_history": val_mse_history,
        "metrics": final_metrics,
    }


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--smoke", action="store_true", help="冒烟测试: 2 折 × 2 epochs, 快速验证流程")
    args = parser.parse_args()
    device = args.device or _auto_device()

    smoke = args.smoke
    epochs = 2 if smoke else args.epochs
    patience = 1 if smoke else args.patience

    print("=" * 60)
    print(f"Early Fusion LOSO {'(SMOKE)' if smoke else '(快速版)'}")
    print(f"Device: {device}")
    print("=" * 60)

    # 加载数据
    dataset = WESADDataset(PROJ / "data" / "wesad")
    unique_subjects = sorted(set(s for s in dataset.subject_ids))
    if smoke:
        unique_subjects = unique_subjects[:2]
    print(f"被试: {unique_subjects} (共 {len(unique_subjects)} 人)")

    hparams = HyperParams(lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size)

    # 逐折训练
    fold_results = []
    for i, holdout in enumerate(unique_subjects):
        print(f"\n[{i+1}/{len(unique_subjects)}] Holdout: {holdout}")
        train_idx = [j for j, s in enumerate(dataset.subject_ids) if s != holdout]
        val_idx = [j for j, s in enumerate(dataset.subject_ids) if s == holdout]

        result = train_early_fusion_fold(
            dataset=dataset,
            train_indices=train_idx,
            val_indices=val_idx,
            hparams=hparams,
            epochs=epochs,
            patience=patience,
            device=device,
        )
        fold_results.append({
            "holdout": holdout,
            "mse": result["metrics"]["mse"],
            "rmse": result["metrics"]["rmse"],
            "mae": result["metrics"]["mae"],
            "pearson": result["metrics"]["pearson"],
            "best_epoch": result["best_epoch"],
        })
        print(f"  MSE={result['metrics']['mse']:.4f}  Pearson={result['metrics']['pearson']:.4f}")

    # 汇总
    mses = [r["mse"] for r in fold_results]
    pearsons = [r["pearson"] for r in fold_results]
    summary = {
        "method": "Early Fusion",
        "encoder": "inceptiontime",
        "n_folds": len(fold_results),
        "mse_mean": float(np.mean(mses)),
        "mse_std": float(np.std(mses)),
        "pearson_mean": float(np.mean(pearsons)),
        "pearson_std": float(np.std(pearsons)),
        "per_fold": fold_results,
    }

    print(f"\n{'='*60}")
    print(f"Early Fusion LOSO 结果:")
    print(f"  MSE = {summary['mse_mean']:.4f} ± {summary['mse_std']:.4f}")
    print(f"  Pearson r = {summary['pearson_mean']:.4f} ± {summary['pearson_std']:.4f}")
    print(f"{'='*60}")

    # 保存
    out_dir = PROJ / "paper" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "early_fusion_loso.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"结果已保存: {out_path}")


if __name__ == "__main__":
    main()
