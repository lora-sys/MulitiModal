#!/usr/bin/env python3
"""
敏感性分析：冥想态映射值对模型性能的影响
分别将冥想态(label=3)映射为 0.5, 0.6, 0.7，各跑一轮 Final Ours LOSO。

优化版：只跑 Final Ours 配置（InceptionTime + TCM + Gate A/B + Cross-Attention），
不跑完整 10-step 矩阵，大幅节省时间。

运行方式：
  cd /path/to/MulitiModal
  python scripts/sensitivity_analysis.py --device cuda

输出：
  - paper/results/sensitivity_analysis.json
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

import src.config as _cfg
import src.data_loader as _dl
from src.config import TrainConfig, EARLY_STOPPING_PATIENCE, TCM_CHECKPOINT_PATH, TCM_SCALER_PATH
from src.data_loader import WESADDataset
from run_experiments import (
    FrozenTCMPrior, HyperParams, train_eval_step,
    _strip_for_json, timestamp
)

RESULTS_DIR = PROJ / "paper" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MEDITATION_VALUES = [0.5, 0.6, 0.7]


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def patch_label_map(meditation_val: float):
    """In-place patch WESAD_LABEL_MAP in both src.config and src.data_loader.

    Python module-level imports create name bindings to the *same dict object*.
    Replacing the dict (e.g. via importlib.reload) breaks the binding — the
    data_loader still sees the old values.  We must mutate the dict in-place
    AND update the reference in data_loader's namespace.
    """
    new_map = {1: 1.0, 2: 0.0, 3: meditation_val}
    # Patch src.config (canonical source)
    _cfg.WESAD_LABEL_MAP.clear()
    _cfg.WESAD_LABEL_MAP.update(new_map)
    # Patch data_loader's imported reference (module-level `from src.config import WESAD_LABEL_MAP`)
    _dl.WESAD_LABEL_MAP = new_map
    print(f"  WESAD_LABEL_MAP patched in-place: {new_map}")


def restore_label_map():
    """Restore default meditation = 0.6"""
    patch_label_map(0.6)
    print("  WESAD_LABEL_MAP restored to default")


def run_final_ours_loso(
    meditation_val: float,
    device: str,
    epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    max_subjects: int = 0,
) -> dict:
    """只跑 Final Ours 的 LOSO (max_subjects=0 表示全部)"""
    print(f"\n{'='*60}")
    print(f"Final Ours LOSO: meditation = {meditation_val}")
    print(f"{'='*60}")

    patch_label_map(meditation_val)

    dataset = WESADDataset(PROJ / "data" / "wesad")
    unique_subjects = sorted(set(s for s in dataset.subject_ids))
    if max_subjects > 0:
        unique_subjects = unique_subjects[:max_subjects]
    print(f"  被试: {unique_subjects} ({len(unique_subjects)} 人)")

    tcm_prior = FrozenTCMPrior(
        checkpoint_path=TCM_CHECKPOINT_PATH,
        scaler_path=TCM_SCALER_PATH,
        device=device,
    )

    hparams = HyperParams(lr=lr, weight_decay=weight_decay, batch_size=batch_size)
    fold_results = []

    for i, holdout in enumerate(unique_subjects):
        print(f"\n  [{i+1}/{len(unique_subjects)}] Holdout: {holdout}")
        train_idx = [j for j, s in enumerate(dataset.subject_ids) if s != holdout]
        val_idx = [j for j, s in enumerate(dataset.subject_ids) if s == holdout]

        result = train_eval_step(
            step_name=f"FinalOurs-m{meditation_val}-{holdout}",
            encoder="inceptiontime",
            use_tcm=True,
            use_gate_a=True,
            use_gate_b=True,
            hparams=hparams,
            epochs=epochs,
            patience=patience,
            dataset=dataset,
            train_indices=train_idx,
            val_indices=val_idx,
            seed=42,
            device=device,
            tcm_prior=tcm_prior,
            gate_a_scale=0.0,
            gate_b_scale=0.35,
            use_cross_attention=True,
            early_fusion=False,
        )
        metrics = result["metrics"]
        fold_results.append({
            "holdout": holdout,
            "mse": metrics["mse"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "pearson": metrics["pearson"],
            "best_epoch": result["best_epoch"],
        })
        print(f"    MSE={metrics['mse']:.4f}  Pearson={metrics['pearson']:.4f}")

    mses = [r["mse"] for r in fold_results]
    pearsons = [r["pearson"] for r in fold_results]
    summary = {
        "meditation_value": meditation_val,
        "method": "Final Ours",
        "encoder": "inceptiontime",
        "n_folds": len(fold_results),
        "mse_mean": float(np.mean(mses)),
        "mse_std": float(np.std(mses)),
        "pearson_mean": float(np.mean(pearsons)),
        "pearson_std": float(np.std(pearsons)),
        "per_fold": fold_results,
    }

    restore_label_map()
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--smoke", action="store_true", help="冒烟测试: 1 组 × 2 折 × 2 epochs")
    args = parser.parse_args()
    device = args.device or _auto_device()

    smoke = args.smoke
    epochs = 2 if smoke else args.epochs
    patience = 1 if smoke else args.patience
    smoke_subjects = 2
    meditation_values = [0.6] if smoke else MEDITATION_VALUES

    print("=" * 60)
    print(f"敏感性分析 {'(SMOKE)' if smoke else '(快速版: 只跑 Final Ours)'}")
    print(f"Device: {device}")
    print("=" * 60)

    all_results = {}
    for med_val in meditation_values:
        summary = run_final_ours_loso(
            meditation_val=med_val,
            device=device,
            epochs=epochs,
            patience=patience,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_subjects=smoke_subjects if smoke else 0,
        )
        all_results[str(med_val)] = summary

    # 汇总
    print(f"\n{'='*60}")
    print("敏感性分析结果汇总")
    print(f"{'='*60}")
    print(f"{'Meditation':>12s} {'MSE (mean±std)':>20s} {'Pearson r (mean±std)':>22s}")
    print("-" * 58)
    for med_val in meditation_values:
        key = str(med_val)
        s = all_results[key]
        print(f"{med_val:>12.1f} {s['mse_mean']:>10.4f} ± {s['mse_std']:.4f}   {s['pearson_mean']:.4f} ± {s['pearson_std']:.4f}")

    out_path = RESULTS_DIR / "sensitivity_analysis.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
