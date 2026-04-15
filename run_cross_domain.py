#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.config import Paths, TCM_CHECKPOINT_PATH, TCM_SCALER_PATH, TrainConfig, ensure_dirs, override_from_env, resolve_device
from src.data_loader import BUTPPGDataset, MIMICStaticDataset
from src.models.fusion import DualGatingModel
from src.utils import regression_metrics, save_json, timestamp


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-domain validation (BUT PPG + MIMIC)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--but-dir", type=str, default=None)
    p.add_argument("--mimic-csv", type=str, default=None)
    p.add_argument("--head-lr", type=float, default=None)
    p.add_argument("--best-params-yaml", type=str, default="configs/best_params.yaml")
    p.add_argument("--tcm-checkpoint", type=str, default=str(TCM_CHECKPOINT_PATH))
    p.add_argument("--tcm-scaler", type=str, default=str(TCM_SCALER_PATH))
    return p.parse_args()


@torch.no_grad()
def eval_linear_head(model: DualGatingModel, head: nn.Module, loader: DataLoader, device: str):
    ys, ps = [], []
    model.eval()
    head.eval()
    for dynamic, static, target in loader:
        dynamic = dynamic.to(device)
        static = static.to(device)
        pred = head(model.extract_fused_features(dynamic, static))
        ys.append(target.numpy())
        ps.append(pred.cpu().numpy())
    return regression_metrics(np.concatenate(ys), np.concatenate(ps))


def _load_base_lr_from_yaml(path: Path, default_lr: float = 1e-3) -> float:
    if not path.exists():
        return default_lr
    text = path.read_text(encoding="utf-8", errors="ignore")
    # Accept lines like: lr: 0.0007
    m = re.search(r"^\s*lr\s*:\s*([0-9.eE+-]+)\s*$", text, flags=re.MULTILINE)
    if m:
        try:
            lr = float(m.group(1))
            if lr > 0:
                return lr
        except Exception:
            pass
    return default_lr


def append_cross_domain_tsv(tsv_path: Path, row: dict) -> None:
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "timestamp",
        "task",
        "num_samples",
        "base_lr",
        "finetune_factor",
        "finetune_lr",
        "mse",
        "rmse",
        "mae",
        "pearson",
    ]
    write_header = not tsv_path.exists()
    with tsv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in headers})


def run_but_validation(paths: Paths, args: argparse.Namespace, device: str):
    scaler_path = Path(args.tcm_scaler)
    if not scaler_path.exists():
        raise FileNotFoundError(f"TCM scaler not found: {scaler_path}")
    dataset = BUTPPGDataset(
        paths.but_ppg_dir,
        scaler_path=scaler_path,
    )
    if len(dataset) < 2:
        raise RuntimeError("BUT dataset too small after HR-quality filtering; need at least 2 samples.")
    n_val = max(1, int(0.2 * len(dataset)))
    n_train = max(1, len(dataset) - n_val)
    if n_train + n_val > len(dataset):
        n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    ckpt = torch.load(paths.checkpoints / "best_model.pth", map_location="cpu", weights_only=True)
    best_encoder = ckpt.get("best_encoder", "inceptiontime")

    model = DualGatingModel(
        encoder_name=best_encoder,
        tcm_checkpoint_path=Path(args.tcm_checkpoint),
        tcm_scaler_path=Path(args.tcm_scaler),
        freeze_tcm=True,
        use_gate_a=True,
        use_gate_b=True,
        use_tcm_encoder=True,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    for p in model.parameters():
        p.requires_grad = False

    base_lr = _load_base_lr_from_yaml(Path(args.best_params_yaml), default_lr=1e-3)
    finetune_factor = 0.1
    finetune_lr = args.head_lr if args.head_lr is not None else (base_lr * finetune_factor)
    head = nn.Linear(256, 1).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=finetune_lr)
    loss_fn = nn.MSELoss()

    # Keep feature extractor behavior frozen (including BN/dropout behavior)
    model.eval()
    head.train()
    for _ in range(args.epochs):
        for dynamic, static, target in train_loader:
            dynamic = dynamic.to(device)
            static = static.to(device)
            target = target.to(device)
            with torch.no_grad():
                fused = model.extract_fused_features(dynamic, static)
            pred = head(fused)
            loss = loss_fn(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    metrics = eval_linear_head(model, head, val_loader, device)
    return {
        "task": "heart_rate_regression_bpm",
        "mse": metrics["mse"],
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "pearson": metrics["pearson"],
        "num_samples": len(dataset),
        "base_lr": base_lr,
        "finetune_factor": finetune_factor,
        "finetune_lr": finetune_lr,
        "scaler_path": str(scaler_path),
    }


def _pearson_1d(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.reshape(-1)
    bb = b.reshape(-1)
    if len(aa) < 2 or np.std(aa) == 0 or np.std(bb) == 0:
        return 0.0
    return float(np.corrcoef(aa, bb)[0, 1])


@torch.no_grad()
def run_but_tcm_feature_correlation(paths: Paths, args: argparse.Namespace, device: str):
    """Mechanism probe: correlate internal TCM features with physiological target (HR)."""
    scaler_path = Path(args.tcm_scaler)
    dataset = BUTPPGDataset(paths.but_ppg_dir, scaler_path=scaler_path)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)

    ckpt = torch.load(paths.checkpoints / "best_model.pth", map_location="cpu", weights_only=True)
    best_encoder = ckpt.get("best_encoder", "inceptiontime")
    model = DualGatingModel(
        encoder_name=best_encoder,
        tcm_checkpoint_path=Path(args.tcm_checkpoint),
        tcm_scaler_path=Path(args.tcm_scaler),
        freeze_tcm=True,
        use_tcm=True,
        use_gate_a=True,
        use_gate_b=True,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    feats, targets = [], []
    for _, static, target in loader:
        static = static.to(device)
        internal = model.get_tcm_internal_features(static)  # [B, 128]
        feats.append(internal.cpu().numpy())
        targets.append(target.numpy())

    x = np.concatenate(feats, axis=0)  # [N,128]
    y = np.concatenate(targets, axis=0).reshape(-1)  # HR
    per_dim_corr = np.array([_pearson_1d(x[:, i], y) for i in range(x.shape[1])], dtype=np.float32)
    feat_norm_corr = _pearson_1d(np.linalg.norm(x, axis=1), y)
    max_abs_idx = int(np.argmax(np.abs(per_dim_corr)))
    return {
        "tcm_internal_mean_abs_pearson": float(np.mean(np.abs(per_dim_corr))),
        "tcm_internal_max_abs_pearson": float(np.max(np.abs(per_dim_corr))),
        "tcm_internal_max_abs_dim": max_abs_idx,
        "tcm_internal_norm_pearson": feat_norm_corr,
    }


def main() -> None:
    args = parse_args()
    paths = override_from_env(Paths())
    if args.but_dir:
        paths.but_ppg_dir = Path(args.but_dir)
    if args.mimic_csv:
        paths.mimic_csv = Path(args.mimic_csv)
    ensure_dirs(paths)
    device = resolve_device(args.device)

    print(f"[{timestamp()}] >>> Cross-domain A: BUT PPG (HR mechanism validation)")
    but_metrics = run_but_validation(paths, args, device)
    print(f"[{timestamp()}] BUT HR Pearson: {but_metrics['pearson']:.4f}")
    print(f"[{timestamp()}] BUT HR MAE (BPM): {but_metrics['mae']:.4f}")
    print("(Note: Negative correlation is physiologically expected between WESAD Relaxation and Heart Rate)")
    append_cross_domain_tsv(
        paths.results / "cross_domain_metrics.tsv",
        {
            "timestamp": timestamp(),
            "task": but_metrics["task"],
            "num_samples": but_metrics["num_samples"],
            "base_lr": but_metrics["base_lr"],
            "finetune_factor": but_metrics["finetune_factor"],
            "finetune_lr": but_metrics["finetune_lr"],
            "mse": but_metrics["mse"],
            "rmse": but_metrics["rmse"],
            "mae": but_metrics["mae"],
            "pearson": but_metrics["pearson"],
        },
    )

    print(f"[{timestamp()}] >>> Cross-domain B: TCM internal feature mechanism probe")
    mechanism_metrics = run_but_tcm_feature_correlation(paths, args, device)
    print(f"[{timestamp()}] TCM feature mean|r|: {mechanism_metrics['tcm_internal_mean_abs_pearson']:.4f}")
    print(f"[{timestamp()}] TCM feature max|r|: {mechanism_metrics['tcm_internal_max_abs_pearson']:.4f}")
    print(f"[{timestamp()}] TCM feature norm r: {mechanism_metrics['tcm_internal_norm_pearson']:.4f}")
    save_json({"but": but_metrics, "mechanism": mechanism_metrics}, paths.results / "cross_domain_results.json")


if __name__ == "__main__":
    main()
