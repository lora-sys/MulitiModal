#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.config import CONSTITUTION_NAMES, Paths, TCM_CHECKPOINT_PATH, TCM_SCALER_PATH, TrainConfig, ensure_dirs, override_from_env, resolve_device
from src.data_loader import BUTPPGDataset, MIMICStaticDataset
from src.models.fusion import DualGatingModel, TCMEncoderAdapter
from src.utils import regression_metrics, save_json, timestamp


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-domain validation (BUT PPG + MIMIC)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--but-dir", type=str, default=None)
    p.add_argument("--mimic-csv", type=str, default=None)
    p.add_argument("--head-lr", type=float, default=1e-4)
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


def run_but_validation(paths: Paths, args: argparse.Namespace, device: str):
    dataset = BUTPPGDataset(
        paths.but_ppg_dir,
        scaler_path=Path(args.tcm_scaler),
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
        freeze_tcm=True,
        use_gate_a=True,
        use_gate_b=True,
        use_tcm_encoder=True,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    for p in model.parameters():
        p.requires_grad = False

    head = nn.Linear(256, 1).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=args.head_lr)
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
        "head_lr": args.head_lr,
    }


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    try:
        from scipy.stats import spearmanr

        return float(spearmanr(x, y).correlation)
    except Exception:
        xr = np.argsort(np.argsort(x))
        yr = np.argsort(np.argsort(y))
        return float(np.corrcoef(xr, yr)[0, 1])


def run_mimic_validation(paths: Paths, args: argparse.Namespace, device: str):
    ds = MIMICStaticDataset(paths.mimic_csv, Path(args.tcm_scaler))
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    tcm = TCMEncoderAdapter(Path(args.tcm_checkpoint), freeze=True).to(device)
    tcm.eval()

    all_probs, all_sbp = [], []
    with torch.no_grad():
        for static_8d, sbp in loader:
            static_8d = static_8d.to(device)
            _, probs = tcm(static_8d)
            all_probs.append(probs.cpu().numpy())
            all_sbp.append(sbp.numpy())

    probs = np.concatenate(all_probs, axis=0)
    sbp = np.concatenate(all_sbp, axis=0).reshape(-1)
    tan_shi_idx = CONSTITUTION_NAMES.index("痰湿质")
    corr = _spearman(probs[:, tan_shi_idx], sbp)
    return {"spearman_tanshizhi_vs_sbp": corr}


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

    # print(f"[{timestamp()}] >>> Cross-domain B: MIMIC static")
    # mimic_metrics = run_mimic_validation(paths, args, device)
    # print(f"[{timestamp()}] MIMIC metrics: {mimic_metrics}")
    print("[INFO] Mechanism Validation (Cross-Domain B) is temporarily skipped as per MVP strategy.")
    mimic_metrics = {"skipped": True}
    save_json({"but": but_metrics, "mimic": mimic_metrics}, paths.results / "cross_domain_results.json")


if __name__ == "__main__":
    main()
