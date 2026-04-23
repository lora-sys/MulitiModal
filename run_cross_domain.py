#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import json
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import Paths, TCM_CHECKPOINT_PATH, TCM_SCALER_PATH, TrainConfig, ensure_dirs, override_from_env, resolve_device
from model import OPLRIRegressor
from run_experiments import FrozenTCMPrior  # reuse identical scaling + caching + smoothing logic
from src.data_loader import BUTPPGDataset
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
    p.add_argument("--tcm-prob-eps", type=float, default=0.0)
    p.add_argument("--tcm-temp", type=float, default=1.0)
    p.add_argument("--gate-a-scale", type=float, default=0.0)
    p.add_argument("--gate-b-scale", type=float, default=0.1)
    p.add_argument("--record-split-seed", type=int, default=42)
    p.add_argument("--paper-dir", type=str, default="paper", help="If exists, copy cross-domain outputs into this folder.")
    return p.parse_args()


@torch.no_grad()
def eval_linear_head(model: OPLRIRegressor, tcm_prior: FrozenTCMPrior, head: nn.Module, loader: DataLoader, args, device: str):
    ys, ps = [], []
    model.eval()
    head.eval()
    for dynamic, static, target, _rid in loader:
        dynamic = dynamic.to(device)
        static = static.to(device)
        tcm_probs = tcm_prior.infer_probs(static)
        z_pure, _ = model.extract_pure_dynamic(
            dynamic,
            tcm_probs,
            gate_a_scale=float(args.gate_a_scale),
            gate_b_scale=float(args.gate_b_scale),
            return_attention=False,
        )
        pred = head(z_pure)
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
        "num_records_train",
        "num_records_val",
        "base_lr",
        "finetune_factor",
        "finetune_lr",
        "mse",
        "rmse",
        "mae",
        "pearson",
        "spearman",
    ]
    write_header = not tsv_path.exists()
    with tsv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in headers})


def _record_level_split_indices(record_ids: List[str], val_ratio: float = 0.2, seed: int = 42) -> Tuple[List[int], List[int]]:
    uniq = sorted(set(record_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_val = max(1, int(len(uniq) * val_ratio))
    val_set = set(uniq[:n_val])
    train_idx, val_idx = [], []
    for i, rid in enumerate(record_ids):
        (val_idx if rid in val_set else train_idx).append(i)
    if not train_idx or not val_idx:
        raise RuntimeError("Record-level split failed (empty train/val).")
    return train_idx, val_idx


def _spearmanr(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.reshape(-1)
    bb = b.reshape(-1)
    if len(aa) < 2:
        return 0.0
    # rank transform
    ar = aa.argsort().argsort().astype(np.float32)
    br = bb.argsort().argsort().astype(np.float32)
    if np.std(ar) == 0 or np.std(br) == 0:
        return 0.0
    return float(np.corrcoef(ar, br)[0, 1])


def run_but_validation(paths: Paths, args: argparse.Namespace, device: str):
    # NOTE: Do not hard-fail here. FrozenTCMPrior has robust path resolution and supports both
    # `tcm_scaler.pkl` and legacy `scaler_params.npz`. Some server layouts may not place the scaler
    # under the exact CLI path.
    scaler_path = Path(args.tcm_scaler)
    dataset = BUTPPGDataset(
        paths.but_ppg_dir,
        scaler_path=scaler_path,
        return_record_id=True,
    )
    if len(dataset) < 2:
        raise RuntimeError("BUT dataset too small after HR-quality filtering; need at least 2 samples.")
    record_ids = [dataset[i][3] for i in range(len(dataset))]
    train_idx, val_idx = _record_level_split_indices(record_ids, val_ratio=0.2, seed=int(args.record_split_seed))
    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds = torch.utils.data.Subset(dataset, val_idx)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    ckpt = torch.load(paths.checkpoints / "best_model.pth", map_location="cpu", weights_only=True)
    best_encoder = ckpt.get("best_encoder", "inceptiontime")

    model = OPLRIRegressor(
        encoder_name=best_encoder,
        use_gate_a=True,
        use_gate_b=True,
        use_cross_attention=bool(ckpt.get("use_cross_attention", False)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    tcm_prior = FrozenTCMPrior(
        Path(args.tcm_checkpoint),
        Path(args.tcm_scaler),
        device,
        prob_eps=float(args.tcm_prob_eps),
        temperature=float(args.tcm_temp),
    )

    base_lr = _load_base_lr_from_yaml(Path(args.best_params_yaml), default_lr=1e-3)
    finetune_factor = 0.1
    finetune_lr = args.head_lr if args.head_lr is not None else (base_lr * finetune_factor)
    head = nn.Linear(128, 1).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=finetune_lr)
    loss_fn = nn.MSELoss()

    # Keep feature extractor behavior frozen (including BN/dropout behavior)
    head.train()
    for _ in range(args.epochs):
        for dynamic, static, target, _rid in train_loader:
            dynamic = dynamic.to(device)
            static = static.to(device)
            target = target.to(device)
            with torch.no_grad():
                tcm_probs = tcm_prior.infer_probs(static)
                z_pure, _ = model.extract_pure_dynamic(
                    dynamic,
                    tcm_probs,
                    gate_a_scale=float(args.gate_a_scale),
                    gate_b_scale=float(args.gate_b_scale),
                    return_attention=False,
                )
            pred = head(z_pure)
            loss = loss_fn(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    metrics = eval_linear_head(model, tcm_prior, head, val_loader, args, device)
    # add spearman for robustness
    # recompute preds to avoid changing regression_metrics
    ys, ps = [], []
    model.eval()
    head.eval()
    with torch.no_grad():
        for dynamic, static, target, _rid in val_loader:
            dynamic = dynamic.to(device)
            static = static.to(device)
            tcm_probs = tcm_prior.infer_probs(static)
            z_pure, _ = model.extract_pure_dynamic(dynamic, tcm_probs, gate_a_scale=float(args.gate_a_scale), gate_b_scale=float(args.gate_b_scale))
            pred = head(z_pure)
            ys.append(target.numpy())
            ps.append(pred.cpu().numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    spearman = _spearmanr(y, p)
    return {
        "task": "heart_rate_regression_bpm",
        "mse": metrics["mse"],
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "pearson": metrics["pearson"],
        "spearman": spearman,
        "num_samples": len(dataset),
        "num_records_train": len(set(record_ids[i] for i in train_idx)),
        "num_records_val": len(set(record_ids[i] for i in val_idx)),
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
    """Mechanism probe: correlate inferred TCM prior probabilities with HR (no waveform)."""
    scaler_path = Path(args.tcm_scaler)
    dataset = BUTPPGDataset(paths.but_ppg_dir, scaler_path=scaler_path, return_record_id=False)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    tcm_prior = FrozenTCMPrior(
        Path(args.tcm_checkpoint),
        Path(args.tcm_scaler),
        device,
        prob_eps=float(args.tcm_prob_eps),
        temperature=float(args.tcm_temp),
    )

    probs_all, targets = [], []
    for _dyn, static, target in loader:
        static = static.to(device)
        probs = tcm_prior.infer_probs(static)  # [B, 9]
        probs_all.append(probs.cpu().numpy())
        targets.append(target.numpy())
    p = np.concatenate(probs_all, axis=0)
    y = np.concatenate(targets, axis=0).reshape(-1)
    per_dim = np.array([_pearson_1d(p[:, i], y) for i in range(p.shape[1])], dtype=np.float32)
    max_abs_idx = int(np.argmax(np.abs(per_dim)))
    return {
        "tcm_probs_mean_abs_pearson": float(np.mean(np.abs(per_dim))),
        "tcm_probs_max_abs_pearson": float(np.max(np.abs(per_dim))),
        "tcm_probs_max_abs_dim": max_abs_idx,
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
    print(f"[{timestamp()}] BUT HR Spearman: {but_metrics['spearman']:.4f}")
    print(f"[{timestamp()}] BUT HR MAE (BPM): {but_metrics['mae']:.4f}")
    print("(Note: Negative correlation is physiologically expected between WESAD Relaxation and Heart Rate)")
    append_cross_domain_tsv(
        paths.results / "cross_domain_metrics.tsv",
        {
            "timestamp": timestamp(),
            "task": but_metrics["task"],
            "num_samples": but_metrics["num_samples"],
            "num_records_train": but_metrics["num_records_train"],
            "num_records_val": but_metrics["num_records_val"],
            "base_lr": but_metrics["base_lr"],
            "finetune_factor": but_metrics["finetune_factor"],
            "finetune_lr": but_metrics["finetune_lr"],
            "mse": but_metrics["mse"],
            "rmse": but_metrics["rmse"],
            "mae": but_metrics["mae"],
            "pearson": but_metrics["pearson"],
            "spearman": but_metrics["spearman"],
        },
    )

    print(f"[{timestamp()}] >>> Cross-domain B: TCM internal feature mechanism probe")
    mechanism_metrics = run_but_tcm_feature_correlation(paths, args, device)
    print(f"[{timestamp()}] TCM probs mean|r|: {mechanism_metrics['tcm_probs_mean_abs_pearson']:.4f}")
    print(f"[{timestamp()}] TCM probs max|r|: {mechanism_metrics['tcm_probs_max_abs_pearson']:.4f}")
    print(f"[{timestamp()}] TCM probs max|r| dim: {mechanism_metrics['tcm_probs_max_abs_dim']}")
    save_json({"but": but_metrics, "mechanism": mechanism_metrics}, paths.results / "cross_domain_results.json")

    # Optional: copy key outputs into paper/ for writing.
    paper_dir = Path(args.paper_dir)
    if paper_dir.exists():
        (paper_dir / "results").mkdir(parents=True, exist_ok=True)
        (paper_dir / "tables").mkdir(parents=True, exist_ok=True)
        (paper_dir / "results" / "cross_domain_results.json").write_text(
            json.dumps({"but": but_metrics, "mechanism": mechanism_metrics}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # Append TSV (already written under paths.results); copy latest snapshot.
        src_tsv = paths.results / "cross_domain_metrics.tsv"
        if src_tsv.exists():
            (paper_dir / "tables" / "cross_domain_metrics.tsv").write_text(src_tsv.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"[{timestamp()}] >>> Copied cross-domain outputs into {paper_dir}", flush=True)


if __name__ == "__main__":
    main()
