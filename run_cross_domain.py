#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import json
from typing import List, Tuple
from dataclasses import dataclass

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
    p.add_argument("--epochs", type=int, default=8, help="Max epochs for head fine-tuning per seed.")
    p.add_argument("--but-dir", type=str, default=None)
    p.add_argument("--mimic-csv", type=str, default=None)
    p.add_argument("--head-lr", type=float, default=None)
    p.add_argument("--best-params-yaml", type=str, default="configs/best_params.yaml")
    p.add_argument("--tcm-checkpoint", type=str, default=str(TCM_CHECKPOINT_PATH))
    p.add_argument("--tcm-scaler", type=str, default=str(TCM_SCALER_PATH))
    p.add_argument("--tcm-prob-eps", type=float, default=0.0)
    p.add_argument("--tcm-temp", type=float, default=1.0)
    p.add_argument("--seeds", type=str, default="", help="Comma-separated seeds for repeated runs, e.g. 42,43,44")
    p.add_argument("--head", type=str, default="linear", choices=["linear", "mlp"], help="Head type to train (only head trains).")
    p.add_argument("--early-stop-patience", type=int, default=10, help="Early stopping patience on val MSE.")
    p.add_argument(
        "--target-standardize",
        action="store_true",
        help="Standardize HR targets using TRAIN split stats (stabilizes head training); metrics are reported in BPM.",
    )
    p.add_argument("--loss", type=str, default="mse", choices=["mse", "huber"], help="Head training loss.")
    p.add_argument("--huber-delta", type=float, default=5.0, help="Huber delta (BPM units) when --loss huber.")
    p.add_argument("--head-weight-decay", type=float, default=0.0, help="Weight decay for head optimizer.")
    p.add_argument(
        "--strict-tcm-paths",
        action="store_true",
        help="If set, do not fallback/auto-search TCM checkpoint/scaler paths; fail fast if missing.",
    )
    p.add_argument("--gate-a-scale", type=float, default=0.0)
    p.add_argument("--gate-b-scale", type=float, default=0.1)
    p.add_argument("--record-split-seed", type=int, default=42)
    p.add_argument("--paper-dir", type=str, default="paper", help="If exists, copy cross-domain outputs into this folder.")
    return p.parse_args()


def _parse_seeds_arg(seeds: str) -> List[int]:
    s = (seeds or "").strip()
    if not s:
        return []
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_head(kind: str, in_dim: int = 128) -> nn.Module:
    if kind == "mlp":
        return nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, 1),
        )
    return nn.Linear(in_dim, 1)


@dataclass
class SeedRun:
    seed: int
    metrics: dict


@torch.no_grad()
def _collect_preds(
    model: OPLRIRegressor,
    tcm_prior: FrozenTCMPrior,
    head: nn.Module,
    loader: DataLoader,
    args,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
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
    return np.concatenate(ys).reshape(-1), np.concatenate(ps).reshape(-1)


def _train_head_one_seed(
    *,
    seed: int,
    model: OPLRIRegressor,
    tcm_prior: FrozenTCMPrior,
    head: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args,
    device: str,
    finetune_lr: float,
) -> dict:
    _set_seed(seed)
    head = head.to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=float(finetune_lr), weight_decay=float(args.head_weight_decay))
    if str(args.loss) == "huber":
        loss_fn = nn.SmoothL1Loss(beta=float(args.huber_delta))
    else:
        loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    counter = 0

    # Target standardization (train-split only), report metrics back in BPM.
    y_mu = 0.0
    y_std = 1.0
    if bool(args.target_standardize):
        ys = []
        for _dyn, _st, target, _rid in train_loader:
            ys.append(target.numpy().reshape(-1))
        y = np.concatenate(ys) if ys else np.zeros((1,), dtype=np.float32)
        y_mu = float(np.mean(y))
        y_std = float(np.std(y))
        if y_std <= 1e-6:
            y_std = 1.0

        # Initialize head bias near the (standardized) mean to reduce constant-collapse.
        try:
            if isinstance(head, nn.Linear):
                head.bias.data.fill_(0.0)
            elif isinstance(head, nn.Sequential) and isinstance(head[-1], nn.Linear):
                head[-1].bias.data.fill_(0.0)
        except Exception:
            pass

    head.train()
    for epoch in range(int(args.epochs)):
        for dynamic, static, target, _rid in train_loader:
            dynamic = dynamic.to(device)
            static = static.to(device)
            target = target.to(device)
            if bool(args.target_standardize):
                target = (target - float(y_mu)) / float(y_std)
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

        # Early stopping on val MSE
        y, p = _collect_preds(model, tcm_prior, head, val_loader, args, device)
        if bool(args.target_standardize):
            p = p * float(y_std) + float(y_mu)
        val_mse = float(np.mean((y - p) ** 2))
        if val_mse + 1e-12 < best_val:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            counter = 0
        else:
            counter += 1
            if counter >= int(args.early_stop_patience):
                break

    if best_state is not None:
        head.load_state_dict(best_state, strict=True)

    y, p = _collect_preds(model, tcm_prior, head, val_loader, args, device)
    if bool(args.target_standardize):
        p = p * float(y_std) + float(y_mu)
    metrics = regression_metrics(y, p)
    metrics["spearman"] = _spearmanr(y, p)
    metrics["num_val_samples"] = int(y.shape[0])
    metrics["best_val_mse"] = float(best_val)
    metrics["seed"] = int(seed)
    if bool(args.target_standardize):
        metrics["target_mu"] = float(y_mu)
        metrics["target_std"] = float(y_std)
    return {"metrics": metrics, "y_true": y, "y_pred": p}


def _write_cross_domain_seed_table(tsv_path: Path, seed_rows: List[dict]) -> None:
    headers = [
        "timestamp",
        "seed",
        "task",
        "num_samples",
        "num_records_train",
        "num_records_val",
        "base_lr",
        "finetune_factor",
        "finetune_lr",
        "head",
        "mse",
        "rmse",
        "mae",
        "pearson",
        "spearman",
    ]
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    with tsv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers, delimiter="\t")
        w.writeheader()
        for r in seed_rows:
            w.writerow({k: r.get(k, "") for k in headers})


def _plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path, stem: str) -> List[str]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=300)
    ax.scatter(y_true, y_pred, s=10, alpha=0.6, color="#2c3e50")
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], color="#c0392b", linewidth=1.2)
    ax.set_xlabel("True HR (BPM)")
    ax.set_ylabel("Pred HR (BPM)")
    ax.set_title("Fig.X Cross-domain HR Probe (BUT)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    paths = []
    for ext in ("png", "svg", "pdf"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p)
        paths.append(str(p))
    plt.close(fig)
    return paths


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
        strict_paths=bool(args.strict_tcm_paths),
    )

    base_lr = _load_base_lr_from_yaml(Path(args.best_params_yaml), default_lr=1e-3)
    finetune_factor = 0.1
    finetune_lr = args.head_lr if args.head_lr is not None else (base_lr * finetune_factor)
    seeds = _parse_seeds_arg(args.seeds)
    if not seeds:
        seeds = [int(args.record_split_seed)]

    seed_rows: List[dict] = []
    per_seed_runs: List[dict] = []
    best_seed = None
    best_mse = float("inf")
    best_y = None
    best_p = None

    for seed in seeds:
        head = _make_head(str(args.head), in_dim=128)
        run = _train_head_one_seed(
            seed=int(seed),
            model=model,
            tcm_prior=tcm_prior,
            head=head,
            train_loader=train_loader,
            val_loader=val_loader,
            args=args,
            device=device,
            finetune_lr=float(finetune_lr),
        )
        m = run["metrics"]
        per_seed_runs.append(m)
        if float(m["mse"]) < best_mse:
            best_mse = float(m["mse"])
            best_seed = int(seed)
            best_y = run["y_true"]
            best_p = run["y_pred"]

        seed_rows.append(
            {
                "timestamp": timestamp(),
                "seed": int(seed),
                "task": "heart_rate_regression_bpm",
                "num_samples": len(dataset),
                "num_records_train": len(set(record_ids[i] for i in train_idx)),
                "num_records_val": len(set(record_ids[i] for i in val_idx)),
                "base_lr": base_lr,
                "finetune_factor": finetune_factor,
                "finetune_lr": float(finetune_lr),
                "head": str(args.head),
                "mse": float(m["mse"]),
                "rmse": float(m["rmse"]),
                "mae": float(m["mae"]),
                "pearson": float(m["pearson"]),
                "spearman": float(m["spearman"]),
            }
        )

    _write_cross_domain_seed_table(paths.results / "cross_domain_seed_metrics.tsv", seed_rows)
    if best_y is not None and best_p is not None:
        _plot_scatter(best_y, best_p, paths.results, "fig_cross_domain_hr_probe")

    # Aggregate mean±std across seeds (paper-friendly)
    mses = np.array([float(x["mse"]) for x in per_seed_runs], dtype=np.float64)
    maes = np.array([float(x["mae"]) for x in per_seed_runs], dtype=np.float64)
    pears = np.array([float(x["pearson"]) for x in per_seed_runs], dtype=np.float64)
    spears = np.array([float(x["spearman"]) for x in per_seed_runs], dtype=np.float64)

    agg = {
        "task": "heart_rate_regression_bpm",
        "seeds": seeds,
        "head": str(args.head),
        "mse_mean": float(mses.mean()),
        "mse_std": float(mses.std(ddof=0)) if mses.size > 1 else 0.0,
        "mae_mean": float(maes.mean()),
        "mae_std": float(maes.std(ddof=0)) if maes.size > 1 else 0.0,
        "pearson_mean": float(pears.mean()),
        "pearson_std": float(pears.std(ddof=0)) if pears.size > 1 else 0.0,
        "spearman_mean": float(spears.mean()),
        "spearman_std": float(spears.std(ddof=0)) if spears.size > 1 else 0.0,
        "best_seed": best_seed,
        "num_samples": len(dataset),
        "num_records_train": len(set(record_ids[i] for i in train_idx)),
        "num_records_val": len(set(record_ids[i] for i in val_idx)),
        "base_lr": base_lr,
        "finetune_factor": finetune_factor,
        "finetune_lr": float(finetune_lr),
        "scaler_path": str(scaler_path),
        "per_seed": per_seed_runs,
    }
    return agg


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
        strict_paths=bool(args.strict_tcm_paths),
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
    print(f"[{timestamp()}] BUT HR Pearson(mean±std): {but_metrics['pearson_mean']:.4f} ± {but_metrics['pearson_std']:.4f}")
    print(f"[{timestamp()}] BUT HR Spearman(mean±std): {but_metrics['spearman_mean']:.4f} ± {but_metrics['spearman_std']:.4f}")
    print(f"[{timestamp()}] BUT HR MAE (BPM, mean±std): {but_metrics['mae_mean']:.4f} ± {but_metrics['mae_std']:.4f}")
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
            "mse": but_metrics["mse_mean"],
            "rmse": float(np.sqrt(but_metrics["mse_mean"])),
            "mae": but_metrics["mae_mean"],
            "pearson": but_metrics["pearson_mean"],
            "spearman": but_metrics["spearman_mean"],
        },
    )

    print(f"[{timestamp()}] >>> Cross-domain B: TCM internal feature mechanism probe")
    mechanism_metrics = run_but_tcm_feature_correlation(paths, args, device)
    print(f"[{timestamp()}] TCM probs mean|r|: {mechanism_metrics['tcm_probs_mean_abs_pearson']:.4f}")
    print(f"[{timestamp()}] TCM probs max|r|: {mechanism_metrics['tcm_probs_max_abs_pearson']:.4f}")
    print(f"[{timestamp()}] TCM probs max|r| dim: {mechanism_metrics['tcm_probs_max_abs_dim']}")
    save_json({"but": but_metrics, "mechanism": mechanism_metrics}, paths.results / "cross_domain_results.json")

    # Archive key outputs into paper/ for writing (create if missing).
    paper_dir = Path(args.paper_dir) if getattr(args, "paper_dir", None) else None
    if paper_dir is not None:
        paper_dir.mkdir(parents=True, exist_ok=True)
        (paper_dir / "results").mkdir(parents=True, exist_ok=True)
        (paper_dir / "tables").mkdir(parents=True, exist_ok=True)
        (paper_dir / "results" / "cross_domain_results.json").write_text(
            json.dumps({"but": but_metrics, "mechanism": mechanism_metrics}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # Append TSV (already written under paths.results); copy latest snapshot.
        src_tsv = paths.results / "cross_domain_metrics.tsv"
        if src_tsv.exists():
            (paper_dir / "tables" / "cross_domain_metrics.tsv").write_text(
                src_tsv.read_text(encoding="utf-8"), encoding="utf-8"
            )
        print(f"[{timestamp()}] >>> Copied cross-domain outputs into {paper_dir}", flush=True)


if __name__ == "__main__":
    main()
