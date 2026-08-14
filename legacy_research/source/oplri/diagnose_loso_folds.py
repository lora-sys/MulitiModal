#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose LOSO fold variability and outliers.")
    p.add_argument("--fold-dir", type=str, default="results/loso_folds")
    p.add_argument("--out-dir", type=str, default="results")
    p.add_argument("--topk", type=int, default=5)
    return p.parse_args()


def _load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_step_best_val_mse(payload: Dict, step: int) -> float | None:
    for r in payload.get("stage3_matrix_best_val_mse", []):
        if int(r.get("step", -1)) == step:
            v = r.get("best_val_mse")
            return None if v is None else float(v)
    return None


def _get_step_metrics(payload: Dict, step: int) -> Dict[str, float] | None:
    for row in payload.get("matrix_logs", []):
        if int(row.get("step", -1)) != step:
            continue
        full = row.get("full")
        if not isinstance(full, dict):
            return None
        m = full.get("metrics", {})
        out = {}
        for k in ["mse", "rmse", "mae", "pearson"]:
            if k in m and m[k] is not None:
                out[k] = float(m[k])
        if "best_epoch" in full and full["best_epoch"] is not None:
            out["best_epoch"] = float(full["best_epoch"])
        return out or None
    return None


def _safe_float(x) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _write_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No rows to write.")
    headers = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _save_fig(fig: plt.Figure, out_dir: Path, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    svg = out_dir / f"{stem}.svg"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(svg)
    fig.savefig(pdf)
    return png


def main() -> None:
    args = parse_args()
    fold_dir = Path(args.fold_dir)
    out_dir = Path(args.out_dir)

    files = sorted(fold_dir.glob("experiments_summary_*.json"))
    files = [p for p in files if not p.name.endswith("_loso.json")]
    if not files:
        raise RuntimeError(f"No fold summaries found under {fold_dir}")

    rows: List[Dict] = []
    per_fold_step8: List[Tuple[str, float]] = []
    per_fold_delta: List[Tuple[str, float]] = []

    for fp in files:
        payload = _load_json(fp)
        sid = str(payload.get("holdout_subject") or fp.stem.replace("experiments_summary_", ""))

        s1 = _get_step_best_val_mse(payload, 1)
        s8 = _get_step_best_val_mse(payload, 8)
        m8 = _get_step_metrics(payload, 8) or {}

        delta = None
        if s1 is not None and s8 is not None:
            delta = float(s1 - s8)

        row = {
            "subject": sid,
            "step1_mse": _safe_float(s1),
            "step8_mse": _safe_float(s8),
            "delta_step1_minus_step8": _safe_float(delta),
            "step8_mae": _safe_float(m8.get("mae")),
            "step8_pearson": _safe_float(m8.get("pearson")),
            "step8_best_epoch": _safe_float(m8.get("best_epoch")),
            "file": str(fp),
        }
        rows.append(row)

        if s8 is not None:
            per_fold_step8.append((sid, float(s8)))
        if delta is not None:
            per_fold_delta.append((sid, float(delta)))

    rows.sort(key=lambda r: (r["step8_mse"] is None, r["step8_mse"] if r["step8_mse"] is not None else 1e9))

    csv_path = out_dir / "loso_diagnosis.csv"
    _write_csv(rows, csv_path)

    # Plot: Step8 MSE distribution + delta distribution
    step8_vals = np.array([v for _, v in per_fold_step8], dtype=float)
    delta_vals = np.array([v for _, v in per_fold_delta], dtype=float) if per_fold_delta else np.array([])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].set_title("LOSO Step8 (Final Ours) MSE per Fold")
    axes[0].boxplot(step8_vals, vert=True)
    axes[0].scatter(np.ones_like(step8_vals), step8_vals, alpha=0.7)
    axes[0].set_ylabel("MSE")
    axes[0].grid(alpha=0.3)

    # Label top-k worst folds
    worst = sorted(per_fold_step8, key=lambda x: x[1], reverse=True)[: max(1, int(args.topk))]
    if worst:
        ymax = float(step8_vals.max())
        axes[0].text(1.15, ymax, "Worst folds:", va="top", fontsize=10)
        for i, (sid, v) in enumerate(worst, start=1):
            axes[0].text(1.15, ymax - 0.06 * i * ymax, f"{sid}: {v:.4f}", va="top", fontsize=9)

    axes[1].set_title("Improvement: Step1 MSE - Step8 MSE")
    if delta_vals.size:
        axes[1].axhline(0.0, color="black", linewidth=0.8)
        axes[1].boxplot(delta_vals, vert=True)
        axes[1].scatter(np.ones_like(delta_vals), delta_vals, alpha=0.7)
        axes[1].set_ylabel("Delta (positive = better than Baseline A)")
        axes[1].grid(alpha=0.3)
        pos_frac = float((delta_vals > 0).mean())
        axes[1].text(1.15, float(delta_vals.max()), f"pos_frac={pos_frac:.2%}", va="top", fontsize=10)
    else:
        axes[1].text(0.5, 0.5, "Delta not available", ha="center", va="center")
        axes[1].axis("off")

    fig.tight_layout()
    fig_path = _save_fig(fig, out_dir, "fig_loso_diagnosis")
    plt.close(fig)

    print(f"[OK] Wrote: {csv_path}")
    print(f"[OK] Wrote: {fig_path}")


if __name__ == "__main__":
    main()

