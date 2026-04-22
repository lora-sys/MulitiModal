#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate LOSO fold summaries into paper-ready tables.")
    p.add_argument("--fold-dir", type=str, default="results/loso_folds")
    p.add_argument("--out-dir", type=str, default="results")
    return p.parse_args()


def _read_fold_jsons(fold_dir: Path) -> List[Dict]:
    files = sorted(fold_dir.glob("experiments_summary_*.json"))
    payloads: List[Dict] = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            obj = json.load(f)
        obj["_file"] = str(fp)
        payloads.append(obj)
    return payloads


def _collect_step_metric(payloads: List[Dict], step: int, metric: str) -> np.ndarray:
    vals = []
    for p in payloads:
        for row in p.get("matrix_logs", []):
            if int(row.get("step", -1)) != step:
                continue
            full = row.get("full")
            if not isinstance(full, dict):
                continue
            m = full.get("metrics", {})
            if metric in m and m[metric] is not None:
                vals.append(float(m[metric]))
    return np.asarray(vals, dtype=np.float64)


def _collect_step_mse(payloads: List[Dict], step: int) -> np.ndarray:
    vals = []
    for p in payloads:
        for row in p.get("stage3_matrix_best_val_mse", []):
            if int(row.get("step", -1)) == step and row.get("best_val_mse") is not None:
                vals.append(float(row["best_val_mse"]))
    return np.asarray(vals, dtype=np.float64)


def main() -> None:
    args = parse_args()
    fold_dir = Path(args.fold_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payloads = _read_fold_jsons(fold_dir)
    if not payloads:
        raise RuntimeError(f"No fold summaries found in {fold_dir}")

    # Paper main table based on Stage-3 MSE (best val MSE per step)
    rows = []
    for step in range(1, 10):
        vals = _collect_step_mse(payloads, step)
        if vals.size == 0:
            rows.append({"step": step, "name": f"Step {step}", "mean_mse": None, "std_mse": None, "n_folds": 0})
            continue
        # Name from first payload that has this step.
        step_name = f"Step {step}"
        for p in payloads:
            for r in p.get("stage3_matrix_best_val_mse", []):
                if int(r.get("step", -1)) == step:
                    step_name = str(r.get("name", step_name))
                    break
        rows.append(
            {
                "step": step,
                "name": step_name,
                "mean_mse": float(vals.mean()),
                "std_mse": float(vals.std(ddof=0)),
                "n_folds": int(vals.size),
            }
        )

    # Final model fold-level stats
    final_step = 8
    final_mse = _collect_step_metric(payloads, final_step, "mse")
    final_rmse = _collect_step_metric(payloads, final_step, "rmse")
    final_mae = _collect_step_metric(payloads, final_step, "mae")
    final_pearson = _collect_step_metric(payloads, final_step, "pearson")
    final_stats = {
        "step": final_step,
        "name": "Final Ours",
        "mse_mean": float(final_mse.mean()) if final_mse.size else None,
        "mse_std": float(final_mse.std(ddof=0)) if final_mse.size else None,
        "rmse_mean": float(final_rmse.mean()) if final_rmse.size else None,
        "rmse_std": float(final_rmse.std(ddof=0)) if final_rmse.size else None,
        "mae_mean": float(final_mae.mean()) if final_mae.size else None,
        "mae_std": float(final_mae.std(ddof=0)) if final_mae.size else None,
        "pearson_mean": float(final_pearson.mean()) if final_pearson.size else None,
        "pearson_std": float(final_pearson.std(ddof=0)) if final_pearson.size else None,
        "n_folds": int(final_mse.size),
    }

    summary = {
        "n_folds": len(payloads),
        "fold_files": [p["_file"] for p in payloads],
        "stage3_main_table": rows,
        "final_ours_fold_stats": final_stats,
    }

    json_path = out_dir / "loso_main_table.json"
    csv_path = out_dir / "loso_main_table.csv"
    tsv_path = out_dir / "loso_main_table.tsv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    headers = ["step", "name", "mean_mse", "std_mse", "n_folds"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with open(tsv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[OK] Aggregated {len(payloads)} LOSO folds")
    print(f"[OK] JSON: {json_path}")
    print(f"[OK] CSV : {csv_path}")
    print(f"[OK] TSV : {tsv_path}")
    print(f"[OK] Final Ours (mean±std MSE): {final_stats['mse_mean']:.6f} ± {final_stats['mse_std']:.6f}")


if __name__ == "__main__":
    main()

