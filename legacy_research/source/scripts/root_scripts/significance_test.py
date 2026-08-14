#!/usr/bin/env python3
"""
统计显著性检验
对 LOSO 15-fold 的 per-fold MSE 进行配对检验。

运行方式：
  cd /path/to/MulitiModal
  python scripts/significance_test.py

输出：
  - 终端打印 t-test 和 Wilcoxon 结果
  - paper/results/significance_test.json
"""

import json
import sys
import numpy as np
from pathlib import Path
from scipy import stats

PROJ = Path(__file__).resolve().parent.parent


def main():
    fold_dir = PROJ / "experiment1" / "1.2" / "loso_folds"
    if not fold_dir.exists():
        print(f"ERROR: {fold_dir} not found")
        sys.exit(1)

    subjects = ["S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9",
                "S10", "S11", "S13", "S14", "S15", "S16", "S17"]

    # 提取 per-fold MSE
    methods = {
        "Baseline A": [],
        "Baseline B": [],
        "w/o Dual Gating": [],
        "w/o TCM Prior": [],
        "Final Ours": [],
    }

    for s in subjects:
        f = fold_dir / f"experiments_summary_{s}.json"
        with open(f) as fp:
            data = json.load(fp)
        matrix = {
            item["name"]: item["best_val_mse"]
            for item in data["stage3_matrix_best_val_mse"]
            if item["best_val_mse"] is not None
        }
        for name in methods:
            methods[name].append(matrix.get(name, np.nan))

    # 转成 numpy
    for name in methods:
        methods[name] = np.array(methods[name], dtype=np.float64)

    final = methods["Final Ours"]

    print("=" * 60)
    print("Statistical Significance Test (15-fold LOSO)")
    print("=" * 60)

    # 定义要比较的 pairs
    pairs = [
        ("Baseline A", "Final Ours"),
        ("Baseline B", "Final Ours"),
        ("w/o Dual Gating", "Final Ours"),
        ("w/o TCM Prior", "Final Ours"),
    ]

    results = {}
    print(f"\n{'Comparison':<30s} {'Paired t-test':>20s} {'Wilcoxon':>20s}")
    print("-" * 72)

    for name_a, name_b in pairs:
        a = methods[name_a]
        b = methods[name_b]

        # 过滤 NaN
        mask = ~(np.isnan(a) | np.isnan(b))
        a_valid, b_valid = a[mask], b[mask]

        t_stat, p_ttest = stats.ttest_rel(a_valid, b_valid)
        w_stat, p_wilcoxon = stats.wilcoxon(a_valid, b_valid)

        sig_t = "***" if p_ttest < 0.001 else "**" if p_ttest < 0.01 else "*" if p_ttest < 0.05 else ""
        sig_w = "***" if p_wilcoxon < 0.001 else "**" if p_wilcoxon < 0.01 else "*" if p_wilcoxon < 0.05 else ""

        print(f"{name_a} vs {name_b:<12s} t={t_stat:>6.3f} p={p_ttest:.4f}{sig_t:<3s}  W={w_stat:>6.1f} p={p_wilcoxon:.4f}{sig_w:<3s}")

        key = f"{name_a}_vs_{name_b.replace(' ', '_')}"
        results[key] = {
            "method_a": name_a,
            "method_b": name_b,
            "n_folds": int(mask.sum()),
            "mean_a": float(np.nanmean(a_valid)),
            "mean_b": float(np.nanmean(b_valid)),
            "delta_mse": float(np.nanmean(a_valid) - np.nanmean(b_valid)),
            "paired_t_test": {
                "t": round(float(t_stat), 4),
                "p": round(float(p_ttest), 6),
                "significant_005": bool(p_ttest < 0.05),
            },
            "wilcoxon": {
                "W": round(float(w_stat), 4),
                "p": round(float(p_wilcoxon), 6),
                "significant_005": bool(p_wilcoxon < 0.05),
            },
        }

    # 保存
    out_dir = PROJ / "paper" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "significance_test.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存: {out_path}")

    # 总结
    print("\n" + "=" * 60)
    print("Summary: Final Ours vs all baselines")
    print("=" * 60)
    print("Wilcoxon signed-rank test (recommended for small-sample, non-normal MSE):")
    for key, val in results.items():
        sig = "SIGNIFICANT" if val["wilcoxon"]["significant_005"] else "n.s."
        print(f"  {val['method_a']} vs {val['method_b']}: "
              f"p={val['wilcoxon']['p']:.4f} ({sig}), "
              f"ΔMSE={val['delta_mse']:+.4f}")


if __name__ == "__main__":
    main()
