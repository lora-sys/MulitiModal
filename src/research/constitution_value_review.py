"""任务 B（体质线）：只读预测价值复核，补均值基线、逐维误差、配对差值区间。

不训练、不调参、不改冠军、不写 results.tsv、不解锁历史 holdout。
只重放冻结的 4 折 × 2 重复折外预测，把冻结冠军与「训练折维度均值」这一
最简单的可复现基线逐维比较：逐维 MAE、配对差值、bootstrap 95% 区间。

口径：选择集折外 CV（LOCAL_EXPERIMENTAL / AGENT_ISOLATION_UNVERIFIED），
不是独立外部验证，不构成临床有效性结论。

运行（acam-lab，PYTHONPATH=src）：
    python -m research.constitution_value_review --out runs/value-review/constitution.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

from research.constitution import _fold_payload
from research.constitution_data import (
    BUDGET_SECONDS,
    EVIDENCE,
    FOLDS,
    N_SUBJECTS,
    N_TARGET_DIMS,
    OBSERVED_DIMS,
    REPEATS,
    SEED,
    load_selection,
    selection_fold_ids,
)
from research.constitution_delivery import (
    CANDIDATE_PATH,
    CHAMPION_COMMIT,
    verify_champion_source,
)
from research.prepare import committed_source, run_candidate

REVIEW_SCHEMA = "tcm-constitution-value-review-v1"
N_BOOT = 2000
BASELINE_ID = "training-fold-per-dimension-mean"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _masked_errors(rows: list[dict], predictions: list) -> list[list[float]]:
    """逐受试者逐维绝对误差；尊重 score_mask，不补零、不伪造未观测维度。"""
    if len(rows) != len(predictions):
        raise ValueError("prediction count does not match dev rows")
    out = []
    for row, pred in zip(rows, predictions, strict=True):
        if not isinstance(pred, list) or len(pred) != N_TARGET_DIMS:
            raise ValueError("candidate must return 8 values per subject")
        errors = []
        for index, value in enumerate(pred):
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(value):
                raise ValueError("candidate produced a non-finite prediction")
            if row["score_mask"][index]:
                errors.append(abs(float(value) - float(row["scores"][index])))
            else:
                errors.append(0.0)
        out.append(errors)
    return out


def mean_baseline_predictions(training_rows: list[dict], n_predict: int) -> list[list[float]]:
    """训练折内每维均值；只在训练折统计，绝不读 dev 标签。"""
    per_dim: list[list[float]] = [[] for _ in range(N_TARGET_DIMS)]
    for row in training_rows:
        for index, (value, observed) in enumerate(
                zip(row["scores"], row["score_mask"], strict=True)):
            if observed:
                per_dim[index].append(float(value))
    means = [statistics.mean(values) if values else 0.0 for values in per_dim]
    return [list(means) for _ in range(n_predict)]


def _bootstrap_ci(values: list[float], seed: int, n_boot: int = N_BOOT) -> list[float]:
    rng = random.Random(seed)
    boot = sorted(statistics.mean(rng.choices(values, k=len(values)))
                  for _ in range(n_boot))
    return [round(boot[int(0.025 * n_boot)], 4), round(boot[int(0.975 * n_boot)], 4)]


def review(manifest_path: Path, dataset_path: Path, xls_path: Path,
           repo: Path, seed: int = SEED, n_boot: int = N_BOOT) -> dict:
    """只读重放：返回完整复核报告（调用方负责写 runs/，不提交）。"""
    manifest, rows = load_selection(manifest_path, dataset_path, xls_path)
    if manifest["split"]["holdout"]:
        raise ValueError("selection manifest has a non-empty holdout")
    if len(rows) != N_SUBJECTS:
        raise ValueError(f"expected {N_SUBJECTS} subjects")

    committed = committed_source(repo, CHAMPION_COMMIT, CANDIDATE_PATH)
    champion_sha = verify_champion_source(committed)

    # subject_id -> {"champion": [per-dim errors per fold], "mean": [...]}
    oof: dict[str, dict[str, list[list[float]]]] = {}
    folds: list[dict] = []
    for repeat in range(REPEATS):
        for fold in range(FOLDS):
            train_ids, dev_ids = selection_fold_ids(manifest, fold, FOLDS, repeat)
            training, inputs, dev_rows = _fold_payload(manifest, rows, train_ids, dev_ids)
            champion_pred = run_candidate(committed, training, inputs,
                                         budget_seconds=BUDGET_SECONDS, seed=SEED)
            mean_pred = mean_baseline_predictions(training["rows"], len(dev_rows))
            champion_err = _masked_errors(dev_rows, champion_pred)
            mean_err = _masked_errors(dev_rows, mean_pred)
            for row, c_err, m_err in zip(dev_rows, champion_err, mean_err, strict=True):
                slot = oof.setdefault(row["subject_id"], {"champion": [], "mean": []})
                slot["champion"].append(c_err)
                slot["mean"].append(m_err)
            folds.append({
                "fold": fold, "repeat": repeat,
                "champion_mae": round(statistics.mean(
                    sum(e) / N_TARGET_DIMS for e in champion_err), 4),
                "mean_baseline_mae": round(statistics.mean(
                    sum(e) / N_TARGET_DIMS for e in mean_err), 4),
                "n_validation": len(dev_rows),
            })

    subjects = sorted(oof)
    champion_subject = {s: statistics.mean(sum(e) / N_TARGET_DIMS
                                           for e in oof[s]["champion"]) for s in subjects}
    mean_subject = {s: statistics.mean(sum(e) / N_TARGET_DIMS
                                       for e in oof[s]["mean"]) for s in subjects}
    paired = [champion_subject[s] - mean_subject[s] for s in subjects]
    fold_paired = [f["champion_mae"] - f["mean_baseline_mae"] for f in folds]

    per_dimension = []
    for index, dim in enumerate(OBSERVED_DIMS):
        champ = [statistics.mean(fold[index] for fold in oof[s]["champion"])
                 for s in subjects]
        base = [statistics.mean(fold[index] for fold in oof[s]["mean"])
                for s in subjects]
        dim_paired = [c - b for c, b in zip(champ, base, strict=True)]
        per_dimension.append({
            "dim": dim,
            "champion_mae": round(statistics.mean(champ), 4),
            "mean_baseline_mae": round(statistics.mean(base), 4),
            "delta": round(statistics.mean(dim_paired), 4),
            "paired_ci95": _bootstrap_ci(dim_paired, seed, n_boot),
            "improved_subjects": sum(d < 0 for d in dim_paired),
        })

    return {
        "schema_version": REVIEW_SCHEMA,
        "status": {
            "evidence": EVIDENCE,
            "trusted_evaluator": "AGENT_ISOLATION_UNVERIFIED",
            "external_validation": "NONE",
            "clinical_claim": False,
        },
        "inputs": {
            "manifest_sha256": _sha256(manifest_path),
            "dataset_sha256": _sha256(dataset_path),
            "xls_sha256": _sha256(xls_path),
            "champion_source_sha256": champion_sha,
            "champion_commit": CHAMPION_COMMIT,
            "holdout_subjects": len(manifest["split"]["holdout"]),
            "scope": "frozen selection-only 4-fold x 2-repeat out-of-fold; no holdout, no training, no tuning",
        },
        "protocol": {"folds": FOLDS, "repeats": REPEATS, "seed": SEED,
                     "n_subjects": len(subjects), "n_folds": len(folds),
                     "n_boot": n_boot, "baseline": BASELINE_ID},
        "acceptance": {
            "out_of_fold_champion_mae": round(statistics.mean(champion_subject.values()), 4),
            "out_of_fold_mean_baseline_mae": round(statistics.mean(mean_subject.values()), 4),
        },
        "paired": {
            "definition": "per-subject champion MAE minus mean-baseline MAE (negative = champion better)",
            "mean_delta": round(statistics.mean(paired), 4),
            "ci95": _bootstrap_ci(paired, seed, n_boot),
            "improved_subjects": sum(d < 0 for d in paired),
            "n_subjects": len(paired),
            "fold_delta_mean": round(statistics.mean(fold_paired), 4),
            "fold_ci95": _bootstrap_ci(fold_paired, seed, n_boot),
            "improved_folds": sum(d < 0 for d in fold_paired),
            "ci_excludes_zero": None,  # filled below
        },
        "per_dimension": {"dimensions": per_dimension, "n_dimensions": len(per_dimension)},
        "fold_details": folds,
        "unverified_generalization": [
            "折外 CV 选择集结果，不是独立外部验证；跨机构/设备/时间泛化未验证",
            "评估器隔离 AGENT_ISOLATION_UNVERIFIED，无独立防篡改认证",
            "分数未校准，不是概率、诊断或临床疗效；GO_RESEARCH_CLAIMS 未放行",
            "均值基线只用训练折统计量，但仍是同一批受试者的内部比较",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--manifest", default=("data/local/mendeley-cxtph3tjsg-v1/"
                                               "protocol-v1/selection_manifest.json"))
    parser.add_argument("--dataset", default=("data/local/mendeley-cxtph3tjsg-v1/"
                                              "protocol-v1/selection_records.json"))
    parser.add_argument("--xls", default="data/local/mendeley-cxtph3tjsg-v1/Share data.xls")
    parser.add_argument("--out", default="runs/value-review/constitution.json")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args()
    report = review(Path(args.manifest), Path(args.dataset), Path(args.xls),
                    Path(args.repo).resolve(), args.seed, args.n_boot)
    low, high = report["paired"]["ci95"]
    report["paired"]["ci_excludes_zero"] = bool(low > 0.0 or high < 0.0)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "champion_mae": report["acceptance"]["out_of_fold_champion_mae"],
        "mean_baseline_mae": report["acceptance"]["out_of_fold_mean_baseline_mae"],
        "paired_delta": report["paired"]["mean_delta"],
        "paired_ci95": report["paired"]["ci95"],
        "ci_excludes_zero": report["paired"]["ci_excludes_zero"],
        "improved_subjects": report["paired"]["improved_subjects"],
        "n_subjects": report["paired"]["n_subjects"],
        "report": str(out),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
