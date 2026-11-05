"""中医体质 AutoResearch 第二阶段：冻结评价协议与 Git 实验循环入口。

复用现有基础设施，不新建训练平台：
- research.prepare.run_candidate：隔离 -I 子进程，只见 train.json / predict.json；
- research.prepare.append_history：results.tsv 的 KEEP / REVERT / CRASH 严格晋级；
- research.prepare.write_new_atomic / open_regular / local_device；
- research.dashboard：round-*.json + status.json + results.tsv 只读监控。

主指标 constitution/masked_mae（8 维 PUBLISHED 分数的平均绝对误差，minimize）。
平和质是完全未观测的第 9 维，ABSTAIN：候选输出必须是 (n, 8)，任何 9 维输出
或任何伪造的平和质分数都会被评分器拒绝。

CLI:
    python -m research.constitution \\
        --manifest ... --dataset ... --xls ... \\
        --candidate src/research/constitution_train.py \\
        --out runs/constitution-v1/round-000.json --folds 4 --repeats 2 \\
        --seed 20261105 --budget 300
    python -m research.constitution record --repo . --results runs/constitution-v1/results.tsv \\
        --commit <sha> --decision KEEP|REVERT|CRASH --description ... \\
        --result runs/constitution-v1/round-000.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import uuid
from pathlib import Path

from research.constitution_data import (
    BUDGET_SECONDS,
    EVIDENCE,
    FOLDS,
    N_TARGET_DIMS,
    OBSERVED_DIMS,
    REPEATS,
    SEED,
    UNOBSERVED_DIMS,
    load_selection,
    selection_fold_ids,
)
from research.prepare import (
    EVIDENCE_LOCAL,
    append_history,
    committed_source,
    local_device,
    run_candidate,
    write_new_atomic,
)

CANDIDATE_PATH = "src/research/constitution_train.py"
METRIC = "constitution/masked_mae"
METRIC_VERSION = "constitution-eval-v1"
DIRECTION = "minimize"
PROMOTION = "strict"


def code_hashes(candidate: bytes) -> dict[str, str]:
    """冻结评估器、数据适配与本轮候选的哈希。"""
    import research.constitution_data as adapter

    return {
        "evaluator": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "adapter": hashlib.sha256(Path(adapter.__file__).read_bytes()).hexdigest(),
        "candidate": hashlib.sha256(candidate).hexdigest(),
    }


def score_predictions(rows: list[dict], predictions: object) -> tuple[float, list[float]]:
    """候选进程外评分：masked MAE over the 8 observed dimensions.

    负例防护：形状错误、非有限值、9 维输出（含伪造的平和质）一律拒绝。
    """
    if not isinstance(predictions, list) or len(predictions) != len(rows):
        raise ValueError("candidate must return one 8-value row per validation subject")
    per_dim_sum = [0.0] * N_TARGET_DIMS
    per_dim_count = [0] * N_TARGET_DIMS
    for row, output in zip(rows, predictions, strict=True):
        if not isinstance(output, list) or len(output) != N_TARGET_DIMS:
            raise ValueError(
                f"candidate row must be exactly {N_TARGET_DIMS} values; "
                "the unobserved balanced dimension must never be predicted"
            )
        for index, value in enumerate(output):
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(value):
                raise ValueError("prediction must be a finite number")
            if row["score_mask"][index]:
                per_dim_sum[index] += abs(float(value) - row["scores"][index])
                per_dim_count[index] += 1
    if any(count == 0 for count in per_dim_count):
        raise ValueError("no observed targets to score")
    per_dim = [total / count for total, count in zip(per_dim_sum, per_dim_count, strict=True)]
    return sum(per_dim) / N_TARGET_DIMS, per_dim


def _fold_payload(manifest: dict, rows: list[dict], train_ids: list[str],
                  dev_ids: list[str]) -> tuple[object, object, list[dict]]:
    by_id = {row["subject_id"]: row for row in rows}
    train_rows = [by_id[sid] for sid in train_ids]
    dev_rows = [by_id[sid] for sid in dev_ids]
    training = {
        "rows": [{
            "features": row["features"],
            "feature_mask": row["feature_mask"],
            "scores": row["scores"],
            "score_dims": row["score_dims"],
            "score_mask": row["score_mask"],
        } for row in train_rows],
        "score_dims": list(OBSERVED_DIMS),
    }
    inputs = {
        "rows": [{
            "features": row["features"],
            "feature_mask": row["feature_mask"],
        } for row in dev_rows],
        "score_dims": list(OBSERVED_DIMS),
    }
    return training, inputs, dev_rows


def evaluate(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--xls", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=FOLDS)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--budget", type=int, default=BUDGET_SECONDS)
    args = parser.parse_args(argv)
    if (args.folds, args.repeats, args.seed, args.budget) != (
            FOLDS, REPEATS, SEED, BUDGET_SECONDS):
        raise ValueError("folds/repeats/seed/budget are frozen; do not change them")
    if not 0 < args.budget <= 300 or args.budget * args.folds * args.repeats > 14400:
        raise ValueError("per-fold budget exceeds the five-minute limit")

    manifest, rows = load_selection(args.manifest, args.dataset, args.xls)
    if manifest["split"]["holdout"]:
        raise ValueError("no holdout evaluation is permitted on the constitution line")
    candidate = Path(args.candidate).read_bytes()
    hashes = code_hashes(candidate)
    device = local_device()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    status_path = args.out.parent / "status.json"

    def status(state: str, **extra: object) -> None:
        payload = {"state": state, "task": manifest["task"], "evidence": EVIDENCE,
                   "folds": args.folds, "updated_at": time.time(), **extra}
        temporary = args.out.parent / (".status-" + uuid.uuid4().hex)
        try:
            with temporary.open("x", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False)
            os.replace(temporary, status_path)
        finally:
            temporary.unlink(missing_ok=True)

    fold_records: list[dict] = []
    try:
        status("TRAINING", fold=0, deadline=time.time() + args.budget * args.folds * args.repeats)
        for repeat in range(args.repeats):
            for fold in range(args.folds):
                train_ids, dev_ids = selection_fold_ids(manifest, fold, args.folds, repeat)
                training, inputs, dev_rows = _fold_payload(manifest, rows, train_ids, dev_ids)
                started = time.monotonic()
                predictions = run_candidate(candidate, training, inputs,
                                            budget_seconds=args.budget, seed=args.seed)
                elapsed = time.monotonic() - started
                masked_mae, per_dim = score_predictions(dev_rows, predictions)
                fold_records.append({
                    "fold": fold,
                    "repeat": repeat,
                    "value": masked_mae,
                    "masked_mae": masked_mae,
                    "per_dim_mae": per_dim,
                    "n_validation": len(dev_rows),
                    "train_seconds": round(elapsed, 3),
                })
                status("TRAINING", fold=fold + 1,
                       deadline=time.time() + args.budget * args.folds * args.repeats)

        values = [record["value"] for record in fold_records]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        result = {
            "metric": METRIC,
            "direction": DIRECTION,
            "metric_version": METRIC_VERSION,
            "code_sha256": hashes,
            "promotion": PROMOTION,
            "masked_mae": mean,
            "fold_std": math.sqrt(variance),
            "fold_count": len(values),
            "repeats": args.repeats,
            "folds": fold_records,
            "device": device["device"],
            "gpu": device["gpu"],
            "dataset_sha256": manifest["dataset_sha256"],
            "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "records_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
            "seed": args.seed,
            "budget_seconds": args.budget,
            "evidence": EVIDENCE,
            "n_target_dims": N_TARGET_DIMS,
            "observed_dims": list(OBSERVED_DIMS),
            "unobserved_dims": list(UNOBSERVED_DIMS),
            "balanced_dimension": "UNOBSERVED_ABSTAIN_NEVER_FABRICATED",
            "holdout_subjects": len(manifest["split"]["holdout"]),
            "label_tier": manifest["label_tier"],
            "label_version": manifest["label_version"],
            "standard_version": manifest["standard_version"],
            "attested_evaluator": "AGENT_ISOLATION_UNVERIFIED",
        }
        write_new_atomic(args.out,
                         (json.dumps(result, ensure_ascii=False, indent=2,
                                     allow_nan=False) + "\n").encode("utf-8"))
        status("COMPLETE", masked_mae=mean)
    except Exception as exc:
        status("FAILED", reason=type(exc).__name__)
        raise
    print(json.dumps(result, ensure_ascii=False))
    return 0


def validate_result(result: dict) -> float:
    """核对结果元数据与冻结体质协议；返回 masked_mae。"""
    if result.get("metric") != METRIC or result.get("direction") != DIRECTION \
            or result.get("metric_version") != METRIC_VERSION:
        raise ValueError("result metadata differs from the frozen constitution protocol")
    if result.get("evidence") != EVIDENCE or result.get("promotion") != PROMOTION:
        raise ValueError("result evidence/promotion is not the frozen constitution one")
    if result.get("balanced_dimension") != "UNOBSERVED_ABSTAIN_NEVER_FABRICATED" \
            or result.get("holdout_subjects") != 0:
        raise ValueError("result must declare balanced unobserved and no holdout")
    value = result.get("masked_mae")
    if value is None:
        raise ValueError("result is missing masked_mae")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("masked_mae must be a finite non-negative number")
    return value


def record_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Append one constitution decision")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--decision", required=True,
                        choices=("KEEP", "REVERT", "CRASH"))
    parser.add_argument("--description", required=True)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args(argv)
    value = None
    protocol = ""
    evidence = EVIDENCE_LOCAL
    peak_vram = "NOT_AVAILABLE"
    if args.decision != "CRASH":
        if args.result is None:
            raise ValueError("a scored decision requires --result")
        result = json.loads(Path(args.result).read_text(encoding="utf-8"))
        value = validate_result(result)
        protocol = result["manifest_sha256"]
        candidate = committed_source(args.repo, args.commit, CANDIDATE_PATH)
        if hashlib.sha256(candidate).hexdigest() != result["code_sha256"]["candidate"]:
            raise ValueError("result candidate hash does not match the committed source")
        peak_vram = result.get("peak_vram_mb", "NOT_AVAILABLE")
    append_history(args.results, commit=args.commit, metric=METRIC, value=value,
                   direction=DIRECTION, decision=args.decision,
                   description=args.description, protocol=protocol,
                   evidence=evidence, peak_vram=peak_vram, promotion=PROMOTION)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "record":
        return record_main(argv[1:])
    return evaluate(argv)


if __name__ == "__main__":
    raise SystemExit(main())
