"""中医体质 AutoResearch 第二阶段：独立数据适配与冻结选择集。

与证候线（src/research/prepare.py 的 protocol-v2）完全独立：数据来自 Mendeley
274 名受试者的八维 PUBLISHED 体质评分，任务是分数回归，不是 148 类文本分类。

冻结约束（不得在实验中修改）：
- 原始 xls 只读，哈希导入前后各校验一次；
- 平和质（balanced）是第 9 维且**完全未观测**，绝不伪造、不补零、不猜测，
  目标只有 CANONICAL_ORDER[1:] 的 8 维；
- 缺失特征以 null + feature_mask=False 显式表达，绝不补零；
- 每条记录一名受试者，行序即匿名身份（UNVERIFIED），同一受试者不跨分区；
- holdout 为空：历史体质 holdout 的 2/2 解锁预算已耗尽，本线只用
  selection-only 的 4 折 × 2 重复分组 CV；
- 标签层级 PUBLISHED，标准 ZYYXH/T157-2009，来源随 provenance 保留。

生成物（selection_manifest.json / selection_records.json）位于 gitignored 的
data/local/ 目录，不入 Git；其 sha256 常量冻结在本模块中。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

from data.mendeley import (
    AUDITED_FEATURE_COLUMNS,
    EXPECTED_SHA256,
    FEATURE_GROUP_SIZES,
    FEATURE_GROUPS,
    load_mendeley_scores,
    verify_mendeley_sha256,
)
from data.real_split import LABEL_VERSION_REGRESSION
from labels.standards import CANONICAL_ORDER

PROTOCOL_VERSION = "constitution-protocol-v1"
STANDARD_VERSION = "ZYYXH/T157-2009"
TASK = "tcm-constitution-8d-score-regression"
N_SUBJECTS = 274
OBSERVED_DIMS = tuple(CANONICAL_ORDER[1:])  # 8 维，balanced 明确排除
UNOBSERVED_DIMS = ("balanced",)
N_TARGET_DIMS = 8
N_FEATURE_COLUMNS = len(AUDITED_FEATURE_COLUMNS)
FOLDS = 4
REPEATS = 2
SEED = 20261105
BUDGET_SECONDS = 300
EVIDENCE = "LOCAL_EXPERIMENTAL"

# 冻结哈希：由 prepare 子命令生成选择集后写入，load_selection 逐一校验。
CONSTITUTION_DATASET_SHA256 = "6206f432b683f8a7b9a75dae23e598a4f01a41a4f8af84b099924b9d85d08a96"
CONSTITUTION_MANIFEST_SHA256 = "611d4552716b81dc9d19b3fae32c0553a51a6c7c530e065f5548ed71467813b9"
CONSTITUTION_RECORDS_SHA256 = "441fc26ba65f34abbfcba208087f3f76ae3f7aecfd017868f893b8e75823fed5"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def subject_id(row_index: int) -> str:
    return f"m{row_index:04d}"


def build_selection(xls_path: Path, out_dir: Path) -> tuple[Path, Path]:
    """从只读 xls 生成冻结选择集；原始文件哈希在解析前后各校验一次。"""
    digest_before = verify_mendeley_sha256(xls_path)
    records = load_mendeley_scores(xls_path)
    digest_after = verify_mendeley_sha256(xls_path)
    if digest_before != digest_after or digest_before != EXPECTED_SHA256:
        raise ValueError("Mendeley source changed during import")
    if len(records) != N_SUBJECTS:
        raise ValueError(f"expected {N_SUBJECTS} subjects, got {len(records)}")

    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if record.score_mask[0] or record.scores[0] is not None:
            raise ValueError("balanced dimension must stay unobserved; never fabricate it")
        if len(record.scores) != 9 or len(record.score_mask) != 9:
            raise ValueError("canonical 9D target with an explicit mask is required")
        scores = record.scores[1:]
        mask = record.score_mask[1:]
        if len(scores) != N_TARGET_DIMS or not all(mask):
            raise ValueError("the 8 observed constitution scores must be complete")
        if any(not isinstance(value, (int, float)) or not math.isfinite(value)
               for value in scores):
            raise ValueError("PUBLISHED constitution scores must be finite")

        flat_features = [value for group in FEATURE_GROUPS
                         for value in record.features[group]]
        if len(flat_features) != N_FEATURE_COLUMNS:
            raise ValueError("audited feature vector has the wrong width")
        features: list[float | None] = []
        feature_mask: list[bool] = []
        for value in flat_features:
            missing = value is None or (isinstance(value, float) and math.isnan(value))
            features.append(None if missing else float(value))
            feature_mask.append(not missing)
            if not missing and not math.isfinite(features[-1]):  # type: ignore[arg-type]
                raise ValueError("non-finite feature value")

        rows.append({
            "subject_index": index,
            "subject_id": subject_id(index),
            "features": features,
            "feature_mask": feature_mask,
            "scores": [float(value) for value in scores],
            "score_dims": list(OBSERVED_DIMS),
            "score_mask": list(mask),
            "provenance": {
                "source_dataset": "mendeley-cxtph3tjsg-v1",
                "dataset_sha256": digest_after,
                "label_tier": "PUBLISHED",
                "label_version": LABEL_VERSION_REGRESSION,
                "standard_version": STANDARD_VERSION,
                "score_column_order": list(OBSERVED_DIMS),
                "unobserved_dimensions": list(UNOBSERVED_DIMS),
                "identity": "row order is the only anonymous identity: UNVERIFIED",
                "feature_missing_policy": "null + feature_mask=False; never zero-filled",
            },
        })

    ids = [row["subject_id"] for row in rows]
    manifest: dict[str, Any] = {
        "version": PROTOCOL_VERSION,
        "task": TASK,
        "task_type": "score_regression",
        "dataset_id": "mendeley-cxtph3tjsg-v1",
        "dataset_sha256": digest_after,
        "label_tier": "PUBLISHED",
        "label_version": LABEL_VERSION_REGRESSION,
        "standard_version": STANDARD_VERSION,
        "score_dims": list(OBSERVED_DIMS),
        "unobserved_dimensions": list(UNOBSERVED_DIMS),
        "n_subjects": len(rows),
        "n_target_dims": N_TARGET_DIMS,
        "n_feature_columns": N_FEATURE_COLUMNS,
        "feature_group_sizes": dict(FEATURE_GROUP_SIZES),
        "modalities": {"features": {"dim": N_FEATURE_COLUMNS, "kind": "numeric_with_mask"}},
        "split": {
            "seed": SEED,
            "folds": FOLDS,
            "repeats": REPEATS,
            "train": ids,
            "dev": [],
            "holdout": [],
            "selection_only": True,
            "policy": "subject-grouped 4-fold x 2-repeat CV over the whole selection pool",
        },
        "subject_row_by_id": {row["subject_id"]: row["subject_index"] for row in rows},
        "user_partition_by_id": {sid: "train" for sid in ids},
        "budget_seconds": BUDGET_SECONDS,
        "metric": "constitution/masked_mae",
        "direction": "minimize",
        "metric_version": "constitution-eval-v1",
        "evidence": EVIDENCE,
        "notes": [
            "平和质标签不存在，ABSTAIN，不伪造、不补零、不猜测",
            "历史体质 holdout 2/2 解锁预算已耗尽，本线不读取任何历史 holdout",
            "各行身份仅由行序表达，UNVERIFIED；一行一名受试者，天然不跨分区",
            "缺失特征以 null 与 mask 显式表达，候选不得把缺失当 0",
        ],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "selection_manifest.json"
    records_path = out_dir / "selection_records.json"
    manifest_payload = (json.dumps(manifest, ensure_ascii=False, sort_keys=True,
                                  indent=2, allow_nan=False) + "\n").encode("utf-8")
    records_payload = (json.dumps(rows, ensure_ascii=False, sort_keys=True,
                                  allow_nan=False) + "\n").encode("utf-8")
    manifest_path.write_bytes(manifest_payload)
    records_path.write_bytes(records_payload)
    return manifest_path, records_path


def load_selection(manifest_path: Path, records_path: Path,
                   xls_path: Path) -> tuple[dict, list[dict]]:
    """读取并校验冻结选择集；任何哈希或结构偏差都拒绝加载。"""
    verify_mendeley_sha256(xls_path)  # 真实数据只读校验（导入后再次核对）
    manifest_bytes = Path(manifest_path).read_bytes()
    records_bytes = Path(records_path).read_bytes()
    if (CONSTITUTION_MANIFEST_SHA256
            and _sha256_bytes(manifest_bytes) != CONSTITUTION_MANIFEST_SHA256):
        raise ValueError("constitution selection manifest hash does not match the frozen value")
    if (CONSTITUTION_RECORDS_SHA256
            and _sha256_bytes(records_bytes) != CONSTITUTION_RECORDS_SHA256):
        raise ValueError("constitution selection records hash does not match the frozen value")
    if _sha256_bytes(xls_path.read_bytes()) != CONSTITUTION_DATASET_SHA256:
        raise ValueError("constitution dataset hash does not match the frozen value")

    manifest = json.loads(manifest_bytes)
    rows = json.loads(records_bytes)
    _validate_manifest(manifest)
    if len(rows) != manifest["n_subjects"] or len(rows) != N_SUBJECTS:
        raise ValueError("selection records count does not match the frozen manifest")
    for index, row in enumerate(rows):
        _validate_row(row, index, manifest)
    return manifest, rows


def _validate_manifest(manifest: dict) -> None:
    expected = {
        "version": PROTOCOL_VERSION,
        "task": TASK,
        "dataset_sha256": EXPECTED_SHA256,
        "label_tier": "PUBLISHED",
        "label_version": LABEL_VERSION_REGRESSION,
        "n_subjects": N_SUBJECTS,
        "n_target_dims": N_TARGET_DIMS,
        "n_feature_columns": N_FEATURE_COLUMNS,
        "metric": "constitution/masked_mae",
        "direction": "minimize",
        "metric_version": "constitution-eval-v1",
        "evidence": EVIDENCE,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"frozen manifest field {key!r} changed")
    if tuple(manifest.get("score_dims", ())) != OBSERVED_DIMS:
        raise ValueError("score_dims must be exactly the 8 observed dimensions")
    if tuple(manifest.get("unobserved_dimensions", ())) != UNOBSERVED_DIMS:
        raise ValueError("balanced must stay declared as unobserved")
    split = manifest.get("split", {})
    if split.get("holdout"):
        raise ValueError("selection-only protocol must have an empty holdout")
    if not split.get("selection_only") or split.get("folds") != FOLDS \
            or split.get("repeats") != REPEATS or split.get("seed") != SEED:
        raise ValueError("frozen split configuration changed")
    ids = split.get("train", [])
    if len(ids) != N_SUBJECTS or len(set(ids)) != N_SUBJECTS or split.get("dev"):
        raise ValueError("selection pool must be the complete 274-subject list")
    rows_by_id = manifest.get("subject_row_by_id", {})
    if set(rows_by_id) != set(ids) or sorted(rows_by_id.values()) != list(range(N_SUBJECTS)):
        raise ValueError("subject_row_by_id does not cover the frozen pool")
    if set(manifest.get("user_partition_by_id", {})) != set(ids):
        raise ValueError("user_partition_by_id does not cover the frozen pool")


def _validate_row(row: dict, index: int, manifest: dict) -> None:
    if row.get("subject_index") != index or row.get("subject_id") != subject_id(index):
        raise ValueError("subject identity / row order mismatch")
    dims = row.get("score_dims")
    if tuple(dims or ()) != OBSERVED_DIMS or "balanced" in (dims or ()):
        raise ValueError("balanced label must never be fabricated into targets")
    scores = row.get("scores")
    score_mask = row.get("score_mask")
    if not isinstance(scores, list) or len(scores) != N_TARGET_DIMS:
        raise ValueError("each subject needs exactly 8 observed scores")
    if score_mask != [True] * N_TARGET_DIMS:
        raise ValueError("the 8 observed score dimensions must be fully masked in")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) for value in scores):
        raise ValueError("non-finite or non-numeric PUBLISHED score")
    features = row.get("features")
    feature_mask = row.get("feature_mask")
    if not isinstance(features, list) or len(features) != N_FEATURE_COLUMNS \
            or not isinstance(feature_mask, list) or len(feature_mask) != N_FEATURE_COLUMNS:
        raise ValueError("feature vector / mask has the wrong width")
    for value, observed in zip(features, feature_mask, strict=True):
        if not isinstance(observed, bool):
            raise ValueError("feature_mask must be boolean")
        if observed:
            if value is None or isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(value):
                raise ValueError("observed feature must be a finite number")
        elif value is not None:
            raise ValueError("missing feature must be null, never zero-filled")
    provenance = row.get("provenance", {})
    if provenance.get("label_tier") != "PUBLISHED" \
            or provenance.get("dataset_sha256") != EXPECTED_SHA256 \
            or provenance.get("unobserved_dimensions") != list(UNOBSERVED_DIMS):
        raise ValueError("PUBLISHED provenance / unobserved-dimension declaration missing")


def selection_fold_ids(manifest: dict, fold: int, folds: int,
                       repeat: int) -> tuple[list[str], list[str]]:
    """确定性受试者分组折：(manifest, fold, folds, repeat) 唯一决定划分。"""
    if folds != FOLDS or not 0 <= fold < folds or not 0 <= repeat < REPEATS:
        raise ValueError(f"invalid fold coordinates: fold={fold}, folds={folds}, repeat={repeat}")
    pool = sorted(manifest["split"]["train"])
    rng = random.Random(f"{PROTOCOL_VERSION}|{SEED}|{repeat}")
    rng.shuffle(pool)
    offset = repeat % folds
    dev = sorted(sid for index, sid in enumerate(pool)
                 if (index + offset) % folds == fold)
    train = sorted(sid for sid in pool if sid not in set(dev))
    if not train or not dev or set(train) & set(dev):
        raise ValueError("fold derivation produced an invalid partition")
    return train, dev


def verify_real_data(manifest_path: Path, records_path: Path, xls_path: Path) -> dict:
    """真实数据与冻结协议的独立核查摘要（不训练）。"""
    manifest, rows = load_selection(manifest_path, records_path, xls_path)
    missing = sum(sum(not observed for observed in row["feature_mask"]) for row in rows)
    duplicate_vectors = len(rows) - len({tuple(row["features"]) for row in rows})
    return {
        "dataset_sha256": EXPECTED_SHA256,
        "manifest_sha256": _sha256_bytes(manifest_path.read_bytes()),
        "records_sha256": _sha256_bytes(records_path.read_bytes()),
        "n_subjects": len(rows),
        "n_target_dims": N_TARGET_DIMS,
        "observed_dims": list(OBSERVED_DIMS),
        "unobserved_dims": list(UNOBSERVED_DIMS),
        "balanced_observed_in_any_row": any(
            "balanced" in row["score_dims"] or len(row["scores"]) != N_TARGET_DIMS
            for row in rows
        ),
        "missing_feature_cells": missing,
        "duplicate_feature_vectors": duplicate_vectors,
        "holdout_subjects": len(manifest["split"]["holdout"]),
        "label_tier": manifest["label_tier"],
        "label_version": manifest["label_version"],
        "standard_version": manifest["standard_version"],
        "identity": "row order only: UNVERIFIED",
        "evidence": EVIDENCE,
        "attested_evaluator": "AGENT_ISOLATION_UNVERIFIED",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify"))
    parser.add_argument("--xls", type=Path,
                        default=Path("data/local/mendeley-cxtph3tjsg-v1/Share data.xls"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("data/local/mendeley-cxtph3tjsg-v1/protocol-v1"))
    args = parser.parse_args(argv)
    manifest_path = args.out_dir / "selection_manifest.json"
    records_path = args.out_dir / "selection_records.json"
    if args.action == "prepare":
        build_selection(args.xls, args.out_dir)
        print(json.dumps({
            "CONSTITUTION_MANIFEST_SHA256": _sha256_bytes(manifest_path.read_bytes()),
            "CONSTITUTION_RECORDS_SHA256": _sha256_bytes(records_path.read_bytes()),
        }, ensure_ascii=False))
    else:
        print(json.dumps(verify_real_data(manifest_path, records_path, args.xls),
                         ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
