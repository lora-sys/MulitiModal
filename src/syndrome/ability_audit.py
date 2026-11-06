"""任务 02：148 类证候逐类能力审计（只读，不训练、不调参、不碰 holdout）。

复用冻结 protocol-v2 选择集 dev 分区与已交付证候冠军工件，重算并核对
Task B 的宏观指标，再补齐逐类证据：训练/开发支撑数、P/R/F1、Top-3 命中率、
主要混淆类别、拒判统计、证据状态，以及 55 个 F1=0 类别的逐一解释。

口径边界：dev 是模型选择时见过的内部选择集分区，不是独立测试集；
结果一律 LOCAL_EXPERIMENTAL / AGENT_ISOLATION_UNVERIFIED，不构成临床结论。
低样本类（dev support < 30）即使 F1 高也标为证据不足。

运行（acam-lab，PYTHONPATH=src）：
    python -m syndrome.ability_audit --out-dir runs/ability-audit
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from data.tcm_sd import load_syndrome_vocab
from research.prepare import load_selection
from syndrome.delivery import N_CLASSES, SEED, assert_holdout_untouched, selection_split
from syndrome.inference import STATUS_OK, validate_probability_matrix
from syndrome.metrics import summarize_evaluation
from syndrome.service import (
    ModelRegistry,
    SymptomAnalysisRequest,
    SyndromeService,
    tfidf_logreg_spec,
)

AUDIT_SCHEMA = "tcm-syndrome-ability-audit-v1"
EXPECTED_MACRO_F1 = 0.320213
EXPECTED_TOP3 = 0.843179
LOW_SAMPLE_THRESHOLD = 30
N_BOOT = 1000
TOP_CONFUSIONS = 3
STATUSES = (STATUS_OK, "UNCERTAIN", "ABSTAIN", "OUT_OF_DOMAIN")

CSV_FIELDS = ("index", "syndrome", "train_support", "dev_support", "precision",
              "recall", "f1", "tp", "fp", "fn", "top3_hits", "top3_hit_rate",
              "top3_predicted_count", "status_ok", "status_uncertain",
              "status_abstain", "status_out_of_domain", "rejected",
              "rejection_rate", "evidence_status", "low_sample",
              "top_misclassified_as", "top_false_positive_from",
              "zero_f1_reason", "evidence_note")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_confusion(y_true: list[int], y_pred: list[int], n_classes: int) -> np.ndarray:
    """完整混淆矩阵：行=真实类别，列=Top-1 预测类别，合计=全部评估样本。"""
    truth = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    if truth.shape != pred.shape or truth.size == 0:
        raise ValueError("confusion needs non-empty, equally sized label sequences")
    if truth.min() < 0 or pred.min() < 0 or truth.max() >= n_classes or pred.max() >= n_classes:
        raise ValueError("label index outside the frozen taxonomy")
    return np.bincount(truth * n_classes + pred,
                       minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def confusion_pairs(confusion: np.ndarray, labels: list[str]) -> list[dict]:
    """所有非零误判类别对，按误判次数降序、索引升序，便于排序复核。"""
    n = len(labels)
    pairs = [
        {"true_index": int(i), "true_syndrome": labels[i],
         "predicted_index": int(j), "predicted_syndrome": labels[j],
         "count": int(confusion[i, j]),
         "share_of_true_class": round(float(confusion[i, j] / confusion[i].sum()), 6)
         if confusion[i].sum() else 0.0}
        for i in range(n) for j in range(n)
        if i != j and confusion[i, j] > 0
    ]
    pairs.sort(key=lambda row: (-row["count"], row["true_index"], row["predicted_index"]))
    return pairs


def class_rows(y_true: list[int], y_pred: list[int], scores: np.ndarray,
               train_labels: list[int], labels: list[str],
               statuses: list[str], confusion: np.ndarray) -> list[dict]:
    """逐类能力表；样本计数覆盖全部 dev 样本，而非只统计预测正确的样本。"""
    n = len(labels)
    truth = np.asarray(y_true, dtype=np.int64)
    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.shape != (len(truth), n):
        raise ValueError("score matrix shape does not match (n_dev, n_classes)")
    if len(statuses) != len(truth) or any(s not in STATUSES for s in statuses):
        raise ValueError("statuses must describe every evaluated sample")
    if confusion.shape != (n, n):
        raise ValueError("confusion matrix shape does not match the taxonomy")

    train_support = np.bincount(np.asarray(train_labels, dtype=np.int64), minlength=n)
    dev_support = confusion.sum(axis=1)
    tp = np.diag(confusion)
    fp = confusion.sum(axis=0) - tp
    fn = confusion.sum(axis=1) - tp
    precision = np.divide(tp, tp + fp, out=np.zeros(n), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros(n), where=(tp + fn) > 0)
    denominator = precision + recall
    f1 = np.divide(2 * precision * recall, denominator, out=np.zeros(n),
                   where=denominator > 0)

    k = min(3, n)
    top3 = np.argpartition(matrix, -k, axis=1)[:, -k:]
    hit = (top3 == truth[:, None]).any(axis=1)
    top3_hits = np.bincount(truth[hit], minlength=n)
    top3_predicted = np.bincount(top3.ravel(), minlength=n)
    status_array = np.asarray(statuses)
    status_counts = {
        status.lower(): np.bincount(truth[status_array == status], minlength=n)
        for status in STATUSES
    }
    rejected = np.bincount(truth[status_array != STATUS_OK], minlength=n)

    rows = []
    for i in range(n):
        misclassified = sorted(
            ((j, int(confusion[i, j])) for j in range(n) if j != i and confusion[i, j] > 0),
            key=lambda item: (-item[1], item[0]))[:TOP_CONFUSIONS]
        false_positive_from = sorted(
            ((j, int(confusion[j, i])) for j in range(n) if j != i and confusion[j, i] > 0),
            key=lambda item: (-item[1], item[0]))[:TOP_CONFUSIONS]
        mis_names = [f"{labels[j]}({count})" for j, count in misclassified]
        fp_names = [f"{labels[j]}({count})" for j, count in false_positive_from]
        low_sample = bool(dev_support[i] < LOW_SAMPLE_THRESHOLD)
        if dev_support[i] == 0:
            evidence_status = "NO_DEV_SAMPLES"
            note = "dev 无该类样本，无法评估；macro-F1 的 148 类分母仍计入 F1=0"
        elif low_sample:
            evidence_status = "LOW_SAMPLE_INSUFFICIENT"
            note = (f"dev 仅 {int(dev_support[i])} 个样本（<{LOW_SAMPLE_THRESHOLD}），"
                    "逐类指标不稳定，即使 F1 高也判为证据不足")
        elif f1[i] == 0.0:
            evidence_status = "FAILED_ON_DEV"
            note = f"dev 有 {int(dev_support[i])} 个样本但 Top-1 全部预测失败"
        elif f1[i] < 0.2:
            evidence_status = "WEAK_DEV_EVIDENCE"
            note = f"dev 有 {int(dev_support[i])} 个样本，F1={float(f1[i]):.6f} 偏低"
        else:
            evidence_status = "DEV_EVIDENCE_ONLY"
            note = "仅在选择集 dev 上有内部证据，未经独立外部验证"

        zero_f1_reason = None
        if round(float(f1[i]), 6) == 0.0:
            if dev_support[i] == 0:
                parts = ["dev_support=0：验证集没有该类样本，F1 按 zero_division=0 记为 0",
                         f"train_support={int(train_support[i])}，"
                         "macro-F1 的 148 类分母仍包含该类"]
            else:
                parts = [f"dev 有 {int(dev_support[i])} 个样本，Top-1 正确 0 个",
                         f"train_support={int(train_support[i])}"]
                if train_support[i] == 0:
                    parts.append("训练集无该类样本，模型不可能学到该类")
                if fp[i] > 0:
                    parts.append(f"模型曾把该类预测为 Top-1 共 {int(fp[i])} 次，但无一正确")
                else:
                    parts.append("模型从未把该类输出为 Top-1")
                if mis_names:
                    parts.append("主要误判为 " + "、".join(mis_names))
            if low_sample:
                parts.append(f"dev_support<{LOW_SAMPLE_THRESHOLD}，属低样本证据不足，"
                             "不能据此断言该类别完全不可用")
            zero_f1_reason = "；".join(parts)

        rows.append({
            "index": i,
            "syndrome": labels[i],
            "train_support": int(train_support[i]),
            "dev_support": int(dev_support[i]),
            "precision": round(float(precision[i]), 6),
            "recall": round(float(recall[i]), 6),
            "f1": round(float(f1[i]), 6),
            "tp": int(tp[i]),
            "fp": int(fp[i]),
            "fn": int(fn[i]),
            "top3_hits": int(top3_hits[i]),
            "top3_hit_rate": round(float(top3_hits[i] / dev_support[i]), 6)
            if dev_support[i] else None,
            "top3_predicted_count": int(top3_predicted[i]),
            "status_ok": int(status_counts[STATUS_OK.lower()][i]),
            "status_uncertain": int(status_counts["UNCERTAIN".lower()][i]),
            "status_abstain": int(status_counts["ABSTAIN".lower()][i]),
            "status_out_of_domain": int(status_counts["OUT_OF_DOMAIN".lower()][i]),
            "rejected": int(rejected[i]),
            "rejection_rate": round(float(rejected[i] / dev_support[i]), 6)
            if dev_support[i] else None,
            "evidence_status": evidence_status,
            "low_sample": low_sample,
            "top_misclassified_as": "；".join(mis_names),
            "top_false_positive_from": "；".join(fp_names),
            "zero_f1_reason": zero_f1_reason or "",
            "evidence_note": note,
        })
    return rows


def verify(labels: list[str], artifact_labels: list[str], rows: list[dict],
           confusion: np.ndarray, evaluation: dict, split: dict,
           expected_macro_f1: float = EXPECTED_MACRO_F1,
           expected_top3: float = EXPECTED_TOP3,
           expected_n_classes: int = N_CLASSES) -> dict:
    """核对 148 类覆盖、索引/名称一致、样本守恒与 Task B 的两个既定数值。"""
    n = len(labels)
    n_dev = len(split["dev_labels"])
    checks = {
        "n_labels_is_148": n == expected_n_classes,
        "labels_unique": len(set(labels)) == n,
        "artifact_labels_match_vocab_order": artifact_labels == labels,
        "rows_cover_all_148_labels":
            len(rows) == n and [r["index"] for r in rows] == list(range(n)),
        "row_names_match_vocab": [r["syndrome"] for r in rows] == labels,
        "holdout_untouched": split["holdout_subjects"] == 0,
        "sum_dev_support_equals_all_evaluated_samples":
            sum(r["dev_support"] for r in rows) == n_dev,
        "confusion_matrix_sums_to_all_evaluated_samples": int(confusion.sum()) == n_dev,
        "tp_plus_fn_equals_all_evaluated_samples":
            sum(r["tp"] for r in rows) + sum(r["fn"] for r in rows) == n_dev,
        "tp_plus_fp_equals_all_evaluated_samples":
            sum(r["tp"] for r in rows) + sum(r["fp"] for r in rows) == n_dev,
        "macro_f1_matches_task_b": evaluation["macro_f1"] == expected_macro_f1,
        "top3_accuracy_matches_task_b": evaluation["top3_accuracy"] == expected_top3,
    }
    report = {
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "n_labels": n,
        "n_dev_samples": n_dev,
        "n_train_samples": len(split["train_labels"]),
        "sum_dev_support": sum(r["dev_support"] for r in rows),
        "sum_confusion_matrix": int(confusion.sum()),
        "sum_tp": sum(r["tp"] for r in rows),
        "sum_fn": sum(r["fn"] for r in rows),
        "sum_fp": sum(r["fp"] for r in rows),
        "macro_f1": evaluation["macro_f1"],
        "expected_macro_f1": expected_macro_f1,
        "top3_accuracy": evaluation["top3_accuracy"],
        "expected_top3_accuracy": expected_top3,
        "top3_hits_total": sum(r["top3_hits"] for r in rows),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"ability audit verification failed: {failed}")
    return report


def render_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render_pairs_csv(pairs: list[dict], path: Path) -> None:
    fields = ("true_index", "true_syndrome", "predicted_index", "predicted_syndrome",
              "count", "share_of_true_class")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(pairs)


def render_markdown(report: dict, path: Path) -> None:
    acceptance = report["acceptance"]
    verification = report["verification"]
    zero = report["zero_f1_summary"]
    rows = report["per_class"]
    lines = [
        "# 148 类证候逐类能力审计（任务 02）",
        "",
        f"- Schema：`{report['schema_version']}`",
        f"- 证据等级：`{report['status']['evidence']}`；"
        f"隔离级别：`{report['status']['trusted_evaluator']}`；"
        f"外部验证：`{report['status']['external_validation']}`",
        "- 数据口径：冻结 protocol-v2 **选择集 dev 分区**，无 holdout、无重训练、无阈值调优；"
        "不是独立测试集，不构成临床有效性结论。",
        f"- 排序与输出：完整逐类表见 `syndrome_ability.csv`（{len(rows)} 行，可按任意列排序）；"
        "机器可读结果见 `syndrome_ability.json`；混淆矩阵见 `confusion_matrix.json`；"
        "误判对见 `confusion_pairs.csv`。",
        "",
        "## 1. 宏观指标与核对",
        "",
        "| 指标 | 值 | 口径 |",
        "| --- | --- | --- |",
        f"| 评估样本数 | {acceptance['n_samples']} | dev 分区全部已标注样本 |",
        f"| Top-1 准确率 | {acceptance['accuracy']} | score 矩阵 argmax |",
        f"| **Top-3 命中率** | **{acceptance['top3_accuracy']}** | "
        "每样本取分数最高 3 类，真值命中即计 1；与 Task B 一致（保留 6 位） |",
        f"| Top-5 命中率 | {acceptance['top5_accuracy']} | 同上取 5 类 |",
        f"| **Macro-F1** | **{acceptance['macro_f1']}** | 对全部 148 类逐类 F1 取算术平均；"
        "dev 缺席类按 zero_division=0 记 F1=0，仍进入分母 |",
        f"| Macro-F1 95% CI | [{acceptance['macro_f1_ci']['low']}, "
        f"{acceptance['macro_f1_ci']['high']}] | 样本级 bootstrap，"
        f"n_boot={acceptance['macro_f1_ci']['n_boot']}, seed={acceptance['macro_f1_ci']['seed']} |",
        "",
        f"核对：`{verification['all_checks_passed']}`。"
        f"逐类 dev_support 之和 = {verification['sum_dev_support']} = 实际评估样本数 "
        f"{verification['n_dev_samples']}；混淆矩阵总和 = {verification['sum_confusion_matrix']}；"
        f"TP+FN = {verification['sum_tp'] + verification['sum_fn']}；"
        f"TP+FP = {verification['sum_tp'] + verification['sum_fp']}。"
        "统计的不是「只算预测正确的样本」，而是全部评估样本。",
        "",
        "## 2. F1=0 的类别逐类解释",
        "",
        f"共 **{zero['n_zero_f1']}** 类 F1=0："
        f"**{zero['n_no_dev_samples']}** 类完全没有 dev 样本，"
        f"**{zero['n_with_dev_samples']}** 类有 dev 样本但 Top-1 全部预测失败；"
        f"其中 {zero['n_no_train_samples']} 类在训练集中也没有样本。"
        f"另有 {zero['n_low_sample_zero_f1']} 个 F1=0 类别同时属于低样本证据不足，"
        "不能据此断言其完全不可用。",
        "",
        "| 索引 | 证候 | train | dev | FP | Top-3 命中 | 证据状态 | 解释 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        if row["zero_f1_reason"]:
            lines.append(
                f"| {row['index']} | {row['syndrome']} | {row['train_support']} | "
                f"{row['dev_support']} | {row['fp']} | {row['top3_hits']} | "
                f"{row['evidence_status']} | {row['zero_f1_reason']} |")
    lines += [
        "",
        "## 3. 证据不足与薄弱类别",
        "",
        f"- 低样本阈值：dev support < {LOW_SAMPLE_THRESHOLD}；"
        f"共 {report['evidence_summary']['n_low_sample']} 类证据不足，"
        f"其中 {report['evidence_summary']['n_high_f1_but_low_sample']} "
        "类 F1≥0.5 仍被标为证据不足。",
        f"- 证据状态分布：{report['evidence_summary']['by_status']}。",
        f"- 最常见的误判类别对（前 {len(report['confusion_summary']['top_pairs'])} 条）：",
        "",
        "| 真值 | 误判为 | 次数 | 占该类 dev 样本 |",
        "| --- | --- | --- | --- |",
    ]
    for pair in report["confusion_summary"]["top_pairs"]:
        lines.append(f"| {pair['true_syndrome']} | {pair['predicted_syndrome']} | "
                     f"{pair['count']} | {pair['share_of_true_class']} |")
    lines += [
        "",
        "## 4. 完整 148 类能力表",
        "",
        "| 索引 | 证候 | train | dev | P | R | F1 | Top-3 命中率 | 拒判率 | "
        "主要误判为 | 证据状态 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['index']} | {row['syndrome']} | {row['train_support']} | "
            f"{row['dev_support']} | {row['precision']} | {row['recall']} | {row['f1']} | "
            f"{row['top3_hit_rate'] if row['top3_hit_rate'] is not None else 'NA'} | "
            f"{row['rejection_rate'] if row['rejection_rate'] is not None else 'NA'} | "
            f"{row['top_misclassified_as'] or '—'} | {row['evidence_status']} |")
    lines += [
        "",
        "## 5. 边界声明",
        "",
    ]
    lines += [f"- {item}" for item in report["unverified_generalization"]]
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def audit(manifest_path: Path, dataset_path: Path, raw_dir: Path,
          artifact_dir: str, seed: int = SEED,
          n_boot: int = N_BOOT) -> tuple[dict, np.ndarray, list[dict]]:
    """只读审计：不写实验历史、不训练、不改冠军、不解锁历史 holdout。"""
    manifest, records = load_selection(manifest_path, dataset_path)
    assert_holdout_untouched(manifest)
    split = selection_split(manifest, records)
    labels = load_syndrome_vocab(raw_dir)
    if len(labels) != N_CLASSES or len(set(labels)) != N_CLASSES:
        raise ValueError("frozen syndrome vocabulary is not the 148-class taxonomy")
    artifact_dir_path = Path(artifact_dir)
    artifact_labels = json.loads((artifact_dir_path / "labels.json").read_text(encoding="utf-8"))

    spec = tfidf_logreg_spec(artifact_dir)
    if spec is None:
        raise ValueError(f"syndrome artifact unavailable: {artifact_dir}")
    registry = ModelRegistry()
    registry.register(spec, default=True)
    service = SyndromeService(registry=registry)
    loaded = spec.loader()

    dev_texts = split["dev_texts"]
    dev_labels = split["dev_labels"]
    dev_scores = validate_probability_matrix(
        loaded.predictor.predict_proba(dev_texts), len(dev_texts), N_CLASSES)
    dev_pred = np.asarray(dev_scores).argmax(axis=1).tolist()
    evaluation = summarize_evaluation(dev_labels, dev_pred, dev_scores, labels,
                                      baseline_pred=None, n_boot=n_boot, seed=seed)
    per_class = evaluation.pop("per_class")

    statuses = [
        service.analyze_symptoms(SymptomAnalysisRequest(symptom_text=text)).to_dict()["status"]
        for text in dev_texts
    ]
    confusion = build_confusion(dev_labels, dev_pred, N_CLASSES)
    rows = class_rows(dev_labels, dev_pred, dev_scores, split["train_labels"],
                      labels, statuses, confusion)
    for row, official in zip(rows, per_class, strict=True):
        if (row["syndrome"] != official["label"] or row["dev_support"] != official["support"]
                or row["precision"] != official["precision"]
                or row["recall"] != official["recall"] or row["f1"] != official["f1"]):
            raise ValueError("per-class metrics disagree with syndrome.metrics")

    verification = verify(labels, artifact_labels, rows, confusion, evaluation, split)
    pairs = confusion_pairs(confusion, labels)
    zero_rows = [row for row in rows if row["zero_f1_reason"]]
    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["evidence_status"]] = status_counts.get(row["evidence_status"], 0) + 1

    report = {
        "schema_version": AUDIT_SCHEMA,
        "status": {
            "evidence": "LOCAL_EXPERIMENTAL",
            "trusted_evaluator": "AGENT_ISOLATION_UNVERIFIED",
            "external_validation": "NONE",
            "clinical_claim": False,
            "purpose": "ABILITY_AUDIT_ONLY",
        },
        "inputs": {
            "manifest_sha256": _sha256(manifest_path),
            "dataset_sha256": _sha256(dataset_path),
            "vocab_sha256": _sha256(raw_dir / "syndrome_vocab.txt"),
            "artifact_dir": artifact_dir,
            "model_sha256": _sha256(artifact_dir_path / "model.joblib"),
            "labels_sha256": _sha256(artifact_dir_path / "labels.json"),
            "registry_sha256": _sha256(artifact_dir_path / "registry_entry.json"),
            "holdout_subjects": split["holdout_subjects"],
            "scope": "frozen protocol-v2 selection dev; no holdout, no retraining, no tuning",
        },
        "protocol": {
            "n_classes": N_CLASSES,
            "seed": seed,
            "n_boot": n_boot,
            "dev_dedup_dropped": split["dev_dedup_dropped"],
            "low_sample_threshold": LOW_SAMPLE_THRESHOLD,
            "statistics": {
                "macro_f1": "全部 148 类逐类 F1 算术平均；"
                            "dev 缺席类 F1=0（zero_division=0）仍计入分母",
                "top3_accuracy": "每样本取分数最高 3 类，真值在其中即命中；"
                                "np.argpartition 平分按索引",
                "per_class_support": "dev_support 统计该类全部真实样本；"
                                    "混淆矩阵总和等于全部评估样本数",
                "top3_hit_rate": "该类别真实样本中真值进入 Top-3 的比例（recall@3）；"
                                "无 dev 样本记 null",
                "top3_predicted_count": "全部 dev 样本的 Top-3 列表中该类别出现次数"
                                       "（可作为候选输出）",
                "rejection": "SyndromeService.analyze_symptoms 在 dev 文本上的状态；非 OK 计为拒判",
                "evidence_status": "无 dev 样本 / 低样本 / 预测失败 / 弱证据 / 仅内部 dev 证据",
            },
        },
        "acceptance": {
            "accuracy": evaluation["accuracy"],
            "top1_accuracy": evaluation["accuracy"],
            "top3_accuracy": evaluation["top3_accuracy"],
            "top5_accuracy": evaluation["top5_accuracy"],
            "macro_f1": evaluation["macro_f1"],
            "macro_f1_ci": evaluation["macro_f1_ci"],
            "n_samples": evaluation["n_samples"],
            "n_classes_observed": evaluation["n_classes_observed"],
        },
        "verification": verification,
        "zero_f1_summary": {
            "n_zero_f1": len(zero_rows),
            "n_no_dev_samples": sum(r["dev_support"] == 0 for r in zero_rows),
            "n_with_dev_samples": sum(r["dev_support"] > 0 for r in zero_rows),
            "n_no_train_samples": sum(r["train_support"] == 0 for r in zero_rows),
            "n_never_predicted": sum(r["fp"] == 0 and r["dev_support"] > 0 for r in zero_rows),
            "n_predicted_but_never_correct": sum(r["fp"] > 0 for r in zero_rows),
            "n_low_sample_zero_f1": sum(r["low_sample"] for r in zero_rows),
            "explained_individually": True,
            "where": "per_class[].zero_f1_reason；CSV 同名列；本报告第 2 节",
        },
        "evidence_summary": {
            "n_low_sample": sum(r["low_sample"] for r in rows),
            "n_high_f1_but_low_sample": sum(r["low_sample"] and r["f1"] >= 0.5 for r in rows),
            "by_status": status_counts,
        },
        "confusion_summary": {
            "n_off_diagonal_pairs": len(pairs),
            "n_misclassified_samples": verification["sum_fn"],
            "top_pairs": pairs[:20],
            "where": "confusion_matrix.json（148×148 稠密矩阵）；"
                     "confusion_pairs.csv（全部非零误判对）",
        },
        "per_class": rows,
        "unverified_generalization": [
            "dev 属于模型选择时见过的内部选择集，不是独立外部测试集；跨来源/时间泛化未验证",
            "F1=0 的类别中，低样本与「有样本但预测失败」已分开标记，二者都不能直接等同于临床无效",
            "低样本类别即使 F1 高也标为证据不足；高分不代表可稳定复现",
            "TCM-SD 数据许可 CC BY-NC-SA 4.0（仅非商业科研）；派生知识同许可",
            "分数未经校准，不是概率、置信度、诊断或临床判断",
            "GO_RESEARCH_CLAIMS 未放行；本报告只回答「开发集内哪些类别有预测证据」",
        ],
    }
    return report, confusion, pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/local/tcm-sd")
    parser.add_argument("--manifest", default=("data/local/tcm-sd/protocol-v2/"
                                               "short/selection_manifest.json"))
    parser.add_argument("--dataset", default=("data/local/tcm-sd/protocol-v2/"
                                              "short/selection_records.json"))
    parser.add_argument("--artifact-dir", default="artifacts/sd-v2-ar1-champion")
    parser.add_argument("--out-dir", default="runs/ability-audit")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args()
    report, confusion, pairs = audit(Path(args.manifest), Path(args.dataset),
                                     Path(args.raw_dir), args.artifact_dir,
                                     args.seed, args.n_boot)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "syndrome_ability.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "confusion_matrix.json").write_text(
        json.dumps({
            "schema_version": "tcm-syndrome-confusion-matrix-v1",
            "labels": [row["syndrome"] for row in report["per_class"]],
            "row_axis": "true_syndrome_index",
            "column_axis": "predicted_top1_syndrome_index",
            "matrix": confusion.tolist(),
            "row_sums": confusion.sum(axis=1).tolist(),
            "total": int(confusion.sum()),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    render_csv(report["per_class"], out / "syndrome_ability.csv")
    render_pairs_csv(pairs, out / "confusion_pairs.csv")
    render_markdown(report, out / "report.md")
    print(json.dumps({
        "n_classes": len(report["per_class"]),
        "macro_f1": report["acceptance"]["macro_f1"],
        "top3_accuracy": report["acceptance"]["top3_accuracy"],
        "n_zero_f1": report["zero_f1_summary"]["n_zero_f1"],
        "verified": report["verification"]["all_checks_passed"],
        "out_dir": str(out),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
