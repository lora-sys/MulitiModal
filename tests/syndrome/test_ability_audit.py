"""任务 02 契约测试：逐类计数守恒、零 F1 归因、证据状态与输出格式。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from syndrome import ability_audit as audit_mod
from syndrome.metrics import summarize_evaluation, topk_accuracy

pytestmark = pytest.mark.synthetic


def _fixture():
    labels = ["A", "B", "C", "D", "E", "F"]
    y_true = [0, 0, 1, 1, 2, 3, 3, 3]
    y_pred = [0, 2, 1, 1, 1, 3, 3, 0]
    scores = np.array([
        [0.90, 0.04, 0.03, 0.02, 0.01, 0.00],
        [0.10, 0.20, 0.60, 0.05, 0.03, 0.02],
        [0.05, 0.80, 0.06, 0.04, 0.03, 0.02],
        [0.04, 0.82, 0.05, 0.04, 0.03, 0.02],
        [0.05, 0.70, 0.10, 0.08, 0.04, 0.03],
        [0.05, 0.05, 0.05, 0.75, 0.06, 0.04],
        [0.05, 0.05, 0.05, 0.75, 0.06, 0.04],
        [0.55, 0.20, 0.10, 0.08, 0.04, 0.03],
    ], dtype=float)
    statuses = ["OK", "UNCERTAIN", "OK", "OK", "ABSTAIN", "OK", "UNCERTAIN", "OK"]
    train_labels = [0, 0, 1, 1, 3, 3, 3, 3]
    confusion = audit_mod.build_confusion(y_true, y_pred, len(labels))
    rows = audit_mod.class_rows(y_true, y_pred, scores, train_labels, labels,
                                statuses, confusion)
    return labels, y_true, y_pred, scores, statuses, train_labels, confusion, rows


def test_class_rows_covers_all_labels_and_counts_every_evaluated_sample():
    (labels, y_true, y_pred, scores, statuses, train_labels,
     confusion, rows) = _fixture()
    n = len(labels)
    assert [row["index"] for row in rows] == list(range(n))
    assert [row["syndrome"] for row in rows] == labels
    assert sum(row["dev_support"] for row in rows) == len(y_true)
    assert int(confusion.sum()) == len(y_true)
    assert sum(row["tp"] for row in rows) + sum(row["fn"] for row in rows) == len(y_true)
    assert sum(row["tp"] for row in rows) + sum(row["fp"] for row in rows) == len(y_true)
    assert sum(row["top3_hits"] for row in rows) == int(
        round(topk_accuracy(y_true, scores, 3) * len(y_true)))
    assert sum(row["rejected"] for row in rows) == sum(s != "OK" for s in statuses)
    assert confusion.shape == (n, n)
    assert rows[0]["top_misclassified_as"] == "C(1)"


def test_zero_f1_reasons_distinguish_no_samples_from_prediction_failure():
    (_, _, _, _, _, _, _, rows) = _fixture()
    zeros = [row for row in rows if row["f1"] == 0.0]
    assert {row["syndrome"] for row in zeros} == {"C", "E", "F"}
    for name in ("E", "F"):
        row = next(r for r in zeros if r["syndrome"] == name)
        assert row["evidence_status"] == "NO_DEV_SAMPLES"
        assert "dev_support=0" in row["zero_f1_reason"]
        assert "denominator" not in row["zero_f1_reason"] or "分母" in row["zero_f1_reason"]
        assert row["top3_hit_rate"] is None
        assert row["rejection_rate"] is None
    failed = next(r for r in zeros if r["syndrome"] == "C")
    assert failed["dev_support"] == 1 and failed["tp"] == 0 and failed["fn"] == 1
    assert failed["train_support"] == 0
    assert "训练集无该类样本" in failed["zero_f1_reason"]
    assert "曾把该类预测为 Top-1 共 1 次" in failed["zero_f1_reason"]
    assert failed["top3_hit_rate"] == 1.0
    assert failed["evidence_status"] == "LOW_SAMPLE_INSUFFICIENT"
    assert "低样本证据不足" in failed["zero_f1_reason"]


def test_low_sample_is_insufficient_even_with_perfect_f1():
    scores = np.array([[0.9, 0.1]], dtype=float)
    confusion = audit_mod.build_confusion([0], [0], 2)
    rows = audit_mod.class_rows([0], [0], scores, [0], ["X", "Y"], ["OK"], confusion)
    assert rows[0]["f1"] == 1.0
    assert rows[0]["low_sample"] is True
    assert rows[0]["evidence_status"] == "LOW_SAMPLE_INSUFFICIENT"
    assert rows[1]["evidence_status"] == "NO_DEV_SAMPLES"

    n = audit_mod.LOW_SAMPLE_THRESHOLD
    scores = np.tile(np.array([0.9, 0.1]), (n, 1))
    confusion = audit_mod.build_confusion([0] * n, [0] * n, 2)
    rows = audit_mod.class_rows([0] * n, [0] * n, scores, [0] * n, ["X", "Y"],
                                ["OK"] * n, confusion)
    assert rows[0]["f1"] == 1.0 and rows[0]["low_sample"] is False
    assert rows[0]["evidence_status"] == "DEV_EVIDENCE_ONLY"


def test_confusion_pairs_are_sorted_and_exclude_correct_predictions():
    (labels, _, _, _, _, _, confusion, _) = _fixture()
    pairs = audit_mod.confusion_pairs(confusion, labels)
    assert pairs
    assert all(pair["true_index"] != pair["predicted_index"] for pair in pairs)
    assert all(pair["count"] > 0 for pair in pairs)
    assert [pair["count"] for pair in pairs] == sorted(
        (pair["count"] for pair in pairs), reverse=True)
    assert sum(pair["count"] for pair in pairs) == int(confusion.sum() - np.trace(confusion))


def test_verify_checks_conservation_and_rejects_wrong_expected_values():
    (labels, y_true, y_pred, scores, _, train_labels, confusion, rows) = _fixture()
    evaluation = summarize_evaluation(y_true, y_pred, scores, labels,
                                      n_boot=10, seed=1)
    split = {"dev_labels": y_true, "train_labels": train_labels,
             "holdout_subjects": 0}
    report = audit_mod.verify(labels, list(labels), rows, confusion, evaluation,
                              split, evaluation["macro_f1"],
                              evaluation["top3_accuracy"], len(labels))
    assert report["all_checks_passed"] is True
    assert report["sum_dev_support"] == len(y_true) == report["sum_confusion_matrix"]
    with pytest.raises(ValueError, match="verification failed"):
        audit_mod.verify(labels, list(labels), rows, confusion, evaluation, split,
                         expected_macro_f1=0.999,
                         expected_top3=evaluation["top3_accuracy"],
                         expected_n_classes=len(labels))
    with pytest.raises(ValueError, match="verification failed"):
        audit_mod.verify(labels, list(reversed(labels)), rows, confusion, evaluation,
                         split, evaluation["macro_f1"], evaluation["top3_accuracy"],
                         len(labels))
    with pytest.raises(ValueError, match="verification failed"):
        audit_mod.verify(labels, list(labels), rows, confusion, evaluation,
                         {**split, "holdout_subjects": 1},
                         evaluation["macro_f1"], evaluation["top3_accuracy"],
                         len(labels))


def test_render_outputs_are_sortable_and_include_every_class(tmp_path):
    (labels, y_true, y_pred, scores, _, train_labels, confusion, rows) = _fixture()
    pairs = audit_mod.confusion_pairs(confusion, labels)
    evaluation = summarize_evaluation(y_true, y_pred, scores, labels,
                                      n_boot=10, seed=1)
    verification = audit_mod.verify(
        labels, list(labels), rows, confusion, evaluation,
        {"dev_labels": y_true, "train_labels": train_labels, "holdout_subjects": 0},
        evaluation["macro_f1"], evaluation["top3_accuracy"], len(labels))
    acceptance = {**{key: evaluation[key] for key in
                     ("accuracy", "top3_accuracy", "top5_accuracy", "macro_f1",
                      "n_samples", "n_classes_observed")},
                  "macro_f1_ci": evaluation["macro_f1_ci"]}
    report = {
        "schema_version": audit_mod.AUDIT_SCHEMA,
        "status": {"evidence": "LOCAL_EXPERIMENTAL",
                   "trusted_evaluator": "AGENT_ISOLATION_UNVERIFIED",
                   "external_validation": "NONE"},
        "acceptance": acceptance,
        "verification": verification,
        "zero_f1_summary": {"n_zero_f1": 3, "n_no_dev_samples": 2,
                            "n_with_dev_samples": 1, "n_no_train_samples": 3,
                            "n_never_predicted": 0,
                            "n_predicted_but_never_correct": 1,
                            "n_low_sample_zero_f1": 3},
        "evidence_summary": {"n_low_sample": 6, "n_high_f1_but_low_sample": 2,
                             "by_status": {}},
        "confusion_summary": {"top_pairs": pairs[:3]},
        "per_class": rows,
        "unverified_generalization": ["仅内部 dev 证据"],
    }
    csv_path = tmp_path / "ability.csv"
    pairs_path = tmp_path / "pairs.csv"
    md_path = tmp_path / "report.md"
    (tmp_path / "ability.json").write_text(json.dumps(report, ensure_ascii=False))
    audit_mod.render_csv(rows, csv_path)
    audit_mod.render_pairs_csv(pairs, pairs_path)
    audit_mod.render_markdown(report, md_path)
    with csv_path.open(encoding="utf-8", newline="") as stream:
        table = list(csv.DictReader(stream))
    assert len(table) == len(labels)
    assert list(table[0]) == list(audit_mod.CSV_FIELDS)
    with pairs_path.open(encoding="utf-8", newline="") as stream:
        assert len(list(csv.DictReader(stream))) == len(pairs)
    markdown = md_path.read_text(encoding="utf-8")
    assert all(label in markdown for label in labels)
    assert "F1=0" in markdown and "完整 148 类能力表" in markdown
    assert "仅内部 dev 证据" in markdown


def test_build_confusion_supports_the_frozen_148_class_taxonomy():
    y_true = [0, 147, 55, 55]
    y_pred = [1, 147, 55, 0]
    confusion = audit_mod.build_confusion(y_true, y_pred, 148)
    assert confusion.shape == (148, 148)
    assert int(confusion.sum()) == 4
    assert confusion[0, 1] == 1 and confusion[147, 147] == 1
    assert confusion[55, 55] == 1 and confusion[55, 0] == 1
    with pytest.raises(ValueError, match="outside the frozen taxonomy"):
        audit_mod.build_confusion([148], [0], 148)


def test_audit_module_is_read_only_and_pins_task_b_numbers():
    source = (Path(__file__).resolve().parents[2] / "src/syndrome/ability_audit.py").read_text()
    assert ".fit(" not in source
    assert "test.json" not in source
    code = source[source.index("def audit("):]
    assert "results.tsv" not in code
    assert "append_history" not in code
    assert "write_new_atomic" not in code
    assert "assert_holdout_untouched" in source
    assert "predict_proba" in source
    assert audit_mod.EXPECTED_MACRO_F1 == 0.320213
    assert audit_mod.EXPECTED_TOP3 == 0.843179
    assert audit_mod.LOW_SAMPLE_THRESHOLD == 30
