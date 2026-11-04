"""metrics 契约：macro-F1 口径、逐类结果、Top-k、bootstrap 区间与配对差值。"""

from __future__ import annotations

import numpy as np
import pytest

from syndrome.metrics import (
    accuracy,
    bootstrap_ci,
    macro_f1,
    majority_class_predictions,
    paired_bootstrap_diff,
    per_class_metrics,
    summarize_evaluation,
    topk_accuracy,
)

pytestmark = pytest.mark.synthetic

LABELS = ("甲", "乙", "丙")


def test_macro_f1_hand_computed():
    # 甲: tp=2 fp=1 fn=0 -> P=2/3 R=1 F1=0.8
    # 乙: tp=0 fp=1 fn=1 -> F1=0 (zero_division=0)
    # 丙: tp=0 fp=0 fn=1 -> F1=0
    y_true = [0, 1, 2, 0]
    y_pred = [0, 0, 1, 0]
    assert macro_f1(y_true, y_pred, n_classes=3) == pytest.approx(0.8 / 3)


def test_macro_f1_perfect_and_degenerate():
    assert macro_f1([0, 1], [0, 1], n_classes=2) == 1.0
    # 全预测同类：缺席类 F1=0 如实计入（不隐藏）
    # 甲类 P=.5 R=1 F1=2/3；乙类 F1=0
    assert macro_f1([0, 1], [0, 0], n_classes=2) == pytest.approx((2 / 3 + 0.0) / 2)
    # 3 类词表只出现 2 类：macro 分母仍是词表大小（冻结口径）
    assert macro_f1([0, 1], [0, 1], n_classes=3) == pytest.approx((1 + 1 + 0) / 3)


def test_per_class_support_and_zero_division():
    report = per_class_metrics([0, 0, 1], [0, 1, 1], LABELS)
    by_label = {r["label"]: r for r in report}
    assert by_label["甲"]["support"] == 2
    assert by_label["丙"]["support"] == 0
    assert by_label["丙"]["f1"] == 0.0


def test_accuracy_and_topk():
    assert accuracy([0, 1], [0, 1]) == 1.0
    scores = np.array([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]])
    assert topk_accuracy([1, 0], scores, k=1) == 1.0
    assert topk_accuracy([1, 1], scores, k=1) == 0.5
    assert topk_accuracy([1, 1], scores, k=3) == 1.0  # k 被裁剪到类数


def test_bootstrap_ci_contains_point_and_is_seeded():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, size=200).tolist()
    y_pred = [y if rng.random() > 0.3 else (y + 1) % 3 for y in y_true]
    a = bootstrap_ci(y_true, y_pred, n_classes=3, n_boot=200, seed=42)
    b = bootstrap_ci(y_true, y_pred, n_classes=3, n_boot=200, seed=42)
    assert a == b  # 固定 seed 可复现
    assert a["low"] <= a["point"] <= a["high"]
    assert a["n_boot"] == 200 and a["seed"] == 42


def test_paired_bootstrap_diff_detects_better_model():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 3, size=300).tolist()
    good = [y if rng.random() > 0.2 else (y + 1) % 3 for y in y_true]
    dummy = [0] * len(y_true)
    diff = paired_bootstrap_diff(y_true, good, dummy, n_classes=3, n_boot=200, seed=1)
    assert diff["point_diff"] > 0.2
    assert diff["ci_low"] > 0  # 明显优于 dummy 时 CI 不跨 0
    assert diff["p_value"] <= 0.05


def test_majority_dummy_predictions():
    assert majority_class_predictions([0, 0, 1], n_eval=4, n_classes=2) == [0, 0, 0, 0]


def test_summarize_evaluation_shape():
    rng = np.random.default_rng(2)
    y_true = rng.integers(0, 3, size=100).tolist()
    y_pred = [y if rng.random() > 0.25 else (y + 1) % 3 for y in y_true]
    scores = np.full((100, 3), 0.3)
    for i, p in enumerate(y_pred):
        scores[i, p] = 0.4
    report = summarize_evaluation(
        y_true, y_pred, scores, LABELS, baseline_pred=[0] * 100, n_boot=50
    )
    for key in (
        "macro_f1",
        "top3_accuracy",
        "top5_accuracy",
        "macro_f1_ci",
        "per_class",
        "vs_baseline",
    ):
        assert key in report
    assert report["n_classes_observed"] <= report["n_classes_total"] == 3
    assert len(report["per_class"]) == 3


def test_empty_input_rejected():
    with pytest.raises(ValueError):
        macro_f1([], [], n_classes=3)
    with pytest.raises(ValueError):
        macro_f1([0], [0, 1], n_classes=3)
