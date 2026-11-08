"""#95 evaluation gate: standard metrics, explicit evidence status."""

from __future__ import annotations

import pytest

from physio.evaluation import (
    STATUS_EVALUATED,
    STATUS_INSUFFICIENT,
    EvaluationError,
    evaluate_classification,
    evaluate_regression,
)

pytestmark = pytest.mark.synthetic


def test_small_sample_regression_stays_insufficient():
    report = evaluate_regression(
        [1.0, 2.0, 3.0], [1.0, 2.5, None], [True, True, False], ["s1", "s2", "s1"],
        min_samples=30,
    )
    assert report.status == STATUS_INSUFFICIENT
    assert report.n_labeled == 2
    assert report.metrics["mae"] == pytest.approx(0.25)
    assert "required 30" in report.reason


def test_enough_samples_and_subjects_are_evaluated():
    predictions = [float(i) for i in range(40)]
    targets = [float(i) + 0.5 for i in range(40)]
    subjects = ["s1" if i % 2 else "s2" for i in range(40)]
    report = evaluate_regression(
        predictions, targets, [True] * 40, subjects, min_samples=30, min_subjects=2
    )
    assert report.status == STATUS_EVALUATED
    assert report.metrics["mae"] == pytest.approx(0.5)
    assert report.metrics["coverage"] == 1.0
    # The mean baseline here is far worse, so the metric is not vacuous.
    assert report.metrics["in_sample_mean_baseline_mae"] > report.metrics["mae"]


def test_single_subject_is_insufficient_even_with_many_rows():
    report = evaluate_regression(
        [1.0] * 40, [1.0] * 40, [True] * 40, ["only-subject"] * 40,
        min_samples=30, min_subjects=2,
    )
    assert report.status == STATUS_INSUFFICIENT
    assert "subjects 1" in report.reason


def test_no_labeled_sample_has_no_fabricated_metric():
    report = evaluate_regression(
        [1.0, 2.0], [None, None], [False, False], ["s1", "s2"], min_samples=1
    )
    assert report.status == STATUS_INSUFFICIENT
    assert report.n_labeled == 0
    assert report.metrics["mae"] is None
    assert report.reason == "no labeled samples"


def test_non_finite_prediction_is_rejected():
    with pytest.raises(EvaluationError):
        evaluate_regression([float("nan")], [1.0], [True], ["s1"])
    with pytest.raises(EvaluationError):
        evaluate_regression([1.0], [float("inf")], [True], ["s1"])


def test_mask_and_target_must_agree():
    with pytest.raises(EvaluationError):
        evaluate_regression([1.0], [None], [True], ["s1"])


def test_classification_reports_confusion_matrix_and_gate():
    report = evaluate_classification(
        ["a", "b", "a"], ["a", "b", "b"], [True, True, True],
        ["s1", "s1", "s2"], min_samples=30,
    )
    assert report.status == STATUS_INSUFFICIENT
    assert report.confusion_matrix == ((1, 0), (1, 1))
    assert report.metrics["accuracy"] == pytest.approx(2 / 3)


def test_classification_rejects_unknown_predicted_class():
    with pytest.raises(EvaluationError):
        evaluate_classification(["z"], ["a"], [True], ["s1"])
