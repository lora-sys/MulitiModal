"""Evaluation interfaces that refuse to overstate evidence (#95).

The functions compute standard metrics, but the report status is the authority:
without enough labeled samples and independently verified subjects the result is
``INSUFFICIENT_EVIDENCE``. A local dev/CV number is never upgraded to a
generalization or clinical claim here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

EVALUATION_SCHEMA_VERSION = "physio-evaluation-v1"

STATUS_EVALUATED = "EVALUATED"
STATUS_INSUFFICIENT = "INSUFFICIENT_EVIDENCE"


class EvaluationError(ValueError):
    """Raised when inputs cannot produce an honest metric."""


@dataclass(frozen=True)
class EvaluationReport:
    status: str
    n_samples: int
    n_labeled: int
    n_subjects: int
    min_samples: int
    min_subjects: int
    metrics: dict[str, float | None]
    confusion_matrix: tuple[tuple[int, ...], ...] | None = None
    reason: str = ""
    schema_version: str = EVALUATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EVALUATION_SCHEMA_VERSION:
            raise ValueError(f"unsupported evaluation schema: {self.schema_version!r}")
        if self.status not in (STATUS_EVALUATED, STATUS_INSUFFICIENT):
            raise ValueError(f"unknown evaluation status: {self.status!r}")
        if self.n_labeled > self.n_samples:
            raise ValueError("n_labeled cannot exceed n_samples")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "n_samples": self.n_samples,
            "n_labeled": self.n_labeled,
            "n_subjects": self.n_subjects,
            "min_samples": self.min_samples,
            "min_subjects": self.min_subjects,
            "metrics": dict(self.metrics),
            "confusion_matrix": (
                [list(row) for row in self.confusion_matrix]
                if self.confusion_matrix is not None
                else None
            ),
            "reason": self.reason,
        }


def _validate(
    predictions: Sequence[float],
    targets: Sequence[float | None],
    mask: Sequence[bool],
    subjects: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    if len(predictions) != len(targets) or len(predictions) != len(mask):
        raise EvaluationError("predictions, targets and mask must have equal length")
    if len(subjects) != len(predictions):
        raise EvaluationError("subjects must match the sample count")
    prediction_array = np.asarray(predictions, dtype=float)
    mask_array = np.asarray(mask, dtype=bool)
    if not np.all(np.isfinite(prediction_array)):
        raise EvaluationError("predictions contain NaN/Inf")
    observed_targets: list[float] = []
    for target in targets:
        if target is None:
            observed_targets.append(float("nan"))
        else:
            value = float(target)
            if not math.isfinite(value):
                raise EvaluationError("targets contain NaN/Inf")
            observed_targets.append(value)
    target_array = np.asarray(observed_targets, dtype=float)
    if np.any(mask_array & np.isnan(target_array)):
        raise EvaluationError("a labeled position has no target value")
    return prediction_array, target_array, mask_array, tuple(subjects)


def evaluate_regression(
    predictions: Sequence[float],
    targets: Sequence[float | None],
    mask: Sequence[bool],
    subjects: Sequence[str],
    *,
    min_samples: int = 30,
    min_subjects: int = 2,
) -> EvaluationReport:
    """Regression report; a small/local sample set stays insufficient."""
    prediction_array, target_array, mask_array, subject_tuple = _validate(
        predictions, targets, mask, subjects
    )
    selected = mask_array
    n_labeled = int(selected.sum())
    unique_subjects = tuple(
        sorted(
            {
                subject
                for subject, keep in zip(subject_tuple, selected, strict=True)
                if keep
            }
        )
    )
    n_subjects = len(unique_subjects)
    reason = ""
    status = STATUS_EVALUATED
    if n_labeled == 0:
        status = STATUS_INSUFFICIENT
        reason = "no labeled samples"
    elif n_labeled < min_samples:
        status = STATUS_INSUFFICIENT
        reason = f"labeled samples {n_labeled} < required {min_samples}"
    elif n_subjects < min_subjects:
        status = STATUS_INSUFFICIENT
        reason = f"subjects {n_subjects} < required {min_subjects}"

    metrics: dict[str, float | None] = {
        "mae": None,
        "rmse": None,
        "coverage": n_labeled / len(predictions) if predictions else 0.0,
        "in_sample_mean_baseline_mae": None,
    }
    if n_labeled:
        errors = prediction_array[selected] - target_array[selected]
        metrics["mae"] = float(np.mean(np.abs(errors)))
        metrics["rmse"] = float(np.sqrt(np.mean(errors**2)))
        mean_target = float(np.mean(target_array[selected]))
        metrics["in_sample_mean_baseline_mae"] = float(
            np.mean(np.abs(mean_target - target_array[selected]))
        )
    return EvaluationReport(
        status=status,
        n_samples=len(predictions),
        n_labeled=n_labeled,
        n_subjects=n_subjects,
        min_samples=min_samples,
        min_subjects=min_subjects,
        metrics=metrics,
        reason=reason,
    )


def evaluate_classification(
    predictions: Sequence[str],
    targets: Sequence[str | None],
    mask: Sequence[bool],
    subjects: Sequence[str],
    *,
    min_samples: int = 30,
    min_subjects: int = 2,
) -> EvaluationReport:
    """Classification report with an explicit confusion matrix and the same gate."""
    if not (len(predictions) == len(targets) == len(mask) == len(subjects)):
        raise EvaluationError("predictions, targets, mask and subjects must have equal length")
    labels = tuple(sorted({str(value) for value in targets if value is not None}))
    index = {label: position for position, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    n_labeled = 0
    subject_set: set[str] = set()
    for prediction, target, keep, subject in zip(
        predictions, targets, mask, subjects, strict=True
    ):
        if not keep:
            continue
        if target is None:
            raise EvaluationError("a labeled position has no target class")
        if str(target) not in index:
            raise EvaluationError(f"target class {target!r} is not in the label set")
        if str(prediction) not in index:
            raise EvaluationError(f"predicted class {prediction!r} is not in the label set")
        matrix[index[str(target)]][index[str(prediction)]] += 1
        n_labeled += 1
        subject_set.add(subject)
    status = STATUS_EVALUATED
    reason = ""
    if n_labeled == 0:
        status, reason = STATUS_INSUFFICIENT, "no labeled samples"
    elif n_labeled < min_samples:
        status = STATUS_INSUFFICIENT
        reason = f"labeled samples {n_labeled} < required {min_samples}"
    elif len(subject_set) < min_subjects:
        status = STATUS_INSUFFICIENT
        reason = f"subjects {len(subject_set)} < required {min_subjects}"

    correct = sum(matrix[i][i] for i in range(len(labels)))
    metrics: dict[str, float | None] = {
        "accuracy": (correct / n_labeled) if n_labeled else None,
        "coverage": (n_labeled / len(predictions)) if predictions else 0.0,
    }
    return EvaluationReport(
        status=status,
        n_samples=len(predictions),
        n_labeled=n_labeled,
        n_subjects=len(subject_set),
        min_samples=min_samples,
        min_subjects=min_subjects,
        metrics=metrics,
        confusion_matrix=tuple(tuple(row) for row in matrix),
        reason=reason,
    )


__all__ = [
    "EVALUATION_SCHEMA_VERSION",
    "STATUS_EVALUATED",
    "STATUS_INSUFFICIENT",
    "EvaluationError",
    "EvaluationReport",
    "evaluate_classification",
    "evaluate_regression",
]
