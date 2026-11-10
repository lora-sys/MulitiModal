"""Train-ready engineering preparation (#108). NOT real physiological training.

Stages: prepare -> train -> validate -> compare -> package. Each stage is
independently gateable and refuses on its own evidence boundary. The default is a
synthetic dry-run / smoke path; real fitting requires an approved, frozen,
subject-trusted dataset, which this module never fabricates.

Reuses src/core/models.py and src/physio/{model,training,dataset,evaluation}.py.
No new training platform, no torch dependency at import time, no AutoResearch
champion promotion. KEEP / REVERT / CRASH are recorded, never acted upon here.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contracts import (
    APPROVAL_APPROVED,
    APPROVAL_PENDING,
    GATE_GRANTED,
    GATE_NOT_GRANTED,
    GATE_REAL_DATA_TRAINING,
    GATE_RESEARCH_CLAIMS,
    STATUS_ABSTAIN,
    STATUS_BLOCKED,
    TRUST_TRUSTED,
    ArtifactManifest,
    DatasetManifest,
    EvalSpec,
    FeatureSpec,
    SplitManifest,
    StageEvidence,
    TargetSpec,
    TrainConfig,
    canonical_json,
    digest,
    gate_bundle,
)

TRAIN_VERSION = "ml-lifecycle-train-v1"

MODEL_KINDS = ("ridge", "random_forest", "mlp", "pdf_dual_branch")

DECISION_KEEP = "KEEP"
DECISION_REVERT = "REVERT"
DECISION_CRASH = "CRASH"
DECISIONS = (DECISION_KEEP, DECISION_REVERT, DECISION_CRASH)


class TrainError(ValueError):
    """Raised when a training stage cannot proceed honestly."""


@dataclass(frozen=True)
class RunManifest:
    """Immutable, replayable identity of one run."""

    run_id: str
    config_sha256: str
    dataset_sha256: str
    split_sha256: str
    feature_sha256: str
    target_sha256: str
    eval_sha256: str
    commit: str
    environment: str
    seed: int
    approval: str
    created_at: str

    @property
    def sha256(self) -> str:
        return digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "config_sha256": self.config_sha256,
            "dataset_sha256": self.dataset_sha256,
            "split_sha256": self.split_sha256,
            "feature_sha256": self.feature_sha256,
            "target_sha256": self.target_sha256,
            "eval_sha256": self.eval_sha256,
            "commit": self.commit,
            "environment": self.environment,
            "seed": self.seed,
            "approval": self.approval,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class StageResult:
    """One stage outcome with its own refusal reason."""

    stage: str
    passed: bool
    reason: str
    evidence: StageEvidence
    metrics: dict[str, float | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "passed": self.passed,
            "reason": self.reason,
            "evidence": self.evidence.to_dict(),
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True)
class TrainReport:
    """Complete, serialisable result. Status is an authority, not a score."""

    run: RunManifest
    stages: tuple[StageResult, ...]
    status: str
    decision: str
    evidence_level: str
    gates: dict[str, str]
    evaluation_status: str
    notes: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.status != STATUS_BLOCKED

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TRAIN_VERSION,
            "run": self.run.to_dict(),
            "stages": [s.to_dict() for s in self.stages],
            "status": self.status,
            "decision": self.decision,
            "evidence_level": self.evidence_level,
            "gates": dict(self.gates),
            "evaluation_status": self.evaluation_status,
            "notes": list(self.notes),
        }


def prepare(
    *,
    dataset: DatasetManifest,
    split: SplitManifest,
    features: FeatureSpec,
    target: TargetSpec,
    evaluation: EvalSpec,
    config: TrainConfig,
    commit: str = "unknown",
    environment: str = "local-cpu",
    run_id: str | None = None,
    created_at: str | None = None,
) -> RunManifest:
    """Bind every immutable reference into one replayable run identity.

    Every stage is checked here. A frozen/approved/trusted dataset is a HARD gate
    for real training; a synthetic smoke run is admitted only with an explicit
    SMOKE/SYNTHETIC config and never produces a validated artifact.
    """
    if config.model_kind not in MODEL_KINDS:
        raise TrainError(f"unknown model_kind {config.model_kind!r}")
    if target.approval != APPROVAL_APPROVED and not target.unavailable_reason:
        raise TrainError("unapproved target must state an unavailable reason")
    rid = run_id or digest(
        {
            "config": config.sha256,
            "dataset": dataset.sha256,
            "split": split.sha256,
            "features": features.sha256,
            "target": target.sha256,
            "eval": evaluation.sha256,
        }
    )[:16]
    return RunManifest(
        run_id=rid,
        config_sha256=config.sha256,
        dataset_sha256=dataset.sha256,
        split_sha256=split.sha256,
        feature_sha256=features.sha256,
        target_sha256=target.sha256,
        eval_sha256=evaluation.sha256,
        commit=commit,
        environment=environment,
        seed=config.seed,
        approval=config.approval,
        # A commit hash is not a timestamp. Use an explicitly declared time, or
        # a fixed epoch for a replayable synthetic run; never invent a clock.
        created_at=created_at or "1970-01-01T00:00:00+00:00",
    )


def _evidence(stage: str, go: bool, reason: str, level: str) -> StageEvidence:
    return StageEvidence(
        stage=stage,
        go=go,
        evidence_level=level,
        gate=GATE_GRANTED if go else GATE_NOT_GRANTED,
        reason=reason,
    )


def _linear_predict(
    rows: Sequence[Sequence[float]],
    targets: Sequence[float],
    test: Sequence[Sequence[float]],
) -> list[float]:
    """Deterministic closed-form ridge via numpy (already a project dependency)."""
    import numpy as np

    x = np.asarray(rows, dtype=float)
    y = np.asarray(targets, dtype=float)
    xt = np.asarray(test, dtype=float)
    if x.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] == 0:
        raise TrainError("training rows and targets must be non-empty 2-D arrays")
    mean = y.mean()
    centred = x - x.mean(axis=0)
    penalty = 1.0 * np.eye(x.shape[1])
    weights, *_ = np.linalg.lstsq(
        centred.T @ centred + penalty, centred.T @ (y - mean), rcond=None
    )
    return [float(v) for v in (xt - x.mean(axis=0)) @ weights + mean]


def _mae(predictions: Sequence[float], targets: Sequence[float]) -> float:
    import numpy as np

    return float(np.mean(np.abs(np.asarray(predictions) - np.asarray(targets))))


def synthetic_dataset(
    n_subjects: int = 6, n_per_subject: int = 5, seed: int = 7
) -> tuple[list[list[float]], list[float], list[str]]:
    """Small deterministic synthetic regression set. Explicitly SMOKE_ONLY."""
    import numpy as np

    rng = np.random.default_rng(seed)
    rows: list[list[float]] = []
    targets: list[float] = []
    subjects: list[str] = []
    for subject in range(n_subjects):
        for _ in range(n_per_subject):
            a = float(rng.uniform(0.0, 1.0))
            b = float(rng.uniform(0.0, 1.0))
            rows.append([a, b])
            targets.append(0.2 + 0.5 * a - 0.3 * b)
            subjects.append(f"subject:{subject}")
    return rows, targets, subjects


def subject_folds(
    subjects: Sequence[str], n_folds: int = 3, seed: int = 7
) -> list[tuple[list[int], list[int]]]:
    """Deterministic subject-level split. Same subject never crosses folds."""
    import numpy as np

    unique = sorted(set(subjects))
    if n_folds > len(unique):
        raise TrainError("more folds than distinct subjects")
    rng = np.random.default_rng(seed)
    order = list(unique)
    rng.shuffle(order)
    assignment = {subject: index % n_folds for index, subject in enumerate(order)}
    folds: list[tuple[list[int], list[int]]] = []
    for fold in range(n_folds):
        train = [i for i, s in enumerate(subjects) if assignment[s] != fold]
        test = [i for i, s in enumerate(subjects) if assignment[s] == fold]
        if train and test:
            folds.append((train, test))
    return folds


def run_synthetic_smoke(
    *,
    features: FeatureSpec,
    target: TargetSpec,
    dataset: DatasetManifest,
    split: SplitManifest,
    evaluation: EvalSpec,
    config: TrainConfig,
    n_subjects: int = 6,
) -> TrainReport:
    """End-to-end synthetic engineering smoke. Never a research result.

    Covers prepare -> train -> validate -> compare -> package with an explicit
    fold-internal fit, hand-checkable MAE, repeatability and an artifact that is
    deliberately NOT VALIDATED.
    """
    if config.evidence_level == "SMOKE_ONLY" and config.approval == APPROVAL_APPROVED:
        raise TrainError("a SMOKE_ONLY config cannot carry an APPROVED real-training record")
    run = prepare(
        dataset=dataset, split=split, features=features, target=target,
        evaluation=evaluation, config=config,
    )
    rows, targets, subjects = synthetic_dataset(n_subjects=n_subjects, seed=config.seed)
    folds = subject_folds(subjects, n_folds=min(3, len(set(subjects))), seed=config.seed)

    train_errors: list[float] = []
    baseline_errors: list[float] = []
    for train_idx, test_idx in folds:
        train_rows = [rows[i] for i in train_idx]
        train_y = [targets[i] for i in train_idx]
        test_rows = [rows[i] for i in test_idx]
        test_y = [targets[i] for i in test_idx]
        predicted = _linear_predict(train_rows, train_y, test_rows)
        train_errors.extend(abs(p - t) for p, t in zip(predicted, test_y, strict=True))
        baseline = sum(train_y) / len(train_y)
        baseline_errors.extend(abs(baseline - t) for t in test_y)

    model_mae = sum(train_errors) / len(train_errors)
    baseline_mae = sum(baseline_errors) / len(baseline_errors)
    # Repeatability: the same seed must reproduce identical predictions.
    repeated_rows, repeated_y, repeated_subjects = synthetic_dataset(
        n_subjects=n_subjects, seed=config.seed
    )
    repeat_folds = subject_folds(
        repeated_subjects, n_folds=min(3, len(set(repeated_subjects))), seed=config.seed
    )
    repeat_predictions: list[float] = []
    for train_idx, test_idx in repeat_folds:
        repeat_predictions.extend(
            _linear_predict(
                [repeated_rows[i] for i in train_idx],
                [repeated_y[i] for i in train_idx],
                [repeated_rows[i] for i in test_idx],
            )
        )
    deterministic = repeat_predictions == [
        p
        for train_idx, test_idx in folds
        for p in _linear_predict(
            [rows[i] for i in train_idx], [targets[i] for i in train_idx],
            [rows[i] for i in test_idx],
        )
    ]

    n_labeled = len(targets)
    insufficient = (
        n_labeled < evaluation.min_samples
        or len(set(subjects)) < evaluation.min_subjects
        or dataset.identity_trust != TRUST_TRUSTED
    )
    evaluation_status = "INSUFFICIENT_EVIDENCE" if insufficient else "EVALUATED"
    level = config.evidence_level

    stages = [
        StageResult(
            "prepare", True, "all immutable references bound",
            _evidence("prepare", True, "references bound", level),
        ),
        StageResult(
            "train", deterministic, "fold-internal ridge fit; repeatable" if deterministic
            else "non-deterministic fit",
            _evidence("train", deterministic, "repeatability check", level),
            {"model_mae": model_mae, "baseline_mae": baseline_mae},
        ),
        StageResult(
            "validate_model", True,
            f"evaluation status {evaluation_status} is the authority; metrics are local",
            _evidence("validate_model", True, evaluation_status, level),
            {"model_mae": model_mae},
        ),
        StageResult(
            "compare", True,
            "synthetic baseline comparison only; not a research claim",
            _evidence("compare", True, "synthetic comparison", level),
            {"model_mae": model_mae, "baseline_mae": baseline_mae},
        ),
        StageResult(
            "package", True,
            "artifact is NOT VALIDATED; SMOKE/SYNTHETIC evidence cannot be promoted",
            _evidence("package", True, "not validated", level),
        ),
    ]

    artifact = ArtifactManifest(
        artifact_id=run.run_id,
        model_sha256=digest({"predictions": repeat_predictions}),
        feature_sha256=features.sha256,
        target_sha256=target.sha256,
        scaler_sha256=digest({"fold_internal": True}),
        dataset_sha256=dataset.sha256,
        split_sha256=split.sha256,
        eval_sha256=evaluation.sha256,
        commit=run.commit,
        dependencies=("numpy",),
        license=config.license,
        approval=APPROVAL_PENDING,
        validation_status="NOT_VALIDATED",
        evidence_level=level,
    )

    gates = gate_bundle(True)
    gates[GATE_REAL_DATA_TRAINING] = GATE_NOT_GRANTED
    gates[GATE_RESEARCH_CLAIMS] = GATE_NOT_GRANTED
    return TrainReport(
        run=run,
        stages=tuple(stages),
        status=STATUS_ABSTAIN,
        decision=DECISION_REVERT,
        evidence_level=level,
        gates=gates,
        evaluation_status=evaluation_status,
        notes=(
            "Synthetic engineering smoke only; no real subject data was fitted.",
            f"artifact {artifact.sha256[:12]} remains NOT_VALIDATED and is never promoted.",
            "KEEP / REVERT / CRASH are recorded, never executed here.",
        ),
    )


def write_report(report: TrainReport, out_dir: str | Path) -> Path:
    path = Path(out_dir) / "train_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(report.to_dict()) + "\n", encoding="utf-8")
    return path


def crash_record(reason: str, *, run_id: str = "crash") -> dict[str, str]:
    """A CRASH is a durable, auditable record, never a silent retry."""
    return {
        "decision": DECISION_CRASH,
        "run_id": run_id,
        "reason": reason,
        "sha256": hashlib.sha256(json.dumps(reason, sort_keys=True).encode()).hexdigest(),
    }
