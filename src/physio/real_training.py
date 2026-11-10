"""Real-data pipeline pre-experiment (#102).

Five de-identified sessions cannot support any performance or clinical claim.
This module therefore answers one engineering question: does the whole path —
subject-level folds, fold-internal imputation, fold-internal scaling, masked-loss
training, checkpoint reload and prediction — run end to end on real rows?

Three things must not be read into any number produced here:

1. **It is a pipeline regression, not a model comparison.** One test row per fold
   cannot rank three baselines, and the module publishes per-model errors only as
   evidence that each path ran.
2. **Both declared targets are vendor proxies of the same family as a Pre
   feature.** ``post_state`` is the Post-endpoint vendor fatigue index, and the
   Pre features already contain the Pre-endpoint vendor fatigue index;
   ``delta_vendor_fatigue`` is the difference of two vendor scores. The
   regression is therefore near-autoregressive on a vendor score and says nothing
   about predicting fatigue from independent physiology.
3. **A zero in the design matrix is a masked placeholder**, never a measured or
   imputed value. ``masked_placeholder_features`` names the columns where that
   still happens after fold imputation.

``evaluate_regression`` is the authority on status and returns
``INSUFFICIENT_EVIDENCE`` for this sample size.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

from .dataset import (
    IDENTITY_SHARED,
    DatasetError,
    FeatureRow,
    FoldImputer,
    IdentityTrustError,
    make_subject_folds,
)
from .evaluation import STATUS_EVALUATED, STATUS_INSUFFICIENT, evaluate_regression
from .preprocessing import PreprocessingError, WindowScaler
from .real_export import RealSession, session_feature_row, session_labels
from .training import (
    TrainingConfig,
    TrainingRefused,
    TrainingUnavailable,
    assert_trainable,
    predict_masked,
    train_realtime_smoke,
)

REAL_TRAINING_SCHEMA_VERSION = "physio-real-training-v1"

#: Declared smoke configuration for the real-data pipeline regression. It is the
#: default ``TrainingConfig``; nothing is relaxed to make the path look better.
CONFIG = TrainingConfig()

TARGET_POST_STATE = "post_state"
TARGET_DELTA_FATIGUE = "delta_vendor_fatigue"
TARGET_MODES: tuple[str, ...] = (TARGET_POST_STATE, TARGET_DELTA_FATIGUE)

MODEL_MEAN = "train_fold_mean"
MODEL_RIDGE = "ridge"
MODEL_NEURAL = "realtime_mlp"
MODELS: tuple[str, ...] = (MODEL_MEAN, MODEL_RIDGE, MODEL_NEURAL)

RIDGE_LAMBDA = 1.0
FOLD_SEED = 20261108

NEURAL_UNAVAILABLE = "torch_unavailable"
NEURAL_COMPLETED = "completed"
NEURAL_REFUSED = "refused"


class RealTrainingError(ValueError):
    """Raised when the real pre-experiment cannot be set up honestly."""


@dataclass(frozen=True)
class FoldResult:
    """One leave-one-subject-out fold with every model's error."""

    fold: int
    train_subjects: tuple[str, ...]
    test_subjects: tuple[str, ...]
    n_train: int
    n_test: int
    predictions: dict[str, tuple[float, ...]]
    targets: tuple[float, ...]
    mae: dict[str, float | None]
    final_loss: float | None = None
    neural_status: str = NEURAL_UNAVAILABLE
    refusal: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "fold": self.fold,
            "train_subjects": list(self.train_subjects),
            "test_subjects": list(self.test_subjects),
            "n_train": self.n_train,
            "n_test": self.n_test,
            "targets": list(self.targets),
            "predictions": {name: list(values) for name, values in self.predictions.items()},
            "mae": dict(self.mae),
            "final_loss": self.final_loss,
            "neural_status": self.neural_status,
            "refusal": self.refusal,
        }


@dataclass(frozen=True)
class RealTrainingReport:
    """The complete pre-experiment result. Status is never upgraded here."""

    schema: str
    status: str
    reason: str
    target: str
    feature_names: tuple[str, ...]
    n_sessions: int
    n_subjects: int
    n_members: int
    identity_trust: str
    folds: tuple[FoldResult, ...]
    model_mae: dict[str, float | None]
    neural_status: str
    imputed_features: tuple[str, ...]
    feature_missing_counts: dict[str, int]
    masked_placeholder_features: tuple[str, ...]
    refused_folds: tuple[int, ...]
    evaluation_status: str
    evaluation_metrics: dict[str, float | None] = dataclass_field(default_factory=dict)
    notes: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.status != "FAILED"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "status": self.status,
            "reason": self.reason,
            "target": self.target,
            "feature_names": list(self.feature_names),
            "n_sessions": self.n_sessions,
            "n_subjects": self.n_subjects,
            "n_members": self.n_members,
            "identity_trust": self.identity_trust,
            "model_mae": dict(self.model_mae),
            "neural_status": self.neural_status,
            "imputed_features": list(self.imputed_features),
            "feature_missing_counts": dict(self.feature_missing_counts),
            "masked_placeholder_features": list(self.masked_placeholder_features),
            "refused_folds": list(self.refused_folds),
            "evaluation_status": self.evaluation_status,
            "evaluation_metrics": dict(self.evaluation_metrics),
            "folds": [fold.to_dict() for fold in self.folds],
            "notes": list(self.notes),
        }


def build_targets(
    sessions: Sequence[RealSession], target: str = TARGET_POST_STATE
) -> tuple[float | None, ...]:
    """Targets for the declared mode. Missing labels stay ``None``, never zero."""
    if target == TARGET_POST_STATE:
        from .real_export import session_post_state

        return tuple(session_post_state(session) for session in sessions)
    if target == TARGET_DELTA_FATIGUE:
        return tuple(
            session_labels(session).values.get("y_delta_vendor_fatigue") for session in sessions
        )
    raise RealTrainingError(f"unknown target mode {target!r}")


@dataclass(frozen=True)
class _ScalerInput:
    """Adapter that lets the shared ``WindowScaler.fit`` consume a ``FeatureRow``.

    ``WindowScaler.fit`` reads ``feature_names``, ``values`` and ``mask``. Reusing
    it instead of reimplementing the statistics keeps one zero-variance and mask
    policy in the codebase; a local copy is what previously let the companion
    transform be forgotten.
    """

    feature_names: tuple[str, ...]
    values: tuple[float, ...]
    mask: tuple[bool, ...]


def _fit_scaler(rows: Sequence[FeatureRow]) -> WindowScaler:
    """Fold-internal standardizer over **observed** entries only.

    Delegates to ``WindowScaler.fit``, which skips unobserved entries, so a
    training-fold median cannot silently shape the scaling of a held-out subject.
    """
    if not rows:
        raise RealTrainingError("cannot fit a scaler without training rows")
    try:
        return WindowScaler.fit(
            [
                _ScalerInput(row.feature_names, row.values, row.observed_mask)
                for row in rows
            ]
        )
    except PreprocessingError as exc:
        raise RealTrainingError(f"fold scaler refused: {exc}") from exc


def _scaled_row(row: FeatureRow, scaler: WindowScaler) -> FeatureRow:
    """Standardize one row with the fold-internal scaler.

    Observed and fold-imputed positions are normalized with the training-fold
    mean and standard deviation. A position that is neither observed nor imputed
    stays ``0.0`` with ``valid_mask`` false: that zero is a masked placeholder,
    never a measurement, and ``masked_placeholder_features`` names the columns
    where it still occurs.

    A position that *is* valid but was never observed while the scaler was fitted
    is refused, exactly as ``WindowScaler.transform`` refuses it: normalizing it
    would publish a raw, unstandardized value that looks standardized. Callers
    record the refusal instead of silently degrading the fold.
    """
    if row.feature_names != scaler.feature_names:
        raise RealTrainingError("row does not share the scaler feature contract")
    values: list[float] = []
    for index, (value, valid) in enumerate(zip(row.values, row.valid_mask, strict=True)):
        if not valid:
            values.append(0.0)
            continue
        if scaler.observed_counts[index] == 0:
            raise RealTrainingError(
                f"feature {scaler.feature_names[index]!r} is valid in this row but was never "
                "observed while fitting the fold scaler; refusing to fabricate a normalization"
            )
        values.append((float(value) - scaler.mean[index]) / scaler.std[index])
    return FeatureRow(
        feature_names=row.feature_names,
        values=tuple(values),
        observed_mask=row.observed_mask,
        valid_mask=row.valid_mask,
        imputed_mask=row.imputed_mask,
        imputation=row.imputation,
        subject_ref=row.subject_ref,
        session_id=row.session_id,
        cutoff=row.cutoff,
    )


def _dense(rows: Sequence[FeatureRow]) -> np.ndarray:
    return np.asarray(
        [
            [
                float(value) if valid else 0.0
                for value, valid in zip(row.values, row.valid_mask, strict=True)
            ]
            for row in rows
        ],
        dtype=float,
    )


def _ridge_predict(
    train_rows: Sequence[FeatureRow],
    train_targets: Sequence[float],
    test_rows: Sequence[FeatureRow],
) -> tuple[float, ...]:
    """Closed-form ridge baseline on the fold-internal scaled matrix.

    ``sklearn`` is already an ``src/``-level dependency (``src/research/train.py``,
    ``src/syndrome/baselines.py``), so the baseline uses it instead of hand-rolled
    linear algebra. The model is fitted on training rows only.
    """
    model = Ridge(alpha=RIDGE_LAMBDA)
    model.fit(_dense(train_rows), np.asarray(train_targets, dtype=float))
    return tuple(float(value) for value in model.predict(_dense(test_rows)))


def _mae(predictions: Sequence[float], targets: Sequence[float]) -> float:
    if not targets:
        return float("nan")
    return float(np.mean(np.abs(np.asarray(predictions) - np.asarray(targets))))


def run_real_pre_experiment(
    sessions: Sequence[RealSession],
    *,
    target: str = TARGET_POST_STATE,
    n_folds: int | None = None,
    out_dir: str | Path | None = None,
    notes: Sequence[str] = (),
) -> RealTrainingReport:
    """Run the declared leave-one-subject-out pipeline regression on real rows."""
    if not sessions:
        raise RealTrainingError("pre-experiment requires at least one session")
    if target not in TARGET_MODES:
        raise RealTrainingError(f"unknown target mode {target!r}")

    rows = [session_feature_row(session) for session in sessions]
    targets = list(build_targets(sessions, target))
    feature_names = rows[0].feature_names
    for row in rows:
        if row.feature_names != feature_names:
            raise RealTrainingError("sessions do not share one feature contract")

    subjects = [session.subject_ref for session in sessions]
    members = sorted({session.member_ref for session in sessions})
    labeled = [index for index, value in enumerate(targets) if value is not None]
    if not labeled:
        raise RealTrainingError("no labelled session; the pipeline cannot start")

    distinct_subjects = sorted(set(subjects))
    requested = n_folds or len(distinct_subjects)
    try:
        folds = make_subject_folds(
            subjects,
            n_folds=requested,
            seed=FOLD_SEED,
            identity_trust=IDENTITY_SHARED,
            allow_untrusted=True,
        )
    except (DatasetError, IdentityTrustError) as exc:
        raise RealTrainingError(f"subject-level folds refused: {exc}") from exc

    fold_results: list[FoldResult] = []
    masked: set[str] = set()
    for fold_index in range(folds.n_folds):
        train_idx = [
            index
            for index in range(len(rows))
            if folds.folds[subjects[index]] != fold_index and targets[index] is not None
        ]
        test_idx = [
            index
            for index in range(len(rows))
            if folds.folds[subjects[index]] == fold_index and targets[index] is not None
        ]
        if not train_idx or not test_idx:
            continue
        try:
            fold_result, fold_masked = _run_fold(
                fold_index=fold_index,
                rows=rows,
                targets=targets,
                subjects=subjects,
                train_idx=train_idx,
                test_idx=test_idx,
                out_dir=out_dir,
            )
        except RealTrainingError as exc:
            # A refused fold is recorded, never silently dropped or degraded.
            fold_results.append(
                FoldResult(
                    fold=fold_index,
                    train_subjects=tuple(sorted({subjects[i] for i in train_idx})),
                    test_subjects=tuple(sorted({subjects[i] for i in test_idx})),
                    n_train=len(train_idx),
                    n_test=len(test_idx),
                    predictions={name: () for name in MODELS},
                    targets=(),
                    mae={name: None for name in MODELS},
                    refusal=str(exc),
                )
            )
            continue
        masked |= fold_masked
        fold_results.append(fold_result)

    # Pool only folds that produced a target, so predictions, targets and subjects
    # stay in lockstep and a refused fold never desynchronises the arrays.
    pooled_predictions: dict[str, list[float]] = {name: [] for name in MODELS}
    pooled_targets: list[float] = []
    pooled_subjects: list[str] = []
    for fold in fold_results:
        if not fold.targets:
            continue
        for name in MODELS:
            pooled_predictions[name].extend(fold.predictions.get(name, ()))
        pooled_targets.extend(fold.targets)
        pooled_subjects.extend(fold.test_subjects)
    model_mae: dict[str, float | None] = {}
    for name in MODELS:
        values = pooled_predictions[name]
        if len(values) != len(pooled_targets) or not pooled_targets:
            model_mae[name] = None
            continue
        model_mae[name] = _mae(values, pooled_targets)

    report = evaluate_regression(
        pooled_predictions[MODEL_MEAN],
        pooled_targets,
        [True] * len(pooled_targets),
        pooled_subjects,
    )
    missing_by_feature: dict[str, int] = {}
    for row in rows:
        for name, valid in zip(row.feature_names, row.valid_mask, strict=True):
            if not valid:
                missing_by_feature[name] = missing_by_feature.get(name, 0) + 1
    imputed_features = tuple(sorted(missing_by_feature))
    masked_placeholder_features = tuple(sorted(masked))

    return RealTrainingReport(
        schema=REAL_TRAINING_SCHEMA_VERSION,
        status=STATUS_INSUFFICIENT if report.status == STATUS_INSUFFICIENT else STATUS_EVALUATED,
        reason=report.reason,
        target=target,
        feature_names=feature_names,
        n_sessions=len(sessions),
        n_subjects=len(distinct_subjects),
        n_members=len(members),
        identity_trust=IDENTITY_SHARED,
        folds=tuple(fold_results),
        model_mae=model_mae,
        neural_status=_neural_status(fold_results),
        imputed_features=imputed_features,
        feature_missing_counts=dict(sorted(missing_by_feature.items())),
        masked_placeholder_features=masked_placeholder_features,
        refused_folds=tuple(fold.fold for fold in fold_results if fold.refusal),
        evaluation_status=report.status,
        evaluation_metrics=report.metrics,
        notes=tuple(notes),
    )


def _run_fold(
    *,
    fold_index: int,
    rows: Sequence[FeatureRow],
    targets: Sequence[float | None],
    subjects: Sequence[str],
    train_idx: Sequence[int],
    test_idx: Sequence[int],
    out_dir: Path | None,
) -> tuple[FoldResult, set[str]]:
    """Run one leave-one-subject-out fold and report its masked placeholder columns."""
    imputer = FoldImputer.fit([rows[index] for index in train_idx])
    imputed = [imputer.transform(row) for row in rows]
    scaler = _fit_scaler([imputed[index] for index in train_idx])
    scaled = [_scaled_row(imputed[index], scaler) for index in range(len(rows))]
    masked = {
        name
        for row in imputed
        for name, valid in zip(row.feature_names, row.valid_mask, strict=True)
        if not valid
    }

    train_rows = [scaled[index] for index in train_idx]
    test_rows = [scaled[index] for index in test_idx]
    train_targets = [float(targets[index]) for index in train_idx]
    test_targets = [float(targets[index]) for index in test_idx]

    predictions: dict[str, tuple[float, ...]] = {
        MODEL_MEAN: tuple(float(np.mean(train_targets)) for _ in test_targets),
        MODEL_RIDGE: _ridge_predict(train_rows, train_targets, test_rows),
    }
    mae: dict[str, float | None] = {
        MODEL_MEAN: _mae(predictions[MODEL_MEAN], test_targets),
        MODEL_RIDGE: _mae(predictions[MODEL_RIDGE], test_targets),
    }
    neural_status = NEURAL_UNAVAILABLE
    final_loss: float | None = None
    refusal = ""
    try:
        assert_trainable(
            train_rows,
            train_targets,
            config=CONFIG,
            identity_trust=IDENTITY_SHARED,
            allow_untrusted_pipeline=True,
        )
    except TrainingRefused as exc:
        neural_status = NEURAL_REFUSED
        refusal = str(exc)
    else:
        fold_dir = _fold_checkpoint_dir(out_dir, fold_index)
        try:
            manifest = train_realtime_smoke(
                train_rows,
                train_targets,
                scaler=scaler,
                config=CONFIG,
                identity_trust=IDENTITY_SHARED,
                allow_untrusted_pipeline=True,
                out_dir=fold_dir,
            )
        except TrainingUnavailable:
            neural_status = NEURAL_UNAVAILABLE
        except TrainingRefused as exc:
            neural_status = NEURAL_REFUSED
            refusal = str(exc)
        else:
            neural_status = NEURAL_COMPLETED
            final_loss = manifest.final_loss
            predictions[MODEL_NEURAL] = predict_masked(
                test_rows,
                checkpoint=fold_dir / "model_state.pt",
                scaler=scaler,
                hidden_dim=CONFIG.hidden_dim,
                fusion_dim=CONFIG.fusion_dim,
                dropout=CONFIG.dropout,
            )
            mae[MODEL_NEURAL] = _mae(predictions[MODEL_NEURAL], test_targets)

    if MODEL_NEURAL not in predictions:
        predictions[MODEL_NEURAL] = ()
        mae[MODEL_NEURAL] = None

    return (
        FoldResult(
            fold=fold_index,
            train_subjects=tuple(sorted({subjects[index] for index in train_idx})),
            test_subjects=tuple(sorted({subjects[index] for index in test_idx})),
            n_train=len(train_idx),
            n_test=len(test_idx),
            predictions=predictions,
            targets=tuple(test_targets),
            mae=mae,
            final_loss=final_loss,
            neural_status=neural_status,
            refusal=refusal,
        ),
        masked,
    )


def _fold_checkpoint_dir(out_dir: Path | None, fold_index: int) -> Path:
    """Every fold owns its checkpoint directory.

    A shared, cwd-relative path would let one fold's reload observe another
    fold's weights, so the fallback is still fold-scoped.
    """
    base = Path(out_dir) if out_dir is not None else Path("runs") / "physio-real-data"
    return base / f"fold-{fold_index}"


def _neural_status(fold_results: Sequence[FoldResult]) -> str:
    """Neural status over the folds that actually ran.

    A refused fold never reached training, so its default status must not drag the
    summary to ``torch_unavailable`` and hide that the other folds completed.
    """
    ran = [fold for fold in fold_results if fold.targets]
    if not ran:
        return NEURAL_UNAVAILABLE
    statuses = {fold.neural_status for fold in ran}
    if statuses == {NEURAL_COMPLETED}:
        return NEURAL_COMPLETED
    if NEURAL_REFUSED in statuses:
        return NEURAL_REFUSED
    return NEURAL_UNAVAILABLE
