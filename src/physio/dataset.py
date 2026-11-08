"""Training-side dataset discipline: masks, fold imputation and subject splits (#95/#98).

The rules are deliberately conservative:

* raw observation, QC validity and imputation are three separate masks;
* an imputer is fitted on training-fold valid observations only;
* an imputed value may feed a model input but may never create a label;
* a subject's rows stay in exactly one fold, and a shared/unknown account may be
  used for pipeline regression only, never as evidence of cross-subject
  generalization;
* ``y_pri`` remains unavailable and is never produced here.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from .response import (
    DELTA_TARGET_PREFIX,
    PRI_TARGET_KEY,
    DeltaMeasure,
    build_delta_measures,
)
from .schema import PHASE_POST, PHASE_PRE
from .windows import WindowFeatures

DATASET_SCHEMA_VERSION = "physio-dataset-v1"

IMPUTATION_TRAIN_MEDIAN = "train_median"
IMPUTATION_NONE = "none"

IDENTITY_TRUSTED = "TRUSTED"
IDENTITY_SHARED = "SHARED_ACCOUNT_UNKNOWN"
IDENTITY_UNKNOWN = "UNKNOWN"

ADMISSION_ACCEPTED = "accepted"
ADMISSION_SAME_RECORD = "same_record"
ADMISSION_INTERVAL_TOO_LONG = "interval_too_long"
ADMISSION_IMPUTED_ENDPOINT = "imputed_endpoint"
ADMISSION_NOT_PRE_POST = "not_pre_post"
ADMISSION_DELTA_INVALID = "delta_invalid"
ADMISSION_UNVERIFIED_RECORD = "unverified_record"

MASK_OBSERVED = "observed"
MASK_VALID = "valid"
MASK_IMPUTED = "imputed"


class DatasetError(ValueError):
    """Raised when dataset construction would fabricate or leak information."""


class IdentityTrustError(DatasetError):
    """Raised when an untrusted identity is used for a generalization claim."""


def _digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class FeatureRow:
    """One model input row with explicit observed/valid/imputed masks.

    ``values`` uses ``0.0`` only as a dense placeholder. A position may be read
    only when ``valid_mask`` is true; ``observed_mask`` says whether the raw
    source supplied the value, and ``imputed_mask`` says whether a training-fold
    imputer filled it.
    """

    feature_names: tuple[str, ...]
    values: tuple[float, ...]
    observed_mask: tuple[bool, ...]
    valid_mask: tuple[bool, ...]
    imputed_mask: tuple[bool, ...]
    imputation: tuple[str, ...]
    subject_ref: str
    session_id: str
    cutoff: datetime

    def __post_init__(self) -> None:
        size = len(self.feature_names)
        if size == 0 or len(set(self.feature_names)) != size:
            raise DatasetError("feature_names must be non-empty and unique")
        for name in ("values", "observed_mask", "valid_mask", "imputed_mask", "imputation"):
            if len(getattr(self, name)) != size:
                raise DatasetError(f"{name} length must match feature_names")
        if not self.subject_ref or not self.session_id:
            raise DatasetError("subject_ref and session_id must be non-empty")
        if self.cutoff.tzinfo is None:
            raise DatasetError("cutoff must be timezone-aware")
        for index in range(size):
            observed = self.observed_mask[index]
            valid = self.valid_mask[index]
            imputed = self.imputed_mask[index]
            method = self.imputation[index]
            if type(observed) is not bool or type(valid) is not bool or type(imputed) is not bool:
                raise DatasetError("masks must be bool values")
            if imputed and not valid:
                raise DatasetError("an imputed value must also be usable (valid_mask=True)")
            if imputed and observed:
                raise DatasetError("an imputed value must not claim to be observed")
            if not valid and not imputed and self.values[index] != 0.0:
                raise DatasetError("an unusable position must use the 0.0 placeholder")
            if not math.isfinite(self.values[index]):
                raise DatasetError("feature values must be finite")
            if imputed and method == IMPUTATION_NONE:
                raise DatasetError("an imputed value must record its method")
            if not imputed and method != IMPUTATION_NONE:
                raise DatasetError("a non-imputed value must use imputation='none'")

    @property
    def observed_features(self) -> tuple[str, ...]:
        return tuple(
            name for name, flag in zip(self.feature_names, self.observed_mask, strict=True) if flag
        )

    @property
    def imputed_features(self) -> tuple[str, ...]:
        return tuple(
            name for name, flag in zip(self.feature_names, self.imputed_mask, strict=True) if flag
        )

    def value_map(self) -> dict[str, float]:
        return {
            name: float(value)
            for name, value, valid in zip(
                self.feature_names, self.values, self.valid_mask, strict=True
            )
            if valid
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "values": list(self.values),
            "observed_mask": list(self.observed_mask),
            "valid_mask": list(self.valid_mask),
            "imputed_mask": list(self.imputed_mask),
            "imputation": list(self.imputation),
            "subject_ref": self.subject_ref,
            "session_id": self.session_id,
            "cutoff": self.cutoff.isoformat(),
        }


def feature_row_from_window(
    window: WindowFeatures,
    *,
    subject_ref: str,
    session_id: str,
    cutoff: datetime,
) -> FeatureRow:
    """Convert a causal window into an explicitly masked row."""
    if not isinstance(window, WindowFeatures):
        raise DatasetError("window must be WindowFeatures")
    return FeatureRow(
        feature_names=tuple(window.feature_names),
        values=tuple(float(value) for value in window.values),
        observed_mask=tuple(bool(flag) for flag in window.mask),
        valid_mask=tuple(bool(flag) for flag in window.mask),
        imputed_mask=tuple(False for _ in window.values),
        imputation=tuple(IMPUTATION_NONE for _ in window.values),
        subject_ref=subject_ref,
        session_id=session_id,
        cutoff=cutoff,
    )


@dataclass(frozen=True)
class FoldImputer:
    """Median imputer fitted on training-fold valid observations only."""

    feature_names: tuple[str, ...]
    medians: tuple[float | None, ...]
    observed_counts: tuple[int, ...]
    min_observed: int = 2
    feature_version: str = ""

    def __post_init__(self) -> None:
        size = len(self.feature_names)
        if size == 0 or len(set(self.feature_names)) != size:
            raise DatasetError("imputer feature_names must be non-empty and unique")
        if len(self.medians) != size or len(self.observed_counts) != size:
            raise DatasetError("imputer arrays must match feature_names")
        if self.min_observed < 1:
            raise DatasetError("min_observed must be at least one")
        for median, count in zip(self.medians, self.observed_counts, strict=True):
            if median is not None and not math.isfinite(median):
                raise DatasetError("imputer medians must be finite")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise DatasetError("observed_counts must be non-negative integers")

    @classmethod
    def fit(
        cls,
        rows: Sequence[FeatureRow],
        *,
        feature_names: Sequence[str] | None = None,
        min_observed: int = 2,
        feature_version: str = "",
    ) -> FoldImputer:
        if not rows:
            raise DatasetError("cannot fit an imputer without training rows")
        names = tuple(feature_names) if feature_names is not None else rows[0].feature_names
        if not names or len(set(names)) != len(names):
            raise DatasetError("feature_names must be non-empty and unique")
        columns: list[list[float]] = [[] for _ in names]
        for row in rows:
            if row.feature_names != names:
                raise DatasetError("training rows do not share one feature contract")
            for index, (value, observed, valid) in enumerate(
                zip(row.values, row.observed_mask, row.valid_mask, strict=True)
            ):
                if observed and valid:
                    if not math.isfinite(value):
                        raise DatasetError("training row contains a non-finite value")
                    columns[index].append(float(value))
        medians: list[float | None] = []
        counts: list[int] = []
        for column in columns:
            counts.append(len(column))
            if len(column) < min_observed:
                medians.append(None)
            else:
                medians.append(float(np.median(np.asarray(column, dtype=float))))
        return cls(
            feature_names=names,
            medians=tuple(medians),
            observed_counts=tuple(counts),
            min_observed=min_observed,
            feature_version=feature_version,
        )

    def transform(self, row: FeatureRow) -> FeatureRow:
        """Fill only positions with an adequate training-fold median."""
        if row.feature_names != self.feature_names:
            raise DatasetError("row feature_names do not match the imputer contract")
        values = list(row.values)
        observed = list(row.observed_mask)
        valid = list(row.valid_mask)
        imputed = list(row.imputed_mask)
        methods = list(row.imputation)
        for index, median in enumerate(self.medians):
            if valid[index] or observed[index] or median is None:
                continue
            values[index] = float(median)
            valid[index] = True
            imputed[index] = True
            methods[index] = IMPUTATION_TRAIN_MEDIAN
        return FeatureRow(
            feature_names=row.feature_names,
            values=tuple(values),
            observed_mask=tuple(observed),
            valid_mask=tuple(valid),
            imputed_mask=tuple(imputed),
            imputation=tuple(methods),
            subject_ref=row.subject_ref,
            session_id=row.session_id,
            cutoff=row.cutoff,
        )

    def payload(self) -> dict[str, Any]:
        return {
            "schema": DATASET_SCHEMA_VERSION,
            "kind": "fold_imputer",
            "feature_names": list(self.feature_names),
            "medians": list(self.medians),
            "observed_counts": list(self.observed_counts),
            "min_observed": self.min_observed,
            "feature_version": self.feature_version,
        }

    def digest(self) -> str:
        return _digest(self.payload())


@dataclass(frozen=True)
class SubjectFold:
    """Subject-level fold assignment that never splits a subject."""

    folds: Mapping[str, int]
    n_folds: int
    seed: int
    identity_trust: str
    trusted: bool
    digest: str = ""

    def __post_init__(self) -> None:
        if self.n_folds < 2:
            raise DatasetError("n_folds must be at least two")
        if self.identity_trust not in (IDENTITY_TRUSTED, IDENTITY_SHARED, IDENTITY_UNKNOWN):
            raise DatasetError(f"unknown identity trust: {self.identity_trust!r}")
        if self.trusted != (self.identity_trust == IDENTITY_TRUSTED):
            raise DatasetError("trusted must match identity_trust")
        for subject, fold in self.folds.items():
            if not subject:
                raise DatasetError("subject ids must be non-empty")
            if not isinstance(fold, int) or isinstance(fold, bool) or not 0 <= fold < self.n_folds:
                raise DatasetError(f"invalid fold {fold!r} for subject {subject!r}")
        object.__setattr__(self, "folds", dict(sorted(self.folds.items())))
        if not self.digest:
            object.__setattr__(self, "digest", _digest(self.payload()))

    def payload(self) -> dict[str, Any]:
        return {
            "schema": DATASET_SCHEMA_VERSION,
            "kind": "subject_fold",
            "n_folds": self.n_folds,
            "seed": self.seed,
            "identity_trust": self.identity_trust,
            "trusted": self.trusted,
            "folds": dict(sorted(self.folds.items())),
        }

    def fold_of(self, row: FeatureRow) -> int:
        try:
            return self.folds[row.subject_ref]
        except KeyError as exc:
            raise DatasetError(f"row subject {row.subject_ref!r} is not in the split") from exc

    def validate_rows(self, rows: Sequence[FeatureRow]) -> None:
        for row in rows:
            self.fold_of(row)
        seen: dict[str, int] = {}
        for row in rows:
            fold = self.fold_of(row)
            previous = seen.setdefault(row.subject_ref, fold)
            if previous != fold:
                raise DatasetError("a subject appears in more than one fold")


def make_subject_folds(
    subjects: Sequence[str],
    *,
    n_folds: int,
    seed: int,
    identity_trust: str = IDENTITY_TRUSTED,
    allow_untrusted: bool = False,
) -> SubjectFold:
    """Assign whole subjects to folds; untrusted identities cannot claim folds."""
    if identity_trust != IDENTITY_TRUSTED and not allow_untrusted:
        raise IdentityTrustError(
            "shared/unknown account identity cannot be used for subject-level "
            "generalization; pass allow_untrusted=True for pipeline regression only"
        )
    unique = tuple(sorted({subject for subject in subjects if subject}))
    if len(unique) < n_folds:
        raise DatasetError("at least one subject per fold is required")
    if any(not subject for subject in subjects):
        raise DatasetError("subject ids must be non-empty")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    folds: dict[str, int] = {}
    for position, index in enumerate(order):
        folds[unique[index]] = position % n_folds
    return SubjectFold(
        folds=folds,
        n_folds=n_folds,
        seed=seed,
        identity_trust=identity_trust,
        trusted=identity_trust == IDENTITY_TRUSTED,
    )


@dataclass(frozen=True)
class TargetAdmission:
    """Per-metric y_delta admission plus the reason for every rejection."""

    values: dict[str, float | None]
    mask: dict[str, bool]
    reasons: dict[str, str]
    y_pri_value: float | None = None
    y_pri_available: bool = False

    def __post_init__(self) -> None:
        if self.y_pri_value is not None or self.y_pri_available:
            raise DatasetError("y_pri must remain unavailable and null")
        if set(self.values) != set(self.mask) or set(self.values) != set(self.reasons):
            raise DatasetError("target values, mask and reasons must share keys")
        for key, available in self.mask.items():
            if type(available) is not bool:
                raise DatasetError("target mask must be bool")
            if available and self.values[key] is None:
                raise DatasetError("an available target cannot be None")
            if not available and self.values[key] is not None:
                raise DatasetError("an unavailable target must be None")

    @property
    def available(self) -> tuple[str, ...]:
        return tuple(key for key, flag in self.mask.items() if flag)


def _is_imputed_measurement(measurement: Any) -> bool:
    notes = getattr(measurement, "notes", ()) or ()
    return any("value_origin=imputed" in str(note) for note in notes)


def admit_delta_targets(
    pre_event: Any,
    post_event: Any,
    *,
    max_interval_s: float = 2 * 3600,
    require_independent_records: bool = True,
) -> TargetAdmission:
    """Build y_delta labels only from independent, valid, non-imputed endpoints."""
    if max_interval_s <= 0:
        raise DatasetError("max_interval_s must be positive")
    values: dict[str, float | None] = {}
    mask: dict[str, bool] = {}
    reasons: dict[str, str] = {}

    if pre_event.phase != PHASE_PRE or post_event.phase != PHASE_POST:
        for metric in sorted(
            {m.metric for m in pre_event.measurements} | {m.metric for m in post_event.measurements}
        ):
            values[f"{DELTA_TARGET_PREFIX}{metric}"] = None
            mask[f"{DELTA_TARGET_PREFIX}{metric}"] = False
            reasons[f"{DELTA_TARGET_PREFIX}{metric}"] = ADMISSION_NOT_PRE_POST
        values[PRI_TARGET_KEY] = None
        mask[PRI_TARGET_KEY] = False
        reasons[PRI_TARGET_KEY] = ADMISSION_NOT_PRE_POST
        return TargetAdmission(values, mask, reasons)

    pre_record = pre_event.provenance.source_record if pre_event.provenance else None
    post_record = post_event.provenance.source_record if post_event.provenance else None
    if pre_record and post_record:
        # Independent records carry distinct source ids even when both phases
        # start their own sequence counter at zero.
        same_record = pre_record == post_record
    elif pre_record or post_record:
        # One side proves its record identity and the other does not; sequence
        # equality cannot establish independence, so refuse the label.
        for metric in sorted(
            {m.metric for m in pre_event.measurements} | {m.metric for m in post_event.measurements}
        ):
            values[f"{DELTA_TARGET_PREFIX}{metric}"] = None
            mask[f"{DELTA_TARGET_PREFIX}{metric}"] = False
            reasons[f"{DELTA_TARGET_PREFIX}{metric}"] = ADMISSION_UNVERIFIED_RECORD
        values[PRI_TARGET_KEY] = None
        mask[PRI_TARGET_KEY] = False
        reasons[PRI_TARGET_KEY] = ADMISSION_UNVERIFIED_RECORD
        return TargetAdmission(values, mask, reasons)
    else:
        same_record = pre_event.sequence == post_event.sequence
    if require_independent_records and same_record:
        for metric in sorted(
            {m.metric for m in pre_event.measurements} | {m.metric for m in post_event.measurements}
        ):
            values[f"{DELTA_TARGET_PREFIX}{metric}"] = None
            mask[f"{DELTA_TARGET_PREFIX}{metric}"] = False
            reasons[f"{DELTA_TARGET_PREFIX}{metric}"] = ADMISSION_SAME_RECORD
        values[PRI_TARGET_KEY] = None
        mask[PRI_TARGET_KEY] = False
        reasons[PRI_TARGET_KEY] = ADMISSION_SAME_RECORD
        return TargetAdmission(values, mask, reasons)

    interval = (post_event.measured_at - pre_event.measured_at).total_seconds()
    if interval <= 0 or interval > max_interval_s:
        for metric in sorted(
            {m.metric for m in pre_event.measurements} | {m.metric for m in post_event.measurements}
        ):
            values[f"{DELTA_TARGET_PREFIX}{metric}"] = None
            mask[f"{DELTA_TARGET_PREFIX}{metric}"] = False
            reasons[f"{DELTA_TARGET_PREFIX}{metric}"] = (
                ADMISSION_INTERVAL_TOO_LONG if interval > max_interval_s else ADMISSION_NOT_PRE_POST
            )
        values[PRI_TARGET_KEY] = None
        mask[PRI_TARGET_KEY] = False
        reasons[PRI_TARGET_KEY] = (
            ADMISSION_INTERVAL_TOO_LONG if interval > max_interval_s else ADMISSION_NOT_PRE_POST
        )
        return TargetAdmission(values, mask, reasons)

    deltas: tuple[DeltaMeasure, ...] = build_delta_measures(pre_event, post_event)
    imputed_metrics = {
        measurement.metric
        for event in (pre_event, post_event)
        for measurement in event.measurements
        if _is_imputed_measurement(measurement)
    }
    for delta in deltas:
        key = f"{DELTA_TARGET_PREFIX}{delta.metric}"
        if delta.metric in imputed_metrics:
            values[key] = None
            mask[key] = False
            reasons[key] = ADMISSION_IMPUTED_ENDPOINT
        elif delta.valid:
            values[key] = delta.delta
            mask[key] = True
            reasons[key] = ADMISSION_ACCEPTED
        else:
            values[key] = None
            mask[key] = False
            reasons[key] = f"{ADMISSION_DELTA_INVALID}:{delta.status}"
    values[PRI_TARGET_KEY] = None
    mask[PRI_TARGET_KEY] = False
    reasons[PRI_TARGET_KEY] = ADMISSION_NOT_PRE_POST
    return TargetAdmission(values, mask, reasons)


def assert_loss_uses_only_observed_labels(admission: TargetAdmission) -> None:
    """Reject a target vector that includes a non-observed or imputed label."""
    for key, available in admission.mask.items():
        if key == PRI_TARGET_KEY and available:
            raise DatasetError("y_pri must not be available")
        reason = admission.reasons[key]
        if available and reason != ADMISSION_ACCEPTED:
            raise DatasetError(f"target {key!r} is available with reason {reason!r}")
        if not available and admission.values[key] is not None:
            raise DatasetError(f"target {key!r} is unavailable but carries a value")


@dataclass(frozen=True)
class DatasetQC:
    rows: int
    subjects: int
    observed_entries: int
    valid_entries: int
    imputed_entries: int
    labels_available: int
    labels_unavailable: int
    identity_trust: str
    digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DATASET_SCHEMA_VERSION,
            "rows": self.rows,
            "subjects": self.subjects,
            "observed_entries": self.observed_entries,
            "valid_entries": self.valid_entries,
            "imputed_entries": self.imputed_entries,
            "labels_available": self.labels_available,
            "labels_unavailable": self.labels_unavailable,
            "identity_trust": self.identity_trust,
            "digest": self.digest,
        }


def dataset_qc(
    rows: Sequence[FeatureRow],
    admissions: Sequence[TargetAdmission] = (),
    *,
    identity_trust: str = IDENTITY_UNKNOWN,
) -> DatasetQC:
    payload = {
        "rows": sorted(
            (row.to_dict() for row in rows),
            key=lambda item: (item["subject_ref"], item["session_id"], item["cutoff"],
                              json.dumps(item, sort_keys=True)),
        ),
        "admissions": [
            {"values": admission.values, "mask": admission.mask, "reasons": admission.reasons}
            for admission in admissions
        ],
        "identity_trust": identity_trust,
    }
    return DatasetQC(
        rows=len(rows),
        subjects=len({row.subject_ref for row in rows}),
        observed_entries=sum(sum(row.observed_mask) for row in rows),
        valid_entries=sum(sum(row.valid_mask) for row in rows),
        imputed_entries=sum(sum(row.imputed_mask) for row in rows),
        labels_available=sum(sum(admission.mask.values()) for admission in admissions),
        labels_unavailable=sum(
            len(admission.mask) - sum(admission.mask.values()) for admission in admissions
        ),
        identity_trust=identity_trust,
        digest=_digest(payload),
    )


__all__ = [
    "ADMISSION_ACCEPTED",
    "ADMISSION_IMPUTED_ENDPOINT",
    "ADMISSION_INTERVAL_TOO_LONG",
    "ADMISSION_SAME_RECORD",
    "ADMISSION_UNVERIFIED_RECORD",
    "DATASET_SCHEMA_VERSION",
    "IDENTITY_SHARED",
    "IDENTITY_TRUSTED",
    "IDENTITY_UNKNOWN",
    "IMPUTATION_NONE",
    "IMPUTATION_TRAIN_MEDIAN",
    "DatasetError",
    "DatasetQC",
    "FeatureRow",
    "FoldImputer",
    "IdentityTrustError",
    "SubjectFold",
    "TargetAdmission",
    "admit_delta_targets",
    "assert_loss_uses_only_observed_labels",
    "dataset_qc",
    "feature_row_from_window",
    "make_subject_folds",
]
