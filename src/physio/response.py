"""Versioned Post-Pre response labels and program-conditioned multi-target data (#98).

``build_delta_measures`` records an observed difference, never an improvement.
A delta is valid only when both endpoints use the same canonical metric and
unit, both measurements passed QC, and ``post_time > pre_time``. Heart rate or
blood pressure falling, or SpO2 rising, is just a signed number here: no
recovery, relief, or causal claim is attached to that direction.

``y_pri`` has no approved formula. It therefore stays ``None`` with
``available=False`` until a registry entry is explicitly approved *and* an
implementation exists. The module never auto-fills a score.

Training discipline: masks, scalers, weights and missing-value handlers are fit
inside the training fold only. ``fit_only_on_train`` builds that fold mask; the
caller remains responsible for applying it before any fold-sensitive fit.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from .metrics import is_known_metric, normalize_unit
from .programs import (
    ProgramCapabilityCatalog,
    ProgramContext,
    ProgramValidationError,
    parse_program_context,
)
from .schema import (
    PHASE_DURING,
    PHASE_POST,
    PHASE_PRE,
    QUALITY_REJECTED,
    QUALITY_STALE,
    MeasurementV1,
    PhysioEventV1,
    ProtocolContextV1,
)

PRI_TARGET_VERSION = "y_pri_v1"
DELTA_SCHEMA_VERSION = "physio-delta-v1"

RULE_STATUS_APPROVED = "APPROVED"
RULE_STATUS_UNAPPROVED = "UNAPPROVED"
RULE_STATUSES = (RULE_STATUS_APPROVED, RULE_STATUS_UNAPPROVED)

DELTA_ACCEPTED = "accepted"
DELTA_MISSING_PRE = "missing_pre"
DELTA_MISSING_POST = "missing_post"
DELTA_UNIT_MISMATCH = "unit_mismatch"
DELTA_NON_POSITIVE_INTERVAL = "non_positive_interval"
DELTA_INVALID_VALUE = "invalid_value"
DELTA_STATUSES = (
    DELTA_ACCEPTED,
    DELTA_MISSING_PRE,
    DELTA_MISSING_POST,
    DELTA_UNIT_MISMATCH,
    DELTA_NON_POSITIVE_INTERVAL,
    DELTA_INVALID_VALUE,
)

DELTA_TARGET_PREFIX = "y_delta_"
PRI_TARGET_KEY = "y_pri"
DELTA_TARGET_STATUS = "OBSERVED_DELTA"


@dataclass(frozen=True)
class DeltaMeasure:
    """One metric comparison between a Pre and a Post event.

    ``delta`` is exactly ``post_value - pre_value`` and carries no direction
    meaning. Invalid rows never expose a delta, so downstream code cannot
    accidentally train on a missing or mismatched pair.
    """

    metric: str
    pre_value: float | int | str | None
    post_value: float | int | str | None
    delta: float | None
    unit: str
    valid: bool
    status: str
    pre_time: datetime | None
    post_time: datetime | None
    provenance: dict[str, Any] | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.metric, str) or not self.metric.strip():
            raise ValueError("DeltaMeasure.metric must be a non-empty string")
        if self.status not in DELTA_STATUSES:
            raise ValueError(f"unknown delta status: {self.status!r}")
        if self.valid and self.status != DELTA_ACCEPTED:
            raise ValueError("a valid delta must have status 'accepted'")
        if not self.valid and self.status == DELTA_ACCEPTED:
            raise ValueError("accepted status requires valid=True")
        if self.valid:
            if self.delta is None or self.pre_value is None or self.post_value is None:
                raise ValueError("a valid delta must carry pre, post and delta values")
            if self.pre_time is None or self.post_time is None:
                raise ValueError("a valid delta must carry pre and post timestamps")
            if self.post_time <= self.pre_time:
                raise ValueError("a valid delta requires post_time > pre_time")
            expected = float(self.post_value) - float(self.pre_value)
            if not math.isclose(self.delta, expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("a valid delta must equal post_value - pre_value")
        elif self.delta is not None:
            raise ValueError("an invalid delta must not expose a delta value")
        object.__setattr__(self, "notes", tuple(self.notes))
        if self.provenance is not None:
            object.__setattr__(self, "provenance", dict(self.provenance))

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "pre_value": self.pre_value,
            "post_value": self.post_value,
            "delta": self.delta,
            "unit": self.unit,
            "valid": self.valid,
            "status": self.status,
            "pre_time": self.pre_time.isoformat() if self.pre_time else None,
            "post_time": self.post_time.isoformat() if self.post_time else None,
            "provenance": dict(self.provenance) if self.provenance else None,
            "notes": list(self.notes),
        }


def _event_provenance(event: PhysioEventV1) -> dict[str, Any]:
    return {
        "source": event.source,
        "sequence": event.sequence,
        "quality": event.quality,
        "adapter": event.provenance.adapter if event.provenance else None,
        "record": event.provenance.source_record if event.provenance else None,
    }


def _measurements_by_metric(event: PhysioEventV1) -> dict[str, list[MeasurementV1]]:
    grouped: dict[str, list[MeasurementV1]] = {}
    for measurement in event.measurements:
        grouped.setdefault(measurement.metric, []).append(measurement)
    return grouped


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _invalid_delta(
    metric: str,
    *,
    status: str,
    pre_event: PhysioEventV1,
    post_event: PhysioEventV1,
    pre_item: MeasurementV1 | None,
    post_item: MeasurementV1 | None,
    unit: str,
    notes: tuple[str, ...],
) -> DeltaMeasure:
    return DeltaMeasure(
        metric=metric,
        pre_value=pre_item.value if pre_item is not None else None,
        post_value=post_item.value if post_item is not None else None,
        delta=None,
        unit=unit,
        valid=False,
        status=status,
        pre_time=pre_event.measured_at,
        post_time=post_event.measured_at,
        provenance={
            "pre": _event_provenance(pre_event),
            "post": _event_provenance(post_event),
            "pre_raw_metric": pre_item.raw_metric if pre_item else None,
            "post_raw_metric": post_item.raw_metric if post_item else None,
            "pre_status": pre_item.status if pre_item else None,
            "post_status": post_item.status if post_item else None,
        },
        notes=notes,
    )


def build_delta_measures(
    pre_event: PhysioEventV1,
    post_event: PhysioEventV1,
) -> tuple[DeltaMeasure, ...]:
    """Build deterministic Post-Pre deltas for the union of observed metrics.

    Missing endpoints, duplicate fields, invalid measurements, stale/rejected
    events, non-numeric fields, unit mismatches and non-positive intervals are
    explicit statuses with ``valid=False``. No status is silently repaired and
    no direction is inferred from the sign of ``delta``.
    """
    if pre_event.phase != PHASE_PRE:
        raise ValueError(f"pre_event must be in phase {PHASE_PRE!r}")
    if post_event.phase != PHASE_POST:
        raise ValueError(f"post_event must be in phase {PHASE_POST!r}")
    if pre_event.session_id != post_event.session_id:
        raise ValueError("pre and post events must share session_id")
    if pre_event.subject_ref != post_event.subject_ref:
        raise ValueError("pre and post events must share subject_ref")

    pre_index = _measurements_by_metric(pre_event)
    post_index = _measurements_by_metric(post_event)
    metrics = sorted(set(pre_index) | set(post_index))
    deltas: list[DeltaMeasure] = []

    for metric in metrics:
        pre_items = pre_index.get(metric, [])
        post_items = post_index.get(metric, [])
        pre_item = pre_items[0] if len(pre_items) == 1 else None
        post_item = post_items[0] if len(post_items) == 1 else None
        unit = normalize_unit(pre_item.unit if pre_item else None) or (
            normalize_unit(post_item.unit if post_item else None) or ""
        )

        if not pre_items:
            deltas.append(
                _invalid_delta(
                    metric,
                    status=DELTA_MISSING_PRE,
                    pre_event=pre_event,
                    post_event=post_event,
                    pre_item=None,
                    post_item=post_item,
                    unit=unit,
                    notes=("metric is absent from the Pre event",),
                )
            )
            continue
        if not post_items:
            deltas.append(
                _invalid_delta(
                    metric,
                    status=DELTA_MISSING_POST,
                    pre_event=pre_event,
                    post_event=post_event,
                    pre_item=pre_item,
                    post_item=None,
                    unit=unit,
                    notes=("metric is absent from the Post event",),
                )
            )
            continue
        if len(pre_items) > 1 or len(post_items) > 1:
            deltas.append(
                _invalid_delta(
                    metric,
                    status=DELTA_INVALID_VALUE,
                    pre_event=pre_event,
                    post_event=post_event,
                    pre_item=pre_item,
                    post_item=post_item,
                    unit=unit,
                    notes=("duplicate measurements for one metric; refusing to choose",),
                )
            )
            continue
        if not is_known_metric(metric):
            deltas.append(
                _invalid_delta(
                    metric,
                    status=DELTA_INVALID_VALUE,
                    pre_event=pre_event,
                    post_event=post_event,
                    pre_item=pre_item,
                    post_item=post_item,
                    unit=unit,
                    notes=(f"{metric!r} is not a canonical metric",),
                )
            )
            continue

        bad_events = [
            name
            for name, event in (("pre", pre_event), ("post", post_event))
            if event.quality in (QUALITY_REJECTED, QUALITY_STALE)
        ]
        if bad_events:
            deltas.append(
                _invalid_delta(
                    metric,
                    status=DELTA_INVALID_VALUE,
                    pre_event=pre_event,
                    post_event=post_event,
                    pre_item=pre_item,
                    post_item=post_item,
                    unit=unit,
                    notes=(f"event quality blocks use: {', '.join(bad_events)}",),
                )
            )
            continue
        if not pre_item.valid or not post_item.valid:
            deltas.append(
                _invalid_delta(
                    metric,
                    status=DELTA_INVALID_VALUE,
                    pre_event=pre_event,
                    post_event=post_event,
                    pre_item=pre_item,
                    post_item=post_item,
                    unit=unit,
                    notes=(
                        f"invalid endpoints: pre={pre_item.status}, post={post_item.status}",
                    ),
                )
            )
            continue
        if not _is_numeric(pre_item.value) or not _is_numeric(post_item.value):
            deltas.append(
                _invalid_delta(
                    metric,
                    status=DELTA_INVALID_VALUE,
                    pre_event=pre_event,
                    post_event=post_event,
                    pre_item=pre_item,
                    post_item=post_item,
                    unit=unit,
                    notes=("non-numeric metric cannot produce a numeric delta",),
                )
            )
            continue

        pre_unit = normalize_unit(pre_item.unit) or ""
        post_unit = normalize_unit(post_item.unit) or ""
        if pre_unit != post_unit:
            deltas.append(
                _invalid_delta(
                    metric,
                    status=DELTA_UNIT_MISMATCH,
                    pre_event=pre_event,
                    post_event=post_event,
                    pre_item=pre_item,
                    post_item=post_item,
                    unit=pre_unit,
                    notes=(f"units differ: pre={pre_unit!r}, post={post_unit!r}",),
                )
            )
            continue
        if post_event.measured_at <= pre_event.measured_at:
            deltas.append(
                _invalid_delta(
                    metric,
                    status=DELTA_NON_POSITIVE_INTERVAL,
                    pre_event=pre_event,
                    post_event=post_event,
                    pre_item=pre_item,
                    post_item=post_item,
                    unit=pre_unit,
                    notes=("post_time must be strictly later than pre_time",),
                )
            )
            continue

        delta = float(post_item.value) - float(pre_item.value)
        deltas.append(
            DeltaMeasure(
                metric=metric,
                pre_value=pre_item.value,
                post_value=post_item.value,
                delta=delta,
                unit=pre_unit,
                valid=True,
                status=DELTA_ACCEPTED,
                pre_time=pre_event.measured_at,
                post_time=post_event.measured_at,
                provenance={
                    "pre": _event_provenance(pre_event),
                    "post": _event_provenance(post_event),
                    "pre_raw_metric": pre_item.raw_metric,
                    "post_raw_metric": post_item.raw_metric,
                    "pre_status": pre_item.status,
                    "post_status": post_item.status,
                },
                notes=("delta is Post-Pre only; sign has no efficacy meaning",),
            )
        )

    return tuple(deltas)


@dataclass(frozen=True)
class PriTarget:
    """``y_pri`` label. Unavailable unless an approved rule has been implemented."""

    value: float | None = None
    available: bool = False
    score_version: str = PRI_TARGET_VERSION
    rule_status: str = RULE_STATUS_UNAPPROVED
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.score_version, str) or not self.score_version.strip():
            raise ValueError("PriTarget.score_version must be a non-empty string")
        if self.rule_status not in RULE_STATUSES:
            raise ValueError(f"unknown PRI rule status: {self.rule_status!r}")
        if self.available:
            if self.rule_status != RULE_STATUS_APPROVED:
                raise ValueError("y_pri cannot be available without an approved rule")
            if self.value is None:
                raise ValueError("an available y_pri must carry a value")
        else:
            if self.value is not None:
                raise ValueError("an unavailable y_pri must not carry a value")

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": PRI_TARGET_KEY,
            "value": self.value,
            "available": self.available,
            "score_version": self.score_version,
            "rule_status": self.rule_status,
            "reason": self.reason,
        }


#: Name used by callers that read the PRI target as a label.
PriLabel = PriTarget


@dataclass(frozen=True)
class RuleSpec:
    """Versioned rule metadata. ``direction`` is descriptive, never an efficacy claim."""

    name: str
    version: str
    source: str
    direction: str
    weights: tuple[tuple[str, float], ...] = ()
    status: str = RULE_STATUS_UNAPPROVED
    notes: str = ""

    def __post_init__(self) -> None:
        for field in ("name", "version", "source", "direction"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"RuleSpec.{field} must be a non-empty string")
        if self.status not in RULE_STATUSES:
            raise ValueError(f"unknown rule status: {self.status!r}")
        weights: list[tuple[str, float]] = []
        for item in self.weights:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError("rule weights must be (name, weight) pairs")
            name, weight = item
            if not isinstance(name, str) or not name.strip():
                raise ValueError("rule weight names must be non-empty strings")
            if not isinstance(weight, (int, float)) or isinstance(weight, bool):
                raise ValueError(f"rule weight for {name!r} must be numeric")
            if not math.isfinite(float(weight)):
                raise ValueError(f"rule weight for {name!r} must be finite")
            weights.append((name, float(weight)))
        object.__setattr__(self, "weights", tuple(weights))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "source": self.source,
            "direction": self.direction,
            "weights": [list(item) for item in self.weights],
            "status": self.status,
            "notes": self.notes,
        }


#: Historical BPSI/EFRI semantics are preserved as unapproved RULE entries.
#: No weight strategy or direction table has been verified, so no weights are set.
RULE_REGISTRY: dict[str, RuleSpec] = {
    "BPSI": RuleSpec(
        name="BPSI",
        version="bpsi-v1",
        source=(
            "src/core/specs.py PHYSIO_TARGET_BPSI_AUX; "
            "pre-massage framework Eq.(2) draft"
        ),
        direction=(
            "Descriptive only: a Pre-derived baseline construct. Higher means a higher "
            "weighted construct score by the chosen weights; it is not an efficacy "
            "direction and no weight strategy is approved."
        ),
        weights=(),
        status=RULE_STATUS_UNAPPROVED,
        notes="historical RULE semantics preserved; no new claim",
    ),
    "EFRI": RuleSpec(
        name="EFRI",
        version="efri-v1",
        source=(
            "pre-massage framework Eq.(3) draft; "
            "src/core/models.py PhysioBPSIEFRIModel"
        ),
        direction=(
            "Descriptive only: a composite response whose sign depends on an "
            "unverified direction table. No causal, relief, or clinical effect is "
            "implied by the sign."
        ),
        weights=(),
        status=RULE_STATUS_UNAPPROVED,
        notes="RULE composite is not a gold-standard fatigue label",
    ),
    "PRI": RuleSpec(
        name="PRI",
        version=PRI_TARGET_VERSION,
        source="issue #98; no formula, weights, or direction table was supplied",
        direction=(
            "Not defined. Heart rate or blood pressure falling, or SpO2 rising, "
            "does not by itself mean recovery; only an approved direction table "
            "could define a composite response."
        ),
        weights=(),
        status=RULE_STATUS_UNAPPROVED,
        notes="y_pri remains null/unavailable; no default weighting",
    ),
}


def pri_target(
    deltas: Sequence[DeltaMeasure] | None = None,
    *,
    registry: Mapping[str, RuleSpec] | None = None,
) -> PriTarget:
    """Return ``y_pri`` only when an approved rule *and* implementation exist.

    No PRI formula is implemented in this module, so the result is always
    ``value=None`` and ``available=False``. The ``deltas`` argument documents
    that observed Post-Pre values alone do not define PRI.
    """
    del deltas
    rules = RULE_REGISTRY if registry is None else registry
    rule = rules.get("PRI")
    if rule is None:
        return PriTarget(reason="PRI rule is not registered; y_pri is not available")
    if rule.status != RULE_STATUS_APPROVED:
        return PriTarget(
            rule_status=rule.status,
            reason=f"PRI rule {rule.version!r} is {rule.status}; y_pri is not available",
        )
    return PriTarget(
        score_version=rule.version,
        rule_status=rule.status,
        reason="approved PRI rule has no implemented scoring formula; refusing to fabricate y_pri",
    )


def validate_pri_claim(label: PriTarget) -> None:
    """Reject a claim that marks y_pri available without an approved value."""
    if not isinstance(label, PriTarget):
        raise TypeError("label must be a PriTarget/PriLabel")
    if label.available:
        if label.value is None:
            raise ValueError("y_pri is marked available but carries no value")
        if label.rule_status != RULE_STATUS_APPROVED:
            raise ValueError("y_pri is marked available under an unapproved rule")
    elif label.value is not None:
        raise ValueError("y_pri is unavailable but carries a value")


@dataclass(frozen=True)
class TargetVector:
    """Per-dimension target values plus an explicit availability mask."""

    values: dict[str, float | None]
    mask: dict[str, bool]
    versions: dict[str, str]
    rule_status: dict[str, str]

    def __post_init__(self) -> None:
        keys = set(self.values)
        for field, mapping in (
            ("mask", self.mask),
            ("versions", self.versions),
            ("rule_status", self.rule_status),
        ):
            if set(mapping) != keys:
                raise ValueError(f"target {field} keys do not match target values")
        for name, available in self.mask.items():
            if type(available) is not bool:
                raise ValueError(f"target mask for {name!r} must be bool")
            value = self.values[name]
            if available and value is None:
                raise ValueError(f"available target {name!r} cannot be None")
            if not available and value is not None:
                raise ValueError(f"unavailable target {name!r} must be None")
        if PRI_TARGET_KEY in self.values:
            if self.mask[PRI_TARGET_KEY] or self.values[PRI_TARGET_KEY] is not None:
                raise ValueError("y_pri must remain unavailable and null")
        object.__setattr__(self, "values", dict(self.values))
        object.__setattr__(self, "mask", dict(self.mask))
        object.__setattr__(self, "versions", dict(self.versions))
        object.__setattr__(self, "rule_status", dict(self.rule_status))

    @property
    def available_dimensions(self) -> tuple[str, ...]:
        return tuple(name for name, available in self.mask.items() if available)

    @property
    def missing_dimensions(self) -> tuple[str, ...]:
        return tuple(name for name, available in self.mask.items() if not available)


def _target_vector(deltas: tuple[DeltaMeasure, ...]) -> TargetVector:
    values: dict[str, float | None] = {}
    mask: dict[str, bool] = {}
    versions: dict[str, str] = {}
    statuses: dict[str, str] = {}
    for delta in deltas:
        key = f"{DELTA_TARGET_PREFIX}{delta.metric}"
        values[key] = delta.delta if delta.valid else None
        mask[key] = delta.valid
        versions[key] = DELTA_SCHEMA_VERSION
        statuses[key] = DELTA_TARGET_STATUS
    values[PRI_TARGET_KEY] = None
    mask[PRI_TARGET_KEY] = False
    versions[PRI_TARGET_KEY] = PRI_TARGET_VERSION
    statuses[PRI_TARGET_KEY] = RULE_STATUS_UNAPPROVED
    return TargetVector(values=values, mask=mask, versions=versions, rule_status=statuses)


@dataclass(frozen=True)
class ResponseSample:
    """One Pre/during/Post sample with program context and masked targets."""

    pre_event: PhysioEventV1
    during_events: tuple[PhysioEventV1, ...]
    post_event: PhysioEventV1
    program: ProgramContext
    deltas: tuple[DeltaMeasure, ...]
    target_vector: TargetVector
    delta_schema_version: str = DELTA_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.pre_event.phase != PHASE_PRE:
            raise ValueError("pre_event must be in phase 'pre'")
        if self.post_event.phase != PHASE_POST:
            raise ValueError("post_event must be in phase 'post'")
        if any(event.phase != PHASE_DURING for event in self.during_events):
            raise ValueError("during_events must all be in phase 'during'")
        if self.delta_schema_version != DELTA_SCHEMA_VERSION:
            raise ValueError(f"unsupported delta schema {self.delta_schema_version!r}")
        expected = {
            *(f"{DELTA_TARGET_PREFIX}{delta.metric}" for delta in self.deltas),
            PRI_TARGET_KEY,
        }
        if set(self.target_vector.values) != expected:
            raise ValueError("target vector does not match the delta measures")

    @property
    def targets(self) -> dict[str, float | None]:
        return self.target_vector.values

    @property
    def target_mask(self) -> dict[str, bool]:
        return self.target_vector.mask

    @property
    def available_dimensions(self) -> tuple[str, ...]:
        return self.target_vector.available_dimensions

    @property
    def missing_dimensions(self) -> tuple[str, ...]:
        return self.target_vector.missing_dimensions


def _same_identity(events: Sequence[PhysioEventV1]) -> None:
    if not events:
        return
    first = events[0]
    for event in events[1:]:
        if event.session_id != first.session_id:
            raise ValueError("all events in one response sample must share session_id")
        if event.subject_ref != first.subject_ref:
            raise ValueError("all events in one response sample must share subject_ref")


def build_response_sample(
    pre: PhysioEventV1,
    during: PhysioEventV1 | Sequence[PhysioEventV1] | None,
    post: PhysioEventV1,
    program_context: ProgramContext,
    catalog: ProgramCapabilityCatalog | None,
) -> ResponseSample:
    """Validate phases/program and build a program-conditioned response sample."""
    if not isinstance(pre, PhysioEventV1) or not isinstance(post, PhysioEventV1):
        raise TypeError("pre and post must be PhysioEventV1")
    if not isinstance(program_context, ProgramContext):
        raise TypeError("program_context must be a ProgramContext")
    if not program_context.reviewed:
        raise ProgramValidationError(
            "program context is unreviewed; refusing to condition labels"
        )
    if pre.phase != PHASE_PRE:
        raise ValueError(f"pre event must be in phase {PHASE_PRE!r}")
    if post.phase != PHASE_POST:
        raise ValueError(f"post event must be in phase {PHASE_POST!r}")
    if during is None:
        during_events: tuple[PhysioEventV1, ...] = ()
    elif isinstance(during, PhysioEventV1):
        during_events = (during,)
    else:
        during_events = tuple(during)
    for event in during_events:
        if not isinstance(event, PhysioEventV1):
            raise TypeError("during events must be PhysioEventV1")
        if event.phase != PHASE_DURING:
            raise ValueError(f"during event must be in phase {PHASE_DURING!r}")

    _same_identity((pre, *during_events, post))
    for event in during_events:
        if not pre.measured_at <= event.measured_at <= post.measured_at:
            raise ValueError("during events must lie between the Pre and Post timestamps")

    try:
        validated_program = parse_program_context(
            ProtocolContextV1(
                program_id=program_context.program_id,
                intensity=program_context.intensity,
                duration_s=program_context.duration_s,
                catalog_version=program_context.catalog_version,
            ),
            catalog,
        )
    except ProgramValidationError:
        raise
    if validated_program.actions != program_context.actions:
        raise ProgramValidationError(
            "program context actions do not match the reviewed catalog; "
            "stale context is rejected"
        )

    deltas = build_delta_measures(pre, post)
    return ResponseSample(
        pre_event=pre,
        during_events=during_events,
        post_event=post,
        program=validated_program,
        deltas=deltas,
        target_vector=_target_vector(deltas),
    )


def masked_multi_target_loss(
    predictions: Any,
    targets: Any,
    mask: Any,
    *,
    weights: Any | None = None,
) -> float:
    """Weighted MAE over available targets only, implemented with numpy.

    ``predictions``/``targets``/``mask`` may be 1-D vectors or 2-D batches.
    Masked-out entries never contribute; an empty mask means real training
    cannot start and raises ``ValueError``. ``weights`` is a per-dimension
    vector applied to the last axis.
    """
    prediction_array = np.asarray(predictions, dtype=float)
    target_array = np.asarray(targets, dtype=float)
    mask_array = np.asarray(mask, dtype=bool)
    if prediction_array.shape != target_array.shape:
        raise ValueError(
            f"predictions shape {prediction_array.shape} != targets shape {target_array.shape}"
        )
    if mask_array.shape != prediction_array.shape:
        raise ValueError(
            f"mask shape {mask_array.shape} != predictions shape {prediction_array.shape}"
        )
    if prediction_array.ndim not in (1, 2):
        raise ValueError("predictions/targets must be 1-D vectors or 2-D batches")
    if prediction_array.size == 0:
        raise ValueError("no targets were supplied; real training cannot start")
    if not np.any(mask_array):
        raise ValueError("empty target mask; no available target, training cannot start")

    if weights is None:
        weight_array = np.ones(prediction_array.shape[-1], dtype=float)
    else:
        weight_array = np.asarray(weights, dtype=float)
        if weight_array.shape != (prediction_array.shape[-1],):
            raise ValueError(
                f"weights shape {weight_array.shape} must match the target dimension "
                f"{(prediction_array.shape[-1],)}"
            )
    if not np.all(np.isfinite(weight_array)) or np.any(weight_array < 0):
        raise ValueError("target weights must be finite and non-negative")

    broadcast_weights = np.broadcast_to(weight_array, prediction_array.shape)
    selected_weights = broadcast_weights[mask_array]
    if not math.isfinite(float(selected_weights.sum())) or selected_weights.sum() <= 0:
        raise ValueError("available target weights must sum to a positive value")

    selected_predictions = prediction_array[mask_array]
    selected_targets = target_array[mask_array]
    if not np.all(np.isfinite(selected_predictions)):
        raise ValueError("predictions contain NaN/Inf for an available target")
    if not np.all(np.isfinite(selected_targets)):
        raise ValueError("targets contain NaN/Inf for an available target")

    errors = np.abs(selected_predictions - selected_targets)
    return float(np.sum(errors * selected_weights) / np.sum(selected_weights))


def fit_only_on_train(indices: Sequence[int], n_samples: int) -> np.ndarray:
    """Boolean mask for fitting fold-sensitive state on training rows only."""
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")
    index_items = list(indices)
    if any(
        isinstance(index, bool) or not isinstance(index, (int, np.integer))
        for index in index_items
    ):
        raise ValueError("training indices must be integers")
    index_array = np.asarray(index_items, dtype=int)
    if index_array.ndim != 1 or index_array.size == 0:
        raise ValueError("training indices must be a non-empty 1-D sequence")
    if np.any(index_array < 0) or np.any(index_array >= n_samples):
        raise ValueError("training indices must lie within [0, n_samples)")
    if np.unique(index_array).size != index_array.size:
        raise ValueError("training indices must not contain duplicates")
    train_mask = np.zeros(n_samples, dtype=bool)
    train_mask[index_array] = True
    return train_mask


__all__ = [
    "DELTA_ACCEPTED",
    "DELTA_INVALID_VALUE",
    "DELTA_MISSING_POST",
    "DELTA_MISSING_PRE",
    "DELTA_NON_POSITIVE_INTERVAL",
    "DELTA_SCHEMA_VERSION",
    "DELTA_STATUSES",
    "DELTA_TARGET_PREFIX",
    "DELTA_TARGET_STATUS",
    "DELTA_UNIT_MISMATCH",
    "PRI_TARGET_KEY",
    "PRI_TARGET_VERSION",
    "RULE_REGISTRY",
    "RULE_STATUS_APPROVED",
    "RULE_STATUSES",
    "RULE_STATUS_UNAPPROVED",
    "DeltaMeasure",
    "PriLabel",
    "PriTarget",
    "ResponseSample",
    "RuleSpec",
    "TargetVector",
    "build_delta_measures",
    "build_response_sample",
    "fit_only_on_train",
    "masked_multi_target_loss",
    "pri_target",
    "validate_pri_claim",
]
