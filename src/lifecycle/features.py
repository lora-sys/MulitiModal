"""Versioned feature and target registry contracts (#107).

Features, targets, and data dictionaries for the ML lifecycle:
- FeatureRegistry and TargetRegistry with versioned FeatureSpec / TargetSpec objects.
- Pre-only frozen contract verification (BODY_DIM=12, PHYSIO_DIM=7, Pre-only).
- BPSI and EFRI declared rules (PDF Eq.2 and Eq.3) with hand-computable arithmetic.
- Target adapters and heads addressing known mismatches (negative delta vs [0, 100]).
- Feature engineering with explicit masks (observed/valid/imputed) and fold-internal fitting.
- Temporal leakage protection and refusal of spoofed pre/post records.
- Unavailable target declaration enforcing explicit unavailable_reason.

Deliberately dependency-free (stdlib + numpy) without torch requirement.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

from lifecycle.contracts import (
    APPROVAL_UNVERIFIED,
    EVIDENCE_SYNTHETIC,
    ContractError,
    FeatureSpec,
    FieldSpec,
    TargetSpec,
)
from physio.dataset import FeatureRow

BODY_DIM = 12
PHYSIO_DIM = 7

EFRI_JOINT_EQUAL_BLOCK_V0 = (0.5, 0.5)


class TemporalLeakageError(ValueError):
    """Raised when during/post data leaks into pre features or pre/post records are spoofed."""


class TargetCompatibilityError(ValueError):
    """Raised when a target legal_range is structurally incompatible with a model output head."""


@dataclass(frozen=True)
class TargetHead:
    """Output head metadata defining the legal output range."""

    name: str
    kind: str
    legal_range: tuple[float, float] | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if self.legal_range is not None and self.legal_range[0] >= self.legal_range[1]:
            raise ValueError("legal_range must be (low, high) with low < high")


def bpsi(
    features: Mapping[str, float] | Sequence[float],
    weights: Mapping[str, float] | Sequence[float] | None = None,
    *,
    rescale_range: tuple[float, float] | None = None,
) -> float:
    """Compute Baseline Physiological State Index (BPSI) per PDF Eq.(2).

    BPSI = sum_{j=1}^p w_j * x_j_tilde.
    Declared rule only: derived from Pre-only variables; NOT a clinically validated
    label or gold standard of fatigue.
    """
    if isinstance(features, Mapping):
        p = len(features)
        if p == 0:
            raise ValueError("features must be non-empty")
        if weights is None:
            w = 1.0 / p
            raw_sum = sum(float(val) * w for val in features.values())
        elif isinstance(weights, Mapping):
            raw_sum = sum(float(features[k]) * float(weights[k]) for k in features)
        elif isinstance(weights, Sequence):
            # A Mapping has no declared pairing to a positional weight vector:
            # zipping it would silently bind weights to insertion order.
            raise TypeError(
                "weights must be a Mapping when features is a Mapping; "
                "a Sequence would depend on dict insertion order"
            )
        else:
            raise TypeError("weights must be a Mapping or None when features is a Mapping")
    elif isinstance(features, Sequence):
        p = len(features)
        if p == 0:
            raise ValueError("features must be non-empty")
        if weights is None:
            w = 1.0 / p
            raw_sum = sum(float(val) * w for val in features)
        elif isinstance(weights, Sequence):
            if len(weights) != p:
                raise ValueError("weights length must match features length")
            raw_sum = sum(float(val) * float(w) for val, w in zip(features, weights, strict=True))
        else:
            raise TypeError("weights must be a Sequence or None when features is a Sequence")
    else:
        raise TypeError("features must be a Mapping or Sequence")

    if rescale_range is not None:
        low, high = rescale_range
        if low >= high:
            raise ValueError("rescale_range must be (low, high) with low < high")
        raw_sum = 100.0 * (raw_sum - low) / (high - low)
    return float(raw_sum)


def _compute_block(
    vals: Mapping[str, float] | Sequence[float],
    weights: Mapping[str, float] | Sequence[float] | None,
    block_name: str,
) -> float:
    if isinstance(vals, Mapping):
        count = len(vals)
        if count == 0:
            raise ValueError(f"{block_name} must be non-empty")
        if weights is None:
            return sum(float(v) for v in vals.values()) / count
        if isinstance(weights, Mapping):
            return sum(float(vals[k]) * float(weights[k]) for k in vals)
        if isinstance(weights, Sequence):
            if len(weights) != count:
                raise ValueError(f"{block_name} weights length must match values")
            return sum(float(v) * float(w) for v, w in zip(vals.values(), weights, strict=True))
        raise TypeError(f"{block_name} weights must be Mapping, Sequence, or None")
    if isinstance(vals, Sequence):
        count = len(vals)
        if count == 0:
            raise ValueError(f"{block_name} must be non-empty")
        if weights is None:
            return sum(float(v) for v in vals) / count
        if isinstance(weights, Sequence):
            if len(weights) != count:
                raise ValueError(f"{block_name} weights length must match values")
            return sum(float(v) * float(w) for v, w in zip(vals, weights, strict=True))
        raise TypeError(f"{block_name} weights must be Sequence or None")
    raise TypeError(f"{block_name} must be Mapping or Sequence")


def efri(
    *,
    eeg_responses: Mapping[str, float] | Sequence[float] | None = None,
    post_responses: Mapping[str, float] | Sequence[float] | None = None,
    variant: str = "joint",
    eeg_weights: Mapping[str, float] | Sequence[float] | None = None,
    post_weights: Mapping[str, float] | Sequence[float] | None = None,
    joint_block_weights: tuple[float, float] = EFRI_JOINT_EQUAL_BLOCK_V0,
) -> float:
    """Compute Expected Fatigue Relief Index (EFRI) per PDF Eq.(3).

    EFRI = sum_{k=1}^q v_k * e_k + sum_{l=1}^m u_l * r_l.
    Declared rule only: experimental composite response, NOT clinical ground truth
    and NOT an independently validated fatigue relief efficacy label.

    Supported variants per PDF:
    - 'eeg_only': evaluates only the EEG response block.
    - 'post_only': evaluates only the post-massage physiological response block.
    - 'joint': evaluates both blocks combined with joint_block_weights (alpha, beta).
      Default block weighting is (0.5, 0.5) (EFRI_JOINT_EQUAL_BLOCK_V0), which is an
      experimental design choice, not validated coefficients.
    """
    normalized_variant = variant.lower()
    if normalized_variant == "eeg_only":
        if eeg_responses is None:
            raise ValueError("eeg_responses must be supplied for eeg_only variant")
        return _compute_block(eeg_responses, eeg_weights, "eeg_responses")
    if normalized_variant == "post_only":
        if post_responses is None:
            raise ValueError("post_responses must be supplied for post_only variant")
        return _compute_block(post_responses, post_weights, "post_responses")
    if normalized_variant == "joint":
        if eeg_responses is None or post_responses is None:
            raise ValueError("Both eeg_responses and post_responses are required for joint variant")
        s_eeg = _compute_block(eeg_responses, eeg_weights, "eeg_responses")
        s_post = _compute_block(post_responses, post_weights, "post_responses")
        alpha, beta = joint_block_weights
        return float(alpha * s_eeg + beta * s_post)
    raise ValueError(f"Unknown EFRI variant {variant!r}; expected eeg_only, post_only, or joint")


def head_for(target: TargetSpec) -> TargetHead:
    """Return an output head whose legal_range strictly contains target.legal_range."""
    if target.legal_range is None:
        return TargetHead(
            name=f"{target.name}_head",
            kind="unbounded_linear",
            legal_range=None,
            description=f"Unbounded linear head for {target.name}",
        )
    low, high = target.legal_range
    if low < 0.0:
        head_low = min(low, -50.0)
        head_high = max(high, 50.0)
        return TargetHead(
            name=f"{target.name}_delta_head",
            kind="linear_delta",
            legal_range=(head_low, head_high),
            description=f"Linear delta head covering [{head_low}, {head_high}] for {target.name}",
        )
    if high <= 1.0 and low >= 0.0:
        return TargetHead(
            name=f"{target.name}_unit_head",
            kind="unit_interval",
            legal_range=(0.0, 1.0),
            description=f"Unit interval head for {target.name}",
        )
    if high <= 100.0 and low >= 0.0:
        return TargetHead(
            name=f"{target.name}_score_head",
            kind="clamped_100",
            legal_range=(0.0, 100.0),
            description=f"Clamped [0, 100] head for {target.name}",
        )
    return TargetHead(
        name=f"{target.name}_bounded_head",
        kind="bounded_linear",
        legal_range=(low, high),
        description=f"Bounded head for {target.name}",
    )


def check_target_head_compatibility(target: TargetSpec, head: TargetHead | Any) -> bool:
    """Validate that the head's legal_range structurally contains target.legal_range.

    Fails loudly (raises TargetCompatibilityError) on any structural mismatch:
    e.g. attempting to emit negative delta values from a [0, 100] clamped head.
    """
    head_range = getattr(head, "legal_range", None)
    head_name = getattr(head, "name", str(head))

    if target.legal_range is None:
        if head_range is not None:
            raise TargetCompatibilityError(
                f"Target '{target.name}' is unbounded, but head '{head_name}' "
                f"is clamped to {head_range}"
            )
        return True

    target_low, target_high = target.legal_range
    if head_range is None:
        return True

    head_low, head_high = head_range
    if target_low < head_low:
        raise TargetCompatibilityError(
            f"Structural mismatch: target '{target.name}' min {target_low} is outside "
            f"head '{head_name}' range [{head_low}, {head_high}]. "
            "Negative target values cannot be produced by this head."
        )
    if target_high > head_high:
        raise TargetCompatibilityError(
            f"Structural mismatch: target '{target.name}' max {target_high} is outside "
            f"head '{head_name}' range [{head_low}, {head_high}]."
        )
    return True


def unavailable_target(
    name: str,
    *,
    version: str = "v1",
    head: str = "none",
    legal_range: tuple[float, float] | None = None,
    unit: str = "score",
    timepoint: str = "derived",
    provenance: str = "unverified rule",
    reason: str = "no fabricated fatigue ground truth; target is unverified",
) -> TargetSpec:
    """Return a TargetSpec with approval=UNVERIFIED and an explicit unavailable_reason."""
    if not isinstance(reason, str) or not reason.strip():
        raise ContractError("unavailable_target requires an explicit, non-empty unavailable_reason")
    return TargetSpec(
        name=name,
        version=version,
        head=head,
        legal_range=legal_range,
        unit=unit,
        timepoint=timepoint,
        provenance=provenance,
        approval=APPROVAL_UNVERIFIED,
        unavailable_reason=reason.strip(),
    )


FROZEN_BODY_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec(
        "age", "years", "float", "pre", "none", "body_pre", "neutral", "train_fold", "observed",
        (0.0, 120.0),
    ),
    FieldSpec(
        "sex", "code", "float", "pre", "none", "body_pre", "neutral", "train_fold", "observed",
        (0.0, 2.0),
    ),
    FieldSpec(
        "height", "cm", "float", "pre", "none", "body_pre", "neutral", "train_fold", "observed",
        (50.0, 250.0),
    ),
    FieldSpec(
        "weight", "kg", "float", "pre", "none", "body_pre", "neutral", "train_fold", "observed",
        (20.0, 250.0),
    ),
    FieldSpec(
        "body_fat_percentage", "percent", "float", "pre", "none", "body_pre", "neutral",
        "train_fold", "observed", (2.0, 60.0),
    ),
    FieldSpec(
        "bmi", "kg/m^2", "float", "pre", "none", "body_pre", "neutral", "train_fold", "observed",
        (10.0, 60.0),
    ),
    FieldSpec(
        "basal_metabolic_rate", "kcal", "float", "pre", "none", "body_pre", "neutral",
        "train_fold", "observed", (500.0, 3500.0),
    ),
    FieldSpec(
        "visceral_fat", "level", "float", "pre", "none", "body_pre", "neutral", "train_fold",
        "observed", (1.0, 30.0),
    ),
    FieldSpec(
        "muscle_mass", "kg", "float", "pre", "none", "body_pre", "neutral", "train_fold",
        "observed", (10.0, 100.0),
    ),
    FieldSpec(
        "body_water_percentage", "percent", "float", "pre", "none", "body_pre", "neutral",
        "train_fold", "observed", (20.0, 80.0),
    ),
    FieldSpec(
        "body_score", "score", "float", "pre", "none", "body_pre", "neutral", "train_fold",
        "observed", (0.0, 100.0),
    ),
    FieldSpec(
        "body_age", "years", "float", "pre", "none", "body_pre", "neutral", "train_fold",
        "observed", (10.0, 120.0),
    ),
)

FROZEN_PHYSIO_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec(
        "heart_rate", "bpm", "float", "pre", "none", "physio_pre", "neutral", "train_fold",
        "observed", (30.0, 220.0),
    ),
    FieldSpec(
        "microcirculation", "index", "float", "pre", "none", "physio_pre", "neutral", "train_fold",
        "observed", (0.0, 100.0),
    ),
    FieldSpec(
        "fatigue_index", "index", "float", "pre", "none", "physio_pre", "neutral", "train_fold",
        "observed", (0.0, 100.0),
    ),
    FieldSpec(
        "blood_oxygen_saturation", "percent", "float", "pre", "none", "physio_pre", "neutral",
        "train_fold", "observed", (50.0, 100.0),
    ),
    FieldSpec(
        "health_score", "score", "float", "pre", "none", "physio_pre", "neutral", "train_fold",
        "observed", (0.0, 100.0),
    ),
    FieldSpec(
        "systolic_bp", "mmHg", "float", "pre", "none", "physio_pre", "neutral", "train_fold",
        "observed", (50.0, 250.0),
    ),
    FieldSpec(
        "diastolic_bp", "mmHg", "float", "pre", "none", "physio_pre", "neutral", "train_fold",
        "observed", (30.0, 150.0),
    ),
)

FROZEN_PRE_ONLY_FEATURE_SPEC = FeatureSpec(
    version="physio-features-pre-only-v1",
    fields=FROZEN_BODY_FIELDS + FROZEN_PHYSIO_FIELDS,
    transform_id="pre_only_transform_v1",
    evidence_level=EVIDENCE_SYNTHETIC,
)


def verify_against_core_spec(
    spec: FeatureSpec,
    core_spec: Any = None,
) -> bool:
    """Verify that a FeatureSpec satisfies the frozen core spec contract.

    Contract rules:
    - Pre-only: no during or post timepoints allowed.
    - BODY_DIM = 12: exactly 12 body features.
    - PHYSIO_DIM = 7: exactly 7 physio features.
    - Total features = 19.
    """
    for field in spec.fields:
        if field.timepoint not in ("pre", "static"):
            raise ContractError(
                f"Pre-only contract violated: field '{field.name}' "
                f"has timepoint '{field.timepoint}'"
            )
        if field.source in ("during", "post", "eeg_during", "physio_post"):
            raise ContractError(
                f"Pre-only contract violated: field '{field.name}' has source '{field.source}'"
            )

    body_fields = [f for f in spec.fields if f.source == "body_pre"]
    physio_fields = [f for f in spec.fields if f.source == "physio_pre"]

    if len(body_fields) != BODY_DIM:
        raise ContractError(
            f"Expected exactly {BODY_DIM} body fields (BODY_DIM=12), got {len(body_fields)}"
        )
    if len(physio_fields) != PHYSIO_DIM:
        raise ContractError(
            f"Expected exactly {PHYSIO_DIM} physio fields (PHYSIO_DIM=7), got {len(physio_fields)}"
        )
    if len(spec.fields) != BODY_DIM + PHYSIO_DIM:
        raise ContractError(
            f"Expected exactly {BODY_DIM + PHYSIO_DIM} total fields, got {len(spec.fields)}"
        )

    if core_spec is not None:
        body_mod = core_spec.get("body_pre")
        physio_mod = core_spec.get("physio_pre")
        if body_mod.dim != len(body_fields):
            raise ContractError(f"core_spec body_pre dim {body_mod.dim} != {len(body_fields)}")
        if physio_mod.dim != len(physio_fields):
            raise ContractError(
                f"core_spec physio_pre dim {physio_mod.dim} != {len(physio_fields)}"
            )

    return True


DEFAULT_TARGET_SPECS: tuple[TargetSpec, ...] = (
    unavailable_target(
        name="bpsi",
        version="bpsi-v1",
        head="bpsi_aux_head",
        legal_range=(0.0, 100.0),
        unit="score",
        timepoint="pre",
        provenance="PDF Eq.(2) declared rule; baseline index",
        reason="unverified rule weights; not a clinically validated fatigue label",
    ),
    unavailable_target(
        name="efri_joint",
        version="efri-joint-v1",
        head="linear_regression",
        legal_range=None,
        unit="standardized_score",
        timepoint="derived",
        provenance="PDF Eq.(3) declared rule; 0.5/0.5 joint composite response",
        reason="experimental composite response; not clinically validated ground truth",
    ),
    unavailable_target(
        name="efri_eeg_only",
        version="efri-eeg-v1",
        head="linear_regression",
        legal_range=None,
        unit="standardized_score",
        timepoint="during",
        provenance="PDF Eq.(3) EEG-only variant",
        reason="experimental EEG composite; not clinically validated ground truth",
    ),
    unavailable_target(
        name="efri_post_only",
        version="efri-post-v1",
        head="linear_regression",
        legal_range=None,
        unit="standardized_score",
        timepoint="post",
        provenance="PDF Eq.(3) post-physio response variant",
        reason="experimental post-physio composite; not clinically validated ground truth",
    ),
    unavailable_target(
        name="fatigue_state",
        version="fatigue_state_v1",
        head="realtime_head_0_100",
        legal_range=(0.0, 100.0),
        unit="score",
        timepoint="during",
        provenance="engineering proxy for current fatigue state",
        reason="engineering proxy; not validated clinical fatigue state",
    ),
    unavailable_target(
        name="delta_vendor_fatigue",
        version="delta-vendor-v1",
        head="linear_delta",
        legal_range=(-18.0, 0.0),
        unit="point_delta",
        timepoint="derived",
        provenance="vendor post-pre fatigue difference",
        reason="vendor proxy difference; unverified clinical efficacy",
    ),
    unavailable_target(
        name="post_state",
        version="post-state-v1",
        head="unit_interval",
        legal_range=(0.25, 0.48),
        unit="ratio",
        timepoint="post",
        provenance="vendor post-massage state ratio",
        reason="unverified vendor state metric",
    ),
)


class FeatureRegistry:
    """Registry holding versioned FeatureSpec objects with data dictionaries."""

    def __init__(self, specs: Sequence[FeatureSpec] | None = None) -> None:
        self._specs: dict[str, FeatureSpec] = {}
        if specs:
            for s in specs:
                self.register(s)
        else:
            self.register(FROZEN_PRE_ONLY_FEATURE_SPEC)

    def register(self, spec: FeatureSpec) -> None:
        if not isinstance(spec, FeatureSpec):
            raise TypeError("spec must be an instance of FeatureSpec")
        self._specs[spec.version] = spec

    def get(self, version: str) -> FeatureSpec:
        if version not in self._specs:
            raise KeyError(f"FeatureSpec version '{version}' not found in registry")
        return self._specs[version]

    def list(self) -> tuple[FeatureSpec, ...]:
        return tuple(self._specs.values())

    def data_dictionary(self, version: str) -> list[dict[str, Any]]:
        spec = self.get(version)
        return [f.to_dict() for f in spec.fields]

    def __contains__(self, version: str) -> bool:
        return version in self._specs

    def __getitem__(self, version: str) -> FeatureSpec:
        return self.get(version)


class TargetRegistry:
    """Registry holding versioned TargetSpec objects with data dictionaries."""

    def __init__(self, specs: Sequence[TargetSpec] | None = None) -> None:
        self._specs: dict[str, TargetSpec] = {}
        if specs:
            for s in specs:
                self.register(s)
        else:
            for target in DEFAULT_TARGET_SPECS:
                self.register(target)

    def register(self, spec: TargetSpec) -> None:
        if not isinstance(spec, TargetSpec):
            raise TypeError("spec must be an instance of TargetSpec")
        self._specs[spec.version] = spec
        self._specs[spec.name] = spec
        self._specs[f"{spec.name}:{spec.version}"] = spec

    def get(self, key: str) -> TargetSpec:
        if key not in self._specs:
            raise KeyError(f"TargetSpec '{key}' not found in registry")
        return self._specs[key]

    def list(self) -> tuple[TargetSpec, ...]:
        seen: set[str] = set()
        res: list[TargetSpec] = []
        for s in self._specs.values():
            if s.sha256 not in seen:
                seen.add(s.sha256)
                res.append(s)
        return tuple(res)

    def data_dictionary(self, key: str | None = None) -> list[dict[str, Any]]:
        if key is not None:
            return [self.get(key).to_dict()]
        return [s.to_dict() for s in self.list()]

    def __contains__(self, key: str) -> bool:
        return key in self._specs

    def __getitem__(self, key: str) -> TargetSpec:
        return self.get(key)


def assert_no_temporal_leakage(rows: Sequence[Any]) -> None:
    """Reject any during/post value in a pre feature, and any Pre/Post same-record spoofing."""
    for row in rows:
        if isinstance(row, Mapping):
            phase = row.get("phase")
            if phase in ("during", "post"):
                raise TemporalLeakageError(
                    f"Temporal leakage: row has phase '{phase}' in pre feature context"
                )
            if row.get("extra_during_post"):
                raise TemporalLeakageError(
                    "Temporal leakage: extra during/post data present in pre feature row"
                )
            pre_rec = (
                row.get("pre_record")
                or row.get("pre_record_id")
                or row.get("source_record_pre")
            )
            post_rec = (
                row.get("post_record")
                or row.get("post_record_id")
                or row.get("source_record_post")
            )
            if pre_rec and post_rec and pre_rec == post_rec:
                raise TemporalLeakageError(
                    f"Pre/Post same-record spoofing detected: record '{pre_rec}' "
                    "used for both pre and post"
                )
            if row.get("same_record_spoof"):
                raise TemporalLeakageError("Pre/Post same-record spoofing detected")
            for k, v in row.items():
                if v is not None and ("during" in k or "post" in k) and k not in (
                    "post_record", "post_record_id", "source_record_post",
                ):
                    raise TemporalLeakageError(
                        f"Temporal leakage: during/post key '{k}' found in pre feature row"
                    )
        elif isinstance(row, FeatureRow):
            for name in row.feature_names:
                if "during" in name or "post" in name:
                    raise TemporalLeakageError(
                        f"Temporal leakage: during/post feature '{name}' found in FeatureRow"
                    )
            if getattr(row, "same_record_spoof", False):
                raise TemporalLeakageError("Pre/Post same-record spoofing detected")
        else:
            phase = getattr(row, "phase", None)
            if phase in ("during", "post"):
                raise TemporalLeakageError(
                    f"Temporal leakage: object phase '{phase}' in pre feature context"
                )
            if getattr(row, "extra_during_post", False):
                raise TemporalLeakageError("Temporal leakage: extra_during_post is True")
            pre_rec = getattr(row, "pre_record", None)
            post_rec = getattr(row, "post_record", None)
            if pre_rec and post_rec and pre_rec == post_rec:
                raise TemporalLeakageError(
                    f"Pre/Post same-record spoofing detected: '{pre_rec}'"
                )
            if getattr(row, "same_record_spoof", False):
                raise TemporalLeakageError("Pre/Post same-record spoofing detected")


def _extract_field_value(event: Any, field_name: str) -> float | None:
    raw: Any = None
    if isinstance(event, Mapping):
        if field_name in event:
            raw = event[field_name]
        elif (
            "values" in event
            and isinstance(event["values"], Mapping)
            and field_name in event["values"]
        ):
            raw = event["values"][field_name]
        elif "measurements" in event:
            for m in event["measurements"]:
                if isinstance(m, Mapping) and m.get("metric") == field_name:
                    raw = m.get("value")
                    break
                if hasattr(m, "metric") and m.metric == field_name:
                    raw = getattr(m, "value", None)
                    break
    elif hasattr(event, field_name):
        raw = getattr(event, field_name)
    elif hasattr(event, "measurements"):
        for m in getattr(event, "measurements", ()):
            if hasattr(m, "metric") and m.metric == field_name:
                raw = getattr(m, "value", None)
                break
            if isinstance(m, Mapping) and m.get("metric") == field_name:
                raw = m.get("value")
                break

    if raw is None:
        return None
    try:
        val = float(raw)
        return val if math.isfinite(val) else None
    except (TypeError, ValueError):
        return None


def build_feature_rows(
    events: Sequence[Any],
    spec: FeatureSpec,
    *,
    train_subjects: Sequence[str] | set[str] | frozenset[str],
) -> list[FeatureRow]:
    """Build explicitly masked FeatureRows with fold-internal median imputation.

    Rules:
    - Pre-only features must not admit any During/Post event values.
    - Imputation and scaling parameters are fitted ONLY on training subjects (fold-internal).
    - Missing values that are imputed have observed_mask=False, valid_mask=True, imputed_mask=True.
    - Missing values that are un-imputed have observed_mask=False, valid_mask=False,
      imputed_mask=False, and use exactly 0.0 placeholder.
    - Observed values have observed_mask=True, valid_mask=True, imputed_mask=False.
    """
    train_subj_set = set(train_subjects)

    # 1. Temporal leakage check across all events for pre-only fields
    has_pre_field = any(f.timepoint == "pre" for f in spec.fields)
    for event in events:
        if isinstance(event, Mapping):
            phase = event.get("phase")
            if has_pre_field and phase in ("during", "post"):
                raise TemporalLeakageError(
                    f"Temporal leakage: cannot extract pre features from {phase} event"
                )
            if has_pre_field and event.get("extra_during_post"):
                raise TemporalLeakageError(
                    "Temporal leakage: extra_during_post data present in pre event"
                )
            pre_rec = (
                event.get("pre_record")
                or event.get("pre_record_id")
                or event.get("source_record_pre")
            )
            post_rec = (
                event.get("post_record")
                or event.get("post_record_id")
                or event.get("source_record_post")
            )
            if pre_rec and post_rec and pre_rec == post_rec:
                raise TemporalLeakageError(
                    f"Pre/Post same-record spoofing detected: record '{pre_rec}'"
                )
            if event.get("same_record_spoof"):
                raise TemporalLeakageError("Pre/Post same-record spoofing detected")
        else:
            phase = getattr(event, "phase", None)
            if has_pre_field and phase in ("during", "post"):
                raise TemporalLeakageError(
                    f"Temporal leakage: cannot extract pre features from {phase} event"
                )
            if has_pre_field and getattr(event, "extra_during_post", False):
                raise TemporalLeakageError("Temporal leakage: extra_during_post is True")
            pre_rec = getattr(event, "pre_record", None)
            post_rec = getattr(event, "post_record", None)
            if pre_rec and post_rec and pre_rec == post_rec:
                raise TemporalLeakageError(
                    f"Pre/Post same-record spoofing detected: '{pre_rec}'"
                )
            if getattr(event, "same_record_spoof", False):
                raise TemporalLeakageError("Pre/Post same-record spoofing detected")

    # 2. Fit medians ONLY on training subjects
    train_medians: dict[str, float] = {}
    for field in spec.fields:
        if field.fit_scope == "train_fold":
            train_vals: list[float] = []
            for event in events:
                subj = (
                    event.get("subject_ref")
                    if isinstance(event, Mapping)
                    else getattr(event, "subject_ref", "")
                )
                if subj in train_subj_set:
                    val = _extract_field_value(event, field.name)
                    if val is not None:
                        train_vals.append(val)
            if train_vals:
                train_medians[field.name] = float(np.median(train_vals))

    # 3. Construct FeatureRows with explicit masks
    rows: list[FeatureRow] = []
    feature_names = tuple(f.name for f in spec.fields)

    for event in events:
        subj = (
            event.get("subject_ref")
            if isinstance(event, Mapping) and event.get("subject_ref")
            else getattr(event, "subject_ref", None) or "unknown_subject"
        )
        session = (
            event.get("session_id")
            if isinstance(event, Mapping) and event.get("session_id")
            else getattr(event, "session_id", None) or "unknown_session"
        )
        cutoff = (
            event.get("cutoff")
            if isinstance(event, Mapping)
            else getattr(event, "cutoff", None)
        )
        if cutoff is None:
            cutoff = datetime.now(UTC)
        elif cutoff.tzinfo is None:
            cutoff = cutoff.replace(tzinfo=UTC)

        row_values: list[float] = []
        observed_mask: list[bool] = []
        valid_mask: list[bool] = []
        imputed_mask: list[bool] = []
        imputation_methods: list[str] = []

        for field in spec.fields:
            val = _extract_field_value(event, field.name)
            if val is not None:
                row_values.append(val)
                observed_mask.append(True)
                valid_mask.append(True)
                imputed_mask.append(False)
                imputation_methods.append("none")
            else:
                if field.fit_scope == "train_fold" and field.name in train_medians:
                    row_values.append(train_medians[field.name])
                    observed_mask.append(False)
                    valid_mask.append(True)
                    imputed_mask.append(True)
                    imputation_methods.append("train_median")
                else:
                    # Unusable missing value: MUST use 0.0 placeholder per FeatureRow contract
                    row_values.append(0.0)
                    observed_mask.append(False)
                    valid_mask.append(False)
                    imputed_mask.append(False)
                    imputation_methods.append("none")

        row = FeatureRow(
            feature_names=feature_names,
            values=tuple(row_values),
            observed_mask=tuple(observed_mask),
            valid_mask=tuple(valid_mask),
            imputed_mask=tuple(imputed_mask),
            imputation=tuple(imputation_methods),
            subject_ref=subj,
            session_id=session,
            cutoff=cutoff,
        )
        rows.append(row)

    return rows


__all__ = [
    "BODY_DIM",
    "DEFAULT_TARGET_SPECS",
    "EFRI_JOINT_EQUAL_BLOCK_V0",
    "FROZEN_BODY_FIELDS",
    "FROZEN_PHYSIO_FIELDS",
    "FROZEN_PRE_ONLY_FEATURE_SPEC",
    "FeatureRegistry",
    "PHYSIO_DIM",
    "TargetCompatibilityError",
    "TargetHead",
    "TargetRegistry",
    "TemporalLeakageError",
    "assert_no_temporal_leakage",
    "bpsi",
    "build_feature_rows",
    "check_target_head_compatibility",
    "efri",
    "head_for",
    "unavailable_target",
    "verify_against_core_spec",
]
