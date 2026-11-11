"""Real de-identified 94-field Session export import (#102).

The vendor export is *research input only*. This module freezes the mapping from
the long-table raw records (one row per device record) onto the same
``PhysioEventV1`` / ``PhysioSessionV1`` contract used by the device, service and
offline adapters, then rebuilds the Pre/Post endpoints and the ``y_delta_*``
labels from those raw records. A pre-computed summary row is never trusted as a
label source; it is only compared against, in ``real_data_audit``.

No real export is committed. The raw file lives under ``data/local/`` and is read
at audit time only.

The mapping this module implements is a **declared implementation hypothesis**,
reverse-engineered from the provided extract. It is not a verified vendor mapping:
without a vendor field dictionary the reconstruction necessarily imposes a
semantic reading on 94 fields, and ``coded_field_semantics_unverified`` /
``not_recoverable`` blockers bound how far that reading may be trusted.

Anything that cannot be recovered from the provided extract is reported through a
QC issue instead of being guessed.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .dataset import IDENTITY_SHARED, IMPUTATION_NONE, FeatureRow, admit_delta_targets
from .metrics import route_measurement
from .schema import (
    PHASE_POST,
    PHASE_PRE,
    SOURCE_OFFLINE,
    MeasurementV1,
    PhysioEventV1,
    PhysioSessionV1,
    ProvenanceV1,
)

REAL_EXPORT_SCHEMA_VERSION = "physio-real-export-v1"

#: EXTRACT-FITTED HYPOTHESIS, NOT A VENDOR RULE.
#:
#: The reference training set for the provided 5-session extract was produced with
#: this endpoint rule, and the rule reproduces every reference value on that
#: extract. It is reverse-engineered from the extract, not supplied by the vendor.
#: A future export with different measurement timing can break it, and the
#: guardrails below can silently discard valid labels if read as a contract.
#: They are constants so they can be tested and changed, never a frozen spec.
ENDPOINT_RULE = "first_and_last_measurement_group_by_measured_at"
ENDPOINT_GROUP_WINDOW_S = 60.0
MAX_ENDPOINT_INTERVAL_S = 2 * 3600.0

#: EXTRACT-FITTED HYPOTHESIS, NOT A CLINICAL OR VENDOR RULE. These thresholds
#: reproduce the reference anomaly verdict on the provided extract only. They are
#: an engineering guardrail and must not be treated as a general contract.
BP_DELTA_SBP_GUARD_MMHG = 30.0
BP_DELTA_DBP_GUARD_MMHG = 20.0

#: EXTRACT-FITTED HYPOTHESIS. Tolerance for cross-module heart-rate agreement
#: (health module versus blood-pressure ``pulse``). A disagreement is a QC
#: observation, never a repair, and the tolerance is not a vendor contract.
HR_CROSS_MODULE_TOLERANCE_BPM = 5.0

#: The export carries naive wall-clock timestamps. Only relative ordering and
#: deltas are used, so attaching UTC cannot change any label; the naive source is
#: still recorded as a QC issue.
TIMESTAMP_TZ_POLICY = "naive export timestamps are read as UTC; only relative ordering is used"

SOURCE_BP = "血压"
SOURCE_HEALTH = "健康"
SOURCE_BODY = "体脂"
SOURCE_TCM = "舌面"
SOURCE_EEG = "脑波"
SOURCE_EEG_POINTS = "脑波逐点"

MODULE_OF_SOURCE: dict[str, str] = {
    SOURCE_BP: "bp",
    SOURCE_HEALTH: "health",
    SOURCE_BODY: "body",
    SOURCE_TCM: "tcm",
    SOURCE_EEG: "eeg",
    SOURCE_EEG_POINTS: "eeg_points",
}

#: Only these modules carry a Pre/Post endpoint pair.
ENDPOINT_MODULES: tuple[str, ...] = ("bp", "health")

#: Module that owns a feature when several modules measure the same metric.
FEATURE_MODULE_PRIORITY: tuple[str, ...] = ("health", "bp")

#: Module that owns a reference label when several modules measure the same
#: metric. The other module's value is kept as a cross-check only.
LABEL_MODULE_PRIORITY: tuple[str, ...] = ("health", "bp")

#: Declared raw column -> canonical metric mapping for endpoint modules. The
#: third element is the reference label key used by the vendor training set.
MODULE_METRIC_COLUMNS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "bp": (
        ("sys", "systolic_bp", "y_delta_sbp"),
        ("dia", "diastolic_bp", "y_delta_dbp"),
        ("pulse", "heart_rate", "y_delta_hr"),
    ),
    "health": (
        ("hr", "heart_rate", "y_delta_hr"),
        ("spo2", "spo2", "y_delta_spo2"),
        ("micro_hrv", "microcirculation_index", "y_delta_micro"),
        ("rr", "respiration_rate", "y_delta_rr"),
        ("fatigue_index", "fatigue_index", "y_delta_vendor_fatigue"),
        ("vendor_score", "vendor_health_index", "y_delta_vendor_health"),
    ),
}

STATIC_MODULE_COLUMNS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "body": (
        ("weight_kg", "weight_kg", "weight_kg"),
        ("body_fat_pct", "body_fat_percentage", "body_fat_pct"),
        ("bmi", "bmi", "bmi"),
    ),
}

#: Coded export fields whose canonical metric uses semantic categories while the
#: export carries raw codes. The code is preserved verbatim for audit only; the
#: code-to-category meaning is unverified and no feature is derived from it.
CODED_FIELDS: tuple[tuple[str, str, str], ...] = (("bp", "sex", "gender"),)

#: Reference label contract. The reference QC report and training set list exactly
#: these keys, so a derived label outside the contract is reported rather than
#: silently admitted or silently dropped.
REFERENCE_LABEL_KEYS: tuple[str, ...] = (
    "y_delta_sbp",
    "y_delta_dbp",
    "y_delta_hr",
    "y_delta_spo2",
    "y_delta_micro",
    "y_delta_vendor_fatigue",
    "y_delta_vendor_health",
)

#: Reference training set stores spo2 as a fraction while the canonical contract
#: keeps percent. The declared scale is applied to both features and labels.
REFERENCE_VALUE_SCALE: dict[str, float] = {"spo2": 0.01}

#: Pre-phase feature contract realized from the provided extract. Names are the
#: canonical metric names so a row can be audited against the frozen
#: ``physio-input-pre-only-v0`` input spec.
REAL_PRE_FEATURE_NAMES: tuple[str, ...] = (
    "weight_kg",
    "bmi",
    "heart_rate",
    "spo2",
    "microcirculation_index",
    "respiration_rate",
    "fatigue_index",
    "vendor_health_index",
    "systolic_bp",
    "diastolic_bp",
)

#: Declared decimal precision for emitted label and feature values. It removes
#: float representation noise only; it never rounds across a guardrail.
VALUE_DECIMALS = 6

#: Frozen Pre-only contract columns mapped to the names above; used only to
#: report contract coverage, never to add a feature. ``sex`` is excluded because
#: the export carries raw codes whose canonical meaning is unverified.
FROZEN_CONTRACT_COVERAGE: dict[str, tuple[str, ...]] = {
    "body_pre": ("weight_kg", "bmi"),
    "physio_pre": (
        "heart_rate",
        "microcirculation_index",
        "fatigue_index",
        "spo2",
        "vendor_health_index",
        "systolic_bp",
        "diastolic_bp",
    ),
}
FROZEN_CONTRACT_UNUSED: dict[str, tuple[str, ...]] = {
    "body_pre": ("sex",),
    "physio_pre": (),
}

REQUIRED_RAW_COLUMNS: tuple[str, ...] = (
    "session_id",
    "source_type",
    "record_id",
    "member_id_hash",
    "subject_key",
    "measured_at",
)

#: Every raw column the mapping declares. A column present in an export but
#: absent here is reported as unmapped rather than silently dropped.
DECLARED_RAW_COLUMNS: frozenset[str] = frozenset(
    REQUIRED_RAW_COLUMNS
    + ("is_complete", "fatigue_level")
    + tuple(column for module in MODULE_METRIC_COLUMNS.values() for column, _, _ in module)
    + tuple(
        column
        for module in STATIC_MODULE_COLUMNS.values()
        for column, _, _ in module
    )
    + tuple(column for _module, column, _reference in CODED_FIELDS)
    + ("constitution", "health_index")
    + (
        "brain_overall",
        "brain_avg_meditation",
        "brain_duration",
        "sample_points",
    )
)

#: Raw columns declared as carrying no recoverable meaning in this contract.
DECLARED_UNUSABLE_RAW_COLUMNS: tuple[str, ...] = ("fatigue_level",)

#: Module families whose absence is a session-level QC observation. ``eeg_points``
#: is excluded: it is a point stream that is never a feature or label, so its
#: absence is reported only when it is present (``QC_POINT_STREAM_EXCLUDED``).
REPORTED_MODULE_FAMILIES: tuple[str, ...] = ("bp", "health", "body", "tcm", "eeg")

_UNIT_OVERRIDES: dict[str, str] = {
    "sys": "mmHg",
    "dia": "mmHg",
    "pulse": "bpm",
    "hr": "bpm",
    "spo2": "%",
    "micro_hrv": "index",
    "rr": "breaths/min",
    "fatigue_index": "index",
    "vendor_score": "index",
    "weight_kg": "kg",
    "body_fat_pct": "%",
    "bmi": "kg/m2",
    "sex": "",
}

QC_NAIVE_TIMESTAMP = "naive_timestamp"
QC_UNKNOWN_SOURCE = "unknown_source_type"
QC_POINT_STREAM_EXCLUDED = "point_stream_excluded"
QC_MISSING_MODULE = "missing_module"
QC_MISSING_FIELD = "missing_field"
QC_INVALID_VALUE = "invalid_value"
QC_SINGLE_ENDPOINT = "single_endpoint_record"
QC_OVERLAPPING_ENDPOINTS = "overlapping_endpoint_records"
QC_RECORD_REUSED = "record_reused_across_sessions"
QC_SESSION_OVERLAP = "overlapping_session_windows"
QC_SNAPSHOT_ENDPOINT = "snapshot_endpoint"
QC_BP_DELTA_ANOMALY = "bp_delta_anomaly"
QC_HR_CROSS_MODULE_MISMATCH = "hr_cross_module_mismatch"
QC_IDENTITY_SHARED = "shared_account_identity"
QC_STATIC_PROFILE_COLLISION = "static_profile_collision"
QC_BRAIN_SELECTION_NOT_RECOVERABLE = "brain_record_selection_not_recoverable"
QC_CODED_FIELD_SEMANTICS_UNVERIFIED = "coded_field_semantics_unverified"
QC_UNMAPPED_RAW_COLUMN = "unmapped_raw_column"
QC_DECLARED_UNUSABLE = "declared_unusable_raw_column"


class RealExportError(ValueError):
    """Raised when a real export cannot be imported without fabricating data."""


@dataclass(frozen=True)
class RealQCIssue:
    """One quality observation about a real session."""

    code: str
    detail: str
    session_id: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"session_id": self.session_id, "code": self.code, "detail": self.detail}


@dataclass(frozen=True)
class RealRecord:
    """One long-table row of the real export, normalised."""

    session_id: str
    source_type: str
    record_id: str
    member_id_hash: str
    subject_key: str
    measured_at: datetime
    fields: Mapping[str, str] = field(default_factory=dict)

    @property
    def module(self) -> str:
        return MODULE_OF_SOURCE.get(self.source_type, "unknown")

    def value(self, column: str) -> str | None:
        raw = self.fields.get(column)
        if raw is None:
            return None
        text = str(raw).strip()
        return text or None


@dataclass(frozen=True)
class RealEndpoint:
    """A merged Pre or Post endpoint for one module, with its canonical event."""

    module: str
    phase: str
    measured_at: datetime
    anchor_record: str
    record_ids: tuple[str, ...]
    measurements: tuple[MeasurementV1, ...]
    completion_flags: tuple[str, ...] = ()
    event: PhysioEventV1 | None = None

    def measurement(self, metric: str) -> MeasurementV1 | None:
        for item in self.measurements:
            if item.metric == metric:
                return item
        return None


@dataclass(frozen=True)
class RealSession:
    """One imported real session with its endpoints, features and QC."""

    session_id: str
    member_ref: str
    subject_ref: str
    device_ref: str
    window_start: datetime
    window_end: datetime
    records: tuple[RealRecord, ...]
    endpoints: Mapping[str, Mapping[str, RealEndpoint]]
    static_values: Mapping[str, float]
    static_origins: Mapping[str, str]
    tcm_values: Mapping[str, str]
    eeg_record_id: str
    eeg_values: Mapping[str, float]
    coded_values: Mapping[str, str]
    identity_trust: str
    issues: tuple[RealQCIssue, ...] = ()

    @property
    def source(self) -> str:
        return SOURCE_OFFLINE

    @property
    def cutoff(self) -> datetime:
        endpoint = self.endpoints.get("bp", {}).get(PHASE_PRE)
        if endpoint is not None:
            return endpoint.measured_at
        for module in ENDPOINT_MODULES:
            endpoint = self.endpoints.get(module, {}).get(PHASE_PRE)
            if endpoint is not None:
                return endpoint.measured_at
        return self.window_start

    def to_session(self) -> PhysioSessionV1:
        events = [
            endpoint.event
            for module in self.endpoints
            for endpoint in self.endpoints[module].values()
            if endpoint.event is not None
        ]
        return PhysioSessionV1(
            session_id=self.session_id,
            subject_ref=self.subject_ref,
            device_ref=self.device_ref,
            events=tuple(events),
            source=self.source,
        )

    def issue_codes(self) -> tuple[str, ...]:
        return tuple(sorted({issue.code for issue in self.issues}))

    def module_present(self, module: str) -> bool:
        """Whether the export carries records for a module family.

        This is record presence, not value emission: a family can be present and
        still yield no value when the selection rule is not recoverable.
        """
        return any(record.module == module for record in self.records)


@dataclass(frozen=True)
class RealSessionLabels:
    """Reconstructed reference labels plus every admission reason.

    ``available`` answers *what the raw records can produce*. It is deliberately
    not a claim about what the frozen reference contract admits; that view is
    produced by ``reconcile_reference_contract``.
    """

    session_id: str
    values: Mapping[str, float | None]
    mask: Mapping[str, bool]
    reasons: Mapping[str, str]
    excluded_bp_labels: tuple[str, ...]
    cross_module_hr: Mapping[str, float | None]
    issues: tuple[RealQCIssue, ...] = ()
    abstained: tuple[str, ...] = ()

    @property
    def available(self) -> tuple[str, ...]:
        return tuple(sorted(key for key, flag in self.mask.items() if flag))

    def to_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "values": dict(self.values),
            "mask": dict(self.mask),
            "reasons": dict(self.reasons),
            "excluded_bp_labels": list(self.excluded_bp_labels),
            "cross_module_hr": dict(self.cross_module_hr),
            "abstained": list(self.abstained),
        }


def _parse_timestamp(raw: str) -> datetime:
    text = str(raw).strip()
    if not text:
        raise RealExportError("empty measured_at")
    for pattern in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, pattern).replace(tzinfo=UTC)
        except ValueError:
            continue
    raise RealExportError(f"unrecognised measured_at {text!r}")


def _round(value: float) -> float:
    """Drop float representation noise at the declared precision."""
    return round(float(value), VALUE_DECIMALS)


def _float(raw: str | None) -> float | None:
    if raw is None:
        return None
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def load_real_records(path: str | Path) -> tuple[RealRecord, ...]:
    """Read the long-table real export. Missing required columns are refused."""
    file_path = Path(path)
    if not file_path.is_file():
        raise RealExportError(f"real export not found: {file_path}")
    with file_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        raw_columns = tuple(reader.fieldnames or ())
        missing = [name for name in REQUIRED_RAW_COLUMNS if name not in raw_columns]
        if missing:
            raise RealExportError(f"real export is missing required columns: {missing}")
        records: list[RealRecord] = []
        for index, row in enumerate(reader, start=2):
            session_id = str(row.get("session_id") or "").strip()
            source_type = str(row.get("source_type") or "").strip()
            record_id = str(row.get("record_id") or "").strip()
            if not session_id or not source_type or not record_id:
                raise RealExportError(
                    f"row {index}: session_id, source_type and record_id are all required"
                )
            if source_type not in MODULE_OF_SOURCE:
                raise RealExportError(
                    f"row {index}: unknown source_type {source_type!r}; "
                    "refusing to guess a module mapping"
                )
            fields = {
                key: value
                for key, value in row.items()
                if key is not None
                and key not in REQUIRED_RAW_COLUMNS
                and str(value).strip() != ""
            }
            records.append(
                RealRecord(
                    session_id=session_id,
                    source_type=source_type,
                    record_id=record_id,
                    member_id_hash=str(row.get("member_id_hash") or "").strip(),
                    subject_key=str(row.get("subject_key") or "").strip(),
                    measured_at=_parse_timestamp(str(row.get("measured_at") or "")),
                    fields=fields,
                )
            )
    if not records:
        raise RealExportError("real export contains no rows")
    return tuple(records)


def _module_records(records: Sequence[RealRecord], module: str) -> list[RealRecord]:
    return sorted(
        (record for record in records if record.module == module),
        key=lambda item: (item.measured_at, item.record_id),
    )


def _endpoint_group(module_records: Sequence[RealRecord], *, last: bool) -> tuple[RealRecord, ...]:
    if not module_records:
        return ()
    anchor_time = module_records[-1].measured_at if last else module_records[0].measured_at
    window = timedelta(seconds=ENDPOINT_GROUP_WINDOW_S)
    if last:
        return tuple(r for r in module_records if r.measured_at >= anchor_time - window)
    return tuple(r for r in module_records if r.measured_at <= anchor_time + window)


def _merge_endpoint(
    session: RealSessionDraft,
    module: str,
    phase: str,
    group: Sequence[RealRecord],
    issues: list[RealQCIssue],
) -> RealEndpoint | None:
    if not group:
        return None
    ordered = sorted(group, key=lambda item: (item.measured_at, item.record_id))
    anchor = ordered[0]
    measurements: list[MeasurementV1] = []
    for raw_column, metric, _label_key in MODULE_METRIC_COLUMNS[module]:
        chosen: MeasurementV1 | None = None
        value_source_record = ""
        for record in ordered:
            value = record.value(raw_column)
            if value is None:
                continue
            routed = route_measurement(raw_column, value, _UNIT_OVERRIDES.get(raw_column, ""))
            if routed.valid:
                chosen = routed
                value_source_record = record.record_id
                break
        if chosen is None:
            value = ordered[0].value(raw_column)
            value_source_record = ordered[0].record_id
            chosen = route_measurement(raw_column, value, _UNIT_OVERRIDES.get(raw_column, ""))
            issues.append(
                RealQCIssue(
                    QC_MISSING_FIELD if value is None else QC_INVALID_VALUE,
                    f"module {module!r} field {raw_column!r}={value!r} has no usable value",
                    session.session_id,
                )
            )
        measurements.append(
            MeasurementV1(
                metric=metric,
                value=chosen.value,
                unit=chosen.unit,
                data_type=chosen.data_type,
                valid=chosen.valid,
                status=chosen.status,
                raw_metric=raw_column,
                raw_value=chosen.raw_value,
                raw_unit=chosen.raw_unit,
                notes=chosen.notes + (
                    f"endpoint_record={anchor.record_id}",
                    f"value_source_record={value_source_record}",
                ),
            )
        )
    event = PhysioEventV1(
        session_id=session.session_id,
        subject_ref=session.subject_ref,
        device_ref=session.device_ref,
        source=SOURCE_OFFLINE,
        phase=phase,
        measured_at=anchor.measured_at,
        received_at=anchor.measured_at,
        sequence=0,
        measurements=tuple(measurements),
        provenance=ProvenanceV1(
            adapter="physio.real_export",
            adapter_version=REAL_EXPORT_SCHEMA_VERSION,
            source_record=anchor.record_id,
            notes=(
                f"endpoint_rule={ENDPOINT_RULE}",
                f"module={module}",
                f"records={','.join(record.record_id for record in ordered)}",
            ),
        ),
    )
    return RealEndpoint(
        module=module,
        phase=phase,
        measured_at=anchor.measured_at,
        anchor_record=anchor.record_id,
        record_ids=tuple(record.record_id for record in ordered),
        measurements=tuple(measurements),
        completion_flags=tuple(record.value("is_complete") or "" for record in ordered),
        event=event,
    )


@dataclass
class RealSessionDraft:
    """Mutable accumulator used while a session is being assembled."""

    session_id: str
    member_ref: str
    subject_ref: str
    device_ref: str
    window_start: datetime
    window_end: datetime
    records: tuple[RealRecord, ...]
    endpoints: dict[str, dict[str, RealEndpoint]]
    static_values: dict[str, float]
    static_origins: dict[str, str]
    tcm_values: dict[str, str]
    eeg_record_id: str
    eeg_values: dict[str, float]
    coded_values: dict[str, str]
    issues: list[RealQCIssue]


def _static_values(
    records: Sequence[RealRecord], module: str
) -> tuple[dict[str, float], dict[str, str]]:
    values: dict[str, float] = {}
    origins: dict[str, str] = {}
    for record in _module_records(records, module):
        for raw_column, _metric, feature_name in STATIC_MODULE_COLUMNS[module]:
            if feature_name in values:
                continue
            number = _float(record.value(raw_column))
            if number is None:
                continue
            routed = route_measurement(raw_column, number, _UNIT_OVERRIDES.get(raw_column, ""))
            if not routed.valid or routed.value is None:
                continue
            scale = REFERENCE_VALUE_SCALE.get(feature_name, 1.0)
            values[feature_name] = float(routed.value) * scale
            origins[feature_name] = record.record_id
    return values, origins


def _static_profile_key(static_values: Mapping[str, float]) -> tuple[float, ...]:
    """Project over the static values actually held, never over a nominal feature list."""
    return tuple(round(float(value), 6) for value in sorted(static_values.values()))


def _draft_session(session_id: str, records: Sequence[RealRecord]) -> RealSessionDraft:
    ordered = sorted(records, key=lambda item: (item.measured_at, item.record_id))
    first = ordered[0]
    draft = RealSessionDraft(
        session_id=session_id,
        member_ref=f"member:{first.member_id_hash}",
        subject_ref=f"subject:{first.subject_key}",
        device_ref=f"legacy-export:{REAL_EXPORT_SCHEMA_VERSION}",
        window_start=ordered[0].measured_at,
        window_end=ordered[-1].measured_at,
        records=tuple(ordered),
        endpoints={},
        static_values={},
        static_origins={},
        tcm_values={},
        eeg_record_id="",
        eeg_values={},
        coded_values={},
        issues=[RealQCIssue(QC_NAIVE_TIMESTAMP, TIMESTAMP_TZ_POLICY, session_id)],
    )
    for module in ENDPOINT_MODULES:
        module_records = _module_records(ordered, module)
        if not module_records:
            draft.issues.append(
                RealQCIssue(QC_MISSING_MODULE, f"module {module!r} has no records", session_id)
            )
            continue
        pre_group = _endpoint_group(module_records, last=False)
        post_group = _endpoint_group(module_records, last=True)
        repeated = {record.record_id for record in pre_group} & {
            record.record_id for record in post_group
        }
        if repeated:
            draft.issues.append(
                RealQCIssue(
                    QC_OVERLAPPING_ENDPOINTS,
                    f"module {module!r} Pre/Post endpoint groups reuse records: "
                    + ",".join(sorted(repeated)),
                    session_id,
                )
            )
            # The same measurement must never count as both Pre and Post.
            continue
        if len(pre_group) < 2 and len(post_group) < 2 and len(module_records) < 2:
            draft.issues.append(
                RealQCIssue(
                    QC_SINGLE_ENDPOINT,
                    f"module {module!r} has a single-record endpoint group",
                    session_id,
                )
            )
        pre = _merge_endpoint(draft, module, PHASE_PRE, pre_group, draft.issues)
        post = _merge_endpoint(draft, module, PHASE_POST, post_group, draft.issues)
        for phase, endpoint in ((PHASE_PRE, pre), (PHASE_POST, post)):
            if endpoint is None:
                continue
            if any(flag == "否" for flag in endpoint.completion_flags):
                draft.issues.append(
                    RealQCIssue(
                        QC_SNAPSHOT_ENDPOINT,
                        f"module {module!r} {phase} endpoint uses is_complete=否 records; the "
                        "vendor completion flag semantics are unknown",
                        session_id,
                    )
                )
        if pre is not None and post is not None:
            draft.endpoints[module] = {PHASE_PRE: pre, PHASE_POST: post}

    body_values, body_origins = _static_values(ordered, "body")
    draft.static_values.update(body_values)
    draft.static_origins.update(body_origins)

    for record in _module_records(ordered, "tcm"):
        for column in ("constitution", "health_index"):
            value = record.value(column)
            if value is not None and column not in draft.tcm_values:
                draft.tcm_values[column] = value

    eeg_records = _module_records(ordered, "eeg")
    if len(eeg_records) == 1:
        selected = eeg_records[0]
        draft.eeg_record_id = selected.record_id
        for raw_column, feature_name in (
            ("brain_overall", "brain_overall"),
            ("brain_avg_meditation", "brain_avg_meditation"),
            ("brain_duration", "brain_duration"),
            ("sample_points", "sample_points"),
        ):
            number = _float(selected.value(raw_column))
            if number is not None:
                draft.eeg_values[feature_name] = number
    elif eeg_records:
        # More than one record and no declared selection rule: emit nothing rather
        # than publish a value chosen by an undeclared "latest wins" heuristic.
        draft.issues.append(
            RealQCIssue(
                QC_BRAIN_SELECTION_NOT_RECOVERABLE,
                f"{len(eeg_records)} 脑波 records are present but the reference selection "
                "rule is not recoverable from this extract; no brain value is emitted",
                session_id,
            )
        )

    point_records = _module_records(ordered, "eeg_points")
    if point_records:
        draft.issues.append(
            RealQCIssue(
                QC_POINT_STREAM_EXCLUDED,
                f"{len(point_records)} 脑波逐点 rows are point streams, not measurements; "
                "they are excluded from every feature and label",
                session_id,
            )
        )

    for module in REPORTED_MODULE_FAMILIES:
        if _module_records(ordered, module):
            continue
        if module in ENDPOINT_MODULES:
            continue
        draft.issues.append(
            RealQCIssue(
                QC_MISSING_MODULE,
                f"module family {module!r} has no records in this session",
                session_id,
            )
        )

    present_columns = {column for record in ordered for column in record.fields}
    unmapped = sorted(present_columns - DECLARED_RAW_COLUMNS)
    if unmapped:
        draft.issues.append(
            RealQCIssue(
                QC_UNMAPPED_RAW_COLUMN,
                "raw columns with no declared mapping: " + ",".join(unmapped),
                session_id,
            )
        )
    for column in DECLARED_UNUSABLE_RAW_COLUMNS:
        if column in present_columns:
            draft.issues.append(
                RealQCIssue(
                    QC_DECLARED_UNUSABLE,
                    f"raw column {column!r} is declared but carries no recoverable meaning "
                    "in this contract; it is excluded from every feature and label",
                    session_id,
                )
            )
    for _module, raw_column, reference_column in CODED_FIELDS:
        for record in _module_records(ordered, _module):
            value = record.value(raw_column)
            if value is None:
                continue
            draft.coded_values[reference_column] = value
            draft.issues.append(
                RealQCIssue(
                    QC_CODED_FIELD_SEMANTICS_UNVERIFIED,
                    f"export field {raw_column!r} carries the raw code {value!r}; the canonical "
                    f"{reference_column!r} meaning is unverified, so no feature is derived from it",
                    session_id,
                )
            )
            break
    return draft


def build_real_sessions(
    records: Sequence[RealRecord], *, identity_trust: str = IDENTITY_SHARED
) -> tuple[RealSession, ...]:
    """Group records into sessions, rebuild endpoints and collect QC issues."""
    if not records:
        raise RealExportError("cannot build sessions without records")
    grouped: dict[str, list[RealRecord]] = {}
    for record in records:
        grouped.setdefault(record.session_id, []).append(record)
    drafts = [_draft_session(session_id, grouped[session_id]) for session_id in sorted(grouped)]

    # A raw measurement cannot independently support labels in two Sessions.
    owners: dict[tuple[str, str], set[str]] = {}
    for draft in drafts:
        for record in draft.records:
            owners.setdefault((record.source_type, record.record_id), set()).add(draft.session_id)
    for draft in drafts:
        reused = sorted(
            {
                f"{module}:{record_id}"
                for (module, record_id), ids in owners.items()
                if draft.session_id in ids and len(ids) > 1
            }
        )
        if reused:
            draft.issues.append(
                RealQCIssue(
                    QC_RECORD_REUSED,
                    "source records appear in more than one Session: " + ",".join(reused),
                    draft.session_id,
                )
            )

    # Check overlap by explicit subject key, not by a legacy shared account.
    by_subject: dict[str, list[RealSessionDraft]] = {}
    for draft in drafts:
        by_subject.setdefault(draft.subject_ref, []).append(draft)
    for subject_drafts in by_subject.values():
        ordered_drafts = sorted(
            subject_drafts, key=lambda item: (item.window_start, item.session_id)
        )
        for left, right in zip(ordered_drafts, ordered_drafts[1:], strict=False):
            if right.window_start < left.window_end:
                detail = f"Session windows overlap: {left.session_id} and {right.session_id}"
                left.issues.append(RealQCIssue(QC_SESSION_OVERLAP, detail, left.session_id))
                right.issues.append(RealQCIssue(QC_SESSION_OVERLAP, detail, right.session_id))

    members = {draft.member_ref for draft in drafts}
    subjects = {draft.subject_ref for draft in drafts}
    shared_identity = len(members) < len(subjects)
    profiles: dict[tuple[float, ...], list[str]] = {}
    for draft in drafts:
        profiles.setdefault(_static_profile_key(draft.static_values), []).append(draft.session_id)

    sessions: list[RealSession] = []
    for draft in drafts:
        issues = list(draft.issues)
        if shared_identity:
            issues.append(
                RealQCIssue(
                    QC_IDENTITY_SHARED,
                    f"{len(members)} member hash maps to {len(subjects)} subject keys; subject "
                    "identity cannot support a generalization claim",
                    draft.session_id,
                )
            )
        profile = _static_profile_key(draft.static_values)
        if len(profiles[profile]) > 1 and any(value != 0.0 for value in profile):
            issues.append(
                RealQCIssue(
                    QC_STATIC_PROFILE_COLLISION,
                    f"{len(profiles[profile])} sessions share an identical static profile",
                    draft.session_id,
                )
            )
        sessions.append(
            RealSession(
                session_id=draft.session_id,
                member_ref=draft.member_ref,
                subject_ref=draft.subject_ref,
                device_ref=draft.device_ref,
                window_start=draft.window_start,
                window_end=draft.window_end,
                records=draft.records,
                endpoints=draft.endpoints,
                static_values=dict(draft.static_values),
                static_origins=dict(draft.static_origins),
                tcm_values=dict(draft.tcm_values),
                eeg_record_id=draft.eeg_record_id,
                eeg_values=dict(draft.eeg_values),
                coded_values=dict(draft.coded_values),
                identity_trust=identity_trust,
                issues=tuple(issues),
            )
        )
    return tuple(sessions)


def _reference_label_key(module: str, canonical_key: str) -> str:
    for _raw_column, metric, reference_key in MODULE_METRIC_COLUMNS[module]:
        if canonical_key == f"y_delta_{metric}":
            return reference_key
    return canonical_key


def _metric_of_reference_key(label_key: str) -> str:
    for module in MODULE_METRIC_COLUMNS.values():
        for _raw_column, metric, reference_key in module:
            if reference_key == label_key:
                return metric
    return label_key.removeprefix("y_delta_")


def _metric_record(endpoint: RealEndpoint, metric: str) -> str:
    """Return the record supplying this metric rather than the group anchor."""
    measurement = endpoint.measurement(metric)
    if measurement is None:
        return ""
    sources = [
        note.removeprefix("value_source_record=")
        for note in measurement.notes
        if note.startswith("value_source_record=")
    ]
    return sources[0] if len(sources) == 1 else ""


def session_labels(session: RealSession) -> RealSessionLabels:
    """Rebuild the reference ``y_delta_*`` labels from the raw endpoints."""
    values: dict[str, float | None] = {}
    mask: dict[str, bool] = {}
    reasons: dict[str, str] = {}
    excluded: list[str] = []
    cross_module: dict[str, float | None] = {}
    issues: list[RealQCIssue] = []

    blocked = next(
        (
            code
            for code in (QC_RECORD_REUSED, QC_SESSION_OVERLAP, QC_OVERLAPPING_ENDPOINTS)
            if code in session.issue_codes()
        ),
        "",
    )
    if blocked:
        for key in REFERENCE_LABEL_KEYS:
            values[key], mask[key], reasons[key] = None, False, blocked
        return RealSessionLabels(
            session_id=session.session_id,
            values=values,
            mask=mask,
            reasons=reasons,
            excluded_bp_labels=(),
            cross_module_hr={},
        )

    for module in LABEL_MODULE_PRIORITY:
        module_endpoints = session.endpoints.get(module, {})
        pre = module_endpoints.get(PHASE_PRE)
        post = module_endpoints.get(PHASE_POST)
        if pre is None or post is None or pre.event is None or post.event is None:
            continue
        admission = admit_delta_targets(
            pre_event=pre.event, post_event=post.event, max_interval_s=MAX_ENDPOINT_INTERVAL_S
        )
        for canonical_key, reason in admission.reasons.items():
            if canonical_key == "y_pri":
                continue
            label_key = _reference_label_key(module, canonical_key)
            scale = REFERENCE_VALUE_SCALE.get(_metric_of_reference_key(label_key), 1.0)
            raw_value = admission.values.get(canonical_key)
            value = None if raw_value is None else _round(float(raw_value) * scale)
            is_available = bool(admission.mask.get(canonical_key))
            if is_available and canonical_key.startswith("y_delta_"):
                metric = canonical_key.removeprefix("y_delta_")
                pre_source = _metric_record(pre, metric)
                post_source = _metric_record(post, metric)
                if not pre_source or not post_source:
                    reason, is_available = "unverified_metric_source_record", False
                elif pre_source == post_source:
                    reason, is_available = "same_metric_source_record", False
                if not is_available:
                    value = None
            if label_key in mask:
                cross_module[label_key] = value
                continue
            values[label_key] = value
            mask[label_key] = is_available
            reasons[label_key] = (
                reason
                if label_key in REFERENCE_LABEL_KEYS
                else f"{reason};outside_reference_label_contract"
            )

    issues.extend(_bp_anomaly_guard(session, values, mask, reasons, excluded))
    issues.extend(_heart_rate_cross_check(session, cross_module))

    return RealSessionLabels(
        session_id=session.session_id,
        values=values,
        mask=mask,
        reasons=reasons,
        excluded_bp_labels=tuple(sorted(excluded)),
        cross_module_hr=cross_module,
        issues=tuple(issues),
    )


def _bp_anomaly_guard(
    session: RealSession,
    values: dict[str, float | None],
    mask: dict[str, bool],
    reasons: dict[str, str],
    excluded: list[str],
) -> tuple[RealQCIssue, ...]:
    sbp = values.get("y_delta_sbp")
    dbp = values.get("y_delta_dbp")
    if sbp is None or dbp is None:
        return ()
    if abs(sbp) <= BP_DELTA_SBP_GUARD_MMHG and abs(dbp) <= BP_DELTA_DBP_GUARD_MMHG:
        return ()
    for key, value in (("y_delta_sbp", sbp), ("y_delta_dbp", dbp)):
        if not mask.get(key):
            continue
        mask[key] = False
        values[key] = None
        reasons[key] = (
            f"bp_delta_anomaly: |{key.removeprefix('y_delta_')}|={value:g} exceeds the declared "
            f"guardrail |ΔSBP|>{BP_DELTA_SBP_GUARD_MMHG:g} mmHg or "
            f"|ΔDBP|>{BP_DELTA_DBP_GUARD_MMHG:g} mmHg"
        )
        excluded.append(key)
    return (
        RealQCIssue(
            QC_BP_DELTA_ANOMALY,
            f"ΔSBP={sbp:g} mmHg and ΔDBP={dbp:g} mmHg exceed the declared guardrail",
            session.session_id,
        ),
    )


def _heart_rate_cross_check(
    session: RealSession, cross_module: dict[str, float | None]
) -> tuple[RealQCIssue, ...]:
    issues: list[RealQCIssue] = []
    pre_bp = session.endpoints.get("bp", {}).get(PHASE_PRE)
    post_bp = session.endpoints.get("bp", {}).get(PHASE_POST)
    pre_health = session.endpoints.get("health", {}).get(PHASE_PRE)
    post_health = session.endpoints.get("health", {}).get(PHASE_POST)
    if None in (pre_bp, post_bp, pre_health, post_health):
        return ()
    for phase, bp_endpoint, health_endpoint in (
        ("pre", pre_bp, pre_health),
        ("post", post_bp, post_health),
    ):
        bp_hr = bp_endpoint.measurement("heart_rate")
        health_hr = health_endpoint.measurement("heart_rate")
        if bp_hr is None or health_hr is None or not (bp_hr.valid and health_hr.valid):
            continue
        bp_value = float(bp_hr.value)
        health_value = float(health_hr.value)
        cross_module[f"hr_{phase}_bp"] = bp_value
        cross_module[f"hr_{phase}_health"] = health_value
        if abs(bp_value - health_value) > HR_CROSS_MODULE_TOLERANCE_BPM:
            issues.append(
                RealQCIssue(
                    QC_HR_CROSS_MODULE_MISMATCH,
                    f"{phase} heart rate differs by {abs(bp_value - health_value):g} bpm between "
                    "the blood-pressure and health modules",
                    session.session_id,
                )
            )
    return tuple(issues)


def session_feature_row(session: RealSession) -> FeatureRow:
    """Build the Pre-phase masked feature row for one real session."""
    values: dict[str, float] = {}
    for module in FEATURE_MODULE_PRIORITY:
        endpoint = session.endpoints.get(module, {}).get(PHASE_PRE)
        if endpoint is None:
            continue
        for measurement in endpoint.measurements:
            if measurement.metric in values:
                continue
            if not measurement.valid or measurement.value is None:
                continue
            values[measurement.metric] = _round(
                float(measurement.value) * REFERENCE_VALUE_SCALE.get(measurement.metric, 1.0)
            )
    for name, value in session.static_values.items():
        values.setdefault(name, _round(float(value)))

    names: list[str] = []
    numbers: list[float] = []
    observed: list[bool] = []
    valid: list[bool] = []
    imputed: list[bool] = []
    imputation: list[str] = []
    for name in REAL_PRE_FEATURE_NAMES:
        names.append(name)
        if name in values:
            numbers.append(float(values[name]))
            observed.append(True)
            valid.append(True)
            imputed.append(False)
            imputation.append(IMPUTATION_NONE)
        else:
            numbers.append(0.0)
            observed.append(False)
            valid.append(False)
            imputed.append(False)
            imputation.append(IMPUTATION_NONE)

    return FeatureRow(
        feature_names=tuple(names),
        values=tuple(numbers),
        observed_mask=tuple(observed),
        valid_mask=tuple(valid),
        imputed_mask=tuple(imputed),
        imputation=tuple(imputation),
        subject_ref=session.subject_ref,
        session_id=session.session_id,
        cutoff=session.cutoff,
    )


def session_post_state(session: RealSession) -> float | None:
    """Vendor Post proxy, only when the independent endpoint pair is verifiable."""
    if any(
        code in session.issue_codes()
        for code in (QC_RECORD_REUSED, QC_SESSION_OVERLAP, QC_OVERLAPPING_ENDPOINTS)
    ):
        return None
    endpoint = session.endpoints.get("health", {}).get(PHASE_POST)
    if endpoint is None:
        return None
    measurement = endpoint.measurement("fatigue_index")
    if measurement is None or not measurement.valid or measurement.value is None:
        return None
    pre = session.endpoints.get("health", {}).get(PHASE_PRE)
    if pre is None:
        return None
    pre_source = _metric_record(pre, "fatigue_index")
    post_source = _metric_record(endpoint, "fatigue_index")
    if not pre_source or not post_source or pre_source == post_source:
        return None
    return float(measurement.value) / 100.0


def reconcile_reference_contract(
    labels: RealSessionLabels, reference_mask: Iterable[str]
) -> RealSessionLabels:
    """Apply the project ABSTAIN rule to labels the reference contract excludes.

    A delta the raw records can produce but the frozen reference contract does
    not admit is marked inadmissible when no exclusion rule is declared
    (``规则缺项一律 ABSTAIN``). The derived value is preserved for audit; only
    admissibility changes, and the reason names the missing rule instead of
    inventing one.
    """
    declared = set(reference_mask)
    mask = dict(labels.mask)
    reasons = dict(labels.reasons)
    values = dict(labels.values)
    abstained: list[str] = []
    for key, admitted in labels.mask.items():
        if not admitted or key in declared:
            continue
        mask[key] = False
        reasons[key] = (
            "exclusion_rule_not_recoverable: the reference contract excludes this label "
            "and no exclusion rule is declared, so the label is not admitted"
        )
        abstained.append(key)
    return RealSessionLabels(
        session_id=labels.session_id,
        values=values,
        mask=mask,
        reasons=reasons,
        excluded_bp_labels=labels.excluded_bp_labels,
        cross_module_hr=labels.cross_module_hr,
        issues=labels.issues,
        abstained=tuple(sorted(abstained)),
    )


def import_real_export(
    path: str | Path, *, identity_trust: str = IDENTITY_SHARED
) -> tuple[RealSession, ...]:
    """Load and rebuild every session of a real long-table export."""
    return build_real_sessions(load_real_records(path), identity_trust=identity_trust)


def contract_coverage(session: RealSession) -> dict[str, dict[str, object]]:
    """Report how much of the frozen Pre-only input contract this extract realizes."""
    row = session_feature_row(session)
    usable = set(row.observed_features)
    report: dict[str, dict[str, object]] = {}
    for modality, columns in FROZEN_CONTRACT_COVERAGE.items():
        report[modality] = {
            "required": list(columns),
            "present": [name for name in columns if name in usable],
            "missing": [name for name in columns if name not in usable],
            "excluded": [
                {"feature": name, "reason": QC_CODED_FIELD_SEMANTICS_UNVERIFIED}
                for name in FROZEN_CONTRACT_UNUSED.get(modality, ())
            ],
        }
    return report


def collect_issues(sessions: Sequence[RealSession]) -> tuple[RealQCIssue, ...]:
    """Every QC issue across sessions plus every label-level issue."""
    collected: list[RealQCIssue] = [issue for session in sessions for issue in session.issues]
    for session in sessions:
        collected.extend(session_labels(session).issues)
    return tuple(collected)
