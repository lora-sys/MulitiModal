"""Versioned importer for legacy offline session exports (#96/#95).

The legacy export is *research input only*. It is never a realtime transport and
never a reason to trust a vendor summary field. This module freezes the raw
field -> canonical field mapping, then imports rows into the same
``PhysioEventV1`` / ``PhysioSessionV1`` contract used by device and service
adapters.

No real export is committed to the repository. Tests use a synthetic 94-column
fixture that exercises identity collisions, snapshot/completed conflicts,
Pre/Post masquerading, duplicates, out-of-order arrival, timezone/cross-day
boundaries and EEG/duration conflicts. Real field names remain
``BLOCKED_BY_DATA_OR_DEVICE`` until a licensed export and field dictionary are
supplied.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .metrics import is_known_metric, route_measurement
from .schema import (
    PHASES,
    QUALITY_DEGRADED,
    SOURCE_OFFLINE,
    MeasurementV1,
    PhysioEventV1,
    PhysioSessionV1,
    ProtocolContextV1,
    ProvenanceV1,
    SchemaError,
    assemble_session,
)

EXPORT_SCHEMA_VERSION = "physio-legacy-export-v1"
IMPORT_SCHEMA_VERSION = "physio-legacy-import-v1"

FIELD_ROLES = (
    "identity",
    "session",
    "phase",
    "timestamp",
    "program",
    "metric",
    "summary",
    "eeg",
    "ignored",
)
SOURCE_KINDS = ("snapshot", "completed", "both")
VALUE_ORIGINS = ("measured", "imputed", "derived", "unknown")
LABEL_ROLES = ("y_delta_input", "y_pri_input", "none")

IDENTITY_TRUSTED = "TRUSTED"
IDENTITY_SHARED = "SHARED_ACCOUNT_UNKNOWN"
IDENTITY_UNKNOWN = "UNKNOWN"
IDENTITY_TRUST_VALUES = (IDENTITY_TRUSTED, IDENTITY_SHARED, IDENTITY_UNKNOWN)

ISSUE_UNKNOWN_FIELD = "unknown_field"
ISSUE_AMBIGUOUS_FIELD = "ambiguous_field_semantics"
ISSUE_MISSING_FIELD = "missing_required_field"
ISSUE_NAIVE_TIMESTAMP = "naive_timestamp"
ISSUE_REVERSED_TIMESTAMP = "reversed_timestamp"
ISSUE_INVALID_TIMESTAMP = "invalid_timestamp"
ISSUE_UNKNOWN_PHASE = "unknown_phase"
ISSUE_PHASE_MISMATCH = "phase_mismatch"
ISSUE_SOURCE_KIND_CONFLICT = "source_kind_conflict"
ISSUE_SNAPSHOT_SUPERSEDED = "snapshot_superseded"
ISSUE_SNAPSHOT_COMPLETED_MISMATCH = "snapshot_completed_mismatch"
ISSUE_DUPLICATE_RECORD = "duplicate_record"
ISSUE_PHASE_CONFLICT = "phase_conflict"
ISSUE_SEQUENCE_COLLISION = "sequence_collision"
ISSUE_SESSION_COLLISION = "session_identity_collision"
ISSUE_MISSING_SESSION_KEY = "missing_session_key"
ISSUE_IDENTITY_UNKNOWN = "identity_not_independently_verified"
ISSUE_EEG_TIME_CONFLICT = "eeg_time_conflict"
ISSUE_DURATION_CONFLICT = "duration_conflict"
ISSUE_CROSS_DAY = "cross_day_session"
ISSUE_IMPUTED_VALUE = "imputed_value"
ISSUE_DERIVED_VALUE = "derived_value"
ISSUE_NON_FINITE = "non_finite_value"
ISSUE_INVALID_VALUE = "invalid_protocol_value"


class ExportImportError(ValueError):
    """Raised when the registry or import request is structurally invalid."""


@dataclass(frozen=True)
class ExportFieldSpec:
    """One raw export column and its frozen semantic mapping."""

    raw_name: str
    role: str
    canonical_metric: str | None = None
    phase: str | None = None
    unit: str | None = None
    source_kind: str = "both"
    value_origin: str = "measured"
    label_role: str = "none"
    required: bool = False
    ambiguous_semantics: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.raw_name, str) or not self.raw_name.strip():
            raise ExportImportError("raw_name must be a non-empty string")
        if self.role not in FIELD_ROLES:
            raise ExportImportError(f"unknown field role: {self.role!r}")
        if self.source_kind not in SOURCE_KINDS:
            raise ExportImportError(f"unknown source_kind: {self.source_kind!r}")
        if self.value_origin not in VALUE_ORIGINS:
            raise ExportImportError(f"unknown value_origin: {self.value_origin!r}")
        if self.label_role not in LABEL_ROLES:
            raise ExportImportError(f"unknown label_role: {self.label_role!r}")
        if self.phase is not None and self.phase not in PHASES:
            raise ExportImportError(f"unknown phase: {self.phase!r}")
        if self.role == "metric":
            if self.ambiguous_semantics and self.canonical_metric is not None:
                raise ExportImportError(
                    "an ambiguous field must not declare a canonical metric"
                )
            if not self.ambiguous_semantics:
                if self.canonical_metric is None or not is_known_metric(
                    self.canonical_metric
                ):
                    raise ExportImportError(
                        f"metric field {self.raw_name!r} needs a known canonical_metric"
                    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_name": self.raw_name,
            "role": self.role,
            "canonical_metric": self.canonical_metric,
            "phase": self.phase,
            "unit": self.unit,
            "source_kind": self.source_kind,
            "value_origin": self.value_origin,
            "label_role": self.label_role,
            "required": self.required,
            "ambiguous_semantics": self.ambiguous_semantics,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class ExportFieldRegistry:
    """Frozen, versioned set of export columns."""

    version: str
    fields: tuple[ExportFieldSpec, ...]
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version.strip():
            raise ExportImportError("registry version must be a non-empty string")
        if not self.fields:
            raise ExportImportError("registry needs at least one field")
        names = [item.raw_name for item in self.fields]
        if len(set(names)) != len(names):
            raise ExportImportError("registry raw_name values must be unique")
        object.__setattr__(self, "fields", tuple(self.fields))

    @property
    def by_raw(self) -> dict[str, ExportFieldSpec]:
        return {item.raw_name: item for item in self.fields}

    def payload(self) -> dict[str, Any]:
        return {
            "schema": EXPORT_SCHEMA_VERSION,
            "version": self.version,
            "description": self.description,
            "fields": [item.to_dict() for item in self.fields],
        }

    def digest(self) -> str:
        encoded = json.dumps(
            self.payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "sha256": self.digest()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExportFieldRegistry:
        if not isinstance(data, Mapping):
            raise ExportImportError("registry must be a mapping")
        if data.get("schema") != EXPORT_SCHEMA_VERSION:
            raise ExportImportError(
                f"unsupported registry schema: {data.get('schema')!r}"
            )
        raw_fields = data.get("fields")
        if not isinstance(raw_fields, (list, tuple)):
            raise ExportImportError("registry fields must be a list")
        try:
            fields = tuple(ExportFieldSpec(**dict(item)) for item in raw_fields)
        except TypeError as exc:
            raise ExportImportError(f"malformed registry field: {exc}") from exc
        registry = cls(
            version=str(data["version"]),
            fields=fields,
            description=str(data.get("description", "")),
        )
        digest = data.get("sha256")
        if digest is not None and digest != registry.digest():
            raise ExportImportError("registry sha256 mismatch (corrupt or edited)")
        return registry

    @classmethod
    def from_json_text(cls, text: str) -> ExportFieldRegistry:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ExportImportError(f"registry is not valid JSON: {exc}") from exc
        return cls.from_dict(data)


@dataclass(frozen=True)
class ImportIssue:
    row: int
    status: str
    raw_name: str | None
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "row": self.row,
            "status": self.status,
            "raw_name": self.raw_name,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class LegacyImportResult:
    """Deterministic import output plus explicit QC issues."""

    schema_version: str
    registry_version: str
    registry_digest: str
    identity_trust: str
    sessions: tuple[PhysioSessionV1, ...]
    issues: tuple[ImportIssue, ...]
    rows_seen: int
    rows_accepted: int
    rows_rejected: int
    imputed_metrics: dict[str, tuple[str, ...]] = field(default_factory=dict)
    reproducibility_digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != IMPORT_SCHEMA_VERSION:
            raise ExportImportError(f"unsupported import schema: {self.schema_version!r}")
        if self.identity_trust not in IDENTITY_TRUST_VALUES:
            raise ExportImportError(f"unknown identity trust: {self.identity_trust!r}")
        object.__setattr__(self, "sessions", tuple(self.sessions))
        object.__setattr__(self, "issues", tuple(self.issues))
        object.__setattr__(
            self,
            "imputed_metrics",
            {key: tuple(value) for key, value in self.imputed_metrics.items()},
        )

    @property
    def trusted_subject_identity(self) -> bool:
        return self.identity_trust == IDENTITY_TRUSTED

    def issue_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for issue in self.issues:
            counts[issue.status] = counts.get(issue.status, 0) + 1
        return dict(sorted(counts.items()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "registry_version": self.registry_version,
            "registry_digest": self.registry_digest,
            "identity_trust": self.identity_trust,
            "trusted_subject_identity": self.trusted_subject_identity,
            "rows_seen": self.rows_seen,
            "rows_accepted": self.rows_accepted,
            "rows_rejected": self.rows_rejected,
            "issue_counts": self.issue_counts(),
            "issues": [issue.to_dict() for issue in self.issues],
            "sessions": [session.to_dict() for session in self.sessions],
            "imputed_metrics": {
                key: list(value) for key, value in self.imputed_metrics.items()
            },
            "reproducibility_digest": self.reproducibility_digest,
        }

    def qc_report(self) -> dict[str, Any]:
        """Compact, reproducible QC summary safe to attach to a private run."""
        return {
            "schema_version": IMPORT_SCHEMA_VERSION,
            "registry_version": self.registry_version,
            "registry_digest": self.registry_digest,
            "identity_trust": self.identity_trust,
            "rows_seen": self.rows_seen,
            "rows_accepted": self.rows_accepted,
            "rows_rejected": self.rows_rejected,
            "sessions": len(self.sessions),
            "events": sum(len(session) for session in self.sessions),
            "issue_counts": self.issue_counts(),
            "reproducibility_digest": self.reproducibility_digest,
        }


def _parse_datetime(
    value: Any, row: int, raw_name: str, issues: list[ImportIssue]
) -> datetime | None:
    if value is None or isinstance(value, bool):
        issues.append(ImportIssue(row, ISSUE_MISSING_FIELD, raw_name, "timestamp is required"))
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        issues.append(
            ImportIssue(row, ISSUE_INVALID_TIMESTAMP, raw_name, f"cannot parse {value!r}")
        )
        return None
    if parsed.tzinfo is None:
        issues.append(
            ImportIssue(row, ISSUE_NAIVE_TIMESTAMP, raw_name, "timestamp lacks a timezone")
        )
        return None
    return parsed


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _measurement(
    spec: ExportFieldSpec,
    raw_value: Any,
    row: int,
    issues: list[ImportIssue],
    imputed: set[str],
) -> MeasurementV1 | None:
    if spec.ambiguous_semantics:
        issues.append(
            ImportIssue(
                row,
                ISSUE_AMBIGUOUS_FIELD,
                spec.raw_name,
                spec.notes or "field semantics are not confirmed by a field dictionary",
            )
        )
        return None
    if raw_value is None or raw_value == "":
        return None
    if isinstance(raw_value, float) and not math.isfinite(raw_value):
        issues.append(
            ImportIssue(row, ISSUE_NON_FINITE, spec.raw_name, "NaN/Inf is not a measurement")
        )
        return MeasurementV1(
            metric=spec.canonical_metric or spec.raw_name,
            value=None,
            unit=spec.unit or "",
            data_type="float",
            valid=False,
            status=ISSUE_NON_FINITE,
            raw_metric=spec.raw_name,
            raw_value=raw_value,
            raw_unit=spec.unit,
        )
    measurement = route_measurement(spec.canonical_metric or spec.raw_name, raw_value, spec.unit)
    if spec.value_origin == "imputed":
        imputed.add(measurement.metric)
        issues.append(
            ImportIssue(
                row,
                ISSUE_IMPUTED_VALUE,
                spec.raw_name,
                "value is imputed in the export; it must never create a label",
            )
        )
        measurement = route_measurement(
            spec.canonical_metric or spec.raw_name, raw_value, spec.unit
        )
        return MeasurementV1(
            **{
                **measurement.__dict__,
                "notes": (*measurement.notes, "value_origin=imputed"),
            }
        )
    if spec.value_origin == "derived":
        issues.append(
            ImportIssue(
                row,
                ISSUE_DERIVED_VALUE,
                spec.raw_name,
                "derived value is kept for audit but is not a raw measurement",
            )
        )
    return measurement


def _row_value(row: Mapping[str, Any], name: str) -> Any:
    return row.get(name)


def _json_safe(value: Any) -> Any:
    """Replace non-finite floats with stable tokens so a QC row stays hashable."""
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _json_safe(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _session_subject(member_id: str, identity_trust: str) -> str:
    prefix = (
        "shared-account"
        if identity_trust in (IDENTITY_SHARED, IDENTITY_UNKNOWN)
        else "subject"
    )
    return f"{prefix}:{member_id}"


def _session_identifier(prefix: str, member_id: str, session_key: str) -> str:
    return f"{prefix}-{member_id}-{session_key}"


def _normalise_rows(
    rows: Sequence[Mapping[str, Any]], issues: list[ImportIssue]
) -> list[tuple[int, dict[str, Any]]]:
    normalised: list[tuple[int, dict[str, Any]]] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            issues.append(
                ImportIssue(row_index, ISSUE_UNKNOWN_FIELD, None, "row is not a mapping")
            )
            continue
        normalised.append((row_index, dict(row)))
    return normalised


def import_legacy_rows(
    rows: Sequence[Mapping[str, Any]],
    registry: ExportFieldRegistry,
    *,
    identity_trust: str = IDENTITY_UNKNOWN,
    session_id_prefix: str = "legacy",
    max_span_s: float = 24 * 3600,
    eeg_duration_tolerance_s: float = 5.0,
) -> LegacyImportResult:
    """Import synthetic or privately supplied rows into canonical sessions.

    The function never invents a value, never repairs an identity collision and
    never uses an imputed/derived value for a label. Every rejection is recorded
    with its row number and raw field name.
    """
    if identity_trust not in IDENTITY_TRUST_VALUES:
        raise ExportImportError(f"unknown identity_trust: {identity_trust!r}")
    if not isinstance(session_id_prefix, str) or not session_id_prefix.strip():
        raise ExportImportError("session_id_prefix must be a non-empty string")
    if max_span_s <= 0 or eeg_duration_tolerance_s < 0:
        raise ExportImportError("span and EEG tolerance must be non-negative")

    by_raw = registry.by_raw
    issues: list[ImportIssue] = []
    normalised = _normalise_rows(rows, issues)

    if identity_trust != IDENTITY_TRUSTED:
        issues.append(
            ImportIssue(
                -1,
                ISSUE_IDENTITY_UNKNOWN,
                "member_id",
                "shared/unknown account: samples are pipeline regression only, "
                "not independent subjects",
            )
        )

    # Freeze session identity before any metric parsing.
    session_identity: dict[str, tuple[str, str]] = {}
    session_id_by_key: dict[tuple[str, str], str] = {}
    used_session_ids: dict[str, tuple[str, str]] = {}
    parsed: list[dict[str, Any]] = []

    for row_index, row in normalised:
        member_id = str(row.get("member_id") or "").strip()
        session_key = str(row.get("session_key") or "").strip()
        record_id = str(row.get("record_id") or "").strip()
        if not member_id:
            issues.append(
                ImportIssue(row_index, ISSUE_MISSING_FIELD, "member_id", "member_id is required")
            )
            continue
        if not session_key:
            issues.append(
                ImportIssue(
                    row_index,
                    ISSUE_MISSING_SESSION_KEY,
                    "session_key",
                    "a stable session key is required; refusing to merge by row order",
                )
            )
            continue
        if not record_id:
            issues.append(
                ImportIssue(row_index, ISSUE_MISSING_FIELD, "record_id", "record_id is required")
            )
            continue

        identity_key = (member_id, session_key)
        requested_id = str(row.get("session_id") or "").strip()
        session_id = requested_id or session_id_by_key.get(identity_key)
        if session_id is None:
            session_id = _session_identifier(session_id_prefix, member_id, session_key)
        owner = used_session_ids.get(session_id)
        if owner is not None and owner != identity_key:
            suffix = hashlib.sha256(f"{member_id}:{session_key}".encode()).hexdigest()[:8]
            session_id = f"{session_id}-{suffix}"
            issues.append(
                ImportIssue(
                    row_index,
                    ISSUE_SESSION_COLLISION,
                    "session_id",
                    "session_id is already bound to another member/session; suffixed instead "
                    "of merging rows",
                )
            )
        used_session_ids.setdefault(session_id, identity_key)
        session_identity[session_id] = identity_key
        session_id_by_key.setdefault(identity_key, session_id)

        record_kind = str(row.get("record_kind") or "completed").strip().lower()
        if record_kind not in ("snapshot", "completed"):
            issues.append(
                ImportIssue(
                    row_index,
                    ISSUE_SOURCE_KIND_CONFLICT,
                    "record_kind",
                    f"unknown record_kind {record_kind!r}",
                )
            )
            continue
        phase = str(row.get("phase") or "").strip().lower()
        if phase not in PHASES:
            issues.append(
                ImportIssue(row_index, ISSUE_UNKNOWN_PHASE, "phase", f"unknown phase {phase!r}")
            )
            continue
        measured_at = _parse_datetime(
            _row_value(row, "measured_at"), row_index, "measured_at", issues
        )
        received_at = _parse_datetime(
            _row_value(row, "received_at") or _row_value(row, "measured_at"),
            row_index,
            "received_at",
            issues,
        )
        if measured_at is None or received_at is None:
            continue
        if received_at < measured_at:
            issues.append(
                ImportIssue(
                    row_index,
                    ISSUE_REVERSED_TIMESTAMP,
                    "received_at",
                    "received_at precedes measured_at",
                )
            )
            continue
        try:
            sequence = int(row.get("sequence", row_index))
        except (TypeError, ValueError):
            issues.append(
                ImportIssue(
                    row_index,
                    ISSUE_MISSING_FIELD,
                    "sequence",
                    "sequence must be an integer",
                )
            )
            continue
        if isinstance(row.get("sequence"), bool) or sequence < 0:
            issues.append(
                ImportIssue(
                    row_index,
                    ISSUE_MISSING_FIELD,
                    "sequence",
                    "sequence must be non-negative",
                )
            )
            continue
        parsed.append(
            {
                "row": row_index,
                "session_id": session_id,
                "member_id": member_id,
                "session_key": session_key,
                "record_id": record_id,
                "record_kind": record_kind,
                "phase": phase,
                "measured_at": measured_at,
                "received_at": received_at,
                "sequence": sequence,
                "raw": row,
            }
        )

    # Resolve snapshot/completed and duplicate records before constructing events.
    by_record: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for item in parsed:
        by_record.setdefault(
            (item["session_id"], item["phase"], item["record_id"]), []
        ).append(item)

    selected: list[dict[str, Any]] = []
    for _key, group in sorted(by_record.items()):
        completed = [item for item in group if item["record_kind"] == "completed"]
        snapshots = [item for item in group if item["record_kind"] == "snapshot"]
        if len(group) > 1:
            for item in group:
                status = (
                    ISSUE_SNAPSHOT_SUPERSEDED
                    if item["record_kind"] == "snapshot" and completed
                    else ISSUE_DUPLICATE_RECORD
                )
                if len(completed) > 1 or (not completed and len(snapshots) > 1):
                    status = ISSUE_DUPLICATE_RECORD
                issues.append(
                    ImportIssue(
                        item["row"],
                        status,
                        "record_id",
                        "duplicate record; completed/latest arrival wins deterministically",
                    )
                )
        pool = completed or snapshots
        chosen = max(
            pool,
            key=lambda item: (
                item["received_at"],
                item["sequence"],
                _canonical_digest(item["raw"]),
            ),
        )
        if completed and snapshots:
            snapshot_values = snapshots[0]["raw"]
            completed_values = chosen["raw"]
            metric_names = [
                spec.raw_name
                for spec in registry.fields
                if spec.role == "metric" and not spec.ambiguous_semantics
            ]
            if any(
                snapshot_values.get(name) != completed_values.get(name)
                for name in metric_names
            ):
                issues.append(
                    ImportIssue(
                        chosen["row"],
                        ISSUE_SNAPSHOT_COMPLETED_MISMATCH,
                        "record_kind",
                        "snapshot and completed metrics differ; completed wins, snapshot kept "
                        "only in QC",
                    )
                )
        for item in group:
            if item is chosen:
                continue
            if item["record_kind"] == "completed":
                issues.append(
                    ImportIssue(
                        item["row"],
                        ISSUE_DUPLICATE_RECORD,
                        "record_id",
                        "later completed record superseded by deterministic arrival tie-break",
                    )
                )
        selected.append(chosen)

    # Reject records whose id is reused across phases (Pre/Post masquerading).
    record_phases: dict[tuple[str, str], set[str]] = {}
    for item in selected:
        record_phases.setdefault((item["session_id"], item["record_id"]), set()).add(
            item["phase"]
        )
    conflicting_records = {
        key for key, phases in record_phases.items() if len(phases) > 1
    }
    if conflicting_records:
        kept: list[dict[str, Any]] = []
        for item in selected:
            key = (item["session_id"], item["record_id"])
            if key in conflicting_records:
                issues.append(
                    ImportIssue(
                        item["row"],
                        ISSUE_PHASE_CONFLICT,
                        "record_id",
                        "same record_id appears in multiple phases; rejecting the whole record",
                    )
                )
                continue
            kept.append(item)
        selected = kept

    # Detect sequence collisions between distinct records.
    by_sequence: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for item in selected:
        by_sequence.setdefault((item["session_id"], item["sequence"]), []).append(item)
    for _identity, group in sorted(by_sequence.items()):
        if len(group) > 1:
            for item in group[1:]:
                issues.append(
                    ImportIssue(
                        item["row"],
                        ISSUE_SEQUENCE_COLLISION,
                        "sequence",
                        "distinct records share a sequence; refusing to overwrite one",
                    )
                )
            selected = [item for item in selected if item not in group[1:]]

    # Build events.
    events_by_session: dict[str, list[PhysioEventV1]] = {}
    imputed_by_session: dict[str, set[str]] = {}
    accepted_rows = 0
    rejected_rows = 0

    for item in sorted(
        selected, key=lambda value: (value["session_id"], value["measured_at"], value["sequence"])
    ):
        row = item["raw"]
        imputed: set[str] = set()
        measurements: list[MeasurementV1] = []
        eeg_started = _row_value(row, "eeg_started_at")
        eeg_ended = _row_value(row, "eeg_ended_at")
        eeg_duration = _row_value(row, "eeg_duration_s")
        if eeg_started is not None and eeg_ended is not None:
            started = _parse_datetime(eeg_started, item["row"], "eeg_started_at", issues)
            ended = _parse_datetime(eeg_ended, item["row"], "eeg_ended_at", issues)
            if started is not None and ended is not None:
                if ended < started:
                    issues.append(
                        ImportIssue(
                            item["row"],
                            ISSUE_EEG_TIME_CONFLICT,
                            "eeg_ended_at",
                            "EEG end precedes start",
                        )
                    )
                elif _is_finite_number(eeg_duration):
                    span = (ended - started).total_seconds()
                    if abs(span - float(eeg_duration)) > eeg_duration_tolerance_s:
                        issues.append(
                            ImportIssue(
                                item["row"],
                                ISSUE_DURATION_CONFLICT,
                                "eeg_duration_s",
                                f"EEG duration {eeg_duration} conflicts with timestamps "
                                f"({span:g}s)",
                            )
                        )

        for raw_name, raw_value in row.items():
            if raw_name in {
                "member_id",
                "session_id",
                "session_key",
                "record_id",
                "record_kind",
                "phase",
                "measured_at",
                "received_at",
                "sequence",
                "device_ref",
                "program_id",
                "intensity",
                "duration_s",
                "catalog_version",
                "eeg_started_at",
                "eeg_ended_at",
                "eeg_duration_s",
            }:
                continue
            spec = by_raw.get(raw_name)
            if spec is None:
                if raw_value not in (None, ""):
                    issues.append(
                        ImportIssue(
                            item["row"],
                            ISSUE_UNKNOWN_FIELD,
                            raw_name,
                            "raw field is not in the frozen registry; value ignored",
                        )
                    )
                continue
            if spec.role != "metric":
                # EEG and summary fields are audit-only: they are never imported
                # as measurements, so they cannot leak into features or labels.
                continue
            if spec.phase is not None and spec.phase != item["phase"]:
                if raw_value not in (None, ""):
                    issues.append(
                        ImportIssue(
                            item["row"],
                            ISSUE_PHASE_MISMATCH,
                            raw_name,
                            f"field is declared for phase {spec.phase!r} but row is "
                            f"{item['phase']!r}",
                        )
                    )
                continue
            measurement = _measurement(spec, raw_value, item["row"], issues, imputed)
            if measurement is not None:
                measurements.append(measurement)

        if not measurements:
            issues.append(
                ImportIssue(
                    item["row"],
                    ISSUE_MISSING_FIELD,
                    "measurements",
                    "row produced no canonical measurement",
                )
            )
            rejected_rows += 1
            continue

        raw_intensity = _row_value(row, "intensity")
        raw_duration = _row_value(row, "duration_s")
        try:
            intensity = int(raw_intensity) if raw_intensity not in (None, "") else None
        except (TypeError, ValueError):
            issues.append(
                ImportIssue(
                    item["row"],
                    ISSUE_INVALID_VALUE,
                    "intensity",
                    "intensity must be an integer",
                )
            )
            intensity = None
        try:
            duration = int(raw_duration) if raw_duration not in (None, "") else None
        except (TypeError, ValueError):
            issues.append(
                ImportIssue(
                    item["row"],
                    ISSUE_INVALID_VALUE,
                    "duration_s",
                    "duration must be an integer",
                )
            )
            duration = None
        if intensity is not None and not 0 <= intensity <= 10:
            issues.append(
                ImportIssue(
                    item["row"],
                    ISSUE_INVALID_VALUE,
                    "intensity",
                    "intensity outside 0..10",
                )
            )
            intensity = None
        if duration is not None and duration <= 0:
            issues.append(
                ImportIssue(
                    item["row"],
                    ISSUE_INVALID_VALUE,
                    "duration_s",
                    "duration must be positive",
                )
            )
            duration = None
        protocol = ProtocolContextV1(
            program_id=_row_value(row, "program_id"),
            intensity=intensity,
            duration_s=duration,
            catalog_version=_row_value(row, "catalog_version"),
        )
        has_imputed = bool(imputed)
        quality = QUALITY_DEGRADED if has_imputed else ""
        provenance = ProvenanceV1(
            adapter="legacy-export",
            adapter_version=EXPORT_SCHEMA_VERSION,
            source_record=f"{item['session_key']}/{item['record_id']}/{item['record_kind']}",
            data_license="PRIVATE_NOT_FOR_COMMIT",
            notes=(
                f"registry={registry.version}",
                f"value_origin={'imputed' if has_imputed else 'measured'}",
            ),
        )
        try:
            event = PhysioEventV1(
                session_id=item["session_id"],
                subject_ref=_session_subject(item["member_id"], identity_trust),
                device_ref=str(row.get("device_ref") or "legacy-export"),
                source=SOURCE_OFFLINE,
                phase=item["phase"],
                measured_at=item["measured_at"],
                received_at=item["received_at"],
                sequence=item["sequence"],
                measurements=tuple(measurements),
                quality=quality,
                protocol=protocol,
                provenance=provenance,
            )
        except SchemaError as exc:
            issues.append(
                ImportIssue(item["row"], ISSUE_MISSING_FIELD, None, f"event rejected: {exc}")
            )
            rejected_rows += 1
            continue
        events_by_session.setdefault(item["session_id"], []).append(event)
        if imputed:
            imputed_by_session.setdefault(item["session_id"], set()).update(imputed)
        accepted_rows += 1

    sessions: list[PhysioSessionV1] = []
    for session_id, events in sorted(events_by_session.items()):
        member_id, _session_key = session_identity[session_id]
        sessions.append(
            assemble_session(
                session_id=session_id,
                subject_ref=_session_subject(member_id, identity_trust),
                device_ref=events[0].device_ref,
                events=events,
                source=SOURCE_OFFLINE,
                provenance=ProvenanceV1(
                    adapter="legacy-export",
                    adapter_version=EXPORT_SCHEMA_VERSION,
                    source_record=session_id,
                    data_license="PRIVATE_NOT_FOR_COMMIT",
                    notes=(f"registry={registry.version}", f"registry_sha256={registry.digest()}"),
                ),
            )
        )

    # Cross-day / excessive-span diagnostic (informational, never a silent fix).
    for session in sessions:
        if not session.events:
            continue
        span = (session.events[-1].measured_at - session.events[0].measured_at).total_seconds()
        if span > max_span_s:
            # Strictly greater: a session whose span is exactly max_span_s is
            # still accepted; the diagnostic flags only spans beyond the bound.
            issues.append(
                ImportIssue(
                    -1,
                    ISSUE_CROSS_DAY,
                    "measured_at",
                    f"session {session.session_id} spans {span:g}s; split by session_key if "
                    "this crosses a day boundary",
                )
            )

    payload = {
        "registry_digest": registry.digest(),
        "identity_trust": identity_trust,
        "events": [
            event.to_dict()
            for session in sessions
            for event in session.events
        ],
        "issues": [
            {"status": issue.status, "raw_name": issue.raw_name, "detail": issue.detail}
            for issue in sorted(
                issues, key=lambda item: (item.status, item.raw_name or "", item.detail)
            )
        ],
    }
    return LegacyImportResult(
        schema_version=IMPORT_SCHEMA_VERSION,
        registry_version=registry.version,
        registry_digest=registry.digest(),
        identity_trust=identity_trust,
        sessions=tuple(sessions),
        issues=tuple(sorted(issues, key=lambda item: (item.row, item.status))),
        rows_seen=len(normalised),
        rows_accepted=accepted_rows,
        rows_rejected=rejected_rows,
        imputed_metrics={
            session_id: tuple(sorted(metrics))
            for session_id, metrics in sorted(imputed_by_session.items())
        },
        reproducibility_digest=_canonical_digest(payload),
    )


__all__ = [
    "EXPORT_SCHEMA_VERSION",
    "IDENTITY_SHARED",
    "IDENTITY_TRUSTED",
    "IDENTITY_TRUST_VALUES",
    "IDENTITY_UNKNOWN",
    "IMPORT_SCHEMA_VERSION",
    "ExportFieldRegistry",
    "ExportFieldSpec",
    "ExportImportError",
    "ImportIssue",
    "LegacyImportResult",
    "import_legacy_rows",
]
