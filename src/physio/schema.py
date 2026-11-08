"""Versioned physiological event and session contracts (#96).

The same protocol is produced by device, vendor-service and offline adapters.
A CSV export is research input only: ``PhysioSessionV1`` never becomes a
realtime transport requirement.

Field-level provenance is explicit. A measurement carries the raw value/unit
it arrived with plus the canonical metric/unit used by downstream code, so an
unreliable or unknown field is preserved as ``valid=False`` instead of being
silently converted into a plausible normal value.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Any

EVENT_SCHEMA_VERSION = "physio-event-v1"
SESSION_SCHEMA_VERSION = "physio-session-v1"

PHASE_PRE = "pre"
PHASE_DURING = "during"
PHASE_POST = "post"
PHASES = (PHASE_PRE, PHASE_DURING, PHASE_POST)

SOURCE_DEVICE = "device"
SOURCE_SERVICE = "service"
SOURCE_OFFLINE = "offline"
SOURCES = (SOURCE_DEVICE, SOURCE_SERVICE, SOURCE_OFFLINE)

QUALITY_OK = "ok"
QUALITY_DEGRADED = "degraded"
QUALITY_REJECTED = "rejected"
QUALITY_STALE = "stale"
QUALITY_VALUES = (QUALITY_OK, QUALITY_DEGRADED, QUALITY_REJECTED, QUALITY_STALE)

DATA_TYPE_FLOAT = "float"
DATA_TYPE_INT = "int"
DATA_TYPE_BOOL = "bool"
DATA_TYPE_STRING = "string"
DATA_TYPES = (DATA_TYPE_FLOAT, DATA_TYPE_INT, DATA_TYPE_BOOL, DATA_TYPE_STRING)

MEASUREMENT_ACCEPTED = "accepted"

#: Default mapping of a measured value to a canonical value. This is a
#: contract, not a clinical interpretation: an accepted value only means the
#: field parsed and passed engineering QC.
DEFAULT_CLOCK_SKEW_TOLERANCE = timedelta(seconds=5)


class SchemaError(ValueError):
    """Raised when an event or session violates the versioned contract."""


@dataclass(frozen=True)
class ProvenanceV1:
    """Where an event came from. Raw subject data must never be committed."""

    adapter: str
    adapter_version: str
    source_record: str | None = None
    raw_sha256: str | None = None
    data_license: str = "PRIVATE_NOT_FOR_COMMIT"
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.adapter or not self.adapter_version:
            raise SchemaError("provenance adapter and adapter_version must be non-empty")
        if self.raw_sha256 is not None and len(self.raw_sha256) != 64:
            raise SchemaError("raw_sha256 must be a 64-character sha256 hex digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter,
            "adapter_version": self.adapter_version,
            "source_record": self.source_record,
            "raw_sha256": self.raw_sha256,
            "data_license": self.data_license,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProvenanceV1:
        return cls(
            adapter=str(data["adapter"]),
            adapter_version=str(data["adapter_version"]),
            source_record=data.get("source_record"),
            raw_sha256=data.get("raw_sha256"),
            data_license=data.get("data_license", "PRIVATE_NOT_FOR_COMMIT"),
            notes=tuple(data.get("notes", ())),
        )


@dataclass(frozen=True)
class ProtocolContextV1:
    """Optional massage-program context. Validation against a capability
    catalog belongs to the program layer (#98/#99); this is only the envelope."""

    program_id: str | None = None
    intensity: int | None = None
    duration_s: int | None = None
    mode: str | None = None
    catalog_version: str | None = None

    def __post_init__(self) -> None:
        if self.intensity is not None and not 0 <= self.intensity <= 10:
            raise SchemaError(f"protocol intensity must be within 0..10, got {self.intensity}")
        if self.duration_s is not None and self.duration_s <= 0:
            raise SchemaError("protocol duration_s must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "program_id": self.program_id,
            "intensity": self.intensity,
            "duration_s": self.duration_s,
            "mode": self.mode,
            "catalog_version": self.catalog_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ProtocolContextV1 | None:
        if data is None:
            return None
        return cls(
            program_id=data.get("program_id"),
            intensity=data.get("intensity"),
            duration_s=data.get("duration_s"),
            mode=data.get("mode"),
            catalog_version=data.get("catalog_version"),
        )


@dataclass(frozen=True)
class MeasurementV1:
    """One field after routing.

    ``metric`` is canonical only when routing recognised it. ``raw_metric``,
    ``raw_value`` and ``raw_unit`` are always retained for audit. ``valid`` is
    the single authority for whether downstream code may use ``value``.
    """

    metric: str
    value: float | int | bool | str | None
    unit: str
    data_type: str
    valid: bool
    status: str
    raw_metric: str | None = None
    raw_value: Any = None
    raw_unit: str | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.metric:
            raise SchemaError("measurement metric must be non-empty")
        if self.data_type not in DATA_TYPES:
            raise SchemaError(f"unknown measurement data_type: {self.data_type!r}")
        if self.valid and self.value is None:
            raise SchemaError("a valid measurement must carry a value")
        if not self.valid and self.value is not None:
            raise SchemaError("an invalid measurement must not expose a canonical value")
        if self.valid and self.status != MEASUREMENT_ACCEPTED:
            raise SchemaError("valid measurements must have status 'accepted'")

    @property
    def observed(self) -> bool:
        return self.valid

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "value": self.value,
            "unit": self.unit,
            "data_type": self.data_type,
            "valid": self.valid,
            "status": self.status,
            "raw_metric": self.raw_metric,
            "raw_value": self.raw_value,
            "raw_unit": self.raw_unit,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MeasurementV1:
        return cls(
            metric=str(data["metric"]),
            value=data.get("value"),
            unit=str(data.get("unit", "")),
            data_type=str(data.get("data_type", DATA_TYPE_FLOAT)),
            valid=bool(data["valid"]),
            status=str(data["status"]),
            raw_metric=data.get("raw_metric"),
            raw_value=data.get("raw_value"),
            raw_unit=data.get("raw_unit"),
            notes=tuple(data.get("notes", ())),
        )


def event_quality(measurements: tuple[MeasurementV1, ...]) -> str:
    """Derive event quality from field validity, never from a vendor score."""
    if not measurements:
        return QUALITY_REJECTED
    accepted = sum(1 for m in measurements if m.valid)
    if accepted == len(measurements):
        return QUALITY_OK
    if accepted == 0:
        return QUALITY_REJECTED
    return QUALITY_DEGRADED


@dataclass(frozen=True)
class PhysioEventV1:
    """One timestamped event from any source."""

    session_id: str
    subject_ref: str
    device_ref: str
    source: str
    phase: str
    measured_at: datetime
    received_at: datetime
    sequence: int
    measurements: tuple[MeasurementV1, ...]
    quality: str = ""
    protocol: ProtocolContextV1 | None = None
    provenance: ProvenanceV1 | None = None
    schema_version: str = EVENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("session_id", "subject_ref", "device_ref"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise SchemaError(f"{name} must be a non-empty string")
        if self.schema_version != EVENT_SCHEMA_VERSION:
            raise SchemaError(
                f"unsupported event schema {self.schema_version!r}; "
                f"expected {EVENT_SCHEMA_VERSION!r}"
            )
        if self.source not in SOURCES:
            raise SchemaError(f"unknown event source: {self.source!r}")
        if self.phase not in PHASES:
            raise SchemaError(f"unknown event phase: {self.phase!r}")
        if not isinstance(self.measured_at, datetime) or not isinstance(self.received_at, datetime):
            raise SchemaError("measured_at and received_at must be datetime values")
        if self.measured_at.tzinfo is None or self.received_at.tzinfo is None:
            raise SchemaError("measured_at and received_at must be timezone-aware")
        if self.received_at < self.measured_at:
            raise SchemaError(
                "received_at precedes measured_at: reject reversed timestamps "
                "instead of inventing latency"
            )
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 0
        ):
            raise SchemaError("sequence must be a non-negative integer")
        if not isinstance(self.measurements, (tuple, list)):
            raise SchemaError("measurements must be a sequence")
        object.__setattr__(self, "measurements", tuple(self.measurements))
        if any(not isinstance(m, MeasurementV1) for m in self.measurements):
            raise SchemaError("measurements must contain MeasurementV1 items")
        expected = event_quality(self.measurements)
        if not self.quality:
            object.__setattr__(self, "quality", expected)
        elif self.quality not in QUALITY_VALUES:
            raise SchemaError(f"unknown event quality: {self.quality!r}")
        elif self.quality == QUALITY_OK and expected != QUALITY_OK:
            raise SchemaError("event quality cannot be 'ok' when fields are invalid")
        if self.protocol is not None and not isinstance(self.protocol, ProtocolContextV1):
            raise SchemaError("protocol must be ProtocolContextV1 or None")
        if self.provenance is not None and not isinstance(self.provenance, ProvenanceV1):
            raise SchemaError("provenance must be ProvenanceV1 or None")

    @property
    def is_valid(self) -> bool:
        return self.quality in (QUALITY_OK, QUALITY_DEGRADED) and any(
            m.valid for m in self.measurements
        )

    def measurement(self, metric: str) -> MeasurementV1 | None:
        for item in self.measurements:
            if item.metric == metric:
                return item
        return None

    def observed_metrics(self) -> tuple[str, ...]:
        return tuple(m.metric for m in self.measurements if m.valid)

    def with_quality(self, quality: str) -> PhysioEventV1:
        return replace(self, quality=quality)

    @property
    def latency_s(self) -> float:
        return (self.received_at - self.measured_at).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "subject_ref": self.subject_ref,
            "device_ref": self.device_ref,
            "source": self.source,
            "phase": self.phase,
            "measured_at": self.measured_at.isoformat(),
            "received_at": self.received_at.isoformat(),
            "sequence": self.sequence,
            "measurements": [m.to_dict() for m in self.measurements],
            "quality": self.quality,
            "protocol": self.protocol.to_dict() if self.protocol else None,
            "provenance": self.provenance.to_dict() if self.provenance else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PhysioEventV1:
        if data.get("schema_version") != EVENT_SCHEMA_VERSION:
            raise SchemaError(f"unsupported event schema: {data.get('schema_version')!r}")
        try:
            measured_at = datetime.fromisoformat(str(data["measured_at"]))
            received_at = datetime.fromisoformat(str(data["received_at"]))
        except (KeyError, ValueError) as exc:
            raise SchemaError(f"invalid event timestamps: {exc}") from exc
        return cls(
            schema_version=EVENT_SCHEMA_VERSION,
            session_id=str(data["session_id"]),
            subject_ref=str(data["subject_ref"]),
            device_ref=str(data["device_ref"]),
            source=str(data["source"]),
            phase=str(data["phase"]),
            measured_at=measured_at,
            received_at=received_at,
            sequence=data["sequence"],
            measurements=tuple(MeasurementV1.from_dict(m) for m in data.get("measurements", ())),
            quality=str(data.get("quality", "")),
            protocol=ProtocolContextV1.from_dict(data.get("protocol")),
            provenance=(
                ProvenanceV1.from_dict(data["provenance"]) if data.get("provenance") else None
            ),
        )


@dataclass
class PhysioSessionV1:
    """A subject/device session: offline import and online stream share it."""

    session_id: str
    subject_ref: str
    device_ref: str
    events: tuple[PhysioEventV1, ...] = ()
    source: str = "mixed"
    protocol: ProtocolContextV1 | None = None
    provenance: ProvenanceV1 | None = None
    schema_version: str = SESSION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SESSION_SCHEMA_VERSION:
            raise SchemaError(
                f"unsupported session schema {self.schema_version!r}; "
                f"expected {SESSION_SCHEMA_VERSION!r}"
            )
        if not self.session_id or not self.subject_ref or not self.device_ref:
            raise SchemaError("session_id, subject_ref and device_ref must be non-empty")
        self.events = tuple(sorted(self.events, key=lambda e: (e.measured_at, e.sequence)))
        for event in self.events:
            if event.session_id != self.session_id:
                raise SchemaError(
                    f"event session_id {event.session_id!r} != session {self.session_id!r}"
                )
            if event.subject_ref != self.subject_ref:
                raise SchemaError("events must share one pseudonymous subject")
            if event.device_ref != self.device_ref:
                raise SchemaError("events must share one device reference")

    def __len__(self) -> int:
        return len(self.events)

    def by_phase(self, phase: str) -> tuple[PhysioEventV1, ...]:
        if phase not in PHASES:
            raise SchemaError(f"unknown phase: {phase!r}")
        return tuple(e for e in self.events if e.phase == phase)

    def cutoff(self, at: datetime, clock: str = "measured") -> tuple[PhysioEventV1, ...]:
        """Events at or before ``at`` in the chosen clock.

        ``measured`` is the scientific cutoff (what the body had produced);
        ``received`` is the online arrival cutoff. Neither can return future
        events, and Post is never included by a ``during`` caller unless it is
        explicitly requested.
        """
        if at.tzinfo is None:
            raise SchemaError("cutoff time must be timezone-aware")
        if clock not in ("measured", "received"):
            raise SchemaError("clock must be 'measured' or 'received'")
        if clock == "measured":
            return tuple(e for e in self.events if e.measured_at <= at)
        return tuple(e for e in self.events if e.received_at <= at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "subject_ref": self.subject_ref,
            "device_ref": self.device_ref,
            "source": self.source,
            "protocol": self.protocol.to_dict() if self.protocol else None,
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "events": [e.to_dict() for e in self.events],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PhysioSessionV1:
        if data.get("schema_version") != SESSION_SCHEMA_VERSION:
            raise SchemaError(f"unsupported session schema: {data.get('schema_version')!r}")
        return cls(
            schema_version=SESSION_SCHEMA_VERSION,
            session_id=str(data["session_id"]),
            subject_ref=str(data["subject_ref"]),
            device_ref=str(data["device_ref"]),
            source=str(data.get("source", "mixed")),
            protocol=ProtocolContextV1.from_dict(data.get("protocol")),
            provenance=(
                ProvenanceV1.from_dict(data["provenance"]) if data.get("provenance") else None
            ),
            events=tuple(PhysioEventV1.from_dict(e) for e in data.get("events", ())),
        )


_SOURCE_PRIORITY = {SOURCE_DEVICE: 0, SOURCE_SERVICE: 1, SOURCE_OFFLINE: 2}


def assemble_session(
    session_id: str,
    subject_ref: str,
    device_ref: str,
    events: list[PhysioEventV1] | tuple[PhysioEventV1, ...],
    *,
    source: str = "mixed",
    protocol: ProtocolContextV1 | None = None,
    provenance: ProvenanceV1 | None = None,
) -> PhysioSessionV1:
    """Deduplicate by sequence and sort deterministically.

    Source of truth: a device event outranks a service copy, which outranks an
    offline re-import. Ties break on the earliest ``received_at`` so replaying
    the same stream twice resolves duplicates identically.
    """
    chosen: dict[int, PhysioEventV1] = {}
    for event in events:
        current = chosen.get(event.sequence)
        if current is None:
            chosen[event.sequence] = event
            continue
        current_key = (_SOURCE_PRIORITY.get(current.source, 9), current.received_at)
        event_key = (_SOURCE_PRIORITY.get(event.source, 9), event.received_at)
        if event_key < current_key:
            chosen[event.sequence] = event
    return PhysioSessionV1(
        session_id=session_id,
        subject_ref=subject_ref,
        device_ref=device_ref,
        events=tuple(chosen.values()),
        source=source,
        protocol=protocol,
        provenance=provenance,
    )


def mark_stale(
    event: PhysioEventV1, reference: datetime, max_age_s: float
) -> PhysioEventV1:
    """Mark an event stale when too old relative to the query time."""
    if reference.tzinfo is None or event.measured_at.tzinfo is None:
        raise SchemaError("stale checks require timezone-aware timestamps")
    age = (reference - event.measured_at).total_seconds()
    if age > max_age_s:
        return replace(
            event,
            quality=QUALITY_STALE,
            provenance=(
                replace(event.provenance, notes=(*event.provenance.notes, "stale"))
                if event.provenance is not None
                else None
            ),
        )
    return event


__all__ = [
    "DATA_TYPES",
    "DEFAULT_CLOCK_SKEW_TOLERANCE",
    "EVENT_SCHEMA_VERSION",
    "MEASUREMENT_ACCEPTED",
    "PHASES",
    "PHASE_DURING",
    "PHASE_POST",
    "PHASE_PRE",
    "QUALITY_DEGRADED",
    "QUALITY_OK",
    "QUALITY_REJECTED",
    "QUALITY_STALE",
    "QUALITY_VALUES",
    "SESSION_SCHEMA_VERSION",
    "SOURCES",
    "SOURCE_DEVICE",
    "SOURCE_OFFLINE",
    "SOURCE_SERVICE",
    "MeasurementV1",
    "PhysioEventV1",
    "PhysioSessionV1",
    "ProtocolContextV1",
    "ProvenanceV1",
    "SchemaError",
    "assemble_session",
    "event_quality",
    "mark_stale",
]
