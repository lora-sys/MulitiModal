"""Replaceable device / vendor-service / offline adapters (#96).

No official device or vendor protocol has been supplied. The device and
service adapters therefore expose ``NOT_CONNECTED`` capabilities and only
parse explicitly provided payloads: they never claim a live connection and
never invent sampling rates or fields.

The offline CSV/JSON adapter is a research import path. It produces the same
``PhysioSessionV1`` contract as the online adapters, but it is not imported by
the realtime service and is never required for online inference.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .metrics import DEFAULT_ROUTER, FieldRouter, metric_names
from .schema import (
    PHASE_DURING,
    QUALITY_REJECTED,
    QUALITY_STALE,
    SOURCE_DEVICE,
    SOURCE_OFFLINE,
    SOURCE_SERVICE,
    MeasurementV1,
    PhysioEventV1,
    PhysioSessionV1,
    ProtocolContextV1,
    ProvenanceV1,
    SchemaError,
    assemble_session,
    mark_stale,
)

ADAPTER_VERSION = "physio-adapters-v1"

#: Explicitly reported capability state until a real protocol is provided.
NOT_CONNECTED = "NOT_CONNECTED"
MOCK_ONLY = "MOCK_ONLY"


def parse_timestamp(value: Any) -> datetime:
    """Parse an ISO-8601 timestamp; naive datetimes are rejected, not guessed."""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise SchemaError(f"invalid ISO-8601 timestamp: {value!r}") from exc
    else:
        raise SchemaError(f"timestamp must be ISO-8601 text, got {value!r}")
    if parsed.tzinfo is None:
        raise SchemaError(
            "timestamp must carry an explicit timezone; refusing to assume a local zone"
        )
    return parsed


def _require_text(payload: Mapping[str, Any], *names: str, default: str | None = None) -> str:
    for name in names:
        value = payload.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if default is not None:
        return default
    raise SchemaError(f"payload is missing a non-empty field among {names}")


def _coerce_sequence(value: Any) -> int:
    if isinstance(value, bool):
        raise SchemaError("sequence must be an integer, not bool")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    raise SchemaError(f"sequence must be a non-negative integer, got {value!r}")


def _measurement_items(
    payload: Mapping[str, Any], router: FieldRouter
) -> tuple[MeasurementV1, ...]:
    raw = payload.get("measurements", payload.get("data"))
    units = payload.get("units") or {}
    if raw is None:
        return ()
    if isinstance(raw, Mapping):
        return router.route_payload(raw, units=units)
    if isinstance(raw, Iterable) and not isinstance(raw, (str, bytes)):
        routed: list[MeasurementV1] = []
        for item in raw:
            if not isinstance(item, Mapping):
                raise SchemaError("measurement list items must be mappings")
            metric = item.get("metric", item.get("field", item.get("name")))
            if metric is None:
                raise SchemaError("measurement item is missing metric/field/name")
            routed.append(
                router.route(
                    str(metric),
                    item.get("value"),
                    item.get("unit"),
                    item.get("data_type"),
                )
            )
        return tuple(routed)
    raise SchemaError("measurements must be an object or a list")


@dataclass(frozen=True)
class AdapterCapabilities:
    """What an adapter can honestly claim to support."""

    adapter: str
    status: str
    protocol: str
    fields: tuple[str, ...]
    realtime: bool
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter,
            "status": self.status,
            "protocol": self.protocol,
            "fields": list(self.fields),
            "realtime": self.realtime,
            "notes": list(self.notes),
        }


class DeviceEventAdapter:
    """Mock/直连边界：在拿到厂商协议前只解析显式注入的事件。"""

    name = "device"

    def __init__(
        self,
        router: FieldRouter | None = None,
        *,
        stale_after_s: float | None = 120.0,
        clock_skew_s: float = 0.0,
    ) -> None:
        self.router = router or DEFAULT_ROUTER
        self.stale_after_s = stale_after_s
        self.clock_skew_s = clock_skew_s

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            adapter=self.name,
            status=NOT_CONNECTED,
            protocol=MOCK_ONLY,
            fields=metric_names(),
            realtime=True,
            notes=(
                "no vendor SDK/protocol has been supplied",
                "only explicitly injected payloads are parsed",
            ),
        )

    def parse_event(self, payload: Mapping[str, Any]) -> PhysioEventV1:
        if not isinstance(payload, Mapping):
            raise SchemaError("device payload must be a mapping")
        measured_at = parse_timestamp(
            payload.get("measured_at", payload.get("timestamp", payload.get("ts")))
        )
        received_at = parse_timestamp(
            payload.get("received_at", payload.get("ingest_at", datetime.now().astimezone()))
        )
        measurements = _measurement_items(payload, self.router)
        event = PhysioEventV1(
            session_id=_require_text(payload, "session_id", "session"),
            subject_ref=_require_text(payload, "subject_ref", "subject", "subject_id"),
            device_ref=_require_text(payload, "device_ref", "device_id", "device"),
            source=SOURCE_DEVICE,
            phase=str(payload.get("phase", PHASE_DURING)).lower(),
            measured_at=measured_at,
            received_at=received_at,
            sequence=_coerce_sequence(payload.get("sequence", payload.get("seq", 0))),
            measurements=measurements,
            protocol=ProtocolContextV1.from_dict(payload.get("protocol")),
            provenance=ProvenanceV1(
                adapter=self.name,
                adapter_version=ADAPTER_VERSION,
                source_record=payload.get("message_id"),
                notes=("mock/device boundary; not a live vendor connection",),
            ),
        )
        return self._apply_staleness(event)

    def _apply_staleness(self, event: PhysioEventV1) -> PhysioEventV1:
        if self.stale_after_s is None:
            return event
        return mark_stale(event, event.received_at, self.stale_after_s)

    def parse_events(self, payloads: Iterable[Mapping[str, Any]]) -> PhysioSessionV1:
        events = [self.parse_event(payload) for payload in payloads]
        if not events:
            raise SchemaError("at least one device event is required")
        first = events[0]
        return assemble_session(
            first.session_id,
            first.subject_ref,
            first.device_ref,
            events,
            source=SOURCE_DEVICE,
            provenance=first.provenance,
        )


class ServiceEventAdapter:
    """厂商云转发/拉取边界：解析显式 envelope，不猜测通信方式。"""

    name = "service"

    def __init__(
        self,
        router: FieldRouter | None = None,
        *,
        stale_after_s: float | None = 120.0,
    ) -> None:
        self.router = router or DEFAULT_ROUTER
        self.stale_after_s = stale_after_s

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            adapter=self.name,
            status=NOT_CONNECTED,
            protocol=MOCK_ONLY,
            fields=metric_names(),
            realtime=True,
            notes=(
                "vendor cloud push/pull protocol is unknown",
                "no polling interval or field list is assumed",
            ),
        )

    def parse_event(self, payload: Mapping[str, Any]) -> PhysioEventV1:
        if not isinstance(payload, Mapping):
            raise SchemaError("service payload must be a mapping")
        received_at = parse_timestamp(
            payload.get("received_at", payload.get("ingest_at", datetime.now().astimezone()))
        )
        measured_at = parse_timestamp(
            payload.get("measured_at", payload.get("timestamp", payload.get("ts")))
        )
        event = PhysioEventV1(
            session_id=_require_text(payload, "session_id", "session"),
            subject_ref=_require_text(payload, "subject_ref", "subject", "user_id"),
            device_ref=_require_text(payload, "device_ref", "device_id", "device"),
            source=SOURCE_SERVICE,
            phase=str(payload.get("phase", PHASE_DURING)).lower(),
            measured_at=measured_at,
            received_at=received_at,
            sequence=_coerce_sequence(payload.get("sequence", payload.get("seq", 0))),
            measurements=_measurement_items(payload, self.router),
            protocol=ProtocolContextV1.from_dict(
                payload.get("protocol", payload.get("program"))
            ),
            provenance=ProvenanceV1(
                adapter=self.name,
                adapter_version=ADAPTER_VERSION,
                source_record=payload.get("message_id"),
                notes=("vendor-service boundary; transport is not implemented",),
            ),
        )
        if self.stale_after_s is not None:
            event = mark_stale(event, event.received_at, self.stale_after_s)
        return event

    def parse_events(self, payloads: Iterable[Mapping[str, Any]]) -> PhysioSessionV1:
        events = [self.parse_event(payload) for payload in payloads]
        if not events:
            raise SchemaError("at least one service event is required")
        first = events[0]
        return assemble_session(
            first.session_id,
            first.subject_ref,
            first.device_ref,
            events,
            source=SOURCE_SERVICE,
            provenance=first.provenance,
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class OfflineSessionAdapter:
    """Read-only research import for CSV/JSON session exports."""

    name = "offline"
    realtime = False

    def __init__(self, router: FieldRouter | None = None) -> None:
        self.router = router or DEFAULT_ROUTER

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            adapter=self.name,
            status=MOCK_ONLY,
            protocol="OFFLINE_IMPORT",
            fields=metric_names(),
            realtime=False,
            notes=(
                "research import only; never a realtime transport",
                "raw files and hashes stay outside the repository",
            ),
        )

    def _check_hash(self, path: Path, expected_sha256: str | None) -> str:
        actual = _sha256_file(path)
        if expected_sha256 is not None and actual != expected_sha256:
            raise SchemaError("offline export sha256 mismatch; refusing to import")
        return actual

    def _provenance(self, path: Path, digest: str, source_record: str | None) -> ProvenanceV1:
        return ProvenanceV1(
            adapter=self.name,
            adapter_version=ADAPTER_VERSION,
            source_record=source_record,
            raw_sha256=digest,
            data_license="PRIVATE_NOT_FOR_COMMIT",
            notes=("offline research export; not a realtime source",),
        )

    def from_json(
        self,
        path: str | Path,
        *,
        session_id: str | None = None,
        subject_ref: str | None = None,
        device_ref: str | None = None,
        expected_sha256: str | None = None,
        source_record: str | None = None,
    ) -> PhysioSessionV1:
        path = Path(path)
        digest = self._check_hash(path, expected_sha256)
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, Mapping) and "events" in data:
            session = PhysioSessionV1.from_dict(data)
            if (session_id and session.session_id != session_id) or (
                subject_ref and session.subject_ref != subject_ref
            ):
                raise SchemaError("JSON session identity does not match the requested import")
            session.source = SOURCE_OFFLINE
            session.provenance = self._provenance(path, digest, source_record)
            return session
        payloads = data if isinstance(data, list) else [data]
        events = [
            self._event_from_offline_mapping(item, self._provenance(path, digest, source_record))
            for item in payloads
        ]
        return self._session_from_events(events, session_id, subject_ref, device_ref)

    def from_csv(
        self,
        path: str | Path,
        *,
        session_id: str,
        subject_ref: str,
        device_ref: str,
        units: Mapping[str, str] | None = None,
        expected_sha256: str | None = None,
        source_record: str | None = None,
        phase_default: str = PHASE_DURING,
    ) -> PhysioSessionV1:
        path = Path(path)
        digest = self._check_hash(path, expected_sha256)
        prov = self._provenance(path, digest, source_record)
        with open(path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise SchemaError("offline CSV has no data rows")
        fieldnames = set(rows[0])
        if {"metric", "value"} <= fieldnames:
            events = self._events_from_long_csv(
                rows, session_id, subject_ref, device_ref, units or {}, phase_default, prov
            )
        else:
            events = self._events_from_wide_csv(
                rows, session_id, subject_ref, device_ref, units or {}, phase_default, prov
            )
        return assemble_session(
            session_id,
            subject_ref,
            device_ref,
            events,
            source=SOURCE_OFFLINE,
            provenance=prov,
        )

    def _events_from_long_csv(
        self,
        rows: list[dict[str, str]],
        session_id: str,
        subject_ref: str,
        device_ref: str,
        units: Mapping[str, str],
        phase_default: str,
        prov: ProvenanceV1,
    ) -> list[PhysioEventV1]:
        grouped: dict[tuple, list[MeasurementV1]] = {}
        meta: dict[tuple, dict[str, Any]] = {}
        for row in rows:
            key = (
                row.get("sequence", "0"),
                row.get("measured_at", row.get("timestamp", "")),
                row.get("received_at", ""),
                (row.get("phase") or phase_default).lower(),
            )
            metric = row.get("metric", "")
            unit = row.get("unit") or units.get(metric)
            grouped.setdefault(key, []).append(self.router.route(metric, row.get("value"), unit))
            meta[key] = row
        events = []
        for (sequence, measured_at, received_at, phase), measurements in grouped.items():
            events.append(
                self._build_event(
                    session_id=session_id,
                    subject_ref=subject_ref,
                    device_ref=device_ref,
                    sequence=sequence,
                    measured_at=measured_at,
                    received_at=received_at,
                    phase=phase,
                    measurements=measurements,
                    protocol=ProtocolContextV1.from_dict(
                        {
                            "program_id": meta[(sequence, measured_at, received_at, phase)].get(
                                "program_id"
                            ),
                            "intensity": _optional_int(
                                meta[(sequence, measured_at, received_at, phase)].get("intensity")
                            ),
                            "duration_s": _optional_int(
                                meta[(sequence, measured_at, received_at, phase)].get(
                                    "duration_s"
                                )
                            ),
                        }
                    ),
                    prov=prov,
                )
            )
        return events

    def _events_from_wide_csv(
        self,
        rows: list[dict[str, str]],
        session_id: str,
        subject_ref: str,
        device_ref: str,
        units: Mapping[str, str],
        phase_default: str,
        prov: ProvenanceV1,
    ) -> list[PhysioEventV1]:
        meta_fields = {
            "sequence", "measured_at", "timestamp", "received_at", "phase",
            "program_id", "intensity", "duration_s",
        }
        events = []
        for index, row in enumerate(rows):
            measurements = tuple(
                self.router.route(
                    key,
                    value,
                    row.get(f"{key}_unit", units.get(key)),
                )
                for key, value in row.items()
                if key is not None and key not in meta_fields and value not in (None, "")
            )
            protocol = ProtocolContextV1.from_dict(
                {
                    "program_id": row.get("program_id"),
                    "intensity": _optional_int(row.get("intensity")),
                    "duration_s": _optional_int(row.get("duration_s")),
                }
            )
            events.append(
                self._build_event(
                    session_id=session_id,
                    subject_ref=subject_ref,
                    device_ref=device_ref,
                    sequence=row.get("sequence", str(index)),
                    measured_at=row.get("measured_at", row.get("timestamp", "")),
                    received_at=row.get("received_at", ""),
                    phase=(row.get("phase") or phase_default).lower(),
                    measurements=measurements,
                    protocol=protocol,
                    prov=prov,
                )
            )
        return events

    def _event_from_offline_mapping(
        self, payload: Mapping[str, Any], prov: ProvenanceV1
    ) -> PhysioEventV1:
        measurements = _measurement_items(payload, self.router)
        return PhysioEventV1(
            session_id=_require_text(payload, "session_id", "session"),
            subject_ref=_require_text(payload, "subject_ref", "subject", "subject_id"),
            device_ref=_require_text(payload, "device_ref", "device_id", "device"),
            source=SOURCE_OFFLINE,
            phase=str(payload.get("phase", PHASE_DURING)).lower(),
            measured_at=parse_timestamp(
                payload.get("measured_at", payload.get("timestamp", payload.get("ts")))
            ),
            received_at=parse_timestamp(
                payload.get("received_at", payload.get("ingest_at"))
            ),
            sequence=_coerce_sequence(payload.get("sequence", payload.get("seq", 0))),
            measurements=measurements,
            protocol=ProtocolContextV1.from_dict(payload.get("protocol")),
            provenance=prov,
        )

    def _build_event(
        self,
        *,
        session_id: str,
        subject_ref: str,
        device_ref: str,
        sequence: Any,
        measured_at: Any,
        received_at: Any,
        phase: str,
        measurements: tuple[MeasurementV1, ...],
        protocol: ProtocolContextV1 | None,
        prov: ProvenanceV1,
    ) -> PhysioEventV1:
        return PhysioEventV1(
            session_id=session_id,
            subject_ref=subject_ref,
            device_ref=device_ref,
            source=SOURCE_OFFLINE,
            phase=phase,
            measured_at=parse_timestamp(measured_at),
            received_at=parse_timestamp(received_at),
            sequence=_coerce_sequence(sequence),
            measurements=measurements,
            protocol=protocol,
            provenance=prov,
        )

    def _session_from_events(
        self,
        events: list[PhysioEventV1],
        session_id: str | None,
        subject_ref: str | None,
        device_ref: str | None,
    ) -> PhysioSessionV1:
        first = events[0]
        return assemble_session(
            session_id or first.session_id,
            subject_ref or first.subject_ref,
            device_ref or first.device_ref,
            events,
            source=SOURCE_OFFLINE,
            provenance=first.provenance,
        )


def _optional_int(value: Any) -> int | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"expected integer program field, got {value!r}") from exc


def event_is_usable(event: PhysioEventV1) -> bool:
    """A stale or fully rejected event must not feed inference."""
    return event.quality not in (QUALITY_REJECTED, QUALITY_STALE)


__all__ = [
    "ADAPTER_VERSION",
    "MOCK_ONLY",
    "NOT_CONNECTED",
    "AdapterCapabilities",
    "DeviceEventAdapter",
    "OfflineSessionAdapter",
    "ServiceEventAdapter",
    "event_is_usable",
    "parse_timestamp",
]
