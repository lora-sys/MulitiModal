"""Streaming source capability, adapters and deterministic replay (#106).

Real devices stay NOT_CONNECTED until a vendor conformance record exists. A
simulated source is always labelled SIMULATED and can never be reported as a
connected real device. Live arrival and replay converge on the same accepted
event sequence: duplicate keys, late/out-of-order arrival, bounded queue
backpressure and reconnect gaps are all explicit, never silent.

Stdlib only; no network, no real device, no torch.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .contracts import (
    CAP_CONNECTED,
    CAP_DEGRADED,
    CAP_NOT_CONNECTED,
    CAP_UNAVAILABLE,
    GATE_BLOCKED,
    SOURCE_DEVICE,
    SOURCE_KINDS,
    SOURCE_OFFLINE,
    SOURCE_SIMULATED,
    STATUS_OK,
    digest,
)

SOURCES_VERSION = "ml-lifecycle-sources-v1"
CAPABILITY_STATES = (CAP_CONNECTED, CAP_NOT_CONNECTED, CAP_DEGRADED, CAP_UNAVAILABLE)


class SourceError(ValueError):
    """Raised when a source would claim an unverified capability."""


@dataclass(frozen=True)
class ConformanceRecord:
    """Vendor-supplied facts. Empty by default: no real device is connected."""

    vendor: str = ""
    api_version: str = ""
    protocol: str = ""
    event_schema: str = ""
    supported_metrics: tuple[str, ...] = ()
    timestamp_semantics: str = ""
    replay_guarantee: str = ""
    verified_at: str = ""

    @property
    def present(self) -> bool:
        return all(
            (
                self.vendor, self.api_version, self.protocol, self.event_schema,
                self.timestamp_semantics, self.replay_guarantee,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "vendor": self.vendor,
            "api_version": self.api_version,
            "protocol": self.protocol,
            "event_schema": self.event_schema,
            "supported_metrics": list(self.supported_metrics),
            "timestamp_semantics": self.timestamp_semantics,
            "replay_guarantee": self.replay_guarantee,
            "verified_at": self.verified_at,
            "present": self.present,
        }


@dataclass(frozen=True)
class SourceCapability:
    """Declared capability of one source. Facts, not inferences from a mock."""

    source_id: str
    kind: str
    capability: str = CAP_NOT_CONNECTED
    conformance: ConformanceRecord = field(default_factory=ConformanceRecord)
    source_type: str = ""

    def __post_init__(self) -> None:
        if self.kind not in SOURCE_KINDS:
            raise SourceError(f"unknown source kind {self.kind!r}")
        if self.capability not in CAPABILITY_STATES:
            raise SourceError(f"unknown capability {self.capability!r}")
        # A simulated source must be labelled SIMULATED and cannot be a
        # connected real device.
        if self.kind == SOURCE_SIMULATED:
            if self.source_type != "SIMULATED":
                raise SourceError("a simulated source must declare source_type=SIMULATED")
            # Unconditional: a mock can never be a connected real device, even if
            # someone attaches a synthetic conformance record to make it look one.
            if self.capability == CAP_CONNECTED:
                raise SourceError(
                    "a simulated source can never be reported as a connected real device"
                )
        if self.kind == SOURCE_DEVICE and self.capability == CAP_CONNECTED:
            if not self.conformance.present:
                raise SourceError(
                    "a real device may not declare CONNECTED without a vendor conformance record"
                )

    def record(self) -> dict[str, Any]:
        """Explicit NOT_CONNECTED / BLOCKED verdict with a reason by default."""
        if self.capability == CAP_CONNECTED and self.conformance.present:
            return {
                "status": STATUS_OK,
                "capability": CAP_CONNECTED,
                "conformance": self.conformance.to_dict(),
                "reason": "vendor conformance record verified",
            }
        return {
            "status": CAP_NOT_CONNECTED,
            "capability": self.capability,
            "gate": GATE_BLOCKED,
            "conformance": self.conformance.to_dict(),
            "reason": (
                "no vendor conformance record; real device stays NOT_CONNECTED"
                if self.kind == SOURCE_DEVICE
                else "source is simulated or offline; no real device connection"
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "kind": self.kind,
            "capability": self.capability,
            "source_type": self.source_type,
            "conformance": self.conformance.to_dict(),
        }


@dataclass(frozen=True)
class SourceEvent:
    """One synthetic event. `key` is the idempotency key."""

    key: str
    session_id: str
    event_time: str
    sequence: int
    metric: str
    value: float
    unit: str
    quality: str = "ok"

    @property
    def timestamp(self) -> datetime:
        return datetime.fromisoformat(self.event_time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key, "session_id": self.session_id, "event_time": self.event_time,
            "sequence": self.sequence, "metric": self.metric, "value": self.value,
            "unit": self.unit, "quality": self.quality,
        }


def make_event(
    key: str,
    *,
    session_id: str = "S1",
    event_time: str = "2026-11-01T09:00:00+00:00",
    sequence: int = 1,
    metric: str = "heart_rate",
    value: float = 70.0,
    unit: str = "bpm",
    quality: str = "ok",
) -> SourceEvent:
    return SourceEvent(key, session_id, event_time, sequence, metric, value, unit, quality)


@runtime_checkable
class SourceAdapter(Protocol):
    """Thin plugin contract. Implementations must be offline/CI-safe."""

    def capability(self) -> SourceCapability: ...
    def open(self) -> None: ...
    def read(self) -> Iterator[SourceEvent]: ...
    def close(self) -> None: ...


def _event_order(event: SourceEvent) -> tuple[str, str, int, str]:
    return (event.session_id, event.event_time, event.sequence, event.key)


class SimulatedAdapter:
    """Deterministic in-memory events. Always SIMULATED, no network."""

    def __init__(self, events: Sequence[SourceEvent], *, source_id: str = "sim-1") -> None:
        self._events = tuple(events)
        self._capability = SourceCapability(
            source_id=source_id, kind=SOURCE_SIMULATED,
            capability=CAP_UNAVAILABLE, source_type="SIMULATED",
        )
        self.opened = False

    def capability(self) -> SourceCapability:
        return self._capability

    def open(self) -> None:
        self.opened = True

    def read(self) -> Iterator[SourceEvent]:
        if not self.opened:
            raise SourceError("adapter is not open")
        yield from self._events

    def close(self) -> None:
        self.opened = False


class OfflineBatchAdapter:
    """CSV-backed events. Reads are deterministic and offline."""

    COLUMNS = ("key", "session_id", "event_time", "sequence", "metric", "value", "unit", "quality")

    def __init__(self, path: str | Path, *, source_id: str = "offline-1") -> None:
        self.path = Path(path)
        self._capability = SourceCapability(
            source_id=source_id, kind=SOURCE_OFFLINE, capability=CAP_UNAVAILABLE,
        )
        self.opened = False

    def capability(self) -> SourceCapability:
        return self._capability

    def open(self) -> None:
        if not self.path.is_file():
            raise SourceError(f"batch file not found: {self.path}")
        self.opened = True

    def read(self) -> Iterator[SourceEvent]:
        if not self.opened:
            raise SourceError("adapter is not open")
        with self.path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = [c for c in self.COLUMNS if c not in (reader.fieldnames or ())]
            if missing:
                raise SourceError(f"batch missing columns {missing}")
            for raw in reader:
                yield SourceEvent(
                    key=raw["key"], session_id=raw["session_id"],
                    event_time=raw["event_time"], sequence=int(raw["sequence"]),
                    metric=raw["metric"], value=float(raw["value"]),
                    unit=raw["unit"], quality=raw["quality"] or "ok",
                )

    def close(self) -> None:
        self.opened = False


ADAPTERS: dict[str, type[SimulatedAdapter] | type[OfflineBatchAdapter]] = {
    SOURCE_SIMULATED: SimulatedAdapter,
    SOURCE_OFFLINE: OfflineBatchAdapter,
}


def adapter_class(kind: str) -> type[SimulatedAdapter] | type[OfflineBatchAdapter]:
    if kind not in ADAPTERS:
        raise SourceError(f"no adapter registered for source kind {kind!r}")
    return ADAPTERS[kind]


class BoundedEventQueue:
    """Bounded buffer. Overflow DROPS with a counted reason, never silently."""

    def __init__(self, max_size: int, *, session_id: str = "S1") -> None:
        if max_size < 1:
            raise SourceError("max_size must be positive")
        self.max_size = max_size
        self.session_id = session_id
        self._accepted: dict[str, SourceEvent] = {}
        self.dropped_count = 0
        self.reasons: list[str] = []
        self.late_event_count = 0
        self.watermark: str = ""

    @property
    def seen_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._accepted))

    @property
    def accepted(self) -> tuple[SourceEvent, ...]:
        return tuple(sorted(self._accepted.values(), key=_event_order))

    def push(self, event: SourceEvent) -> bool:
        """Return False on duplicate or overflow; both are recorded."""
        if event.key in self._accepted:
            self.dropped_count += 1
            self.reasons.append(f"duplicate key {event.key!r} dropped")
            return False
        if self.watermark and event.event_time < self.watermark:
            self.late_event_count += 1
        if len(self._accepted) >= self.max_size:
            self.dropped_count += 1
            self.reasons.append(
                f"backpressure overflow at max_size={self.max_size}: dropped {event.key!r}"
            )
            return False
        self._accepted[event.key] = event
        if event.event_time > self.watermark:
            self.watermark = event.event_time
        return True

    def ingest(self, events: Iterable[SourceEvent]) -> None:
        for event in events:
            self.push(event)


def replay(
    events: Sequence[SourceEvent], *, max_size: int | None = None
) -> tuple[SourceEvent, ...]:
    """Deterministic replay: sort by event time/sequence/key, then dedup by key.

    Feeding the same logical set in any order yields the same accepted tuple.
    """
    limit = max_size if max_size is not None else len(events) + 1
    queue = BoundedEventQueue(max(1, limit))
    for event in sorted(events, key=_event_order):
        queue.push(event)
    return queue.accepted


def live(events: Iterable[SourceEvent], *, max_size: int | None = None) -> BoundedEventQueue:
    """Live arrival. Order is unconstrained; `accepted` must still converge."""
    buffer = list(events)
    limit = max_size if max_size is not None else len(buffer) + 1
    queue = BoundedEventQueue(max(1, limit))
    queue.ingest(buffer)
    return queue


@dataclass(frozen=True)
class Gap:
    """One reconnect hole with a measured span and a reason."""

    start: str
    end: str
    seconds: float
    reason: str = "reconnect gap: no events between these event times"

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start, "end": self.end, "seconds": self.seconds,
            "reason": self.reason,
        }


class SessionWindow:
    """Causal session view. `cutoff()` never exposes future events."""

    def __init__(
        self,
        session_id: str,
        *,
        max_events: int = 1000,
        gap_threshold_s: float = 60.0,
    ) -> None:
        if max_events < 1:
            raise SourceError("max_events must be positive")
        self.session_id = session_id
        self.max_events = max_events
        self.gap_threshold_s = gap_threshold_s
        self._events: dict[str, SourceEvent] = {}
        self.gaps: list[Gap] = []
        self.closed = False

    def push(self, event: SourceEvent) -> bool:
        if self.closed:
            raise SourceError("session window is closed")
        if event.key in self._events:
            return False
        if len(self._events) >= self.max_events:
            raise SourceError(f"session window overflow at max_events={self.max_events}")
        self._events[event.key] = event
        return True

    def cutoff(self, at: str) -> tuple[SourceEvent, ...]:
        """Events strictly at or before `at`. No future leakage."""
        return tuple(
            sorted(
                (e for e in self._events.values() if e.event_time <= at),
                key=_event_order,
            )
        )

    def close(self) -> tuple[Gap, ...]:
        """Detect reconnect holes between consecutive event times."""
        ordered = sorted(self._events.values(), key=_event_order)
        gaps: list[Gap] = []
        for left, right in zip(ordered, ordered[1:], strict=False):
            span = (right.timestamp - left.timestamp).total_seconds()
            if span > self.gap_threshold_s:
                gaps.append(
                    Gap(
                        start=left.event_time, end=right.event_time, seconds=span,
                        reason=f"reconnect gap of {span:g}s exceeds {self.gap_threshold_s:g}s",
                    )
                )
        self.gaps = gaps
        self.closed = True
        return tuple(gaps)

    @property
    def sha256(self) -> str:
        return digest({"session": self.session_id, "events": [e.to_dict() for e in self.accepted]})

    @property
    def accepted(self) -> tuple[SourceEvent, ...]:
        return tuple(sorted(self._events.values(), key=_event_order))


def conformance_record(capability: SourceCapability) -> dict[str, Any]:
    """Public helper: default verdict is NOT_CONNECTED / BLOCKED with a reason."""
    return capability.record()
