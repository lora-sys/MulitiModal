"""Online ingestion, duplicate/out-of-order handling and gap detection (#96).

The ingestor is transport-agnostic. A disconnected link stops accepting events;
after reconnect, late events are sorted by measurement time and deduplicated by
sequence, so replay and live arrival converge on the same session.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .schema import (
    PhysioEventV1,
    PhysioSessionV1,
    SchemaError,
    assemble_session,
    mark_stale,
)

STATUS_CONNECTED = "CONNECTED"
STATUS_DISCONNECTED = "DISCONNECTED"
STATUS_STALE = "STALE"


class StreamError(RuntimeError):
    """Raised when a caller pushes into a disconnected stream."""


@dataclass(frozen=True)
class Gap:
    start: datetime
    end: datetime

    @property
    def seconds(self) -> float:
        return (self.end - self.start).total_seconds()


class OnlineIngestor:
    """Accept events as they arrive; never claim a live link is healthy."""

    def __init__(
        self,
        session_id: str,
        subject_ref: str,
        device_ref: str,
        *,
        max_gap_s: float = 30.0,
        stale_after_s: float = 120.0,
    ) -> None:
        if max_gap_s < 0 or stale_after_s < 0:
            raise ValueError("gap/staleness budgets must be non-negative")
        self.session_id = session_id
        self.subject_ref = subject_ref
        self.device_ref = device_ref
        self.max_gap_s = max_gap_s
        self.stale_after_s = stale_after_s
        self.connected = True
        self.last_received_at: datetime | None = None
        self.gaps: list[Gap] = []
        self.duplicates = 0
        self._events: dict[int, PhysioEventV1] = {}
        self._disconnected_at: datetime | None = None

    def mark_disconnected(self, at: datetime) -> None:
        if at.tzinfo is None:
            raise SchemaError("disconnect time must be timezone-aware")
        self.connected = False
        self._disconnected_at = at

    def reconnect(self, at: datetime) -> None:
        if at.tzinfo is None:
            raise SchemaError("reconnect time must be timezone-aware")
        if self._disconnected_at is not None and at >= self._disconnected_at:
            self.gaps.append(Gap(self._disconnected_at, at))
        self.connected = True
        self._disconnected_at = None

    def push(self, event: PhysioEventV1) -> PhysioEventV1:
        if not self.connected:
            raise StreamError("stream is disconnected; reconnect before pushing events")
        if event.session_id != self.session_id:
            raise SchemaError("event belongs to a different session")
        if event.subject_ref != self.subject_ref or event.device_ref != self.device_ref:
            raise SchemaError("event subject/device does not match the ingestor")

        if self.last_received_at is not None and event.received_at > self.last_received_at:
            gap_s = (event.received_at - self.last_received_at).total_seconds()
            if gap_s > self.max_gap_s:
                self.gaps.append(Gap(self.last_received_at, event.received_at))
        if self.last_received_at is None or event.received_at > self.last_received_at:
            self.last_received_at = event.received_at

        existing = self._events.get(event.sequence)
        if existing is not None:
            self.duplicates += 1
            candidate = assemble_session(
                self.session_id,
                self.subject_ref,
                self.device_ref,
                [existing, event],
            )
            event = candidate.events[0]
        self._events[event.sequence] = event
        return event

    def connectivity(self, now: datetime) -> str:
        if not self.connected:
            return STATUS_DISCONNECTED
        if self.last_received_at is not None:
            age = (now - self.last_received_at).total_seconds()
            if age > self.stale_after_s:
                return STATUS_STALE
        return STATUS_CONNECTED

    def session(self, reference: datetime | None = None) -> PhysioSessionV1:
        events = [mark_stale(e, reference, self.stale_after_s) for e in self._events.values()] \
            if reference is not None else list(self._events.values())
        return assemble_session(
            self.session_id,
            self.subject_ref,
            self.device_ref,
            events,
            source="mixed",
        )

    def latest(self) -> PhysioEventV1 | None:
        if not self._events:
            return None
        return self.session().events[-1]


__all__ = [
    "STATUS_CONNECTED",
    "STATUS_DISCONNECTED",
    "STATUS_STALE",
    "Gap",
    "OnlineIngestor",
    "StreamError",
]
