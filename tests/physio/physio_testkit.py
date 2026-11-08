"""Shared synthetic builders for physiological tests (#96–#100)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from physio.schema import MeasurementV1, PhysioEventV1, ProtocolContextV1

BASE = datetime(2026, 11, 7, 10, 0, tzinfo=UTC)


def ts(seconds: int = 0) -> datetime:
    return BASE + timedelta(seconds=seconds)


def measurement(metric="heart_rate", value=72, unit="bpm"):
    return MeasurementV1(
        metric=metric,
        value=value,
        unit=unit,
        data_type="float",
        valid=True,
        status="accepted",
    )


def event(
    sequence=0,
    *,
    at=0,
    received=None,
    source="device",
    phase="during",
    measurements=None,
    program=None,
):
    return PhysioEventV1(
        session_id="s1",
        subject_ref="p1",
        device_ref="d1",
        source=source,
        phase=phase,
        measured_at=ts(at),
        received_at=ts(received if received is not None else at + 1),
        sequence=sequence,
        measurements=tuple(measurements or [measurement()]),
        protocol=ProtocolContextV1.from_dict(program) if program else None,
    )


def device_payload():
    return {
        "session_id": "sess-demo-1",
        "subject_ref": "subj-pseudo-01",
        "device_ref": "chair-mock-01",
        "phase": "during",
        "measured_at": ts(0).isoformat(),
        "received_at": ts(1).isoformat(),
        "sequence": 0,
        "measurements": {
            "hr": 72,
            "spo2": 0.97,
            "skin_temp": 33.5,
            "rr": "0",
            "sbp": 0,
            "dbp": 80,
        },
        "units": {
            "hr": "bpm",
            "spo2": "fraction",
            "skin_temp": "°C",
            "rr": "breaths/min",
            "sbp": "mmHg",
            "dbp": "mmHg",
        },
        "protocol": {"program_id": "M01", "intensity": 2, "duration_s": 600},
    }
