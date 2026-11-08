"""#96 contract tests: timestamps, quality, masks, dedupe and replay safety."""

from __future__ import annotations

import pytest
from physio_testkit import event, measurement, ts

from physio.schema import (
    EVENT_SCHEMA_VERSION,
    PHASE_DURING,
    PHASE_POST,
    PHASE_PRE,
    QUALITY_DEGRADED,
    QUALITY_OK,
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

pytestmark = pytest.mark.synthetic


def test_event_roundtrip_and_quality():
    original = event(measurements=[measurement(), measurement("spo2", 97.0)])
    restored = PhysioEventV1.from_dict(original.to_dict())
    assert restored == original
    assert original.quality == QUALITY_OK
    assert original.observed_metrics() == ("heart_rate", "spo2")


def test_event_rejects_naive_timestamps():
    with pytest.raises(SchemaError, match="timezone-aware"):
        PhysioEventV1(
            session_id="s1", subject_ref="p1", device_ref="d1",
            source=SOURCE_DEVICE, phase=PHASE_DURING,
            measured_at=ts(0).replace(tzinfo=None), received_at=ts(1),
            sequence=0, measurements=(measurement(),),
        )


def test_event_rejects_reversed_timestamps():
    with pytest.raises(SchemaError, match="precedes"):
        PhysioEventV1(
            session_id="s1", subject_ref="p1", device_ref="d1",
            source=SOURCE_DEVICE, phase=PHASE_DURING,
            measured_at=ts(10), received_at=ts(1), sequence=0,
            measurements=(measurement(),),
        )


def test_invalid_measurement_cannot_carry_canonical_value():
    with pytest.raises(SchemaError, match="must not expose"):
        MeasurementV1(
            metric="heart_rate", value=72, unit="bpm", data_type="float",
            valid=False, status="out_of_range",
        )


def test_partial_missing_is_degraded_and_all_missing_is_rejected():
    bad = MeasurementV1(
        metric="heart_rate", value=None, unit="bpm", data_type="float",
        valid=False, status="missing",
    )
    partial = event(measurements=[measurement(), bad])
    assert partial.quality == QUALITY_DEGRADED
    assert partial.is_valid
    assert partial.observed_metrics() == ("heart_rate",)

    rejected = event(measurements=[bad])
    assert rejected.quality == QUALITY_REJECTED
    assert not rejected.is_valid


def test_event_quality_cannot_be_ok_with_invalid_fields():
    bad = MeasurementV1(
        metric="heart_rate", value=None, unit="bpm", data_type="float",
        valid=False, status="missing",
    )
    with pytest.raises(SchemaError, match="cannot be 'ok'"):
        event(measurements=[bad]).with_quality(QUALITY_OK)


def test_unknown_phase_and_source_rejected():
    with pytest.raises(SchemaError, match="phase"):
        PhysioEventV1(
            session_id="s1", subject_ref="p1", device_ref="d1",
            source=SOURCE_DEVICE, phase="intra", measured_at=ts(0),
            received_at=ts(1), sequence=0, measurements=(measurement(),),
        )
    with pytest.raises(SchemaError, match="source"):
        PhysioEventV1(
            session_id="s1", subject_ref="p1", device_ref="d1",
            source="cloud", phase=PHASE_DURING, measured_at=ts(0),
            received_at=ts(1), sequence=0, measurements=(measurement(),),
        )


def test_assemble_session_sorts_out_of_order_and_prefers_device():
    dup_service = PhysioEventV1(
        session_id="s1", subject_ref="p1", device_ref="d1",
        source=SOURCE_SERVICE, phase=PHASE_DURING,
        measured_at=ts(0), received_at=ts(2), sequence=0,
        measurements=(measurement("heart_rate", 99),),
    )
    session = assemble_session(
        "s1", "p1", "d1",
        [event(1, at=10), event(0, at=0), dup_service],
    )
    assert [e.sequence for e in session.events] == [0, 1]
    assert session.events[0].source == SOURCE_DEVICE
    assert session.events[0].measurements[0].value == 72


def test_assemble_session_replay_identical():
    session = assemble_session(
        "s1", "p1", "d1", [event(1, at=10), event(0, at=0)]
    )
    again = assemble_session(
        "s1", "p1", "d1", [event(0, at=0), event(1, at=10)]
    )
    assert session.to_dict() == again.to_dict()


def test_session_rejects_cross_subject_events():
    other = PhysioEventV1(
        session_id="s1", subject_ref="p2", device_ref="d1",
        source=SOURCE_DEVICE, phase=PHASE_DURING, measured_at=ts(0),
        received_at=ts(1), sequence=1, measurements=(measurement(),),
    )
    with pytest.raises(SchemaError, match="one pseudonymous subject"):
        PhysioSessionV1("s1", "p1", "d1", (event(), other))


def test_cutoff_never_returns_future_or_post():
    session = assemble_session(
        "s1", "p1", "d1",
        [
            event(0, at=0),
            event(1, at=30),
            event(2, at=60, phase=PHASE_POST),
        ],
    )
    cutoff = session.cutoff(ts(30))
    assert [e.sequence for e in cutoff] == [0, 1]
    assert all(e.measured_at <= ts(30) for e in cutoff)
    assert session.by_phase(PHASE_PRE) == ()
    assert len(session.by_phase(PHASE_POST)) == 1


def test_session_source_clock_cutoff_and_roundtrip():
    late = PhysioEventV1(
        session_id="s1", subject_ref="p1", device_ref="d1",
        source=SOURCE_DEVICE, phase=PHASE_DURING, measured_at=ts(0),
        received_at=ts(100), sequence=2, measurements=(measurement(),),
    )
    session = PhysioSessionV1("s1", "p1", "d1", (event(), late), source=SOURCE_DEVICE)
    assert len(session.cutoff(ts(50), clock="measured")) == 2
    assert [e.sequence for e in session.cutoff(ts(50), clock="received")] == [0]
    assert PhysioSessionV1.from_dict(session.to_dict()) == session


def test_stale_event_is_marked_and_unusable():
    stale = mark_stale(event(0, at=0), ts(1000), max_age_s=60)
    assert stale.quality == QUALITY_STALE
    assert not stale.is_valid


def test_protocol_context_validation():
    with pytest.raises(SchemaError, match="intensity"):
        ProtocolContextV1(program_id="M01", intensity=11)
    with pytest.raises(SchemaError, match="duration"):
        ProtocolContextV1(program_id="M01", duration_s=0)


def test_provenance_requires_adapter_and_hash_shape():
    with pytest.raises(SchemaError, match="adapter"):
        ProvenanceV1(adapter="", adapter_version="v1")
    with pytest.raises(SchemaError, match="sha256"):
        ProvenanceV1(adapter="offline", adapter_version="v1", raw_sha256="abc")


def test_schema_version_is_explicit():
    assert EVENT_SCHEMA_VERSION == "physio-event-v1"
    assert event().schema_version == EVENT_SCHEMA_VERSION
    with pytest.raises(SchemaError, match="unsupported"):
        PhysioEventV1.from_dict({**event().to_dict(), "schema_version": "v0"})


def test_offline_and_service_sources_are_distinct():
    assert len({SOURCE_DEVICE, SOURCE_SERVICE, SOURCE_OFFLINE}) == 3
