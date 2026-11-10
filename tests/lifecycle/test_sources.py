"""#106 streaming sources: capability honesty, dedup, replay, backpressure."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from lifecycle.contracts import (
    CAP_CONNECTED,
    CAP_NOT_CONNECTED,
    GATE_BLOCKED,
    SOURCE_DEVICE,
    SOURCE_OFFLINE,
    SOURCE_SIMULATED,
)
from lifecycle.sources import (
    ADAPTERS,
    ConformanceRecord,
    OfflineBatchAdapter,
    SessionWindow,
    SimulatedAdapter,
    SourceCapability,
    SourceError,
    adapter_class,
    conformance_record,
    live,
    make_event,
    replay,
)

pytestmark = pytest.mark.synthetic

T0 = "2026-11-01T09:00:00+00:00"
T1 = "2026-11-01T09:01:00+00:00"
T2 = "2026-11-01T09:02:00+00:00"


def _events() -> list:
    return [
        make_event("a", event_time=T0, sequence=1),
        make_event("b", event_time=T1, sequence=2),
        make_event("c", event_time=T2, sequence=3),
    ]


def test_real_device_cannot_be_connected_without_conformance():
    with pytest.raises(SourceError, match="vendor conformance record"):
        SourceCapability("dev1", SOURCE_DEVICE, CAP_CONNECTED)
    unconnected = SourceCapability("dev1", SOURCE_DEVICE, CAP_NOT_CONNECTED)
    assert unconnected.capability == CAP_NOT_CONNECTED


def test_simulated_source_is_labelled_simulated_and_never_connected_device():
    with pytest.raises(SourceError, match="source_type=SIMULATED"):
        SourceCapability("sim1", SOURCE_SIMULATED)
    sim = SourceCapability(
        "sim1", SOURCE_SIMULATED, capability="UNAVAILABLE", source_type="SIMULATED"
    )
    assert sim.kind == SOURCE_SIMULATED
    record = conformance_record(sim)
    assert record["status"] == CAP_NOT_CONNECTED
    with pytest.raises(SourceError, match="never be reported"):
        SourceCapability(
            "sim1", SOURCE_SIMULATED, CAP_CONNECTED,
            conformance=_full_conformance(), source_type="SIMULATED",
        )


def _full_conformance() -> ConformanceRecord:
    return ConformanceRecord(
        vendor="Acme", api_version="v1", protocol="USB", event_schema="physio-event-v1",
        supported_metrics=("heart_rate",), timestamp_semantics="measured_at UTC",
        replay_guarantee="sequence-ordered", verified_at="2026-11-01",
    )


def test_conformance_record_reports_blocked_by_default():
    record = conformance_record(SourceCapability("dev1", SOURCE_DEVICE))
    assert record["status"] == CAP_NOT_CONNECTED
    assert record["gate"] == GATE_BLOCKED
    assert "no vendor conformance record" in record["reason"]
    assert record["conformance"]["present"] is False


def test_verified_conformance_allows_connected_device():
    cap = SourceCapability(
        "dev1", SOURCE_DEVICE, CAP_CONNECTED, conformance=_full_conformance()
    )
    record = conformance_record(cap)
    assert record["status"] == "OK"
    assert record["conformance"]["present"] is True


def test_unknown_source_kind_is_refused():
    with pytest.raises(SourceError, match="unknown source kind"):
        SourceCapability("x", "invented")
    with pytest.raises(SourceError, match="no adapter registered"):
        adapter_class("invented")


def test_adapter_registry_covers_the_offline_kinds():
    assert set(ADAPTERS) == {SOURCE_SIMULATED, SOURCE_OFFLINE}
    assert adapter_class(SOURCE_SIMULATED) is SimulatedAdapter
    assert adapter_class(SOURCE_OFFLINE) is OfflineBatchAdapter


def test_duplicate_events_are_not_emitted_twice():
    events = _events() + [make_event("a", event_time=T0, sequence=1)]
    assert [e.key for e in replay(events)] == ["a", "b", "c"]
    queue = live(events)
    assert queue.dropped_count == 1
    assert any("duplicate" in r for r in queue.reasons)


def test_live_and_replay_converge_for_any_arrival_order():
    forward = replay(_events())
    reversed_order = replay(list(reversed(_events())))
    shuffled = replay([_events()[2], _events()[0], _events()[1]])
    assert forward == reversed_order == shuffled
    assert live(list(reversed(_events()))).accepted == forward


def test_late_and_out_of_order_events_are_counted_not_lost():
    queue = live([_events()[2], _events()[0], _events()[1]])
    assert queue.late_event_count >= 1
    assert [e.key for e in queue.accepted] == ["a", "b", "c"]


def test_watermark_advances_monotonically_and_never_moves_backwards():
    queue = live([])
    queue.push(make_event("b", event_time=T2))
    assert queue.watermark == T2
    queue.push(make_event("a", event_time=T0))
    assert queue.watermark == T2
    assert queue.late_event_count == 1


def test_backpressure_overflow_drops_with_counter_and_reason():
    from lifecycle.sources import BoundedEventQueue

    queue = BoundedEventQueue(2)
    assert queue.push(_events()[0]) is True
    assert queue.push(_events()[1]) is True
    assert queue.push(_events()[2]) is False
    assert queue.dropped_count == 1
    assert any("backpressure overflow" in r for r in queue.reasons)
    assert [e.key for e in queue.accepted] == ["a", "b"]


def test_empty_queue_is_refused():
    from lifecycle.sources import BoundedEventQueue

    with pytest.raises(SourceError, match="positive"):
        BoundedEventQueue(0)


def test_session_cutoff_never_exposes_future_events():
    window = SessionWindow("S1")
    for event in _events():
        window.push(event)
    assert [e.key for e in window.cutoff(T0)] == ["a"]
    assert [e.key for e in window.cutoff(T1)] == ["a", "b"]
    assert [e.key for e in window.cutoff(T2)] == ["a", "b", "c"]
    assert window.sha256


def test_session_gap_detection_reports_a_reason():
    window = SessionWindow("S1", gap_threshold_s=30.0)
    window.push(make_event("a", event_time=T0))
    window.push(make_event("b", event_time=T2))
    gaps = window.close()
    assert len(gaps) == 1
    assert gaps[0].seconds == 120.0
    assert "reconnect gap" in gaps[0].reason
    with pytest.raises(SourceError, match="closed"):
        window.push(make_event("c"))


def test_offline_batch_adapter_reproduces_simulated_events(tmp_path: Path):
    events = _events()
    path = tmp_path / "events.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OfflineBatchAdapter.COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(e.to_dict() for e in events)

    offline = OfflineBatchAdapter(path)
    offline.open()
    offline_events = tuple(offline.read())
    offline.close()
    simulated = SimulatedAdapter(events)
    simulated.open()
    simulated_events = tuple(simulated.read())
    simulated.close()
    assert offline_events == simulated_events
    assert replay(offline_events) == replay(simulated_events)


def test_offline_adapter_refuses_missing_file_and_bad_schema(tmp_path: Path):
    with pytest.raises(SourceError, match="not found"):
        OfflineBatchAdapter(tmp_path / "absent.csv").open()
    bad = tmp_path / "bad.csv"
    bad.write_text("key,session_id\nA,S1\n", encoding="utf-8")
    adapter = OfflineBatchAdapter(bad)
    adapter.open()
    with pytest.raises(SourceError, match="missing columns"):
        tuple(adapter.read())


def test_adapter_must_be_open_before_reading():
    adapter = SimulatedAdapter(_events())
    with pytest.raises(SourceError, match="not open"):
        tuple(adapter.read())


def test_same_logical_events_are_idempotent_under_replay():
    first = replay(_events())
    second = replay(_events() + _events())
    assert first == second
