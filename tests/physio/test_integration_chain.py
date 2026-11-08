"""#95 end-to-end chain: import -> stream -> window -> inference -> recommendation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from physio.adapters import DeviceEventAdapter
from physio.inference import STATUS_NOT_AVAILABLE, InferenceSession
from physio.stream import OnlineIngestor
from physio.windows import PhysioWindowBuilder, WindowConfig
from scripts import physio_realtime_demo as demo

pytestmark = pytest.mark.synthetic

BASE = datetime(2026, 11, 7, 10, 0, tzinfo=UTC)


def _ts(seconds: int) -> str:
    return (BASE + timedelta(seconds=seconds)).isoformat()


def _payload(sequence: int, seconds: int) -> dict:
    return {
        "session_id": "s1",
        "subject_ref": "p1",
        "device_ref": "d1",
        "phase": "during",
        "measured_at": _ts(seconds),
        "received_at": _ts(seconds + 1),
        "sequence": sequence,
        "measurements": {"hr": 70 + sequence % 5, "spo2": 0.97, "rr": 16},
        "units": {"hr": "bpm", "spo2": "fraction", "rr": "breaths/min"},
    }


def _session(count: int):
    adapter = DeviceEventAdapter()
    ingestor = OnlineIngestor("s1", "p1", "d1")
    for sequence in range(count):
        ingestor.push(adapter.parse_event(_payload(sequence, sequence * 10)))
    return ingestor.session()


def test_demo_reaches_acceptance_pass():
    report = demo.run()
    assert report["acceptance"]["status"] == "PASS"
    assert report["acceptance"]["passed"] == report["acceptance"]["total"] == 13
    assert report["inference"]["status"] == STATUS_NOT_AVAILABLE
    assert report["recommendation"]["control_command"] is None


def test_timeline_is_monotonic_and_never_controls_the_device():
    timeline = demo.run()["timeline"]
    cutoffs = [datetime.fromisoformat(item["cutoff"]) for item in timeline]
    assert cutoffs == sorted(cutoffs)
    assert len(set(cutoffs)) == len(cutoffs)
    for item in timeline:
        assert item["inference_status"] in {"NOT_AVAILABLE", "ABSTAIN"}
        assert item["control_command"] is None


def test_future_event_never_enters_an_earlier_window():
    session = _session(7)
    builder = PhysioWindowBuilder(WindowConfig(window_s=60, min_events=1))
    early = builder.build(session, BASE + timedelta(seconds=30))
    late = builder.build(session, BASE + timedelta(seconds=60))
    assert early.n_events < late.n_events
    assert early.end is not None and early.end <= BASE + timedelta(seconds=30)
    assert late.end is not None and late.end <= BASE + timedelta(seconds=90)


def test_late_arrival_does_not_rewrite_history():
    adapter = DeviceEventAdapter()
    ingestor = OnlineIngestor("s1", "p1", "d1")
    for sequence in range(4):
        ingestor.push(adapter.parse_event(_payload(sequence, sequence * 10)))
    builder = PhysioWindowBuilder(WindowConfig(window_s=60, min_events=1))
    first = builder.build(ingestor.session(), BASE + timedelta(seconds=30))
    ingestor.push(adapter.parse_event(_payload(4, 300)))
    second = builder.build(ingestor.session(), BASE + timedelta(seconds=30))
    assert first.to_dict() == second.to_dict()


def test_out_of_order_arrival_yields_the_same_window():
    adapter = DeviceEventAdapter()
    payloads = [_payload(sequence, sequence * 10) for sequence in range(6)]
    forward = OnlineIngestor("s1", "p1", "d1")
    backward = OnlineIngestor("s1", "p1", "d1")
    for payload in payloads:
        forward.push(adapter.parse_event(payload))
    for payload in reversed(payloads):
        backward.push(adapter.parse_event(payload))
    builder = PhysioWindowBuilder(WindowConfig(window_s=60, min_events=1))
    cutoff = BASE + timedelta(seconds=60)
    assert builder.build(forward.session(), cutoff).to_dict() == (
        builder.build(backward.session(), cutoff).to_dict()
    )


def test_unvalidated_inference_abstains_instead_of_scoring():
    session = _session(4)
    builder = PhysioWindowBuilder(WindowConfig(window_s=60, min_events=1))
    result = InferenceSession(session, builder).update(BASE + timedelta(seconds=40))
    assert result.status == STATUS_NOT_AVAILABLE
    assert result.prediction is None
    assert result.reason == "model_not_available"
