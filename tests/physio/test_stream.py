"""#96 stream tests: disconnect/reconnect, gap detection, replay convergence."""

from __future__ import annotations

import pytest
from physio_testkit import event as base_event
from physio_testkit import measurement, ts

from physio.schema import QUALITY_STALE, SOURCE_DEVICE
from physio.stream import (
    STATUS_CONNECTED,
    STATUS_DISCONNECTED,
    STATUS_STALE,
    OnlineIngestor,
    StreamError,
)

pytestmark = pytest.mark.synthetic


def event(sequence, seconds, *, received=None, source=SOURCE_DEVICE, value=70):
    return base_event(
        sequence,
        at=seconds,
        received=received,
        source=source,
        measurements=[measurement(value=value)],
    )


def test_disconnected_stream_rejects_and_reconnect_records_gap():
    ingestor = OnlineIngestor("s1", "p1", "d1", max_gap_s=30)
    ingestor.push(event(0, 0))
    ingestor.mark_disconnected(ts(10))
    assert ingestor.connectivity(ts(11)) == STATUS_DISCONNECTED
    with pytest.raises(StreamError, match="disconnected"):
        ingestor.push(event(1, 11))
    ingestor.reconnect(ts(40))
    assert ingestor.connectivity(ts(41)) == STATUS_CONNECTED
    ingestor.push(event(1, 41))
    assert any(gap.seconds == 30.0 for gap in ingestor.gaps)
    assert [e.sequence for e in ingestor.session().events] == [0, 1]


def test_gap_between_events_is_recorded():
    ingestor = OnlineIngestor("s1", "p1", "d1", max_gap_s=10)
    ingestor.push(event(0, 0))
    ingestor.push(event(1, 60))
    assert len(ingestor.gaps) == 1
    assert ingestor.gaps[0].seconds == pytest.approx(60.0)


def test_out_of_order_arrival_still_sorts_by_measurement_time():
    ingestor = OnlineIngestor("s1", "p1", "d1")
    ingestor.push(event(1, 30))
    ingestor.push(event(0, 0, received=31))
    session = ingestor.session()
    assert [e.sequence for e in session.events] == [0, 1]


def test_duplicate_sequence_counts_and_prefers_device():
    ingestor = OnlineIngestor("s1", "p1", "d1")
    ingestor.push(event(0, 0, source="service", value=99))
    resolved = ingestor.push(event(0, 0, source=SOURCE_DEVICE, value=70))
    assert ingestor.duplicates == 1
    assert resolved.source == SOURCE_DEVICE
    assert len(ingestor.session()) == 1
    assert ingestor.session().events[0].measurements[0].value == 70


def test_stale_reference_marks_session_events():
    ingestor = OnlineIngestor("s1", "p1", "d1", stale_after_s=5)
    ingestor.push(event(0, 0))
    session = ingestor.session(reference=ts(100))
    assert session.events[0].quality == QUALITY_STALE


def test_connectivity_goes_stale_without_disconnect_signal():
    ingestor = OnlineIngestor("s1", "p1", "d1", stale_after_s=10)
    ingestor.push(event(0, 0))
    assert ingestor.connectivity(ts(5)) == STATUS_CONNECTED
    assert ingestor.connectivity(ts(30)) == STATUS_STALE
