"""Deterministic property checks across the physiology contract (#95)."""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

import pytest

from physio.adapters import DeviceEventAdapter
from physio.dataset import FeatureRow, FoldImputer, admit_delta_targets, feature_row_from_window
from physio.legacy_export import (
    IDENTITY_TRUSTED,
    ExportFieldRegistry,
    ExportFieldSpec,
    import_legacy_rows,
)
from physio.metrics import route_measurement
from physio.stream import OnlineIngestor
from physio.windows import PhysioWindowBuilder, WindowConfig

pytestmark = pytest.mark.synthetic

BASE = datetime(2026, 11, 7, 10, 0, tzinfo=UTC)
CUTOFF = BASE + timedelta(seconds=60)


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
        "measurements": {"hr": 70, "spo2": 0.97, "rr": 16},
        "units": {"hr": "bpm", "spo2": "fraction", "rr": "breaths/min"},
    }


def _registry() -> ExportFieldRegistry:
    return ExportFieldRegistry(
        version="property-v1",
        fields=tuple(
            ExportFieldSpec(name, role)
            for name, role in (
                ("member_id", "identity"),
                ("session_key", "session"),
                ("record_id", "session"),
                ("record_kind", "session"),
                ("phase", "phase"),
                ("measured_at", "timestamp"),
                ("received_at", "timestamp"),
                ("sequence", "session"),
            )
        )
        + (
            ExportFieldSpec(
                "heart_rate", "metric", canonical_metric="heart_rate", unit="bpm"
            ),
        ),
    )


def _import_row(sequence: int, seconds: int) -> dict:
    return {
        "member_id": "m1",
        "session_key": "s1",
        "record_id": f"r{sequence}",
        "record_kind": "completed",
        "phase": "during",
        "measured_at": _ts(seconds),
        "received_at": _ts(seconds + 1),
        "sequence": sequence,
        "heart_rate": 60 + sequence,
    }


@pytest.mark.parametrize("seed", [1, 7, 42])
def test_import_digest_and_events_are_permutation_invariant(seed):
    rows = [_import_row(sequence, sequence * 10) for sequence in range(8)]
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    baseline = import_legacy_rows(rows, _registry(), identity_trust=IDENTITY_TRUSTED)
    other = import_legacy_rows(shuffled, _registry(), identity_trust=IDENTITY_TRUSTED)
    assert baseline.reproducibility_digest == other.reproducibility_digest
    assert [event.to_dict() for event in baseline.sessions[0].events] == [
        event.to_dict() for event in other.sessions[0].events
    ]


@pytest.mark.parametrize("seed", [3, 11])
def test_window_n_events_is_monotonic_and_permutation_invariant(seed):
    adapter = DeviceEventAdapter()
    payloads = [_payload(sequence, sequence * 5) for sequence in range(10)]
    shuffled = list(payloads)
    random.Random(seed).shuffle(shuffled)
    ingestor = OnlineIngestor("s1", "p1", "d1")
    for payload in shuffled:
        ingestor.push(adapter.parse_event(payload))
    builder = PhysioWindowBuilder(WindowConfig(window_s=60, min_events=1))
    counts = [
        builder.build(ingestor.session(), BASE + timedelta(seconds=step)).n_events
        for step in range(0, 60, 5)
    ]
    assert counts == sorted(counts)
    assert builder.build(ingestor.session(), CUTOFF).to_dict() == (
        builder.build(ingestor.session(), CUTOFF).to_dict()
    )


@pytest.mark.parametrize(
    ("metric", "value", "unit", "expected_valid"),
    [
        ("heart_rate", 72, "bpm", True),
        ("heart_rate", 0, "bpm", False),
        ("heart_rate", 500, "bpm", False),
        ("spo2", 0.97, "fraction", True),
        ("spo2", 97, "%", True),
        ("spo2", 0, "%", False),
    ],
)
def test_metric_routing_never_rescues_an_invalid_number(metric, value, unit, expected_valid):
    measurement = route_measurement(metric, value, unit)
    assert measurement.metric == metric
    assert measurement.valid is expected_valid
    if not expected_valid:
        assert measurement.value is None


def test_imputation_never_claims_to_observe_the_test_value():
    train = [
        FeatureRow(
            feature_names=("f1", "f2"),
            values=(value, value / 10),
            observed_mask=(True, True),
            valid_mask=(True, True),
            imputed_mask=(False, False),
            imputation=("none", "none"),
            subject_ref=f"s{index % 3}",
            session_id=f"train{index}",
            cutoff=CUTOFF,
        )
        for index, value in enumerate((10.0, 20.0, 30.0, 40.0))
    ]
    imputer = FoldImputer.fit(train)
    target = feature_row_from_window(
        _window((999.0, 0.0), (True, False)),  # type: ignore[arg-type]
        subject_ref="s9",
        session_id="test",
        cutoff=CUTOFF,
    )
    transformed = imputer.transform(target)
    assert transformed.values[0] == 999.0
    assert transformed.imputed_mask[0] is False
    assert transformed.imputed_mask[1] is True
    assert transformed.values[1] == imputer.medians[1]
    assert transformed.observed_mask[1] is False


@pytest.mark.parametrize("phase_pair", [("pre", "post"), ("pre", "during"), ("during", "post")])
def test_y_pri_is_never_produced(phase_pair):
    adapter = DeviceEventAdapter()
    pre = adapter.parse_event({**_payload(0, 0), "phase": phase_pair[0]})
    post = adapter.parse_event({**_payload(1, 300), "phase": phase_pair[1]})
    admission = admit_delta_targets(pre, post)
    assert admission.y_pri_value is None
    assert admission.y_pri_available is False
    assert admission.mask["y_pri"] is False


def _window(values, mask):
    from physio.windows import WindowFeatures

    return WindowFeatures(
        usable=any(mask),
        feature_names=("f1", "f2"),
        values=values,
        mask=mask,
        coverage=sum(mask) / len(mask),
        n_events=1,
        start=CUTOFF,
        end=CUTOFF,
        quality="ok",
        metrics_present=("heart_rate",) if mask[0] else (),
    )
