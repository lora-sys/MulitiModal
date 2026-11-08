"""#97 window tests: causality, phases, masks and determinism."""

from __future__ import annotations

from datetime import timedelta

import pytest
from physio_testkit import ts

from physio.schema import (
    QUALITY_STALE,
    MeasurementV1,
    PhysioEventV1,
    PhysioSessionV1,
)
from physio.windows import (
    WINDOW_FEATURE_NAMES,
    PhysioWindowBuilder,
    WindowConfig,
)

pytestmark = pytest.mark.synthetic

HR = ("heart_rate", "bpm")
SPO2 = ("spo2", "%")


def measurement(metric: str, value: object, unit: str) -> MeasurementV1:
    return MeasurementV1(
        metric=metric,
        value=value,
        unit=unit,
        data_type="float",
        valid=True,
        status="accepted",
    )


def missing(metric: str, unit: str) -> MeasurementV1:
    return MeasurementV1(
        metric=metric,
        value=None,
        unit=unit,
        data_type="float",
        valid=False,
        status="missing",
    )


def event(
    sequence: int,
    seconds: int,
    measurements: tuple[MeasurementV1, ...],
    *,
    phase: str = "during",
) -> PhysioEventV1:
    at = ts(seconds)
    return PhysioEventV1(
        session_id="s1",
        subject_ref="p1",
        device_ref="d1",
        source="device",
        phase=phase,
        measured_at=at,
        received_at=at + timedelta(seconds=1),
        sequence=sequence,
        measurements=measurements,
    )


def hr_value(value: float) -> MeasurementV1:
    return measurement(HR[0], value, HR[1])


def spo2_value(value: float) -> MeasurementV1:
    return measurement(SPO2[0], value, SPO2[1])


def hr_event(sequence: int, seconds: int, value: float, *, phase: str = "during"):
    return event(sequence, seconds, (hr_value(value),), phase=phase)


def session(*events: PhysioEventV1) -> PhysioSessionV1:
    return PhysioSessionV1(
        session_id="s1", subject_ref="p1", device_ref="d1", events=events
    )


def feature_map(features) -> dict[str, float]:
    return features.feature_map()


def test_fixed_replay_at_same_cutoff_is_identical():
    replay = session(hr_event(0, 0, 60), hr_event(1, 30, 70))
    builder = PhysioWindowBuilder()
    first = builder.build(replay, ts(60))
    second = builder.build(replay, ts(60))
    assert first.to_dict() == second.to_dict()
    assert first.feature_names == WINDOW_FEATURE_NAMES
    assert len(first.values) == len(WINDOW_FEATURE_NAMES)


def test_future_events_do_not_enter_the_window():
    cutoff = ts(60)
    causal = session(hr_event(0, 0, 60), hr_event(1, 30, 70))
    with_future = session(
        hr_event(0, 0, 60),
        hr_event(1, 30, 70),
        hr_event(2, 90, 200),
    )
    builder = PhysioWindowBuilder()
    assert builder.build(causal, cutoff).to_dict() == builder.build(with_future, cutoff).to_dict()
    assert builder.build(causal, cutoff).n_events == 2


def test_post_events_cannot_enter_the_window():
    cutoff = ts(60)
    causal = session(hr_event(0, 0, 60), hr_event(1, 30, 70))
    with_post = session(
        hr_event(0, 0, 60),
        hr_event(1, 30, 70),
        hr_event(2, 45, 200, phase="post"),
    )
    builder = PhysioWindowBuilder()
    causal_features = builder.build(causal, cutoff)
    post_features = builder.build(with_post, cutoff)
    assert causal_features.to_dict() == post_features.to_dict()
    assert "post" not in post_features.phase_counts


def test_whole_session_aggregate_is_not_used():
    all_events = session(
        hr_event(0, 0, 60),
        hr_event(1, 30, 70),
        hr_event(2, 90, 80),
    )
    features = PhysioWindowBuilder().build(all_events, ts(60))
    assert feature_map(features)["heart_rate.mean"] == pytest.approx(65.0)
    assert feature_map(features)["heart_rate.mean"] != pytest.approx(70.0)


def test_out_of_order_input_still_sorts_by_measured_time():
    builder = PhysioWindowBuilder()
    ordered = session(hr_event(0, 0, 60), hr_event(1, 30, 70))
    unordered = session(hr_event(1, 30, 70), hr_event(0, 0, 60))
    assert builder.build(ordered, ts(60)).to_dict() == builder.build(
        unordered, ts(60)
    ).to_dict()


def test_too_few_points_is_explicitly_unusable():
    features = PhysioWindowBuilder(WindowConfig(min_events=2)).build(
        session(hr_event(0, 0, 60)), ts(30)
    )
    assert features.usable is False
    assert features.reason == "too_few_events"
    assert features.n_events == 1


def test_phase_switch_is_deterministic():
    builder = PhysioWindowBuilder()
    events = session(
        hr_event(0, 0, 60, phase="pre"),
        hr_event(1, 30, 70, phase="during"),
    )
    before_switch = builder.build(events, ts(20))
    after_switch = builder.build(events, ts(30))
    assert before_switch.phase_counts == {"pre": 1, "during": 0}
    assert after_switch.phase_counts == {"pre": 1, "during": 1}
    assert before_switch.to_dict() == builder.build(events, ts(20)).to_dict()
    assert after_switch.to_dict() == builder.build(events, ts(30)).to_dict()


def test_partially_missing_fields_keep_the_rest_of_the_modality():
    events = session(
        event(0, 0, (hr_value(70), missing(SPO2[0], SPO2[1]))),
        event(1, 30, (hr_value(80), spo2_value(97))),
    )
    features = PhysioWindowBuilder().build(events, ts(60))
    observed = feature_map(features)
    assert features.usable is True
    assert features.metrics_present == ("heart_rate", "spo2")
    assert observed["heart_rate.mean"] == pytest.approx(75.0)
    assert observed["spo2.mean"] == pytest.approx(97.0)
    assert observed["spo2.coverage"] == pytest.approx(0.5)
    assert features.mask.count(True) < len(features.mask)


def test_empty_window_never_zero_fills():
    features = PhysioWindowBuilder().build(session(), ts(60))
    assert features.usable is False
    assert features.reason == "empty_window"
    assert features.coverage == 0.0
    assert not any(features.mask)
    assert features.feature_map() == {}
    assert all(value == 0.0 for value in features.values)


def test_no_numeric_values_is_explicitly_unusable():
    sex = MeasurementV1(
        metric="sex",
        value="female",
        unit="",
        data_type="string",
        valid=True,
        status="accepted",
    )
    features = PhysioWindowBuilder().build(
        session(event(0, 0, (sex,)), event(1, 30, (sex,))), ts(60)
    )
    assert features.usable is False
    assert features.reason == "no_numeric_values"
    assert not any(features.mask)
    assert features.metrics_present == ()


def test_rejected_and_stale_events_are_excluded():
    rejected = event(1, 20, (missing(HR[0], HR[1]),))
    stale = hr_event(2, 40, 200).with_quality(QUALITY_STALE)
    events = session(hr_event(0, 0, 60), rejected, stale)
    features = PhysioWindowBuilder(WindowConfig(min_events=1)).build(events, ts(60))
    assert features.n_events == 1
    assert feature_map(features)["heart_rate.mean"] == pytest.approx(60.0)
    assert features.coverage == pytest.approx(1.0 / 3.0)


def test_baseline_uses_pre_events_before_cutoff_only():
    events = session(
        hr_event(0, -30, 60, phase="pre"),
        hr_event(1, 30, 80),
        hr_event(2, 40, 100, phase="post"),
        hr_event(3, 60, 90),
        hr_event(4, 90, 200, phase="pre"),
    )
    builder = PhysioWindowBuilder()
    baseline = builder.baseline(events, ts(60))
    assert baseline == {"heart_rate": pytest.approx(60.0)}
    features = builder.build(events, ts(60), baseline=baseline)
    observed = feature_map(features)
    assert observed["heart_rate.mean"] == pytest.approx(85.0)
    assert observed["heart_rate.baseline_delta"] == pytest.approx(25.0)


def test_mean_trend_variance_and_coverage_math():
    events = session(
        hr_event(0, 0, 60),
        hr_event(1, 30, 70),
        hr_event(2, 60, 80),
    )
    features = PhysioWindowBuilder().build(events, ts(60))
    observed = feature_map(features)
    assert observed["heart_rate.mean"] == pytest.approx(70.0)
    assert observed["heart_rate.trend"] == pytest.approx(1.0 / 3.0)
    assert observed["heart_rate.variance"] == pytest.approx(200.0 / 3.0)
    assert observed["heart_rate.coverage"] == pytest.approx(1.0)


def test_overall_coverage_counts_valid_numeric_rows_only():
    events = session(
        event(0, 0, (hr_value(70), missing(SPO2[0], SPO2[1]))),
        event(1, 30, (hr_value(80), spo2_value(97))),
    )
    features = PhysioWindowBuilder().build(events, ts(60))
    assert features.coverage == pytest.approx(0.75)


def test_baseline_delta_is_masked_without_a_baseline():
    features = PhysioWindowBuilder().build(
        session(hr_event(0, 0, 60), hr_event(1, 30, 70)), ts(60)
    )
    observed = feature_map(features)
    assert "heart_rate.mean" in observed
    assert "heart_rate.baseline_delta" not in observed
