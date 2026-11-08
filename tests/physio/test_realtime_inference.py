"""#97 realtime inference tests: cutoff causality, gates, cache and recovery."""

from __future__ import annotations

import hashlib
from datetime import timedelta

import pytest
from physio_testkit import ts

from physio.artifacts import ModelArtifact
from physio.inference import (
    STATUS_ABSTAIN,
    STATUS_NOT_AVAILABLE,
    STATUS_OK,
    InferenceSession,
)
from physio.model import REALTIME_TARGET_VERSION, VALIDATION_VALIDATED
from physio.preprocessing import WindowScaler
from physio.schema import MeasurementV1, PhysioEventV1, PhysioSessionV1
from physio.stream import OnlineIngestor
from physio.windows import WINDOW_FEATURE_VERSION, PhysioWindowBuilder

pytestmark = pytest.mark.synthetic

HR = "heart_rate"
SPO2 = "spo2"


def measurement(metric: str, value: float, unit: str) -> MeasurementV1:
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


def hr_event(sequence: int, seconds: int, value: float, *, phase: str = "during"):
    return event(
        sequence,
        seconds,
        (measurement(HR, value, "bpm"),),
        phase=phase,
    )


def session(*events: PhysioEventV1) -> PhysioSessionV1:
    return PhysioSessionV1(
        session_id="s1", subject_ref="p1", device_ref="d1", events=events
    )


class FakeOutput:
    def __init__(self, value: float) -> None:
        self.predictions = {"fatigue_state": [[value]]}


class FakeModel:
    """Tiny duck-typed model used only to exercise the OK path locally."""

    model_type = "synthetic-test-model"

    def __init__(self, value: float = 42.0, input_spec=None) -> None:
        self.value = value
        self.input_spec = input_spec
        self.calls = 0
        self.last_mask = None

    def state_sha256(self) -> str:
        return hashlib.sha256(f"fake:{self.value}".encode()).hexdigest()

    def __call__(self, batch, feature_mask=None):
        self.calls += 1
        self.last_mask = feature_mask
        return FakeOutput(self.value)


class FakeModalitySpec:
    def __init__(self, dim: int) -> None:
        self.dim = dim


class FakeInputSpec:
    def __init__(self, dim: int) -> None:
        self.dim = dim

    def get(self, name: str):
        if name != "window_features":
            raise KeyError(name)
        return FakeModalitySpec(self.dim)


def scaler_for(events_session: PhysioSessionV1, at=None) -> WindowScaler:
    builder = PhysioWindowBuilder()
    reference = ts(60) if at is None else at
    return WindowScaler.fit([builder.build(events_session, reference)])


def artifact_for(
    model: FakeModel,
    scaler: WindowScaler,
    *,
    validation_status: str = VALIDATION_VALIDATED,
    model_version: str = "synthetic-test-model-v1",
) -> ModelArtifact:
    return ModelArtifact(
        model_version=model_version,
        scaler_sha256=scaler.digest(),
        state_sha256=model.state_sha256(),
        validation_status=validation_status,
    )


def validated_session(
    events_session: PhysioSessionV1,
    *,
    builder: PhysioWindowBuilder | None = None,
    model: FakeModel | None = None,
    scaler: WindowScaler | None = None,
) -> InferenceSession:
    builder = builder or PhysioWindowBuilder()
    model = model or FakeModel()
    scaler = scaler or scaler_for(events_session)
    return InferenceSession(
        events_session,
        builder,
        model=model,
        scaler=scaler,
        artifact=artifact_for(model, scaler),
    )


def test_fixed_replay_at_same_cutoff_is_identical_and_cached():
    events_session = session(hr_event(0, 0, 60), hr_event(1, 30, 70))
    model = FakeModel()
    inference = validated_session(events_session, model=model)
    first = inference.update(ts(60))
    second = inference.update(ts(60))
    assert first.status == STATUS_OK
    assert first.prediction == pytest.approx(42.0)
    assert first.to_dict() == second.to_dict()
    assert model.calls == 1


def test_future_and_post_events_do_not_change_the_result():
    cutoff = ts(60)
    causal = session(hr_event(0, 0, 60), hr_event(1, 30, 70))
    contaminated = session(
        hr_event(0, 0, 60),
        hr_event(1, 30, 70),
        hr_event(2, 45, 200, phase="post"),
        hr_event(3, 90, 200),
    )
    clean_result = validated_session(causal).update(cutoff).to_dict()
    contaminated_result = validated_session(contaminated).update(cutoff).to_dict()
    assert clean_result == contaminated_result
    assert "post" not in clean_result["window"]["phase_counts"]


def test_empty_session_abstains_without_a_score():
    model = FakeModel()
    inference = InferenceSession(
        session(),
        PhysioWindowBuilder(),
        model=model,
        scaler=WindowScaler.fit(
            [
                PhysioWindowBuilder().build(
                    session(hr_event(0, 0, 60), hr_event(1, 30, 70)), ts(60)
                )
            ]
        ),
    )
    result = inference.update(ts(60))
    assert result.status == STATUS_ABSTAIN
    assert result.reason == "no_valid_measurements"
    assert result.prediction is None
    assert result.measured == ()
    assert model.calls == 0


def test_stale_measurements_abstain_and_never_predict():
    events_session = session(hr_event(0, 0, 60), hr_event(1, 30, 70))
    model = FakeModel()
    inference = validated_session(events_session, model=model)
    result = inference.update(ts(300))
    assert result.status == STATUS_ABSTAIN
    assert "stale" in result.reason
    assert result.prediction is None
    assert model.calls == 0


def test_no_model_is_not_available_and_keeps_measured_summary():
    events_session = session(hr_event(0, 0, 60), hr_event(1, 30, 70))
    inference = InferenceSession(
        events_session,
        PhysioWindowBuilder(),
        model=None,
        scaler=scaler_for(events_session),
    )
    result = inference.update(ts(60))
    assert result.status == STATUS_NOT_AVAILABLE
    assert result.reason == "model_not_available"
    assert result.prediction is None
    assert result.measured == (
        {"metric": HR, "value": pytest.approx(65.0), "unit": "bpm", "valid": True},
    )


def test_no_scaler_is_not_available():
    events_session = session(hr_event(0, 0, 60), hr_event(1, 30, 70))
    model = FakeModel()
    inference = InferenceSession(
        events_session,
        PhysioWindowBuilder(),
        model=model,
        scaler=None,
        artifact=artifact_for(model, scaler_for(events_session)),
    )
    result = inference.update(ts(60))
    assert result.status == STATUS_NOT_AVAILABLE
    assert result.reason == "scaler_not_available"
    assert result.prediction is None
    assert model.calls == 0


def test_unvalidated_model_never_emits_a_score():
    events_session = session(hr_event(0, 0, 60), hr_event(1, 30, 70))
    model = FakeModel()
    inference = InferenceSession(
        events_session,
        PhysioWindowBuilder(),
        model=model,
        scaler=scaler_for(events_session),
        artifact=artifact_for(
            model,
            scaler_for(events_session),
            validation_status="NOT_VALIDATED",
        ),
    )
    result = inference.update(ts(60))
    assert result.status == STATUS_NOT_AVAILABLE
    assert result.reason == "model_not_validated"
    assert result.prediction is None
    assert model.calls == 0


def test_wrong_model_artifact_dimension_is_not_available():
    events_session = session(hr_event(0, 0, 60), hr_event(1, 30, 70))
    builder = PhysioWindowBuilder()
    scaler = scaler_for(events_session)
    model = FakeModel(input_spec=FakeInputSpec(len(scaler.feature_names) - 1))
    inference = InferenceSession(
        events_session,
        builder,
        model=model,
        scaler=scaler,
        artifact=artifact_for(model, scaler),
    )
    result = inference.update(ts(60))
    assert result.status == STATUS_NOT_AVAILABLE
    assert "artifact_error" in result.reason
    assert result.prediction is None
    assert model.calls == 0


def test_scaler_never_observed_feature_is_not_available():
    training = session(hr_event(0, 0, 60), hr_event(1, 30, 70))
    builder = PhysioWindowBuilder()
    scaler = scaler_for(training)
    with_spo2 = session(
        event(0, 0, (measurement(HR, 70, "bpm"), measurement(SPO2, 97, "%"))),
        event(1, 30, (measurement(HR, 80, "bpm"), measurement(SPO2, 98, "%"))),
    )
    model = FakeModel()
    inference = InferenceSession(
        with_spo2,
        builder,
        model=model,
        scaler=scaler,
        artifact=artifact_for(model, scaler),
    )
    result = inference.update(ts(60))
    assert result.status == STATUS_NOT_AVAILABLE
    assert "artifact_error" in result.reason
    assert result.prediction is None
    assert model.calls == 0


def test_partially_missing_fields_keep_the_modality_available():
    events_session = session(
        event(0, 0, (measurement(HR, 70, "bpm"), missing(SPO2, "%"))),
        event(1, 30, (measurement(HR, 80, "bpm"), measurement(SPO2, 97, "%"))),
    )
    model = FakeModel()
    inference = validated_session(events_session, model=model)
    result = inference.update(ts(60))
    assert result.status == STATUS_OK
    assert {item["metric"] for item in result.measured} == {HR, SPO2}
    assert model.last_mask is not None
    assert any(model.last_mask)


def test_saved_and_loaded_scaler_is_used_by_the_session(tmp_path):
    events_session = session(hr_event(0, 0, 60), hr_event(1, 30, 70))
    scaler = scaler_for(events_session)
    path = tmp_path / "scaler.json"
    scaler.save(path)
    loaded = WindowScaler.load(path)
    inference = validated_session(events_session, scaler=loaded)
    assert inference.update(ts(60)).status == STATUS_OK


def test_disconnect_then_reconnect_recovers_without_fabricating_a_score():
    ingestor = OnlineIngestor("s1", "p1", "d1", max_gap_s=30, stale_after_s=120)
    ingestor.push(hr_event(0, 0, 60))
    ingestor.push(hr_event(1, 30, 70))
    builder = PhysioWindowBuilder()
    scaler = WindowScaler.fit([builder.build(ingestor.session(), ts(60))])
    model = FakeModel()
    inference = InferenceSession(
        ingestor.session(),
        builder,
        model=model,
        scaler=scaler,
        artifact=artifact_for(model, scaler),
    )
    assert inference.update(ts(60)).status == STATUS_OK

    ingestor.mark_disconnected(ts(70))
    stale = inference.update(ts(200))
    assert stale.status == STATUS_ABSTAIN
    assert stale.prediction is None

    ingestor.reconnect(ts(210))
    ingestor.push(hr_event(2, 210, 80))
    ingestor.push(hr_event(3, 240, 90))
    inference.session = ingestor.session()
    inference.reset()
    recovered = inference.update(ts(240))
    assert recovered.status == STATUS_OK
    assert recovered.prediction == pytest.approx(42.0)


def test_normalization_is_consistent_across_equivalent_windows():
    early = session(hr_event(0, 0, 60), hr_event(1, 30, 70))
    late = session(hr_event(0, 100, 60), hr_event(1, 130, 70))
    builder = PhysioWindowBuilder()
    scaler = WindowScaler.fit([builder.build(early, ts(60))])
    early_scaled = scaler.transform(builder.build(early, ts(60)))
    late_scaled = scaler.transform(builder.build(late, ts(160)))
    assert early_scaled.values == late_scaled.values

    early_result = validated_session(early, builder=builder, scaler=scaler).update(ts(60))
    late_result = validated_session(late, builder=builder, scaler=scaler).update(ts(160))
    assert early_result.prediction == late_result.prediction
    assert early_result.measured == late_result.measured


def test_twenty_synthetic_rows_cannot_unlock_a_score():
    builder = PhysioWindowBuilder()
    training = [
        builder.build(
            session(hr_event(0, 0, 60 + index), hr_event(1, 30, 70 + index)),
            ts(60),
        )
        for index in range(20)
    ]
    events_session = session(hr_event(0, 0, 60), hr_event(1, 30, 70))
    model = FakeModel()
    training_scaler = WindowScaler.fit(training)
    inference = InferenceSession(
        events_session,
        builder,
        model=model,
        scaler=training_scaler,
        artifact=artifact_for(
            model,
            training_scaler,
            validation_status="NOT_VALIDATED",
            model_version="synthetic-20-row-model-v1",
        ),
    )
    result = inference.update(ts(60))
    assert result.status == STATUS_NOT_AVAILABLE
    assert result.reason == "model_not_validated"
    assert result.prediction is None
    assert model.calls == 0


def test_result_dict_has_the_frozen_top_level_fields():
    events_session = session(hr_event(0, 0, 60), hr_event(1, 30, 70))
    result = validated_session(events_session).update(ts(60))
    payload = result.to_dict()
    assert set(payload) == {
        "status",
        "reason",
        "target_version",
        "feature_version",
        "model_version",
        "input_cutoff",
        "quality",
        "coverage",
        "measured",
        "prediction",
        "window",
        "artifact_sha256",
    }
    assert payload["target_version"] == REALTIME_TARGET_VERSION
    assert payload["feature_version"] == WINDOW_FEATURE_VERSION
    assert payload["input_cutoff"] == ts(60).isoformat()
    assert payload["prediction"] == pytest.approx(42.0)
    assert payload["artifact_sha256"] is not None
    assert len(payload["artifact_sha256"]) == 64
    assert set(payload["window"]) == {
        "start",
        "end",
        "window_s",
        "min_events",
        "n_events",
        "phase_counts",
        "reason",
    }
    assert set(payload["measured"][0]) == {"metric", "value", "unit", "valid"}


def test_naive_now_and_received_clock_are_rejected():
    events_session = session(hr_event(0, 0, 60), hr_event(1, 30, 70))
    inference = validated_session(events_session)
    with pytest.raises(ValueError, match="timezone-aware"):
        inference.update(ts(60).replace(tzinfo=None))
    with pytest.raises(ValueError, match="measured clock"):
        InferenceSession(
            events_session,
            PhysioWindowBuilder(),
            clock="received",
        )


def test_model_module_imports_without_torch():
    import physio.model as model_module

    assert model_module.REALTIME_TARGET_VERSION == "fatigue_state_v1"
    try:
        import torch
    except ImportError:
        with pytest.raises(RuntimeError, match="requires PyTorch"):
            model_module.RealtimeFatigueModel(1)
    else:
        assert torch is not None
