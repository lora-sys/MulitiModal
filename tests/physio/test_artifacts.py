"""#97 artifact binding: scaler/state digests and version contracts."""

from __future__ import annotations

import hashlib

import pytest
from physio_testkit import event, measurement

from physio.artifacts import ArtifactError, ModelArtifact
from physio.inference import STATUS_NOT_AVAILABLE, STATUS_OK, InferenceSession
from physio.model import (
    REALTIME_TARGET_VERSION,
    VALIDATION_VALIDATED,
)
from physio.preprocessing import WindowScaler
from physio.windows import WINDOW_FEATURE_VERSION, PhysioWindowBuilder, WindowConfig

pytestmark = pytest.mark.synthetic


class FakeModel:
    def __init__(self, weight: float = 1.0) -> None:
        self.weight = weight

    def state_sha256(self) -> str:
        return hashlib.sha256(f"weight:{self.weight}".encode()).hexdigest()

    def __call__(self, batch, feature_mask=None):
        return type("Output", (), {"predictions": {"fatigue_state": [42.0]}})()


def _real_session():
    from physio.schema import PhysioSessionV1

    events = (
        event(sequence=0, at=0, measurements=[measurement(value=60)]),
        event(sequence=1, at=30, measurements=[measurement(value=70)]),
    )
    return PhysioSessionV1(
        session_id="s1", subject_ref="p1", device_ref="d1", events=events
    )


def _scaler(session=None):
    session = session or _real_session()
    builder = PhysioWindowBuilder(WindowConfig(window_s=60, min_events=2))
    return WindowScaler.fit([builder.build(session, event(sequence=2, at=60).measured_at)])


def _artifact(model=None, scaler=None, **overrides):
    model = model or FakeModel()
    scaler = scaler or _scaler()
    values = {
        "model_version": "synthetic-artifact-v1",
        "scaler_sha256": scaler.digest(),
        "state_sha256": model.state_sha256(),
        "validation_status": VALIDATION_VALIDATED,
    }
    values.update(overrides)
    return ModelArtifact(**values)


def test_scaler_digest_is_stable_across_save_and_load(tmp_path):
    scaler = _scaler()
    path = tmp_path / "scaler.json"
    scaler.save(path)
    loaded = WindowScaler.load(path)
    assert loaded.digest() == scaler.digest()


def test_artifact_roundtrip_and_tamper_detection(tmp_path):
    artifact = _artifact()
    path = tmp_path / "artifact.json"
    artifact.save(path)
    assert ModelArtifact.load(path) == artifact

    data = artifact.to_dict()
    data["state_sha256"] = hashlib.sha256(b"tampered").hexdigest()
    with pytest.raises(ArtifactError, match="sha256 mismatch"):
        ModelArtifact.from_dict(data)


def test_verify_rejects_mismatched_scaler_and_model():
    scaler = _scaler()
    model = FakeModel()
    artifact = _artifact(model=model, scaler=scaler)

    shifted = WindowScaler(
        feature_names=scaler.feature_names,
        mean=tuple(value + 1.0 for value in scaler.mean),
        std=scaler.std,
        observed_counts=scaler.observed_counts,
    )
    with pytest.raises(ArtifactError, match="scaler"):
        artifact.verify(shifted, model)

    with pytest.raises(ArtifactError, match="state"):
        artifact.verify(scaler, FakeModel(weight=2.0))


def test_verify_rejects_wrong_feature_or_target_version():
    scaler = _scaler()
    model = FakeModel()
    wrong_feature = _artifact(scaler=scaler, model=model, feature_version="window-features-v0")
    with pytest.raises(ArtifactError, match="feature_version"):
        wrong_feature.verify(scaler, model)

    wrong_target = _artifact(scaler=scaler, model=model, target_version="fatigue_state_v0")
    with pytest.raises(ArtifactError, match="target_version"):
        wrong_target.verify(scaler, model)

    assert wrong_target.target_version != REALTIME_TARGET_VERSION


def test_from_manifest_refuses_a_different_scaler():
    from physio.training import TrainingConfig, TrainingManifest

    scaler = _scaler()
    manifest = TrainingManifest(
        model_version="synthetic-artifact-v1",
        feature_names=scaler.feature_names,
        scaler_sha256=scaler.digest(),
        config=TrainingConfig(),
        n_samples=4,
        n_labeled=4,
        identity_trust="TRUSTED",
        status="SMOKE_ONLY",
        checkpoint_sha256=hashlib.sha256(b"checkpoint").hexdigest(),
        state_sha256=FakeModel().state_sha256(),
    )
    artifact = ModelArtifact.from_manifest(manifest, scaler)
    assert artifact.scaler_sha256 == scaler.digest()
    assert artifact.validation_status == "NOT_VALIDATED"
    assert artifact.manifest_sha256 == manifest.digest()

    other = WindowScaler(
        feature_names=scaler.feature_names,
        mean=tuple(value + 2.0 for value in scaler.mean),
        std=scaler.std,
        observed_counts=scaler.observed_counts,
    )
    with pytest.raises(ArtifactError, match="manifest scaler"):
        ModelArtifact.from_manifest(manifest, other)


def test_inference_refuses_an_unbound_or_tampered_artifact():
    session = _real_session()
    builder = PhysioWindowBuilder(WindowConfig(window_s=60, min_events=2))
    scaler = _scaler(session)
    model = FakeModel()
    cutoff = event(sequence=2, at=60).measured_at

    unbound = InferenceSession(session, builder, model=model, scaler=scaler).update(cutoff)
    assert unbound.status == STATUS_NOT_AVAILABLE
    assert unbound.reason == "artifact_not_bound"
    assert unbound.artifact_sha256 is None

    artifact = _artifact(model=model, scaler=scaler)
    ok = InferenceSession(
        session, builder, model=model, scaler=scaler, artifact=artifact
    ).update(cutoff)
    assert ok.status == STATUS_OK

    tampered = _artifact(
        model=model,
        scaler=scaler,
        state_sha256=hashlib.sha256(b"other-weights").hexdigest(),
    )
    refused = InferenceSession(
        session, builder, model=model, scaler=scaler, artifact=tampered
    ).update(cutoff)
    assert refused.status == STATUS_NOT_AVAILABLE
    assert refused.reason.startswith("artifact_error")


def test_inference_requires_a_typed_artifact():
    session = _real_session()
    builder = PhysioWindowBuilder(WindowConfig(window_s=60, min_events=2))
    with pytest.raises(TypeError, match="ModelArtifact"):
        InferenceSession(session, builder, artifact="VALIDATED")  # type: ignore[arg-type]


def test_feature_version_constant_matches_artifact_default():
    assert _artifact().feature_version == WINDOW_FEATURE_VERSION


def test_verify_requires_a_model_state_digest():
    scaler = _scaler()
    artifact = _artifact(scaler=scaler)
    with pytest.raises(ArtifactError, match="state_sha256"):
        artifact.verify(scaler, object())


def test_smoke_manifest_cannot_be_promoted_without_torch():
    from physio.training import TrainingConfig, TrainingManifest

    scaler = _scaler()
    manifest = TrainingManifest(
        model_version="synthetic-artifact-v1",
        feature_names=scaler.feature_names,
        scaler_sha256=scaler.digest(),
        config=TrainingConfig(),
        n_samples=4,
        n_labeled=4,
        identity_trust="TRUSTED",
        status="SMOKE_ONLY",
        checkpoint_sha256=hashlib.sha256(b"checkpoint").hexdigest(),
        state_sha256=FakeModel().state_sha256(),
    )
    with pytest.raises(ArtifactError, match="SMOKE_ONLY"):
        ModelArtifact.from_manifest(
            manifest, scaler, validation_status=VALIDATION_VALIDATED
        )
