"""#95/#97 torch-optional smoke training: masks, gradients, manifest and resume."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from physio.artifacts import ArtifactError, ModelArtifact
from physio.dataset import IDENTITY_SHARED, IDENTITY_TRUSTED, FeatureRow
from physio.model import VALIDATION_VALIDATED
from physio.preprocessing import WindowScaler
from physio.training import (
    TrainingConfig,
    TrainingManifest,
    TrainingRefused,
    load_realtime_model,
    predict_masked,
    train_realtime_smoke,
)
from physio.windows import WindowFeatures

pytestmark = pytest.mark.synthetic
torch = pytest.importorskip("torch")

CUTOFF = datetime(2026, 11, 7, 10, 1, tzinfo=UTC)


def _rows():
    specs = [
        ((1.0, 2.0), (True, True)),
        ((2.0, 3.0), (True, True)),
        ((3.0, 4.0), (True, True)),
        ((4.0, 0.0), (True, False)),
    ]
    rows = []
    for index, (values, mask) in enumerate(specs):
        rows.append(
            FeatureRow(
                feature_names=("f1", "f2"),
                values=values,
                observed_mask=mask,
                valid_mask=mask,
                imputed_mask=(False, False),
                imputation=("none", "none"),
                subject_ref=f"s{index % 2}",
                session_id=f"sess{index}",
                cutoff=CUTOFF,
            )
        )
    return rows


def _scaler(rows):
    windows = [
        WindowFeatures(
            usable=True,
            feature_names=row.feature_names,
            values=row.values,
            mask=row.valid_mask,
            coverage=sum(row.valid_mask) / len(row.valid_mask),
            n_events=1,
            start=CUTOFF,
            end=CUTOFF,
            quality="ok",
            metrics_present=("heart_rate",),
        )
        for row in rows
    ]
    return WindowScaler.fit(windows)


def test_training_refuses_without_labels():
    rows = _rows()
    with pytest.raises(TrainingRefused):
        train_realtime_smoke(
            rows, [None, None, None, None], scaler=_scaler(rows),
            config=TrainingConfig(min_samples=2),
        )


def test_training_refuses_untrusted_identity_without_opt_in():
    rows = _rows()
    with pytest.raises(TrainingRefused):
        train_realtime_smoke(
            rows, [1.0, 2.0, 3.0, 4.0], scaler=_scaler(rows),
            config=TrainingConfig(min_samples=2), identity_trust=IDENTITY_SHARED,
        )


def test_smoke_training_saves_manifest_and_checkpoint(tmp_path):
    rows = _rows()
    manifest = train_realtime_smoke(
        rows,
        [10.0, 20.0, 30.0, 40.0],
        scaler=_scaler(rows),
        config=TrainingConfig(epochs=2, seed=3, hidden_dim=8, fusion_dim=8),
        identity_trust=IDENTITY_TRUSTED,
        out_dir=tmp_path,
    )
    assert manifest.n_labeled == 4
    assert manifest.final_loss is not None
    assert (tmp_path / "model_state.pt").exists()
    assert (tmp_path / "manifest.json").exists()
    loaded = TrainingManifest.load(tmp_path / "manifest.json")
    assert loaded.digest() == manifest.digest()


def test_checkpoint_reload_is_deterministic(tmp_path):
    rows = _rows()
    manifest = train_realtime_smoke(
        rows,
        [10.0, 20.0, 30.0, 40.0],
        scaler=_scaler(rows),
        config=TrainingConfig(epochs=1, seed=5, hidden_dim=8, fusion_dim=8),
        out_dir=tmp_path,
    )
    first = predict_masked(
        rows, checkpoint=tmp_path / "model_state.pt", scaler=_scaler(rows),
        hidden_dim=8, fusion_dim=8,
    )
    second = predict_masked(
        rows, checkpoint=tmp_path / "model_state.pt", scaler=_scaler(rows),
        hidden_dim=8, fusion_dim=8,
    )
    assert first == second
    assert all(0.0 <= value <= 100.0 for value in first)
    assert manifest.final_loss is not None


def test_shared_identity_pipeline_smoke_is_explicit(tmp_path):
    rows = _rows()
    manifest = train_realtime_smoke(
        rows,
        [10.0, 20.0, 30.0, 40.0],
        scaler=_scaler(rows),
        config=TrainingConfig(epochs=1, seed=1, hidden_dim=8, fusion_dim=8),
        identity_trust=IDENTITY_SHARED,
        allow_untrusted_pipeline=True,
        out_dir=tmp_path,
    )
    assert manifest.identity_trust == IDENTITY_SHARED
    assert manifest.status == "SMOKE_ONLY"


def test_state_digest_matches_checkpoint_and_changes_with_weights(tmp_path):
    rows = _rows()
    scaler = _scaler(rows)
    first = train_realtime_smoke(
        rows,
        [10.0, 20.0, 30.0, 40.0],
        scaler=scaler,
        config=TrainingConfig(epochs=1, seed=3, hidden_dim=8, fusion_dim=8),
        out_dir=tmp_path / "first",
    )
    model = load_realtime_model(
        checkpoint=tmp_path / "first" / "model_state.pt",
        scaler=scaler,
        hidden_dim=8,
        fusion_dim=8,
    )
    assert model.state_sha256() == first.state_sha256
    assert first.scaler_sha256 == scaler.digest()

    second = train_realtime_smoke(
        rows,
        [10.0, 20.0, 30.0, 40.0],
        scaler=scaler,
        config=TrainingConfig(epochs=1, seed=4, hidden_dim=8, fusion_dim=8),
        out_dir=tmp_path / "second",
    )
    assert second.state_sha256 != first.state_sha256


def test_trained_artifact_binds_scaler_and_runtime_state(tmp_path):
    rows = _rows()
    scaler = _scaler(rows)
    manifest = train_realtime_smoke(
        rows,
        [10.0, 20.0, 30.0, 40.0],
        scaler=scaler,
        config=TrainingConfig(epochs=1, seed=7, hidden_dim=8, fusion_dim=8),
        out_dir=tmp_path,
    )
    model = load_realtime_model(
        checkpoint=tmp_path / "model_state.pt",
        scaler=scaler,
        hidden_dim=8,
        fusion_dim=8,
    )
    artifact = ModelArtifact.from_manifest(manifest, scaler)
    artifact.verify(scaler, model)
    assert artifact.validation_status == "NOT_VALIDATED"

    shifted = WindowScaler(
        feature_names=scaler.feature_names,
        mean=tuple(value + 1.0 for value in scaler.mean),
        std=scaler.std,
        observed_counts=scaler.observed_counts,
    )
    with pytest.raises(ArtifactError, match="scaler"):
        artifact.verify(shifted, model)


def test_smoke_manifest_cannot_be_promoted_to_validated(tmp_path):
    rows = _rows()
    scaler = _scaler(rows)
    manifest = train_realtime_smoke(
        rows,
        [10.0, 20.0, 30.0, 40.0],
        scaler=scaler,
        config=TrainingConfig(epochs=1, seed=9, hidden_dim=8, fusion_dim=8),
        out_dir=tmp_path,
    )
    with pytest.raises(ArtifactError, match="SMOKE_ONLY"):
        ModelArtifact.from_manifest(
            manifest, scaler, validation_status=VALIDATION_VALIDATED
        )
