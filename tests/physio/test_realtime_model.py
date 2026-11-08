"""#97 realtime model tests. Skipped cleanly when PyTorch is unavailable."""

from __future__ import annotations

import pytest

from physio.model import (
    REALTIME_INPUT_VERSION,
    REALTIME_TARGET_VERSION,
    RealtimeFatigueModel,
    RealtimeModelConfig,
    WindowModelBatch,
    make_realtime_input_spec,
)
from physio.windows import WINDOW_FEATURE_NAMES

pytestmark = pytest.mark.synthetic

torch = pytest.importorskip("torch", reason="torch 未安装")

from core.batch import ModelBatch  # noqa: E402

N_FEATURES = len(WINDOW_FEATURE_NAMES)


def test_input_spec_declares_the_window_modality():
    spec = make_realtime_input_spec(N_FEATURES)
    assert spec.version == REALTIME_INPUT_VERSION
    assert spec.names() == ("window_features",)
    modality = spec.get("window_features")
    assert modality.kind == "tabular"
    assert modality.dim == N_FEATURES


def test_config_defaults_are_not_validated():
    config = RealtimeModelConfig()
    assert config.target_version == REALTIME_TARGET_VERSION
    assert config.validation_status == "NOT_VALIDATED"
    with pytest.raises(ValueError, match="validation_status"):
        RealtimeModelConfig(validation_status="MAYBE")
    with pytest.raises(ValueError, match="dropout"):
        RealtimeModelConfig(dropout=1.0)


def test_forward_returns_one_bounded_fatigue_state():
    model = RealtimeFatigueModel(
        N_FEATURES,
        RealtimeModelConfig(hidden_dim=8, fusion_dim=8, dropout=0.0),
    )
    values = [0.0] * N_FEATURES
    values[0] = 1.5
    mask = [False] * N_FEATURES
    mask[0] = True
    batch = ModelBatch(
        features={"window_features": torch.tensor([values], dtype=torch.float32)},
        availability={"window_features": torch.tensor([True], dtype=torch.bool)},
    )
    batch.validate(model.input_spec)
    output = model(batch, feature_mask=mask)
    prediction = output.predictions["fatigue_state"]
    assert tuple(prediction.shape) == (1, 1)
    assert 0.0 <= float(prediction[0, 0]) <= 100.0
    model.eval()
    assert 0.0 <= float(model(batch, feature_mask=mask).predictions["fatigue_state"][0, 0]) <= 100.0


def test_window_model_batch_converts_to_model_batch():
    values = tuple(0.0 for _ in WINDOW_FEATURE_NAMES)
    mask = tuple(False for _ in WINDOW_FEATURE_NAMES)
    window_batch = WindowModelBatch(
        feature_names=WINDOW_FEATURE_NAMES,
        values=values,
        mask=mask,
    )
    batch = window_batch.to_model_batch(make_realtime_input_spec(N_FEATURES))
    assert batch.batch_size == 1
    assert bool(batch.availability["window_features"][0]) is False
    assert tuple(batch.features["window_features"].shape) == (1, N_FEATURES)


def test_model_rejects_a_wrong_input_dimension():
    model = RealtimeFatigueModel(
        N_FEATURES,
        RealtimeModelConfig(hidden_dim=8, fusion_dim=8, dropout=0.0),
    )
    batch = ModelBatch(
        features={
            "window_features": torch.zeros((1, N_FEATURES - 1), dtype=torch.float32)
        },
        availability={"window_features": torch.tensor([True], dtype=torch.bool)},
    )
    with pytest.raises(ValueError, match="expects dim"):
        model(batch)
