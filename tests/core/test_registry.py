"""Registry save/load/provenance and device contract tests (#48)."""

import json

import pytest
import torch
from conftest import HIDDEN, physio_batch, tcm_batch

from core.models import default_physio_model, default_tcm_model
from core.registry import (
    MODEL_BUILDERS,
    batch_predict,
    load_model,
    resolve_device,
    save_model,
)

pytestmark = pytest.mark.synthetic


def build_and_save_tcm(tmp_path, **overrides):
    torch.manual_seed(7)
    model, config = default_tcm_model(hidden_dim=HIDDEN, fusion_dim=HIDDEN, **overrides)
    out_dir = save_model(model, config, tmp_path / "tcm_artifact", source_commit="test-sha")
    return model, config, out_dir


def test_registered_model_types():
    assert set(MODEL_BUILDERS) == {"tcm_constitution_v0", "physio_bpsi_efri_v0"}


def test_save_load_prediction_consistency(tmp_path):
    model, config, out_dir = build_and_save_tcm(tmp_path)
    batch = tcm_batch()
    before = batch_predict(model, batch, device="cpu")

    loaded, metadata = load_model(out_dir, device="cpu")
    after = batch_predict(loaded, batch, device="cpu")

    torch.testing.assert_close(
        before.predictions["constitution_scores"],
        after.predictions["constitution_scores"],
    )
    assert metadata["model_type"] == "tcm_constitution_v0"
    assert metadata["source_commit"] == "test-sha"
    assert metadata["input_spec"]["version"] == "tcm-input-v0"
    assert len(metadata["target_specs"]) == 1


def test_metadata_contains_provenance_fields(tmp_path):
    _, _, out_dir = build_and_save_tcm(tmp_path)
    metadata = json.loads((out_dir / "metadata.json").read_text())
    for key in ("model_type", "format_version", "config", "input_spec",
                "target_specs", "torch_version", "source_commit", "weights_sha256"):
        assert key in metadata, key
    assert len(metadata["weights_sha256"]) == 64


def test_tampered_weights_are_rejected(tmp_path):
    _, _, out_dir = build_and_save_tcm(tmp_path)
    weights = out_dir / "weights.pt"
    data = weights.read_bytes() + b"\x00"
    weights.write_bytes(data)
    with pytest.raises(ValueError, match="checksum"):
        load_model(out_dir)


def test_architecture_mismatch_fails_loudly(tmp_path):
    _, _, out_dir = build_and_save_tcm(tmp_path)
    import json as jsonlib

    metadata_path = out_dir / "metadata.json"
    metadata = jsonlib.loads(metadata_path.read_text())
    metadata["config"]["hidden_dim"] = HIDDEN * 4
    metadata_path.write_text(jsonlib.dumps(metadata))
    with pytest.raises(RuntimeError):  # strict load_state_dict must not stay silent
        load_model(out_dir)


def test_physio_save_load_roundtrip(tmp_path):
    torch.manual_seed(9)
    model, config = default_physio_model(hidden_dim=HIDDEN, fusion_dim=HIDDEN)
    out_dir = save_model(model, config, tmp_path / "physio_artifact")
    loaded, metadata = load_model(out_dir)
    assert metadata["model_type"] == "physio_bpsi_efri_v0"
    assert metadata["input_spec"]["version"] == "physio-input-pre-only-v0"

    batch = physio_batch()
    torch.testing.assert_close(
        batch_predict(model, batch).predictions["efri"],
        batch_predict(loaded, batch).predictions["efri"],
    )


def test_resolve_device_defaults_to_cpu():
    assert resolve_device().type == "cpu"
    assert resolve_device("cpu").type == "cpu"
    with pytest.raises(ValueError, match="unknown device preference"):
        resolve_device("cuda_nope")
    # "auto" must never raise: falls back to cpu when no accelerator exists.
    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        assert resolve_device("auto").type == "cpu"


def test_unregistered_model_cannot_be_saved(tmp_path):
    from torch import nn

    with pytest.raises(ValueError, match="not registered"):
        save_model(nn.Linear(2, 2), {}, tmp_path / "bogus")
