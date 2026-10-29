"""Spec and batch contract tests: rejection paths are the point (#48)."""

import pytest
import torch
from conftest import physio_batch, physio_spec, tcm_batch, tcm_spec

from core.batch import ModelBatch
from core.specs import (
    InputSpecSet,
    ModalitySpec,
    TargetSpec,
    make_physio_input_spec,
)

pytestmark = pytest.mark.synthetic


def test_modality_spec_rejects_bad_dims():
    with pytest.raises(ValueError):
        ModalitySpec("x", "tabular", 0)
    with pytest.raises(ValueError):
        ModalitySpec("x", "quantum", 4)


def test_input_spec_rejects_duplicate_names():
    with pytest.raises(ValueError, match="duplicate"):
        InputSpecSet("v", (ModalitySpec("a", "tabular", 1), ModalitySpec("a", "tabular", 2)))


def test_spec_get_unknown_modality_raises():
    with pytest.raises(KeyError, match="not.*approved|not part"):
        tcm_spec().get("eeg_during")


def test_spec_dict_roundtrip():
    spec = make_physio_input_spec(5, 6)
    assert InputSpecSet.from_dict(spec.to_dict()) == spec


def test_batch_validate_accepts_b1_and_b16():
    for n in (1, 16):
        ModelBatch(features=tcm_batch(n, with_targets=False).features).validate(tcm_spec())


def test_batch_rejects_during_post_into_physio_contract():
    # The #23/#31 Pre-only boundary: During/Post must fail loudly, never slip in.
    batch = physio_batch(extra_during_post=True)
    with pytest.raises(ValueError, match="not an approved input"):
        batch.validate(physio_spec())


def test_batch_rejects_dim_mismatch():
    batch = tcm_batch(with_targets=False)
    batch.features["body_features"] = torch.rand(8, 3)
    with pytest.raises(ValueError, match="dim"):
        batch.validate(tcm_spec())


def test_batch_rejects_nan_features():
    batch = tcm_batch(with_targets=False, inject_nan=True)
    with pytest.raises(ValueError, match="NaN"):
        batch.validate(tcm_spec())


def test_batch_rejects_bad_availability():
    batch = tcm_batch(with_targets=False, zero_out_availability="body_features")
    batch.availability["body_features"] = batch.availability["body_features"].float()
    with pytest.raises(ValueError, match="bool"):
        batch.validate(tcm_spec())


def test_batch_targets_require_masks():
    batch = tcm_batch(with_targets=True)
    batch.target_mask = None
    with pytest.raises(ValueError, match="together"):
        batch.validate(tcm_spec())


def test_target_spec_range_validation():
    with pytest.raises(ValueError):
        TargetSpec("t", 1, "s", value_range=(5.0, 5.0))
