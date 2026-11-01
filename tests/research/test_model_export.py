"""Frozen architecture export/rebuild tests (F-01's substrate).

The evaluator never imports the candidate and never torch.loads a pickled
model: the training subprocess saves a state_dict plus the frozen-architecture
JSON produced here. These tests pin the properties the isolation rests on:

* export -> rebuild -> strict state_dict load reproduces the model's outputs
  exactly, with no candidate code on the rebuild path;
* anything outside the frozen vocabulary refuses to export (fails the round
  honestly instead of widening evaluation);
* the weights file is loadable with weights_only=True (no pickled code).
"""

from __future__ import annotations

import json
import random

import pytest

from data.mendeley import FEATURE_GROUP_SIZES, SubjectRecord
from data.real_split import build_real_manifest
from research.candidate import FusionMLP, build_model
from research.frozen import build_fold
from research.model_export import (
    ArchitectureExportRefused,
    export_architecture,
    load_eval_model,
    rebuild_model,
    save_eval_model,
)

pytestmark = pytest.mark.synthetic

DATASET_SHA = "e" * 64


def fake_records(n=30, seed=5):
    rng = random.Random(seed)
    records = []
    for i in range(1, n + 1):
        scores = [None] + [float(rng.randint(10, 90)) for _ in range(8)]
        records.append(
            SubjectRecord(
                subject_index=i,
                features={
                    name: [rng.random() for _ in range(size)]
                    for name, size in FEATURE_GROUP_SIZES.items()
                },
                scores=scores,
                score_mask=[s is not None for s in scores],
                provenance={"source": "test-fixture"},
            )
        )
    return records


def _fold(tmp_path, records):
    base = build_real_manifest(records, dataset_sha256=DATASET_SHA).data
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(base, ensure_ascii=False))
    config = {
        "data": {"manifest_path": str(path), "dataset_path": "unused"},
        "train_provider": {"batch_size": 8},
        "trainer": {"seed": 0},
    }
    fold_data, _ = build_fold(
        config, str(path), "unused", fold=0, folds=4, repeat=0, records=records
    )
    return fold_data


def test_export_rebuild_reproduces_the_model_exactly(tmp_path):
    import torch

    records = fake_records()
    fold = _fold(tmp_path, records)
    model = build_model(fold)
    batch = next(iter(fold.train)).batch

    arch = export_architecture(model)
    rebuilt = rebuild_model(arch)
    rebuilt.load_state_dict(model.state_dict(), strict=True)

    with torch.no_grad():
        original = model(batch)
        copy = rebuilt(batch)
    for key in original.predictions:
        assert torch.equal(original.predictions[key], copy.predictions[key])
    for key in original.auxiliary:
        assert torch.equal(original.auxiliary[key], copy.auxiliary[key])
    # the rebuilt model carries the contract the trusted evaluator checks
    assert rebuilt.input_spec.version == model.input_spec.version
    assert rebuilt.model_type == model.model_type
    # the rebuild is NOT the candidate's class: no candidate code ran
    assert not isinstance(rebuilt, FusionMLP)


def test_saved_weights_load_with_weights_only_and_score_identically(tmp_path):
    import torch

    records = fake_records()
    fold = _fold(tmp_path, records)
    model = build_model(fold)
    saved = save_eval_model(model, tmp_path, "m0")

    # the weights file must be safe deserialization: tensors only, no code
    state = torch.load(saved["weights_path"], weights_only=True)
    assert isinstance(state, dict) and state
    assert (tmp_path / "m0-architecture.json").exists()

    loaded = load_eval_model(saved["weights_path"], saved["architecture_path"])
    batch = next(iter(fold.train)).batch
    with torch.no_grad():
        assert torch.equal(
            model(batch).predictions["category_probs"],
            loaded(batch).predictions["category_probs"],
        )


def test_non_whitelisted_module_refuses_to_export(tmp_path):
    import torch.nn as nn

    class Odd(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(4, 4)

        def forward(self, x):
            out, _ = self.gru(x)
            return out

    with pytest.raises(ArchitectureExportRefused, match="frozen"):
        export_architecture(Odd())


def test_non_whitelisted_call_target_refuses_to_export(tmp_path):
    import torch
    import torch.nn as nn

    class Odd(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, x):
            # plain-python helpers are traced through transparently (their
            # tensor ops get recorded individually), but a call to something
            # outside the frozen function whitelist is refused
            return torch.norm(self.lin(x))

    with pytest.raises(ArchitectureExportRefused, match="whitelist"):
        export_architecture(Odd())


def test_a_forward_used_registered_buffer_exports(tmp_path):
    # H-04: the get_attr check compared attribute NAME strings against the
    # tensor objects collected from named_parameters/named_buffers, so every
    # get_attr node was refused — including a plain register_buffer the
    # forward actually uses. Names must be collected, and the export must
    # round-trip.
    import torch
    from torch import nn

    records = fake_records()
    fold = _fold(tmp_path, records)
    dim = fold.input_spec.modalities[0].dim

    class Gated(nn.Module):
        input_spec = fold.input_spec
        model_type = "test-gated"

        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(dim, dim)
            self.register_buffer("gate", torch.full((dim,), 0.5))

        def forward(self, x):
            return self.lin(x) * self.gate

    model = Gated()
    arch = export_architecture(model)
    rebuilt = rebuild_model(arch)
    rebuilt.load_state_dict(model.state_dict(), strict=True)
    x = torch.randn(2, dim)
    with torch.no_grad():
        assert torch.equal(model(x), rebuilt(x))


def test_rebuild_refuses_a_non_whitelisted_class_in_the_json(tmp_path):
    records = fake_records()
    fold = _fold(tmp_path, records)
    arch = export_architecture(build_model(fold))
    arch["modules"]["enc"] = {"class": "nn.GRU", "args": {}}
    with pytest.raises(ValueError, match="non-whitelisted"):
        rebuild_model(arch)
