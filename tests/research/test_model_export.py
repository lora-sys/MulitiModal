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
from data.real_split import build_real_manifest, pool_manifest
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
    selection = pool_manifest(base, records).data
    selection_rows = set(selection["subject_row_by_id"].values())
    selection_records = [
        record for record in records if record.subject_index in selection_rows
    ]
    config = {
        "train_provider": {"batch_size": 8},
        "trainer": {"seed": 0},
    }
    fold_data, _ = build_fold(
        config,
        selection,
        fold=0,
        folds=4,
        repeat=0,
        records=selection_records,
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


def _tiny_linear_architecture():
    import torch.nn as nn

    from core.specs import InputSpecSet, ModalitySpec

    input_spec = InputSpecSet(
        version="test-input-v0",
        modalities=(ModalitySpec("features", "tabular", 2),),
    )

    class TinyLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_spec = input_spec
            self.model_type = "synthetic-tiny"
            self.linear = nn.Linear(2, 2)

        def forward(self, values):
            return self.linear(values)

    return export_architecture(TinyLinear())


def test_rebuild_allows_hashed_text_dimension_plus_availability_flag():
    arch = _tiny_linear_architecture()
    arch["modules"]["linear"]["args"].update(
        in_features=65_537,
        out_features=2,
    )

    rebuilt = rebuild_model(arch, task="syndrome")

    assert rebuilt.linear.weight.shape == (2, 65_537)


def test_rebuild_rejects_oversized_linear_before_module_allocation(monkeypatch):
    from research import model_export

    arch = _tiny_linear_architecture()
    arch["modules"]["linear"]["args"].update(
        in_features=65_538,
        out_features=140,
    )

    def allocation_must_not_start(**_kwargs):
        pytest.fail("oversized architecture reached an nn.Module constructor")

    monkeypatch.setitem(
        model_export.MODULE_BUILDERS, "nn.Linear", allocation_must_not_start
    )
    with pytest.raises(ValueError, match="Linear in_features"):
        rebuild_model(arch, task="syndrome", expected_n_classes=148)


def test_rebuild_rejects_oversized_get_attr_before_tensor_allocation(monkeypatch):
    import torch
    import torch.nn as nn

    from core.specs import InputSpecSet, ModalitySpec
    from research import model_export

    input_spec = InputSpecSet(
        version="test-input-v0",
        modalities=(ModalitySpec("features", "tabular", 2),),
    )

    class TinyBuffer(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_spec = input_spec
            self.model_type = "synthetic-buffer"
            self.register_buffer("offset", torch.ones(2))

        def forward(self, values):
            return values + self.offset

    arch = export_architecture(TinyBuffer())
    attr = next(node["attr"] for node in arch["graph"] if node["op"] == "get_attr")
    attr["shape"] = [10_000_000, 10_000_000]

    def allocation_must_not_start(shape, *args, **kwargs):
        pytest.fail(f"oversized get_attr reached torch.empty: {shape!r}")

    monkeypatch.setattr(model_export.torch, "empty", allocation_must_not_start)
    with pytest.raises(ValueError, match="tensor|element|budget|shape"):
        rebuild_model(arch, task="constitution", expected_n_classes=9)


def test_load_eval_model_rejects_architecture_json_over_limit_before_weights(
    tmp_path, monkeypatch,
):
    import torch

    from research import model_export

    weights = tmp_path / "weights.pt"
    architecture = tmp_path / "architecture.json"
    weights.write_bytes(b"synthetic")
    architecture.write_bytes(b" " * (model_export.MAX_ARCHITECTURE_BYTES + 1))

    def weights_must_not_load(*_args, **_kwargs):
        pytest.fail("weights must not load before bounded architecture parsing")

    monkeypatch.setattr(torch, "load", weights_must_not_load)
    with pytest.raises(ValueError, match="architecture.*limit"):
        load_eval_model(weights, architecture, task="constitution")


def test_checkpoint_huge_tensor_shape_is_rejected_before_state_copy(tmp_path, monkeypatch):
    import torch

    from research import model_export

    arch = _tiny_linear_architecture()
    architecture_path = tmp_path / "architecture.json"
    architecture_path.write_text(json.dumps(arch))
    backing = torch.tensor([1.0])
    huge_view = torch.as_strided(backing, (1_000_000_000,), (0,))
    weights_path = tmp_path / "oversized-shape.pt"
    torch.save(
        {"linear.weight": huge_view, "linear.bias": torch.zeros(2)},
        weights_path,
    )
    assert weights_path.stat().st_size < 1024 * 1024

    def allocation_must_not_start(**_kwargs):
        pytest.fail("checkpoint shape mismatch must fail before model construction")
    monkeypatch.setitem(
        model_export.MODULE_BUILDERS, "nn.Linear", allocation_must_not_start
    )
    def state_copy_must_not_start(*_args, **_kwargs):
        pytest.fail("checkpoint tensor shape must be checked before state copy")

    monkeypatch.setattr(
        torch.nn.Module, "load_state_dict", state_copy_must_not_start
    )
    with pytest.raises(ValueError, match="shape|element|tensor|weights"):
        load_eval_model(weights_path, architecture_path, task="constitution")
