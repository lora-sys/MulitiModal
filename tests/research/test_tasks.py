"""Synthetic task-adapter regression tests for candidate preflight inputs."""
from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.synthetic


def test_resolve_editable_anchor_and_denylist():
    """Editable surface v2: `editable` is an anchor, any writable src file is
    allowed, frozen paths are refused."""
    from research.tasks import adapter_for

    constitution = adapter_for("constitution")
    syndrome = adapter_for("syndrome")

    # default: each task starts from its own candidate anchor
    assert constitution.resolve_editable() == "src/research/candidate.py"
    assert syndrome.resolve_editable() == "src/research/candidate_sd.py"
    # any other model/train/feature source is now a legal anchor
    assert syndrome.resolve_editable("src/core/models.py") == "src/core/models.py"
    assert constitution.resolve_editable("src/research/new_encoder.py") == (
        "src/research/new_encoder.py"
    )

    for frozen in ("src/research/evaluator.py", "src/research/frozen.py",
                   "src/data/x.py", "AGENTS.md", "tests/x.py"):
        with pytest.raises(ValueError, match="frozen path"):
            syndrome.resolve_editable(frozen)
    with pytest.raises(ValueError, match="src/\*\*/\*.py"):
        syndrome.resolve_editable("README.md")


def test_write_synthetic_syndrome_preflight_inputs(tmp_path):
    from research.tasks import write_synthetic_preflight_inputs

    reference_manifest = {
        "modalities": {
            "text": {
                "kind": "hashed_text",
                "dim": 65536,
            }
        },
        "n_classes": 3,
        "label_taxonomy_version": "synthetic-tcm-v1",
    }

    manifest_path, records_path = write_synthetic_preflight_inputs(
        "syndrome",
        reference_manifest,
        tmp_path,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = json.loads(records_path.read_text(encoding="utf-8"))
    assert manifest["task"] == "syndrome"
    assert records
    assert all(
        record["provenance"]["source"] == "synthetic-candidate-preflight"
        for record in records
    )
