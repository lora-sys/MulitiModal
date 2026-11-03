"""Synthetic task-adapter regression tests for candidate preflight inputs."""
from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.synthetic


def test_resolve_editable_defaults_and_rejects_cross_task_paths():
    from research.task_sd import SyndromeAdapter
    from research.tasks import adapter_for

    constitution = adapter_for("constitution")
    syndrome = adapter_for("syndrome")

    # default: each task starts from its own frozen candidate path
    assert constitution.resolve_editable() == "src/research/candidate.py"
    assert syndrome.resolve_editable() == "src/research/candidate_sd.py"
    assert syndrome.resolve_editable("src/research/candidate_sd.py") == (
        "src/research/candidate_sd.py"
    )

    # an explicitly routed foreign candidate fails closed before any run
    with pytest.raises(ValueError, match="does not match task"):
        syndrome.resolve_editable("src/research/candidate.py")
    with pytest.raises(ValueError, match="does not match task"):
        constitution.resolve_editable("src/research/candidate_sd.py")
    with pytest.raises(ValueError, match="does not match task"):
        SyndromeAdapter().resolve_editable("src/research/nonsense.py")


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
