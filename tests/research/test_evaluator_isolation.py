"""Evaluator-process isolation tests (adversarial review finding, P0).

The declared contract was "the evaluator runs outside the candidate's
process", but the first implementation scored in the same interpreter that
executed `candidate.train()`: a candidate could rebind
`runtime.tcm_eval.TCMTrustedEvaluator` for the in-process late import, or
poison the shared FoldData's dev targets in place. These tests pin the fixed
design: the runner saves the model, the evaluator REBUILDS the fold from the
frozen manifest in its own process and scores there, and the training inputs
are covered by an integrity snapshot.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from data.mendeley import FEATURE_GROUP_SIZES, SubjectRecord
from data.real_split import build_real_manifest

pytestmark = pytest.mark.synthetic

DATASET_SHA = "b" * 64


def fake_records(n=30, seed=11):
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


def _config(tmp_path, records):
    base = build_real_manifest(records, dataset_sha256=DATASET_SHA).data
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(base, ensure_ascii=False))
    dataset_path = tmp_path / "dataset.xls"
    dataset_path.write_bytes(b"fixture-workbook")
    return (
        {
            "data": {"manifest_path": str(manifest_path), "dataset_path": str(dataset_path)},
            "train_provider": {"batch_size": 8},
            "trainer": {"seed": 0},
        },
        base,
        manifest_path,
    )


def test_the_evaluator_rebuilds_the_fold_so_in_place_tampering_is_futile(tmp_path):
    import torch

    from research.candidate import build_model
    from research.evaluator import integrity_snapshot, score_fold
    from research.frozen import build_fold

    records = fake_records()
    config, base, manifest_path = _config(tmp_path, records)
    fold_data, _ = build_fold(
        config, config["data"]["manifest_path"], config["data"]["dataset_path"],
        0, 4, 0, records=records,
    )
    model = build_model(fold_data)
    model_path = tmp_path / "model.pt"
    torch.save(model, model_path)
    request = {
        "fold": 0, "folds": 4, "repeat": 0,
        "model_path": str(model_path),
        "integrity": integrity_snapshot(config),
    }

    honest = score_fold(config, request, records=records)

    # what a candidate could do under the old in-process design: poison the
    # dev targets of the FoldData it was handed. The evaluator rebuilds its
    # own fold from the frozen manifest, so the poisoned copy must not move
    # the score.
    fold_data.dev._targets["category_labels"][:, 0] = 1.0
    poisoned_trainer_copy = score_fold(config, request, records=records)
    assert poisoned_trainer_copy == honest
    assert isinstance(honest["primary_value"], float)


def test_the_evaluator_refuses_tampered_training_inputs(tmp_path):
    from research.evaluator import integrity_snapshot, score_fold

    records = fake_records()
    config, base, manifest_path = _config(tmp_path, records)
    snapshot = integrity_snapshot(config)
    # the candidate rewrote the manifest during training and left it changed
    base["notes"] = "tampered during training"
    manifest_path.write_text(json.dumps(base, ensure_ascii=False))
    request = {
        "fold": 0, "folds": 4, "repeat": 0,
        "model_path": "unused", "integrity": snapshot,
    }
    with pytest.raises(RuntimeError, match="modified during training"):
        score_fold(config, request, records=records)


def test_the_candidate_import_guard_blocks_frozen_machinery(tmp_path):
    import importlib.util

    from research.evaluator import install_candidate_import_guard

    evil = tmp_path / "candidate.py"
    # a module name that cannot already be in sys.modules: the audit event
    # only fires for imports that actually reach the import machinery, so a
    # cached module would silently bypass the hook
    evil.write_text("import runtime.nonexistent_frozen_probe\n")
    install_candidate_import_guard(evil)
    spec = importlib.util.spec_from_file_location("evil_candidate", evil)
    module = importlib.util.module_from_spec(spec)
    with pytest.raises(RuntimeError, match="may not import"):
        spec.loader.exec_module(module)


def test_runner_hands_the_evaluator_a_request_not_a_fold(tmp_path):
    import research.runner as runner_mod

    records = fake_records(40)
    config, base, _ = _config(tmp_path, records)
    config["time_budget_seconds"] = 0.01
    calls: list[tuple[dict, dict]] = []

    def spawn(cfg, req):
        # the request hands the evaluator a saved model file, not the fold
        assert Path(req["model_path"]).exists()
        calls.append((cfg, req))
        return {"primary_value": 0.5}

    result = runner_mod.run(
        config, folds=2, repeats=1, records=records, spawn_evaluator=spawn
    )
    assert len(calls) == 2 and result["n_folds"] == 2
    assert result["primary_value"] == 0.5
    for cfg, req in calls:
        assert cfg["data"]["manifest_path"] == config["data"]["manifest_path"]
        assert set(req["integrity"]) == {"manifest_sha256", "dataset_sha256"}
        assert req["fold"] in (0, 1) and req["folds"] == 2


def test_runner_never_touches_the_scoring_machinery_in_process():
    source = Path(runner_source()).read_text()
    # the in-process evaluator path is what made "the evaluator runs outside
    # the candidate's process" untrue: the runner must not import or call it
    assert "TCMTrustedEvaluator" not in source
    assert "evaluate_model" not in source
    assert "research.evaluator" in source


def runner_source() -> str:
    import research.runner

    return research.runner.__file__
