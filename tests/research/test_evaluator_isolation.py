"""Evaluator-process isolation tests (F-01 round, 2026-11-02).

The first round already moved scoring out of the trainer's process; the
second hardening round closes the remaining hole: `torch.load` of the full
pickled model executed candidate code inside the evaluator (module-level
code at unpickle import + a weights_only=False pickle could rebind
`sys.modules["runtime.tcm_eval"]._average_precision_from_tensors`). These
tests pin the closed design:

* the evaluator loads state_dict only (`weights_only=True`) into a model
  rebuilt from the frozen architecture whitelist — no candidate import, no
  executable pickle anywhere on the evaluation path;
* the saved model files are hash-checked against the parent's record;
* the fold is rebuilt from the frozen manifest, so in-place tampering with
  the trainer's objects cannot reach the score;
* the runner never imports the candidate either (F-02).
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


def _saved_model(tmp_path, config, records, stem="m0"):
    from research.candidate import build_model
    from research.evaluator import integrity_snapshot
    from research.frozen import build_fold
    from research.model_export import save_eval_model

    fold_data, _ = build_fold(
        config, config["data"]["manifest_path"], config["data"]["dataset_path"],
        0, 4, 0, records=records,
    )
    model = build_model(fold_data)
    saved = save_eval_model(model, tmp_path, stem)
    request = {
        "fold": 0, "folds": 4, "repeat": 0,
        **saved,
        "integrity": integrity_snapshot(config),
    }
    return fold_data, model, request


def test_the_evaluator_scores_state_dict_only(tmp_path):
    from research.evaluator import score_fold
    from runtime.real_data import real_eval_manifest
    from runtime.tcm_eval import TCMTrustedEvaluator

    records = fake_records()
    config, _base, _manifest_path = _config(tmp_path, records)
    fold_data, model, request = _saved_model(tmp_path, config, records)

    scored = score_fold(config, request, records=records)

    # the score must equal the model's own dev score under the trusted
    # evaluator: state_dict + frozen rebuild, nothing else travelled
    direct = TCMTrustedEvaluator(real_eval_manifest(
        _fold_manifest(config, records)
    )).evaluate(model, fold_data.dev, partition="dev")
    assert scored["primary_value"] == pytest.approx(float(direct.primary_value))


def _fold_manifest(config, records):
    from research.frozen import build_fold

    _, data = build_fold(
        config, config["data"]["manifest_path"], config["data"]["dataset_path"],
        0, 4, 0, records=records,
    )
    return data


def test_the_evaluator_refuses_model_files_modified_after_training(tmp_path):
    from research.evaluator import score_fold

    records = fake_records()
    config, _base, _manifest_path = _config(tmp_path, records)
    _fold_data, _model, request = _saved_model(tmp_path, config, records)
    Path(request["weights_path"]).write_bytes(
        Path(request["weights_path"]).read_bytes() + b"x"
    )
    with pytest.raises(RuntimeError, match="changed between training"):
        score_fold(config, request, records=records)


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
        "weights_path": "unused", "architecture_path": "unused",
        "integrity": snapshot,
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


def test_the_evaluation_path_never_executes_candidate_code():
    src_root = Path(__file__).resolve().parents[2] / "src" / "research"
    evaluator_src = (src_root / "evaluator.py").read_text()
    # F-01: the evaluation path has no candidate import and no torch.load at
    # all — deserialization is state_dict + frozen architecture rebuild, in
    # model_export, behind the safe loader
    assert "import research.candidate" not in evaluator_src
    assert "torch.load" not in evaluator_src
    assert "weights_only" not in evaluator_src
    model_export = (src_root / "model_export.py").read_text()
    assert "torch.load(" in model_export
    assert "weights_only=True" in model_export
    assert "weights_only=False" not in model_export
    assert "import research.candidate" not in model_export


def test_runner_hands_the_evaluator_files_not_folds(tmp_path):
    import research.runner as runner_mod

    records = fake_records(40)
    config, base, _ = _config(tmp_path, records)
    config["time_budget_seconds"] = 0.01
    config["worktree_path"] = str(tmp_path / "wt")
    config["editable"] = "src/research/candidate.py"
    worktree = tmp_path / "wt"
    (worktree / "src" / "research").mkdir(parents=True)
    (worktree / "src" / "research" / "candidate.py").write_text("WIDTH = 32\n")
    calls: list[tuple[dict, dict]] = []

    def spawn_training(worker_config, worker_config_path):
        # the training subprocess's config carries trimmed file paths only:
        # no dataset path, no frozen-manifest path, no holdout
        assert "dataset_path" not in worker_config
        assert "manifest_path" not in worker_config
        fit_manifest = json.loads(
            Path(worker_config["fit_manifest_path"]).read_text()
        )
        assert fit_manifest["split"]["holdout"] == []
        assert fit_manifest["split"]["dev"] == []
        fit_records = json.loads(
            Path(worker_config["fit_records_path"]).read_text()
        )
        fit_rows = set(fit_manifest["subject_row_by_id"].values())
        assert fit_records
        assert {r["subject_index"] for r in fit_records} == fit_rows
        return {
            "ok": True,
            "weights_path": "unused.pt",
            "architecture_path": "unused.json",
            "weights_sha256": "0" * 64,
            "architecture_sha256": "0" * 64,
        }

    def spawn(cfg, req):
        # the request hands the evaluator saved model files, not the fold
        calls.append((cfg, req))
        return {"primary_value": 0.5}

    result = runner_mod.run(
        config, folds=2, repeats=1, records=records,
        spawn_training=spawn_training, spawn_evaluator=spawn,
    )
    assert len(calls) == 2 and result["n_folds"] == 2
    assert result["primary_value"] == 0.5
    for _cfg, req in calls:
        assert set(req["integrity"]) == {"manifest_sha256", "dataset_sha256",
                                         "candidate_sha256"}
        assert req["fold"] in (0, 1) and req["folds"] == 2
        assert req["weights_path"] == "unused.pt"


def test_runner_never_touches_the_scoring_machinery_in_process():
    source = runner_source()
    # the in-process evaluator path is what made "the evaluator runs outside
    # the candidate's process" untrue: the runner must not import or call it,
    # and F-02 removes the candidate import entirely
    assert "TCMTrustedEvaluator" not in source
    assert "evaluate_model" not in source
    assert "research.candidate" not in source
    assert "research.evaluator" in source
    assert "research.train_worker" in source


def runner_source() -> str:
    import research.runner

    return Path(research.runner.__file__).read_text()
