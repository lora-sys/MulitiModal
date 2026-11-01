"""One-shot training subprocess tests (F-02, F-06, F-08).

The structural rule: candidate code runs ONLY in `research.train_worker`, a
one-shot subprocess whose data inputs are trimmed files (no dev, no holdout,
no full-set path) and whose outputs are a state_dict file + a frozen
architecture description. These tests pin:

* the runner never imports the candidate (mechanical source check);
* the candidate-facing FoldData has dev=None and holds only fold rows;
* the worker refuses code whose SHA is not the recorded one;
* the real subprocess chain (train_worker -> evaluator) produces a scoreable
  state_dict-only model.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from data.mendeley import FEATURE_GROUP_SIZES, SubjectRecord, records_to_json
from data.real_split import build_real_manifest, fold_manifest
from research.frozen import selection_fold_ids, training_fold_data

pytestmark = pytest.mark.synthetic

DATASET_SHA = "c" * 64
SRC_ROOT = Path(__file__).resolve().parents[2] / "src"


def fake_records(n=30, seed=13):
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


def _fit_inputs(tmp_path, records, fold=0, folds=4):
    base = build_real_manifest(records, dataset_sha256=DATASET_SHA).data
    manifest_path = tmp_path / "frozen.json"
    manifest_path.write_text(json.dumps(base, ensure_ascii=False))
    train_ids, _dev_ids = selection_fold_ids(base, fold, folds, repeat=0)
    fit_manifest = fold_manifest(base, records, train_ids, [], notes="fit").data
    fit_rows = set(fit_manifest["subject_row_by_id"].values())
    fit_records = [r for r in records if r.subject_index in fit_rows]
    fit_manifest_path = tmp_path / "fit-manifest.json"
    fit_manifest_path.write_text(json.dumps(fit_manifest, ensure_ascii=False))
    fit_records_path = tmp_path / "fit-records.json"
    fit_records_path.write_text(records_to_json(fit_records))
    return fit_manifest, fit_manifest_path, fit_records_path


def test_training_fold_data_has_no_dev_and_only_train_rows(tmp_path):
    records = fake_records()
    base = build_real_manifest(records, dataset_sha256=DATASET_SHA).data
    fit_manifest, _, fit_records_path = _fit_inputs(tmp_path, records)
    from data.mendeley import records_from_json

    fit_records = records_from_json(fit_records_path.read_text())
    config = {"train_batch_size": 8, "seed": 0}
    fold = training_fold_data(fit_manifest, fit_records, config)

    # F-06: the candidate-visible fold has NO dev provider at all
    assert fold.dev is None
    assert fold.n_dev_subjects == 0
    train_rows = set(fold.train.records_by_row)
    manifest_rows = set(fit_manifest["subject_row_by_id"].values())
    assert train_rows == manifest_rows
    holdout_rows = {
        base["subject_row_by_id"][sid] for sid in base["split"]["holdout"]
    }
    pool_rows = {
        base["subject_row_by_id"][sid]
        for sid in base["split"]["train"] + base["split"]["dev"]
    }
    assert holdout_rows
    assert not train_rows & holdout_rows
    # the selection pool is train+dev, so fold-train ⊆ pool is the contract
    assert train_rows <= pool_rows


def test_worker_refuses_code_that_is_not_the_recorded_sha(tmp_path):
    records = fake_records()
    fit_manifest, fit_manifest_path, fit_records_path = _fit_inputs(
        tmp_path, records
    )
    worktree = tmp_path / "wt"
    (worktree / "src" / "research").mkdir(parents=True)
    candidate = worktree / "src" / "research" / "candidate.py"
    candidate.write_text("WIDTH = 32\n")
    config = {
        "worktree_path": str(worktree),
        "editable": "src/research/candidate.py",
        "candidate_sha256": "0" * 64,
        "seed": 0,
        "time_budget_seconds": 0.05,
        "train_batch_size": 8,
        "fit_manifest_path": str(fit_manifest_path),
        "fit_records_path": str(fit_records_path),
        "output_dir": str(tmp_path),
        "output_stem": "m0",
    }
    from research.train_worker import fit

    with pytest.raises(RuntimeError, match="refusing to run code"):
        fit(config)


def test_subprocess_chain_trains_and_scores_state_dict_only(tmp_path):
    """The real F-02 chain: runner-spawned train_worker subprocess, then the
    evaluator subprocess, over files only — no candidate import in between."""
    records = fake_records()
    _fit_manifest, fit_manifest_path, fit_records_path = _fit_inputs(
        tmp_path, records
    )
    worktree = tmp_path / "wt"
    _link_src(worktree)
    candidate_sha = _file_sha(worktree / "src" / "research" / "candidate.py")
    config = {
        "worktree_path": str(worktree),
        "editable": "src/research/candidate.py",
        "candidate_sha256": candidate_sha,
        "seed": 0,
        "time_budget_seconds": 0.05,
        "train_batch_size": 8,
        "fit_manifest_path": str(fit_manifest_path),
        "fit_records_path": str(fit_records_path),
        "output_dir": str(tmp_path / "out"),
        "output_stem": "m0",
    }
    config_path = tmp_path / "worker-config.json"
    config_path.write_text(json.dumps(config))
    result = subprocess.run(
        [sys.executable, "-m", "research.train_worker",
         "--config", str(config_path)],
        cwd=worktree, text=True, capture_output=True, timeout=120,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(tmp_path),
             "PYTHONPATH": str(worktree / "src")},
    )
    payload = _last_json(result)
    assert payload["ok"], payload
    assert payload["fit_seconds"] >= 0

    import torch

    state = torch.load(payload["weights_path"], weights_only=True)
    assert isinstance(state, dict) and state

    (tmp_path / "dataset.xls").write_bytes(b"fixture-workbook")
    (tmp_path / "all-records.json").write_text(records_to_json(records))
    eval_config = {
        "data": {
            "manifest_path": str(tmp_path / "frozen.json"),
            "dataset_path": str(tmp_path / "dataset.xls"),
            "records_path": str(tmp_path / "all-records.json"),
        },
        "train_provider": {"batch_size": 8},
        "trainer": {"seed": 0},
    }
    from research.evaluator import integrity_snapshot

    request = {
        "fold": 0, "folds": 4, "repeat": 0,
        "weights_path": payload["weights_path"],
        "architecture_path": payload["architecture_path"],
        "weights_sha256": payload["weights_sha256"],
        "architecture_sha256": payload["architecture_sha256"],
        "integrity": integrity_snapshot(eval_config),
    }
    eval_config_path = tmp_path / "eval-config.json"
    eval_config_path.write_text(json.dumps(eval_config))
    request_path = tmp_path / "eval-request.json"
    request_path.write_text(json.dumps(request))
    scored = subprocess.run(
        [sys.executable, "-m", "research.evaluator",
         "--config", str(eval_config_path), "--request", str(request_path)],
        cwd=worktree, text=True, capture_output=True, timeout=120,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(tmp_path),
             "PYTHONPATH": str(worktree / "src")},
    )
    score = _last_json(scored)
    assert score["ok"], score
    assert isinstance(score["primary_value"], float)


def test_runner_source_never_imports_the_candidate():
    runner_src = (SRC_ROOT / "research" / "runner.py").read_text()
    # F-02: the runner is an orchestrator — candidate code never runs there,
    # so a monkeypatched subprocess.run cannot forge its own score
    assert "import research.candidate" not in runner_src
    assert "research.candidate" not in runner_src
    assert "TCMTrustedEvaluator" not in runner_src
    # the one-shot worker is the only module that imports the candidate
    worker_src = (SRC_ROOT / "research" / "train_worker.py").read_text()
    assert "import research.candidate as candidate" in worker_src
    # finalize.py is itself a one-shot FIT subprocess for the champion refit
    # (fit only — its holdout read lives in the evaluator subprocess, pinned
    # by the finalize tests), so it may import the candidate there
    for module in ("evaluator.py", "frozen.py", "guard.py",
                   "loop.py", "model_export.py"):
        assert "import research.candidate" not in (SRC_ROOT / "research" / module).read_text(), (
            f"{module} must not execute candidate code"
        )


def _link_src(worktree: Path) -> None:
    """A fake worktree whose src is the real package: subprocesses import the
    real frozen machinery while the candidate file is the recorded baseline."""
    src = worktree / "src"
    src.mkdir(parents=True)
    for item in SRC_ROOT.iterdir():
        (src / item.name).symlink_to(item)


def _file_sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _last_json(result: subprocess.CompletedProcess) -> dict:
    lines = [line for line in result.stdout.strip().splitlines() if line.strip()]
    assert lines, result.stdout + result.stderr
    return json.loads(lines[-1])
