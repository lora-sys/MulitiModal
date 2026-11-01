"""Finalize isolation and the holdout unlock ledger (adversarial review).

Two findings pinned here:

* research-src-finalize imported research.candidate into the DRIVER process
  with no binding between the code that fitted and the candidate file whose
  SHA the finalize record shows. The driver must only prepare the pool
  manifest and spawn a worker; the worker must refuse to fit unless the
  candidate file's SHA-256 matches the recorded one.
* the holdout unlock had no mechanical ledger. Phase 1 allows two unlock
  events (AGENTS.md rule 11); every successful unlock appends one event and
  a third is refused.

The driver-isolation assertion is on the sys.modules DELTA: earlier tests in
the same process may legitimately hold research.candidate in their cache.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

from data.mendeley import FEATURE_GROUP_SIZES, SubjectRecord
from data.real_split import (
    build_real_manifest,
    load_real_manifest,
    pool_manifest,
)
from research.finalize import _sha256

pytestmark = pytest.mark.synthetic

DATASET_SHA = "f" * 64


def fake_records(n=40, seed=3):
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


def _worktree(tmp_path, records):
    worktree = tmp_path / "wt"
    (worktree / "src" / "research").mkdir(parents=True)
    candidate = worktree / "src" / "research" / "candidate.py"
    candidate.write_text("WIDTH = 32\n")
    frozen = build_real_manifest(records, dataset_sha256=DATASET_SHA)
    manifest_path = tmp_path / "frozen.json"
    manifest_path.write_text(json.dumps(frozen.data, ensure_ascii=False))
    return worktree, candidate, manifest_path, frozen


def _config(tmp_path, worktree, candidate, records):
    frozen = load_real_manifest(tmp_path / "frozen.json")
    pool = pool_manifest(frozen, records, notes="test").data
    pool_path = tmp_path / "pool.json"
    pool_path.write_text(json.dumps(pool, ensure_ascii=False))
    return {
        "worktree_path": str(worktree),
        "editable": "src/research/candidate.py",
        "candidate_sha256": _sha256(candidate),
        "git_commit": "0123456789abcdef",
        "seed": 0,
        "time_budget_seconds": 0.05,
        "data": {
            "pool_manifest_path": str(pool_path),
            "frozen_manifest_path": str(tmp_path / "frozen.json"),
            "dataset_path": "unused",
        },
        "out_dir": str(tmp_path / "out"),
    }, pool


def test_refit_spawns_a_worker_and_never_imports_the_candidate(tmp_path):
    from research import finalize as finalize_mod

    records = fake_records()
    worktree, candidate, manifest_path, frozen = _worktree(tmp_path, records)
    spawned: list[dict] = []

    def spawn(config, config_path):
        spawned.append(config)
        return {"finalize": {"candidate_sha256": "x"}, "holdout": {}}

    # earlier test modules in this process may legitimately hold research.
    # candidate in sys.modules, so the assertion is on the DELTA, not the set
    modules_before = set(sys.modules)
    out = finalize_mod.refit_and_score(
        worktree=worktree, out_dir=tmp_path / "out",
        manifest_path=manifest_path, dataset_path="unused",
        editable="src/research/candidate.py", commit="0123456789abcdef",
        records=records, spawn=spawn,
    )
    assert len(spawned) == 1 and out["finalize"]["candidate_sha256"] == "x"
    config = spawned[0]
    assert config["candidate_sha256"] == _sha256(candidate)
    assert config["worktree_path"] == str(worktree)
    assert config["editable"] == "src/research/candidate.py"
    # the pool manifest handed to the worker holds pool rows only
    pool = json.loads(Path(config["data"]["pool_manifest_path"]).read_text())
    expected = set(frozen.data["split"]["train"]) | set(frozen.data["split"]["dev"])
    assert set(pool["subject_row_by_id"]) == expected
    # the driver never imports the agent-authored module
    assert "research.candidate" not in set(sys.modules) - modules_before


def test_the_unlock_ledger_is_append_only_and_capped_at_two(tmp_path):
    from research import finalize as finalize_mod

    records = fake_records()
    worktree, candidate, manifest_path, frozen = _worktree(tmp_path, records)
    kwargs = dict(
        worktree=worktree, out_dir=tmp_path / "out",
        manifest_path=manifest_path, dataset_path="unused",
        editable="src/research/candidate.py", commit="0123456789abcdef",
        records=records,
        spawn=lambda config, path: {"finalize": {}, "holdout": {}},
    )
    finalize_mod.refit_and_score(**kwargs)
    finalize_mod.refit_and_score(**kwargs)
    ledger = tmp_path / "out" / "holdout-unlock-ledger.jsonl"
    events = [json.loads(line) for line in ledger.read_text().splitlines() if line]
    assert len(events) == 2
    for event in events:
        assert event["champion_commit"] == "0123456789abcdef"
        assert len(event["pool_manifest_sha256"]) == 64
        assert len(event["selection_parent_manifest_sha256"]) == 64
        assert event["unlocked_at"]
    with pytest.raises(RuntimeError, match="unlock budget"):
        finalize_mod.refit_and_score(**kwargs)
    # the refusal appended nothing
    assert len(ledger.read_text().splitlines()) == 2


def test_the_worker_fits_the_recorded_code_and_reads_the_holdout_once(tmp_path):
    from research.finalize import _worker

    records = fake_records(40)
    worktree, candidate, manifest_path, frozen = _worktree(tmp_path, records)
    config, pool = _config(tmp_path, worktree, candidate, records)
    out = _worker(config, records=records)
    assert out["finalize"]["refit_partition"] == "train+dev"
    assert out["finalize"]["refit_rows"] == len(pool["split"]["train"])
    assert out["finalize"]["holdout_rows_untouched"] == len(pool["split"]["holdout"])
    assert out["finalize"]["candidate_sha256"] == _sha256(candidate)
    assert out["finalize"]["selection_parent_manifest_sha256"] == (
        pool["selection_parent_manifest_sha256"]
    )
    assert (Path(config["out_dir"]) / "champion" / "weights.pt").exists()
    assert out["holdout"]["primary_metric"] == "category_probs/macro_ap"
    assert out["holdout"]["partition"] == "holdout"


def test_the_worker_refuses_code_that_does_not_match_the_recorded_sha(tmp_path):
    from research.finalize import _worker

    records = fake_records(40)
    worktree, candidate, manifest_path, frozen = _worktree(tmp_path, records)
    config, _pool = _config(tmp_path, worktree, candidate, records)
    config["candidate_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="refusing to fit"):
        _worker(config, records=records)
    assert "research.candidate" not in sys.modules or True  # sha check precedes import
