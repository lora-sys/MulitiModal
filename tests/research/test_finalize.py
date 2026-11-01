"""Finalize isolation and the holdout unlock ledger (adversarial review, two
rounds).

Round 1 pinned: the driver must not import the candidate, and unlocks must be
capped. Round 2 (F-03/F-04/F-09) pins the closed design:

* the fit worker ONLY fits — its config carries no frozen-manifest path, no
  dataset path and the pool manifest carries no holdout; the holdout is
  scored in an isolated evaluator subprocess from state_dict + the frozen
  architecture rebuild;
* the unlock event is appended BEFORE the fit worker spawns, under an flock,
  into the repo-global ledger (not inside --out): a crashed worker has still
  consumed the event, because the event is the grant of read permission, not
  a successful read. The worker entry re-checks the budget and its specific
  event, so invoking `python -m research.finalize` by hand cannot bypass it;
* the worker verifies the worktree HEAD is the recorded commit and the tree
  is clean beyond the editable file, so the fitted code is the committed,
  recorded one (F-09).

The driver-isolation assertion is on the sys.modules DELTA: earlier tests in
the same process may legitimately hold research.candidate in their cache.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from data.mendeley import FEATURE_GROUP_SIZES, SubjectRecord, records_to_json
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


def _git_init(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)


def _worktree(tmp_path, records):
    """A git worktree with the candidate committed: the F-09 binding needs a
    real HEAD and status even in the worker-level tests."""
    worktree = tmp_path / "wt"
    (worktree / "src" / "research").mkdir(parents=True)
    candidate = worktree / "src" / "research" / "candidate.py"
    candidate.write_text("WIDTH = 32\n")
    _git_init(worktree)
    subprocess.run(["git", "add", "-A"], cwd=worktree, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=worktree, check=True)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=worktree, check=True,
        text=True, capture_output=True,
    ).stdout.strip()
    frozen = build_real_manifest(records, dataset_sha256=DATASET_SHA)
    manifest_path = tmp_path / "frozen.json"
    manifest_path.write_text(json.dumps(frozen.data, ensure_ascii=False))
    return worktree, candidate, manifest_path, frozen, head


def _config(tmp_path, worktree, candidate, records, head=None):
    frozen = load_real_manifest(tmp_path / "frozen.json")
    pool = pool_manifest(frozen, records, notes="test").data
    pool_path = tmp_path / "pool.json"
    pool_path.write_text(json.dumps(pool, ensure_ascii=False))
    return {
        "worktree_path": str(worktree),
        "editable": "src/research/candidate.py",
        "candidate_sha256": _sha256(candidate),
        "git_commit": head or "0123456789abcdef",
        "seed": 0,
        "time_budget_seconds": 0.05,
        "train_batch_size": 8,
        "ledger_path": str(tmp_path / "ledger" / "holdout-unlock-ledger.jsonl"),
        "unlock_event_id": "a" * 32,
        "data": {
            "pool_manifest_path": str(pool_path),
            "pool_records_path": str(tmp_path / "pool-records.json"),
        },
        "out_dir": str(tmp_path / "out"),
    }, pool


def _write_ledger_event(config, event_id=None) -> None:
    """Record ONLY this run's intent event: the worker's budget check needs
    room for it (a full ledger refuses any further fit)."""
    from research.finalize import UNLOCK_LEDGER_FILE

    ledger = Path(config["ledger_path"])
    ledger.parent.mkdir(parents=True, exist_ok=True)
    (ledger.parent / UNLOCK_LEDGER_FILE).write_text(
        json.dumps({"event_id": event_id or config["unlock_event_id"]}) + "\n"
    )


def test_refit_spawns_a_worker_and_never_imports_the_candidate(tmp_path):
    from research import finalize as finalize_mod

    records = fake_records()
    worktree, candidate, manifest_path, frozen, _head = _worktree(tmp_path, records)
    spawned: list[dict] = []

    def spawn(config, config_path):
        # the fit worker's config carries NO dataset path and NO frozen
        # manifest path: the fitting code cannot reach the holdout (F-03/F-08)
        assert "dataset_path" not in config["data"]
        assert "frozen_manifest_path" not in config["data"]
        assert config["candidate_sha256"] == _sha256(candidate)
        spawned.append(config)
        return {
            "finalize": {"candidate_sha256": "x"},
            "weights_path": "unused.pt", "architecture_path": "unused.json",
            "weights_sha256": "0" * 64, "architecture_sha256": "0" * 64,
        }

    def spawn_holdout(config, request):
        # the holdout is scored by an evaluator subprocess whose config has
        # the frozen manifest and pool manifest — never the fit worker's
        assert config["data"]["frozen_manifest_path"]
        assert "pool_records_path" not in config["data"]
        return {"holdout": {"partition": "holdout"}, "baselines": {}}

    # earlier test modules in this process may legitimately hold research.
    # candidate in sys.modules, so the assertion is on the DELTA, not the set
    modules_before = set(sys.modules)
    out = finalize_mod.refit_and_score(
        worktree=worktree, out_dir=tmp_path / "out",
        manifest_path=manifest_path, dataset_path="unused",
        editable="src/research/candidate.py", commit="0123456789abcdef",
        records=records, spawn=spawn, spawn_holdout=spawn_holdout,
    )
    assert len(spawned) == 1 and out["finalize"]["candidate_sha256"] == "x"
    assert out["holdout"]["partition"] == "holdout"
    assert out["finalize"]["holdout_rows_untouched"] == len(
        frozen.data["split"]["holdout"]
    )
    config = spawned[0]
    # the pool manifest handed to the worker holds pool rows only, and no
    # holdout partition at all (F-05)
    pool = json.loads(Path(config["data"]["pool_manifest_path"]).read_text())
    expected = set(frozen.data["split"]["train"]) | set(frozen.data["split"]["dev"])
    assert set(pool["subject_row_by_id"]) == expected
    assert pool["split"]["holdout"] == []
    # the driver never imports the agent-authored module
    assert "research.candidate" not in set(sys.modules) - modules_before


def _ledger(out_dir: Path) -> Path:
    return out_dir / "holdout-unlock-ledger.jsonl"


def test_the_unlock_is_billed_before_the_worker_spawns(tmp_path):
    """F-04: the intent event is appended before spawn — a worker that
    crashes, hangs or is killed has still consumed the unlock event. Two
    failed runs consume both events; the third attempt is refused without
    spawning a worker."""
    from research import finalize as finalize_mod

    records = fake_records()
    worktree, _candidate, manifest_path, _frozen, _head = _worktree(tmp_path, records)
    seen_events: list[str] = []

    def crashing_spawn(config, config_path):
        seen_events.append(config["unlock_event_id"])
        raise RuntimeError("worker crashed mid-fit")

    kwargs = dict(
        worktree=worktree, out_dir=tmp_path / "out",
        manifest_path=manifest_path, dataset_path="unused",
        editable="src/research/candidate.py", commit="0123456789abcdef",
        records=records, spawn=crashing_spawn,
    )
    with pytest.raises(RuntimeError, match="crashed"):
        finalize_mod.refit_and_score(**kwargs)
    with pytest.raises(RuntimeError, match="crashed"):
        finalize_mod.refit_and_score(**kwargs)
    assert len(seen_events) == 2
    ledger = finalize_mod.global_ledger_path(worktree)
    events = [json.loads(line) for line in ledger.read_text().splitlines() if line]
    assert [e["event_id"] for e in events] == seen_events
    assert events[0]["event"] == "holdout_unlock_intent"
    # the third attempt is refused BEFORE any worker spawns, appending nothing
    refused: list = []
    with pytest.raises(RuntimeError, match="unlock budget"):
        finalize_mod.refit_and_score(
            **{**kwargs, "spawn": lambda c, p: refused.append(c)}
        )
    assert refused == []
    assert finalize_mod.unlock_count(ledger) == 2


def test_the_ledger_is_global_and_not_inside_out(tmp_path):
    """F-04: the ledger path does not drift with --out; it resolves through
    the worktree's git common dir, so a new out_dir cannot reset the count."""
    from research import finalize as finalize_mod

    records = fake_records()
    worktree, _candidate, manifest_path, _frozen, _head = _worktree(tmp_path, records)
    ledger = finalize_mod.global_ledger_path(worktree)
    assert ledger.name == "holdout-unlock-ledger.jsonl"
    assert ledger.parent.name == "runs"
    assert ledger == worktree / "runs" / "holdout-unlock-ledger.jsonl"

    # two unlocks via different out dirs consume the SAME budget
    kwargs = dict(
        worktree=worktree, manifest_path=manifest_path, dataset_path="unused",
        editable="src/research/candidate.py", commit="0123456789abcdef",
        records=records,
        spawn=lambda config, path: {
            "finalize": {}, "weights_path": "w", "architecture_path": "a",
            "weights_sha256": "0" * 64, "architecture_sha256": "0" * 64,
        },
        spawn_holdout=lambda config, req: {"holdout": {}, "baselines": {}},
    )
    finalize_mod.refit_and_score(out_dir=tmp_path / "out-a", **kwargs)
    finalize_mod.refit_and_score(out_dir=tmp_path / "out-b", **kwargs)
    assert finalize_mod.unlock_count(ledger) == 2
    with pytest.raises(RuntimeError, match="unlock budget"):
        finalize_mod.refit_and_score(out_dir=tmp_path / "out-c", **kwargs)
    # and --out never grew a ledger of its own
    assert not _ledger(tmp_path / "out-a").exists()
    assert not _ledger(tmp_path / "out-b").exists()


def test_the_worker_refits_without_reading_the_holdout(tmp_path):
    from research.finalize import _worker

    records = fake_records(40)
    worktree, candidate, manifest_path, frozen, head = _worktree(tmp_path, records)
    config, pool = _config(tmp_path, worktree, candidate, records, head=head)
    _write_ledger_event(config)
    pool_records = [
        r for r in records
        if r.subject_index in set(pool["subject_row_by_id"].values())
    ]
    (tmp_path / "pool-records.json").write_text(records_to_json(pool_records))
    out = _worker(config, records=pool_records)
    finalize = out["finalize"]
    assert finalize["refit_partition"] == "train+dev"
    assert finalize["refit_rows"] == len(pool["split"]["train"])
    assert finalize["candidate_sha256"] == _sha256(candidate)
    assert finalize["worktree_commit"] == head
    assert finalize["selection_parent_manifest_sha256"] == (
        pool["selection_parent_manifest_sha256"]
    )
    # fit-only: no holdout report is produced in the worker (F-03)
    assert "holdout" not in out
    weights = Path(config["out_dir"]) / "champion" / "weights.pt"
    assert weights.exists()
    import torch

    state = torch.load(weights, weights_only=True)
    assert isinstance(state, dict) and state


def test_the_worker_refuses_code_that_does_not_match_the_recorded_sha(tmp_path):
    from research.finalize import _worker

    records = fake_records(40)
    worktree, candidate, manifest_path, frozen, head = _worktree(tmp_path, records)
    config, _pool = _config(tmp_path, worktree, candidate, records, head=head)
    _write_ledger_event(config)
    config["candidate_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="refusing to fit"):
        _worker(config, records=records)


def test_the_worker_refuses_a_missing_ledger_event(tmp_path):
    """F-04: `python -m research.finalize` by hand cannot skip the ledger."""
    from research import finalize as finalize_mod
    from research.finalize import _worker

    records = fake_records(40)
    worktree, candidate, manifest_path, frozen, head = _worktree(tmp_path, records)
    config, _pool = _config(tmp_path, worktree, candidate, records, head=head)
    # budget has room, but only some OTHER run's event is on file
    ledger = Path(config["ledger_path"])
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(json.dumps({"event_id": "c" * 32}) + "\n")
    with pytest.raises(RuntimeError, match="unlock intent event"):
        _worker(config, records=records)
    # with the event recorded but the budget exhausted, the worker still
    # refuses: the worker entry enforces the budget itself
    ledger = Path(config["ledger_path"])
    event_id = config["unlock_event_id"]
    extras = "\n".join(
        json.dumps({"event_id": "z" * 32})
        for _ in range(finalize_mod.MAX_UNLOCKS)
    )
    ledger.write_text(json.dumps({"event_id": event_id}) + "\n" + extras + "\n")
    with pytest.raises(RuntimeError, match="unlock budget"):
        _worker(config, records=records)


def test_the_worker_binds_the_git_state_to_the_record(tmp_path):
    """F-09: HEAD must be the recorded commit and the tree clean beyond the
    editable file, or the fitted code is not the committed, recorded one."""
    from research.finalize import _worker

    records = fake_records(40)
    repo = tmp_path / "repo"
    (repo / "src" / "research").mkdir(parents=True)
    candidate = repo / "src" / "research" / "candidate.py"
    candidate.write_text("WIDTH = 32\n")
    (repo / "src" / "other.txt").write_text("frozen machinery\n")
    _git_init(repo)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=repo, check=True)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
        text=True, capture_output=True,
    ).stdout.strip()

    frozen = build_real_manifest(records, dataset_sha256=DATASET_SHA)
    pool = pool_manifest(frozen.data, records, notes="test").data
    (repo.parent / "ledger.jsonl").write_text(
        json.dumps({"event_id": "b" * 32}) + "\n"
    )
    config = {
        "worktree_path": str(repo),
        "editable": "src/research/candidate.py",
        "candidate_sha256": _sha256(candidate),
        "git_commit": head,
        "seed": 0,
        "time_budget_seconds": 0.05,
        "train_batch_size": 8,
        "ledger_path": str(repo.parent / "ledger.jsonl"),
        "unlock_event_id": "b" * 32,
        "data": {
            "pool_manifest_path": str(tmp_path / "pool.json"),
            "pool_records_path": str(tmp_path / "pool-records.json"),
        },
        "out_dir": str(repo.parent / "out"),
    }
    (tmp_path / "pool.json").write_text(json.dumps(pool, ensure_ascii=False))
    pool_records = [
        r for r in records
        if r.subject_index in set(pool["subject_row_by_id"].values())
    ]
    (tmp_path / "pool-records.json").write_text(records_to_json(pool_records))
    out = _worker(config, records=pool_records)
    assert out["finalize"]["worktree_commit"] == head

    # a wrong recorded commit is refused
    config["git_commit"] = "0" * 40
    with pytest.raises(RuntimeError, match="unrecorded tree state"):
        _worker(config, records=records)

    # a dirty TRACKED file beyond the editable file is refused
    config["git_commit"] = head
    (repo / "src" / "other.txt").write_text("uncommitted work\n")
    with pytest.raises(RuntimeError, match="dirty beyond the editable"):
        _worker(config, records=records)
    # untracked work products (the ledger, out_dir payloads) do not refuse
    subprocess.run(["git", "checkout", "--", "src/other.txt"], cwd=repo, check=True)
    (repo / "runs").mkdir(exist_ok=True)
    (repo / "runs" / "holdout-unlock-ledger.jsonl").write_text("{}\n")
    out = _worker(config, records=pool_records)
    assert out["finalize"]["worktree_commit"] == head
