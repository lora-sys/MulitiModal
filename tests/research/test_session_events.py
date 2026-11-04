"""Trusted event log and run session: single writer, atomic status, stop/resume.

These tests are about the properties the dashboard and `resume` depend on, not
about the exact prose of a message:

* sequence numbers are monotonic across a restart, so `after=<seq>` paging
  cannot skip or repeat an event;
* a torn tail (process killed mid-write) is dropped rather than raising, because
  the log is read most often right after something went wrong;
* `status.json` is replaced atomically, so a reader never sees half a document;
* exactly one coordinator may hold a run;
* a stop ends the live training child and the completed rounds are the manifest's,
  so `resume` continues instead of replaying.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from research.events import STALE_AFTER_SECONDS, read_events, read_status
from research.session import ResearchSession

pytestmark = pytest.mark.synthetic


def _session(tmp_path: Path, **overrides) -> ResearchSession:
    kwargs = {
        "task": "toy",
        "editable": "src/research/candidate_toy.py",
        "worktree": str(tmp_path / "wt"),
        "branch": "autoresearch/toy",
        "folds": 2,
        "repeats": 1,
        "max_rounds": 3,
        "budget_seconds": 1.0,
        "primary_metric": "toy/accuracy",
    }
    kwargs.update(overrides)
    return ResearchSession.create(tmp_path / "run", **kwargs)


def test_sequence_is_monotonic_across_writer_restart(tmp_path):
    session = _session(tmp_path)
    with session.writer() as writer:
        first = writer.emit("DOCTOR", "started", message="preflight")
        second = writer.emit("TRAINING", "started", round_index=0, fold=0, folds_total=2)
    assert (first["seq"], second["seq"]) == (1, 2)

    # a restarted coordinator continues the sequence instead of colliding with it
    with session.writer() as writer:
        third = writer.emit("DECIDING", "finished", round_index=0, message="KEEP")
    assert third["seq"] == 3
    seqs = [e["seq"] for e in read_events(session.events_path)]
    assert seqs == [1, 2, 3]
    assert read_events(session.events_path, after_seq=2) == [third]


def test_only_one_coordinator_owns_a_run(tmp_path):
    session = _session(tmp_path)
    with session.writer():
        with pytest.raises(RuntimeError, match="another coordinator already owns"):
            session.writer()
    # and the lock is released when that writer goes away
    with session.writer() as writer:
        writer.emit("IDLE", "info", message="reacquired")


def test_torn_tail_is_dropped_not_fatal(tmp_path):
    session = _session(tmp_path)
    with session.writer() as writer:
        writer.emit("TRAINING", "started", message="ok")
    with session.events_path.open("a", encoding="utf-8") as handle:
        handle.write('{"schema_version": "autoresearch-events-v1", "seq": 99')  # killed mid-write
    events = read_events(session.events_path)
    assert [e["seq"] for e in events] == [1]

    # the next writer heals the tail by starting on a fresh line
    with session.writer() as writer:
        healed = writer.emit("TRAINING", "finished", message="after crash")
    assert healed["seq"] == 2, "seq must not be poisoned by the torn line"
    assert [e["seq"] for e in read_events(session.events_path)] == [1, 2]


def test_status_is_replaced_atomically_and_flags_stale(tmp_path):
    session = _session(tmp_path)
    with session.writer() as writer:
        writer.emit("DOCTOR", "started", message="preflight")
        writer.emit("PROPOSING", "started", round_index=0, message="hypothesis")
    status = read_status(session.status_path)
    assert status["stage"] == "PROPOSING" and status["round"] == 0
    assert status["stale"] is False
    assert not list(session.root.glob("status.json.tmp")), "temp file must be renamed away"

    # a reader evaluating the same file later must see STALE, not a frozen stage
    later = time.monotonic() + STALE_AFTER_SECONDS + 5
    assert read_status(session.status_path, now_monotonic=later)["stale"] is True


def test_malformed_and_oversized_events_are_contained(tmp_path):
    session = _session(tmp_path)
    with session.writer() as writer:
        event = writer.emit(
            "TRAINING", "started",
            message="x" * 5000,
            metrics={"loss": 0.5, "note": "y" * 5000, **{f"k{i}": i for i in range(50)}},
        )
    assert len(event["message"]) <= 500
    assert len(event["metrics"]) <= 24
    assert len(event["metrics"]["note"]) <= 500
    assert len(json.dumps(event)) < 4096


def test_bad_stage_and_status_are_refused(tmp_path):
    session = _session(tmp_path)
    with session.writer() as writer:
        with pytest.raises(ValueError, match="unknown stage"):
            writer.emit("NONSENSE", "started")
        with pytest.raises(ValueError, match="unknown status"):
            writer.emit("TRAINING", "nonsense")


def test_stop_terminates_the_live_child_and_resume_keeps_history(tmp_path):
    session = _session(tmp_path)
    # a real child in its own process group, exactly how the loop spawns training
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    try:
        session.mark_active_child(child.pid, os.getpgid(child.pid))
        assert session.stop_requested() is False
        write = session.request_stop("operator request")
        assert write["child_terminated"]["signalled"] is True
        assert session.stop_requested() is True
        child.wait(timeout=10)
        assert child.returncode is not None
    finally:
        if child.poll() is None:
            child.kill()

    # a finished round lives in the manifest, so the next index accounts for it
    session.manifest_path.write_text(
        json.dumps({"run_id": "toy-0000", "decision": "KEEP", "status": "completed",
                    "reason": "baseline 0.5", "best_so_far": 0.5}) + "\n",
        encoding="utf-8",
    )
    assert session.rounds_completed() == 1
    assert session.next_round_index() == 1
    assert session.rounds_remaining() == 2
    session.clear_stop()
    assert session.stop_requested() is False


def test_creating_a_second_run_in_the_same_directory_is_refused(tmp_path):
    _session(tmp_path)
    with pytest.raises(RuntimeError, match="already holds a run"):
        _session(tmp_path)


def test_status_view_is_read_only_and_reports_history(tmp_path):
    session = _session(tmp_path)
    session.manifest_path.write_text(
        json.dumps({"run_id": "toy-0000", "round_index": 0, "intent": "baseline",
                    "hypothesis": "the baseline is the reference",
                    "decision": "KEEP", "status": "completed",
                    "reason": "baseline 0.5", "best_so_far": 0.5,
                    "metrics": {"primary_value": 0.5}}) + "\n",
        encoding="utf-8",
    )
    with session.writer() as writer:
        writer.emit("IDLE", "info", message="waiting for next hypothesis")
    view = session.status()
    assert view["run_id"] == session.run_id
    assert view["rounds_completed"] == 1 and view["rounds_remaining"] == 2
    assert view["best_metric"] == 0.5
    assert view["events"][-1]["stage"] == "IDLE"
    history = session.history()["rounds"]
    assert history[0]["hypothesis"] == "the baseline is the reference"
    # a status view must not take the coordinator lock: the writer above closed
    with session.writer():
        pass