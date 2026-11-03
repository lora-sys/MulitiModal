"""Persistent state for one research run: identity, budget, history, stop.

A run is a directory under `runs/`:

    run.json       the frozen spec (task, editable file, worktree, budget, policy)
    manifest.jsonl THE AUTHORITY: one line per finished round, hash-chained
    events.jsonl   host progress inside rounds (events.py)
    status.json    latest progress snapshot (events.py)
    doctor.json    the latest diagnosis (doctor.py)
    STOP           present while a stop has been requested

`run.json` is written once and never edited: changing the budget or the policy
of a run in flight would make its earlier rounds incomparable, which is the
exact failure mode the issue warns about. A new budget is a new run.

`manifest.jsonl` decides what has already happened. `resume` therefore needs no
separate bookkeeping: the next round index is the number of finished rounds, so
a restart continues instead of replaying.

Readers (`status`, `report`, `dashboard`) never take the coordinator lock; only
the process that drives training does.
"""

from __future__ import annotations

import json
import os
import signal
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from research.events import (
    EVENT_SCHEMA_VERSION,
    EventWriter,
    read_events,
    read_status,
    write_status_atomic,
)

RUN_SPEC_FILE = "run.json"
RUN_SPEC_SCHEMA_VERSION = "autoresearch-run-v1"
MANIFEST_FILE_NAME = "manifest.jsonl"
DOCTOR_FILE = "doctor.json"
STOP_FILE = "STOP"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _new_run_id(task: str) -> str:
    return f"{task}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}"


@dataclass
class ResearchSession:
    root: Path
    spec: dict

    # -- lifecycle -----------------------------------------------------------

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        task: str,
        editable: str,
        worktree: str,
        branch: str,
        folds: int,
        repeats: int,
        max_rounds: int,
        budget_seconds: float,
        primary_metric: str,
        max_wall_clock_s: float | None = None,
        policy: dict | None = None,
        notes: str = "",
    ) -> ResearchSession:
        root = Path(root)
        if (root / RUN_SPEC_FILE).exists():
            raise RuntimeError(
                f"{root} already holds a run ({RUN_SPEC_FILE}); choose a new "
                "--out or use resume on the existing one"
            )
        root.mkdir(parents=True, exist_ok=True)
        spec = {
            "schema_version": RUN_SPEC_SCHEMA_VERSION,
            "run_id": _new_run_id(task),
            "created_at": _now_iso(),
            "task": task,
            "editable": editable,
            "worktree": str(worktree),
            "branch": branch,
            "folds": int(folds),
            "repeats": int(repeats),
            "max_rounds": int(max_rounds),
            "budget_seconds": float(budget_seconds),
            "max_wall_clock_s": float(max_wall_clock_s) if max_wall_clock_s else None,
            "primary_metric": primary_metric,
            "policy": dict(policy or {}),
            "notes": notes,
        }
        (root / RUN_SPEC_FILE).write_text(
            json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return cls(root=root, spec=spec)

    @classmethod
    def load(cls, root: str | Path) -> ResearchSession:
        root = Path(root)
        path = root / RUN_SPEC_FILE
        if not path.exists():
            raise FileNotFoundError(f"no research run at {root} (missing {RUN_SPEC_FILE})")
        spec = json.loads(path.read_text(encoding="utf-8"))
        return cls(root=root, spec=spec)

    # -- identity ------------------------------------------------------------

    @property
    def run_id(self) -> str:
        return str(self.spec["run_id"])

    @property
    def events_path(self) -> Path:
        return self.root / "events.jsonl"

    @property
    def status_path(self) -> Path:
        return self.root / "status.json"

    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST_FILE_NAME

    @property
    def doctor_path(self) -> Path:
        return self.root / DOCTOR_FILE

    def writer(self, **kwargs) -> EventWriter:
        """Open the single-writer event log. Raises if another coordinator owns it."""
        return EventWriter(self.root, self.run_id, **kwargs)

    # -- history (the manifest is the authority) -----------------------------

    def entries(self) -> list[dict]:
        from runtime.manifest import RunManifestStore

        if not self.manifest_path.exists():
            return []
        return RunManifestStore(self.manifest_path).load()

    def rounds_completed(self) -> int:
        return len(self.entries())

    def next_round_index(self) -> int:
        return self.rounds_completed()

    def rounds_remaining(self) -> int:
        return max(0, int(self.spec["max_rounds"]) - self.rounds_completed())

    def champion(self) -> dict | None:
        from runtime.manifest import RunManifestStore

        if not self.manifest_path.exists():
            return None
        return RunManifestStore(self.manifest_path).champion()

    def keep_revert_counts(self) -> dict:
        counts = {"KEEP": 0, "REVERT": 0}
        for entry in self.entries():
            decision = str(entry.get("decision"))
            counts[decision] = counts.get(decision, 0) + 1
        return counts

    # -- stop ----------------------------------------------------------------

    @property
    def stop_path(self) -> Path:
        return self.root / STOP_FILE

    def stop_requested(self) -> bool:
        return self.stop_path.exists()

    def request_stop(self, reason: str = "operator request") -> dict:
        """Ask the run to stop, and end the training child if one is live.

        The marker is what the coordinator checks between stages, so a stop
        never cuts a round in half and leaves a decision unrecorded. The SIGTERM
        is what makes it prompt: a training subprocess is spawned in its own
        session, so the recorded process group is terminated as a unit.
        """
        stopped_child = self._terminate_active_child()
        write = {
            "at": _now_iso(),
            "reason": reason,
            "child_terminated": stopped_child,
        }
        self.stop_path.write_text(json.dumps(write, ensure_ascii=False) + "\n", encoding="utf-8")
        return write

    def clear_stop(self) -> None:
        self.stop_path.unlink(missing_ok=True)

    def _terminate_active_child(self) -> dict:
        status = read_status(self.status_path) or {}
        pgid = status.get("active_pgid")
        pid = status.get("active_pid")
        result = {"pid": pid, "pgid": pgid, "signalled": False, "not_found": False}
        if not isinstance(pid, int):
            result["not_found"] = True
            return result
        try:
            target = int(pgid) if isinstance(pgid, int) and pgid > 1 else int(pid)
            os.killpg(target, signal.SIGTERM)
            result["signalled"] = True
        except ProcessLookupError:
            result["not_found"] = True
        except (PermissionError, OSError) as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    # -- active child bookkeeping (written through status.json) --------------

    def mark_active_child(self, pid: int, pgid: int) -> None:
        status = read_status(self.status_path) or {}
        status.pop("stale", None)
        status.pop("heartbeat_age_seconds", None)
        status.update({"active_pid": int(pid), "active_pgid": int(pgid)})
        write_status_atomic(self.status_path, status)

    def clear_active_child(self) -> None:
        status = read_status(self.status_path) or {}
        status.pop("stale", None)
        status.pop("heartbeat_age_seconds", None)
        status.update({"active_pid": None, "active_pgid": None})
        write_status_atomic(self.status_path, status)

    # -- doctor --------------------------------------------------------------

    def write_doctor(self, report: dict) -> None:
        self.doctor_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    def read_doctor(self) -> dict | None:
        if not self.doctor_path.exists():
            return None
        try:
            return json.loads(self.doctor_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    # -- read-only views -----------------------------------------------------

    def status(self, after_seq: int = 0, events_limit: int = 100) -> dict:
        status = read_status(self.status_path)
        events = read_events(self.events_path, after_seq=after_seq, limit=events_limit)
        champion = self.champion()
        return {
            "schema_version": RUN_SPEC_SCHEMA_VERSION,
            "run_id": self.run_id,
            "task": self.spec.get("task"),
            "editable": self.spec.get("editable"),
            "branch": self.spec.get("branch"),
            "worktree": self.spec.get("worktree"),
            "primary_metric": self.spec.get("primary_metric"),
            "policy": self.spec.get("policy") or {},
            "rounds_completed": self.rounds_completed(),
            "max_rounds": int(self.spec["max_rounds"]),
            "rounds_remaining": self.rounds_remaining(),
            "decisions": self.keep_revert_counts(),
            "best_metric": champion.get("best_so_far") if champion else None,
            "champion_run_id": champion.get("run_id") if champion else None,
            "champion_commit": champion.get("git_commit_sha") if champion else None,
            "stop_requested": self.stop_requested(),
            "progress": status,
            "events": events,
            "last_event_seq": events[-1]["seq"] if events else after_seq,
            "event_schema_version": EVENT_SCHEMA_VERSION,
        }

    def history(self) -> dict:
        """Every round, from the manifest authority plus the in-flight status."""
        rows = []
        for entry in self.entries():
            metrics = entry.get("metrics") or {}
            rows.append(
                {
                    "run_id": entry.get("run_id"),
                    "round": entry.get("round_index"),
                    "intent": entry.get("intent"),
                    "hypothesis": entry.get("hypothesis"),
                    "controlled_change": entry.get("controlled_change"),
                    "expected_observation": entry.get("expected_observation"),
                    "next_action": entry.get("next_action"),
                    "observed_evidence": entry.get("observed_evidence"),
                    "decision": entry.get("decision"),
                    "outcome": entry.get("outcome"),
                    "status": entry.get("status"),
                    "reason": entry.get("reason"),
                    "primary_value": metrics.get("primary_value"),
                    "primary_std": metrics.get("primary_std"),
                    "primary_sem": metrics.get("primary_sem"),
                    "best_so_far": entry.get("best_so_far"),
                    "complexity": entry.get("complexity"),
                    "git_commit_sha": entry.get("git_commit_sha"),
                    "proposer": entry.get("proposer"),
                    "agent_kind": entry.get("agent_kind"),
                    "elapsed_s": entry.get("elapsed_s"),
                }
            )
        return {
            "run_id": self.run_id,
            "rounds": rows,
            "champion": self.champion(),
            "decisions": self.keep_revert_counts(),
        }

    def uptime_seconds(self) -> float | None:
        try:
            created = datetime.fromisoformat(str(self.spec["created_at"]))
        except (ValueError, KeyError):
            return None
        return round((datetime.now(UTC) - created).total_seconds(), 1)

    def within_wall_clock(self, started_monotonic: float) -> bool:
        limit = self.spec.get("max_wall_clock_s")
        if not limit:
            return True
        return (time.monotonic() - started_monotonic) < float(limit)