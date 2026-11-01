"""Finalise the source-level run: refit the champion on the selection pool in
a one-shot training subprocess, then score the holdout once — in an isolated
evaluator subprocess.

What this must not do:

* read the holdout while choosing anything. The champion is already fixed by
  the manifest before this module runs, and the holdout is touched exactly
  once, at the end, to produce a number;
* execute candidate code anywhere but a one-shot training subprocess. The
  first implementation imported research.candidate into the driver; the
  second let the finalize worker fit AND score in one interpreter, so the
  candidate's code ran alongside the holdout read. Now the driver only
  prepares the pool manifest and the trimmed pool records, appends the
  unlock-intent event to the ledger, and spawns `python -m research.finalize`
  (fit only) and then `research.evaluator --mode holdout` (state_dict only,
  holdout rebuilt from the frozen parent manifest) — the same isolated
  evaluation process the selection folds use;
* unlock the holdout without paying. Phase 1 allows two unlock events
  (AGENTS.md rule 11). The unlock event is the GRANT of read permission, not
  a successful read: the driver appends the intent event BEFORE spawning the
  fit worker (under an flock, into the repo-global ledger), and a crashed or
  killed worker has still consumed the event. The worker re-checks the
  ledger — its own budget assertion and the presence of its specific event —
  before importing the candidate, and `python -m research.finalize` invoked
  by hand cannot skip that check: the worker validates the event itself. The
  ledger lives at `<repo-root>/runs/holdout-unlock-ledger.jsonl` (resolved
  through the worktree's git common dir), never inside --out, so changing
  the output directory cannot reset the count.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from pathlib import Path

#: AGENTS.md rule 11: at most two holdout unlock events in Phase 1.
MAX_UNLOCKS = 2
UNLOCK_LEDGER_FILE = "holdout-unlock-ledger.jsonl"
UNLOCK_LEDGER_DIR = "runs"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def global_ledger_path(worktree: Path) -> Path:
    """The repo-global ledger, resolved from the worktree's git common dir.

    Anchored to the repository, not to --out: the unlock budget is a property
    of the project's lifetime, and a ledger inside the output directory would
    reset on every new output path (or on deleting one file).
    """
    worktree = Path(worktree)
    try:
        result = subprocess.run(
            ["git", "-C", str(worktree), "rev-parse", "--path-format=absolute",
             "--git-common-dir"],
            text=True, capture_output=True, check=True,
        )
        git_dir = Path(result.stdout.strip())
        return git_dir.parent / UNLOCK_LEDGER_DIR / UNLOCK_LEDGER_FILE
    except (subprocess.CalledProcessError, OSError):
        return worktree / UNLOCK_LEDGER_DIR / UNLOCK_LEDGER_FILE


def _read_ledger_lines(ledger_path: Path) -> list[str]:
    """Read the ledger under a SHARED lock: readers see a consistent prefix
    even while a driver is inside its count-and-append critical section."""
    path = Path(ledger_path)
    if not path.exists():
        return []
    with path.open("r") as fh:
        fcntl.flock(fh, fcntl.LOCK_SH)
        try:
            lines = fh.read().splitlines()
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
    return [line for line in lines if line.strip()]


def unlock_count(ledger_path: Path) -> int:
    return len(_read_ledger_lines(ledger_path))


def ledger_contains(ledger_path: Path, event_id: str) -> bool:
    """Exact match on the parsed event_id, line by line — not a substring
    scan, which any longer id (or a corrupt line embedding the query) could
    satisfy."""
    for line in _read_ledger_lines(ledger_path):
        try:
            if json.loads(line).get("event_id") == event_id:
                return True
        except json.JSONDecodeError:
            continue
    return False


def _assert_unlock_budget(ledger_path: Path) -> int:
    """DRIVER-side check, run BEFORE charging: refuse when no slot is left."""
    used = unlock_count(ledger_path)
    if used >= MAX_UNLOCKS:
        raise _unlock_budget_error(ledger_path, used)
    return used


def _assert_worker_budget(ledger_path: Path) -> int:
    """WORKER-side check, run AFTER this run's event was charged: the
    driver's own charge for the current run may legitimately have filled
    the budget, so only an OVER-spend (a ledger beyond MAX_UNLOCKS, i.e.
    someone bypassed the driver) is refused here. Using the driver-side
    `>=` check here would crash every legal final unlock (round-4 review
    finding 1)."""
    used = unlock_count(ledger_path)
    if used > MAX_UNLOCKS:
        raise _unlock_budget_error(ledger_path, used)
    return used


def _unlock_budget_error(ledger_path: Path, used: int) -> RuntimeError:
    return RuntimeError(
        f"holdout unlock budget exhausted: {used} of {MAX_UNLOCKS} events "
        f"recorded in {ledger_path}; a further unlock needs a new data "
        "milestone or explicit user authorisation (AGENTS.md rule 11)"
    )


def _charge_unlock_event(ledger_path: Path, event: dict) -> None:
    """Count and append in ONE exclusive critical section (H-03).

    The count used to happen outside the lock, so two concurrent drivers
    could both see room in the budget and both append — overspending the
    Phase-1 unlock budget the ledger exists to enforce. Now: LOCK_EX → count
    → refuse if exhausted → append → fsync → unlock, atomically. The unlock
    event is the GRANT of read permission, not a successful read: a worker
    that crashes after this has still consumed the event.
    """
    ledger_path = Path(ledger_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            fh.seek(0)
            used = sum(1 for line in fh if line.strip())
            if used >= MAX_UNLOCKS:
                raise _unlock_budget_error(ledger_path, used)
            fh.write(json.dumps(event, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def refit_and_score(
    worktree: Path,
    out_dir: Path,
    manifest_path: str | Path,
    dataset_path: str | Path,
    editable: str,
    commit: str,
    seed: int = 0,
    budget_seconds: float | None = None,
    records=None,
    spawn=None,
    spawn_holdout=None,
) -> dict:
    """Drive the finalize: pool manifest here, fitting + holdout in children.

    The driver never imports research.candidate: the champion's code runs in
    the fit subprocess only, and the holdout is scored in an evaluator
    subprocess. `records`, `spawn` and `spawn_holdout` exist so tests can
    inject a parsed fixture and observe the spawns without a real workbook or
    interpreter.
    """
    from data.mendeley import load_mendeley_scores, records_to_json
    from data.real_split import load_real_manifest, pool_manifest
    from research.frozen import TIME_BUDGET_SECONDS

    worktree = Path(worktree)
    out_dir = Path(out_dir)
    budget = TIME_BUDGET_SECONDS if budget_seconds is None else budget_seconds

    candidate_path = worktree / editable
    expected_sha = _sha256(candidate_path)
    frozen = load_real_manifest(manifest_path)
    dataset_records = None if records is None else list(records)
    if records is None:
        records = load_mendeley_scores(dataset_path)
    pool = pool_manifest(
        frozen, records,
        notes=f"source-level champion {commit[:8]} refit: selection pool train+dev",
    ).data
    champion_dir = out_dir / "champion"
    champion_dir.mkdir(parents=True, exist_ok=True)
    pool_path = champion_dir / "pool-manifest.json"
    pool_path.write_text(json.dumps(pool, indent=2, ensure_ascii=False))
    # the fit subprocess's only records: the pool rows (F-08 — no full-set
    # file path enters its config, and pool_manifest carries no holdout)
    pool_rows = set(pool["subject_row_by_id"].values())
    pool_records_path = champion_dir / "pool-records.json"
    pool_records_path.write_text(records_to_json(
        [r for r in records if r.subject_index in pool_rows]
    ))

    # ---- pay BEFORE spawning: the unlock event is the grant of read access,
    # not a successful read — a crashed worker has still consumed it. Count
    # and append happen inside one exclusive critical section, so concurrent
    # drivers cannot both slip under the budget (H-03).
    ledger = global_ledger_path(worktree)
    event = {
        "event_id": uuid.uuid4().hex,
        "event": "holdout_unlock_intent",
        "unlocked_at": _iso(),
        "champion_commit": commit,
        "candidate_sha256": expected_sha,
        "pool_manifest_sha256": pool["manifest_sha256"],
        "selection_parent_manifest_sha256": frozen["manifest_sha256"],
    }
    _charge_unlock_event(ledger, event)

    config = {
        "worktree_path": str(worktree),
        "editable": editable,
        "candidate_sha256": expected_sha,
        "git_commit": commit,
        "seed": seed,
        "time_budget_seconds": budget,
        "train_batch_size": 32,
        "ledger_path": str(ledger),
        "unlock_event_id": event["event_id"],
        "data": {
            "pool_manifest_path": str(pool_path),
            "pool_records_path": str(pool_records_path),
        },
        "out_dir": str(out_dir),
    }
    config_path = out_dir / "finalize-worker-config.json"
    config_path.write_text(json.dumps(config, indent=2))
    fitted = (spawn or _spawn_worker)(config, config_path)

    holdout_config = {
        "worktree_path": str(worktree),
        "data": {
            "pool_manifest_path": str(pool_path),
            "frozen_manifest_path": str(manifest_path),
        },
        "out_dir": str(out_dir),
    }
    if dataset_records:
        # records were injected (tests / smoke): pass them to the evaluator
        # subprocess as a file instead of a workbook path
        records_path = champion_dir / "holdout-input-records.json"
        records_path.write_text(records_to_json(dataset_records))
        holdout_config["data"]["records_path"] = str(records_path)
    else:
        holdout_config["data"]["dataset_path"] = str(dataset_path)
    holdout_request = {
        "weights_path": fitted["weights_path"],
        "architecture_path": fitted["architecture_path"],
        "weights_sha256": fitted["weights_sha256"],
        "architecture_sha256": fitted["architecture_sha256"],
    }
    holdout = (spawn_holdout or _spawn_holdout_evaluator)(
        holdout_config, holdout_request
    )

    finalize = {**fitted["finalize"],
                "holdout_rows_untouched": len(frozen["split"]["holdout"])}
    (champion_dir / "finalize.json").write_text(json.dumps(finalize, indent=2))
    return {"finalize": finalize, "holdout": holdout["holdout"],
            "baselines": holdout["baselines"]}


def _child_env() -> dict:
    """Minimal whitelist env for every child (H-02/H-05): no PYTHONPATH — the
    isolated bootstrap injects the import path; PATH is a minimal fixed set
    because this worker itself needs `git`."""
    from research.spawn import child_env

    return child_env()


def _spawn_worker(config: dict, config_path: Path) -> dict:
    """Run the finalize fit worker: isolated one-shot start, whitelist env."""
    from research.spawn import bootstrap_command

    worktree = config["worktree_path"]
    budget = float(config["time_budget_seconds"])
    timeout = budget + 600.0
    command = bootstrap_command(
        worktree, "research.finalize", "--config", str(config_path),
    )
    try:
        result = subprocess.run(
            command, cwd=worktree, text=True, capture_output=True,
            timeout=timeout, env=_child_env(),
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"finalize worker exceeded {timeout:.0f}s"
        ) from exc
    return _parse_worker_payload(result, "finalize worker")


def _spawn_holdout_evaluator(config: dict, request: dict) -> dict:
    """Score the holdout in an isolated evaluator subprocess (state_dict
    only, holdout rebuilt from the frozen parent manifest)."""
    from research.spawn import bootstrap_command

    worktree = config["worktree_path"]
    with tempfile.TemporaryDirectory(prefix="research-holdout-io-") as io_dir:
        io = Path(io_dir)
        config_path = io / "holdout-config.json"
        config_path.write_text(json.dumps(config))
        request_path = io / "holdout-request.json"
        request_path.write_text(json.dumps(request))
        command = bootstrap_command(
            worktree, "research.evaluator", "--mode", "holdout",
            "--config", str(config_path), "--request", str(request_path),
        )
        try:
            result = subprocess.run(
                command, cwd=worktree, text=True, capture_output=True,
                timeout=600.0, env=_child_env(),
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("holdout evaluator exceeded 600s") from exc
    return _parse_worker_payload(result, "holdout evaluator")


def _parse_worker_payload(result: subprocess.CompletedProcess, name: str) -> dict:
    payload = result.stdout
    if result.returncode != 0:
        # stdout may carry candidate/library noise; the traceback
        # lives on stderr, so it wins when the child failed
        detail = (result.stderr or payload).strip().splitlines()
        last = detail[-1][:400] if detail else "no output"
        raise RuntimeError(f"{name} failed ({result.returncode}): {last}")
    try:
        parsed = json.loads(payload.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        raise RuntimeError(
            f"{name} produced no JSON result: {payload[-300:]!r}"
        ) from exc
    if not parsed.get("ok"):
        raise RuntimeError(f"{name} failed: {parsed.get('error')}")
    return parsed


def _assert_git_binding(config: dict) -> str:
    """The fitted code is the recorded commit, on a clean tree (F-09).

    A candidate-file SHA alone does not pin the run: the rest of the worktree
    (frozen machinery, data adapters) must be exactly the committed state the
    manifest records, and HEAD must be the champion commit.
    """
    worktree = Path(config["worktree_path"])
    editable = config["editable"]

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(worktree), *args], text=True, capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"git {' '.join(args)} failed in {worktree}: "
                f"{result.stderr.strip()[:200]}"
            )
        return result.stdout.strip()

    head = git("rev-parse", "HEAD")
    expected_commit = config.get("git_commit")
    if expected_commit and head != expected_commit:
        raise RuntimeError(
            f"worktree HEAD {head[:12]}… is not the recorded commit "
            f"{expected_commit[:12]}…; refusing to fit code from an "
            "unrecorded tree state"
        )
    # tracked-file drift is the threat: an uncommitted edit to any frozen
    # module would make the fitted code something no commit recorded.
    # UNTRACKED work products (the unlock ledger, out_dir payloads, run
    # configs) are not code bindings and must not refuse the fit.
    dirty = [
        line[3:].strip('"')
        for line in git("status", "--porcelain", "--untracked-files=no").splitlines()
        if line[3:].strip('"') != editable
    ]
    if dirty:
        raise RuntimeError(
            f"worktree is dirty beyond the editable file: {dirty[:5]} — the "
            "fitted code would not be the committed, recorded one"
        )
    return head


def _worker(config: dict, records=None) -> dict:
    """Fit the champion on the pool (worker side). No holdout is read here:
    this process's config carries no frozen-manifest path, no dataset path
    and no holdout rows — the holdout is scored by the evaluator subprocess."""
    from data.real_split import load_real_manifest
    from research.frozen import training_fold_data
    from research.model_export import save_eval_model

    # the code↔SHA binding is checked BEFORE the agent-authored module is
    # imported: what fits is provably what the record binds
    candidate_path = Path(config["worktree_path"]) / config["editable"]
    actual_sha = _sha256(candidate_path)
    expected_sha = config["candidate_sha256"]
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"candidate file {candidate_path} hashes to {actual_sha[:12]}… but "
            f"the finalize record binds {expected_sha[:12]}…; refusing to fit "
            "code that is not the recorded one"
        )
    _assert_git_binding(config)

    # ---- ledger re-check (the worker entry cannot be bypassed by invoking
    # `python -m research.finalize` by hand): budget must not be exhausted
    # and THIS run's intent event must already be recorded
    ledger = Path(config["ledger_path"])
    _assert_worker_budget(ledger)
    if not ledger_contains(ledger, config["unlock_event_id"]):
        raise RuntimeError(
            f"unlock intent event {config['unlock_event_id'][:8]}… is not in "
            f"{ledger}; refusing to fit — the holdout read permission is "
            "granted by the ledger, not by this worker's success"
        )

    import research.candidate as candidate

    if records is None:
        from data.mendeley import records_from_json

        records = records_from_json(
            Path(config["data"]["pool_records_path"]).read_text()
        )
    pool = load_real_manifest(config["data"]["pool_manifest_path"])
    if pool["split"]["holdout"]:
        raise RuntimeError(
            "pool manifest still carries a holdout partition; refusing to fit "
            "against data the pool contract must not contain"
        )
    seed = int(config["seed"])
    budget = float(config["time_budget_seconds"])
    pool_rows = set(pool["subject_row_by_id"].values())
    leaked = sorted({r.subject_index for r in records} - pool_rows)
    if leaked:
        raise RuntimeError(
            f"pool records contain rows outside the pool manifest: {leaked[:3]} "
            "— refusing to fit on data the pool contract did not trim"
        )
    import torch

    torch.manual_seed(seed)  # before training_fold_data and build_model (H-06)
    fold = training_fold_data(pool, records, config)
    started = time.monotonic()
    model = candidate.build_model(fold)
    candidate.train(model, fold, budget, seed)
    fit_seconds = round(time.monotonic() - started, 2)

    saved = save_eval_model(
        model, Path(config["out_dir"]) / "champion", "weights"
    )
    finalize = {
        "candidate_file": config["editable"],
        "candidate_sha256": actual_sha,
        "worktree_commit": _assert_git_binding(config),
        "git_commit": config["git_commit"],
        "pool_manifest_sha256": pool["manifest_sha256"],
        "selection_parent_manifest_sha256": pool["selection_parent_manifest_sha256"],
        "refit_partition": "train+dev",
        "refit_rows": len(pool["split"]["train"]),
        "seed": seed,
        "time_budget_seconds": budget,
        "fit_seconds": fit_seconds,
        "input_spec_version": fold.input_spec.version,
        **saved,
    }
    champion_dir = Path(config["out_dir"]) / "champion"
    (champion_dir / "finalize.json").write_text(json.dumps(finalize, indent=2))
    return {"finalize": finalize, **saved}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="research.finalize")
    parser.add_argument("--config", required=True,
                        help="JSON written by the driver: worktree, editable, "
                             "candidate SHA, ledger binding, data paths")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text())
    try:
        result = _worker(config)
    except Exception as exc:  # noqa: BLE001 — the driver records the failure
        print(json.dumps({
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-1500:],
        }))
        return 1
    print(json.dumps({"ok": True, **result}))
    return 0


def _iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


if __name__ == "__main__":
    sys.exit(main())
