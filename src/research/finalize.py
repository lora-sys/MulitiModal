"""Finalise a source-level run with a fit-validated holdout release.

The driver validates task routing, candidate/commit bindings and both
task-specific manifest schemas before running the candidate. It then runs a
synthetic output-contract preflight and the selection-pool refit in the
restricted Docker worker. Candidate containers see Python source, trimmed
train+dev JSON inputs and temporary output only; they never receive Git,
holdout inputs, the unlock ledger, HOME or the Docker socket.

After the candidate worker exits, the bounded container supervisor starts
a trusted child to validate the state_dict, frozen architecture and first-batch
output contract. The host checks only the child's attestation and streamed file
hashes before atomically charging the unlock event. Only after the event is
charged does it pass the frozen and pool manifest paths to the trusted holdout
evaluator. The evaluator validates that exact event before materializing
holdout records.
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
import traceback
import uuid
from pathlib import Path

#: AGENTS.md rule 11: at most two holdout unlock events in Phase 1 for the
#: constitution task. The syndrome task (#53) has its own ledger file and its
#: own budget (MAX_UNLOCKS_SD): a task neither shares nor extends another
#: task's exhausted budget.
MAX_UNLOCKS = 2
MAX_UNLOCKS_SD = 1
UNLOCK_LEDGER_FILE = "holdout-unlock-ledger.jsonl"
UNLOCK_LEDGER_FILE_SD = "holdout-unlock-ledger-sd.jsonl"
UNLOCK_LEDGER_DIR = "runs"


def ledger_params(task: str = "constitution") -> tuple[str, int]:
    """(ledger filename, unlock budget) for one research task."""
    if task == "constitution":
        return UNLOCK_LEDGER_FILE, MAX_UNLOCKS
    if task == "syndrome":
        return UNLOCK_LEDGER_FILE_SD, MAX_UNLOCKS_SD
    raise ValueError(
        f"unknown research task {task!r}; expected 'constitution' or 'syndrome'"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def global_ledger_path(worktree: Path, task: str = "constitution") -> Path:
    """The task's repo-global ledger, resolved from the worktree's git common
    dir.

    Anchored to the repository, not to --out: the unlock budget is a property
    of the task's lifetime, and a ledger inside the output directory would
    reset on every new output path (or on deleting one file). Tasks get
    separate files, so the syndrome task's unlocks never touch the
    constitution task's exhausted ledger."""
    ledger_file, _ = ledger_params(task)
    worktree = Path(worktree)
    try:
        result = subprocess.run(
            ["git", "-C", str(worktree), "rev-parse", "--path-format=absolute",
             "--git-common-dir"],
            text=True, capture_output=True, check=True,
        )
        git_dir = Path(result.stdout.strip())
        return git_dir.parent / UNLOCK_LEDGER_DIR / ledger_file
    except (subprocess.CalledProcessError, OSError):
        return worktree / UNLOCK_LEDGER_DIR / ledger_file


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


def _validated_ledger_events(lines: list[str]) -> list[dict]:
    import re
    events = []
    seen = set()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"holdout unlock ledger has invalid JSON on line {line_number}"
            ) from exc
        event_id = event.get("event_id") if isinstance(event, dict) else None
        if not isinstance(event_id, str) or re.fullmatch(r"[0-9a-f]{32}", event_id) is None:
            raise RuntimeError(
                f"holdout unlock ledger line {line_number} has an invalid event_id"
            )
        if event_id in seen:
            raise RuntimeError("holdout unlock ledger contains a duplicate event_id")
        seen.add(event_id)
        events.append(event)
    return events


def unlock_count(ledger_path: Path) -> int:
    return len(_validated_ledger_events(_read_ledger_lines(ledger_path)))


def _assert_unlock_budget(ledger_path: Path,
                          max_unlocks: int = MAX_UNLOCKS) -> int:
    """DRIVER-side check, run BEFORE charging: refuse when no slot is left."""
    used = unlock_count(ledger_path)
    if used >= max_unlocks:
        raise _unlock_budget_error(ledger_path, used, max_unlocks)
    return used


def _unlock_budget_error(ledger_path: Path, used: int,
                         max_unlocks: int) -> RuntimeError:
    return RuntimeError(
        f"holdout unlock budget exhausted: {used} of {max_unlocks} events "
        f"recorded in {ledger_path}; a further unlock needs a new data "
        "milestone or explicit user authorisation (AGENTS.md rule 11)"
    )


def _charge_unlock_event(ledger_path: Path, event: dict,
                         max_unlocks: int = MAX_UNLOCKS) -> None:
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
            events = _validated_ledger_events(fh.readlines())
            used = len(events)
            event_id = event.get("event_id")
            if (
                not isinstance(event_id, str)
                or any(existing["event_id"] == event_id for existing in events)
            ):
                raise RuntimeError("refusing to charge a duplicate or malformed event_id")
            if used >= max_unlocks:
                raise _unlock_budget_error(ledger_path, used, max_unlocks)
            fh.seek(0, os.SEEK_END)
            fh.write(json.dumps(event, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)



def verify_holdout_unlock(
    config: dict, request: dict, pool: dict, frozen: dict,
) -> dict:
    """Fail closed unless the trusted driver recorded this exact release.

    This check runs in the trusted evaluator before it loads any dataset
    records. The candidate worker never receives the ledger path or event ID.
    """
    import re

    from research.tasks import validate_manifest_contract

    task = config.get("task", "constitution")
    if task not in ("constitution", "syndrome"):
        raise RuntimeError(f"unknown holdout task {task!r}")
    validate_manifest_contract(task, frozen)
    validate_manifest_contract(task, pool, selection_pool=True)
    if pool.get("selection_parent_manifest_sha256") != frozen.get("manifest_sha256"):
        raise RuntimeError("holdout pool is not bound to the frozen parent manifest")

    unlock = config.get("unlock")
    if not isinstance(unlock, dict) or unlock.get("task") != task:
        raise RuntimeError("holdout configuration has no matching unlock binding")
    event_id = unlock.get("event_id")
    candidate_sha = unlock.get("candidate_sha256")
    commit = unlock.get("champion_commit")
    if not isinstance(event_id, str) or re.fullmatch(r"[0-9a-f]{32}", event_id) is None:
        raise RuntimeError("holdout unlock event_id is malformed")
    if not isinstance(candidate_sha, str) or re.fullmatch(r"[0-9a-f]{64}", candidate_sha) is None:
        raise RuntimeError("holdout candidate SHA is malformed")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise RuntimeError("holdout champion commit is malformed")
    if unlock.get("pool_manifest_sha256") != pool.get("manifest_sha256"):
        raise RuntimeError("holdout unlock is bound to a different pool manifest")
    if unlock.get("selection_parent_manifest_sha256") != frozen.get("manifest_sha256"):
        raise RuntimeError("holdout unlock is bound to a different frozen manifest")

    preflight = unlock.get("candidate_preflight")
    fit_artifacts = unlock.get("fit_artifacts")
    if not isinstance(preflight, dict) or not isinstance(fit_artifacts, dict):
        raise RuntimeError("holdout unlock is missing candidate validation evidence")
    for digest in (
        preflight.get("synthetic_manifest_sha256"),
        preflight.get("weights_sha256"),
        preflight.get("architecture_sha256"),
        fit_artifacts.get("weights_sha256"),
        fit_artifacts.get("architecture_sha256"),
    ):
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise RuntimeError("holdout unlock contains a malformed artifact SHA")
    expected_output_key = "syndrome_probs" if task == "syndrome" else "category_probs"
    fit_shape = fit_artifacts.get("output_shape")
    preflight_shape = preflight.get("output_shape")
    if (
        fit_artifacts.get("output_key") != expected_output_key
        or preflight.get("output_key") != expected_output_key
        or not isinstance(fit_shape, list)
        or len(fit_shape) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in fit_shape
        )
        or not isinstance(preflight_shape, list)
        or len(preflight_shape) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in preflight_shape
        )
        or fit_shape[1] != preflight_shape[1]
    ):
        raise RuntimeError("holdout unlock fit artifact has an invalid output contract")
    if request.get("weights_sha256") != fit_artifacts["weights_sha256"]:
        raise RuntimeError("holdout request weights differ from the charged artifacts")
    if request.get("architecture_sha256") != fit_artifacts["architecture_sha256"]:
        raise RuntimeError("holdout request architecture differs from the charged artifacts")

    raw_ledger = unlock.get("ledger_path")
    if not isinstance(raw_ledger, str) or not raw_ledger:
        raise RuntimeError("holdout unlock ledger path is missing")
    ledger_path = Path(raw_ledger)
    expected_ledger = global_ledger_path(Path(config["worktree_path"]), task)
    if ledger_path.is_symlink() or not ledger_path.is_file():
        raise RuntimeError("holdout unlock ledger is unavailable")
    try:
        resolved_ledger = ledger_path.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError("holdout unlock ledger is unavailable") from exc
    if resolved_ledger != expected_ledger.resolve():
        raise RuntimeError("holdout unlock ledger path does not match the task ledger")
    try:
        fd = os.open(
            ledger_path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        with os.fdopen(fd, "r", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_SH)
            try:
                events = _validated_ledger_events(handle.readlines())
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
    except OSError as exc:
        raise RuntimeError("holdout unlock ledger is unavailable") from exc

    _ledger_file, max_unlocks = ledger_params(task)
    if len(events) > max_unlocks:
        raise RuntimeError("holdout unlock ledger exceeds the task budget")
    matches = [event for event in events if event.get("event_id") == event_id]
    if len(matches) != 1:
        raise RuntimeError("holdout unlock event is missing or duplicated")
    event = matches[0]
    expected_fields = {
        "event": "holdout_unlock_intent",
        "event_id": event_id,
        "task": task,
        "candidate_sha256": candidate_sha,
        "champion_commit": commit,
        "candidate_preflight": preflight,
        "fit_artifacts": fit_artifacts,
        "pool_manifest_sha256": pool["manifest_sha256"],
        "selection_parent_manifest_sha256": frozen["manifest_sha256"],
    }
    if any(event.get(key) != value for key, value in expected_fields.items()):
        raise RuntimeError("holdout unlock event does not match this evaluation request")
    return event


def _assert_candidate_commit_binding(
    worktree: Path, editable: str, commit: str, candidate_sha256: str,
) -> None:
    import re

    from research.train_worker import _candidate_path

    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise RuntimeError("champion commit must be a full lowercase 40-character SHA")
    from research.train_worker import (
        MAX_CANDIDATE_SOURCE_BYTES,
        _read_regular_file_limited,
    )

    candidate_path = _candidate_path({
        "worktree_path": str(worktree), "editable": editable,
    })
    size = subprocess.run(
        ["git", "-C", str(worktree), "cat-file", "-s", f"{commit}:{editable}"],
        capture_output=True, text=True, timeout=15,
    )
    if size.returncode != 0:
        raise RuntimeError(f"candidate {editable!r} is absent from commit {commit[:12]} &")
    try:
        committed_size = int(size.stdout.strip())
    except ValueError as exc:
        raise RuntimeError("could not determine committed candidate size") from exc
    if committed_size > MAX_CANDIDATE_SOURCE_BYTES:
        raise RuntimeError(
            f"committed candidate exceeds the {MAX_CANDIDATE_SOURCE_BYTES}-byte limit"
        )
    result = subprocess.run(
        ["git", "-C", str(worktree), "show", f"{commit}:{editable}"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        timeout=15,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"candidate {editable!r} is absent from commit {commit[:12]}…"
        )
    committed_sha = hashlib.sha256(result.stdout).hexdigest()
    current_source = _read_regular_file_limited(
        candidate_path,
        max_bytes=MAX_CANDIDATE_SOURCE_BYTES,
        label="candidate source",
    )
    current_sha = hashlib.sha256(current_source).hexdigest()
    if current_sha != candidate_sha256 or committed_sha != candidate_sha256:
        raise RuntimeError(
            "candidate file SHA does not match both the worktree and the "
            "recorded champion commit; refusing finalization"
        )


def _bounded_artifact_sha256(path: Path, maximum: int, label: str) -> str:
    import stat
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError(f"trusted artifact {label} cannot be opened safely") from exc
    digest = hashlib.sha256()
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise RuntimeError(f"trusted artifact {label} is not a regular file")
        if info.st_size > maximum:
            raise RuntimeError(f"trusted artifact {label} exceeds its size limit")
        remaining = info.st_size
        while remaining:
            chunk = os.read(fd, min(65_536, remaining))
            if not chunk:
                raise RuntimeError(f"trusted artifact {label} was truncated")
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            raise RuntimeError(f"trusted artifact {label} grew while being read")
        return digest.hexdigest()
    finally:
        os.close(fd)

def _trusted_artifact_pair(
    result: dict, expected_dir: Path, *, task: str,
) -> tuple[dict, dict[str, Path]]:
    from research.model_export import MAX_ARCHITECTURE_BYTES, MAX_WEIGHTS_BYTES
    expected_dir = Path(expected_dir).resolve(strict=True)
    paths: dict[str, Path] = {}
    digests: dict[str, str] = {}
    for field, digest_field, maximum in (
        ("weights_path", "weights_sha256", MAX_WEIGHTS_BYTES),
        ("architecture_path", "architecture_sha256", MAX_ARCHITECTURE_BYTES),
    ):
        raw_value = result.get(field)
        if not isinstance(raw_value, str):
            raise RuntimeError(f"trusted artifact {field} path is missing")
        raw_path = Path(raw_value)
        if not raw_path.is_absolute() or raw_path.is_symlink():
            raise RuntimeError(f"trusted artifact {field} is a symlink or non-absolute path")
        try:
            parent = raw_path.parent.resolve(strict=True)
        except OSError as exc:
            raise RuntimeError(f"trusted artifact {field} is unavailable") from exc
        if not parent.is_relative_to(expected_dir):
            raise RuntimeError(f"trusted artifact {field} is outside its output directory")
        path = parent / raw_path.name
        digest = _bounded_artifact_sha256(path, maximum, field)
        expected_digest = result.get(digest_field)
        if not isinstance(expected_digest, str) or digest != expected_digest:
            raise RuntimeError(f"trusted artifact {field} SHA does not match")
        paths[field] = path
        digests[field] = digest
    attestation = result.get("trusted_validation")
    if not isinstance(attestation, dict):
        raise RuntimeError("sandbox did not return trusted artifact validation")
    expected_output_key = "syndrome_probs" if task == "syndrome" else "category_probs"
    shape = attestation.get("output_shape")
    if (
        attestation.get("task") != task
        or attestation.get("weights_sha256") != digests["weights_path"]
        or attestation.get("architecture_sha256") != digests["architecture_path"]
        or attestation.get("output_key") != expected_output_key
        or not isinstance(shape, list)
        or len(shape) != 2
        or any(type(value) is not int or value < 1 for value in shape)
    ):
        raise RuntimeError("sandbox artifact validation attestation is malformed")
    return attestation, paths

def _validate_preflight_artifacts(
    result: dict, task: str, candidate_sha256: str,
    preflight_config: dict,
) -> dict:
    if result.get("candidate_sha256") != candidate_sha256 or result.get("task") != task:
        raise RuntimeError("synthetic candidate preflight did not bind task and candidate SHA")
    attestation, _paths = _trusted_artifact_pair(
        result, Path(preflight_config["output_dir"]), task=task,
    )
    if (
        result.get("output_key") != attestation["output_key"]
        or result.get("output_shape") != attestation["output_shape"]
    ):
        raise RuntimeError("candidate preflight output differs from trusted validation")
    return {
        "output_key": attestation["output_key"],
        "output_shape": attestation["output_shape"],
        "weights_sha256": attestation["weights_sha256"],
        "architecture_sha256": attestation["architecture_sha256"],
        "sandbox_image_id": result.get("sandbox_image_id"),
    }

def _validate_fitted_artifacts(
    result: dict,
    *,
    task: str,
    editable: str,
    candidate_sha256: str,
    commit: str,
    pool: dict,
    output_dir: Path,
    preflight_config: dict,
) -> dict:
    report = result.get("finalize")
    if not isinstance(report, dict):
        raise RuntimeError("finalize worker returned no fit report")
    expected = {
        "candidate_file": editable,
        "candidate_sha256": candidate_sha256,
        "git_commit": commit,
        "pool_manifest_sha256": pool["manifest_sha256"],
        "selection_parent_manifest_sha256": pool["selection_parent_manifest_sha256"],
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise RuntimeError("finalize worker report does not match the validated fit inputs")
    attestation, _paths = _trusted_artifact_pair(result, output_dir, task=task)
    actual_output = report.get("candidate_output")
    if (
        not isinstance(actual_output, dict)
        or actual_output.get("output_key") != attestation["output_key"]
        or actual_output.get("output_shape") != attestation["output_shape"]
    ):
        raise RuntimeError("fitted artifact output differs from trusted sandbox validation")
    return {
        "weights_sha256": attestation["weights_sha256"],
        "architecture_sha256": attestation["architecture_sha256"],
        "output_key": attestation["output_key"],
        "output_shape": attestation["output_shape"],
    }

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
    task: str = "constitution",
    spawn=None,
    spawn_holdout=None,
) -> dict:
    """Drive the finalize: pool manifest here, fitting + holdout in children.

    The driver never imports the candidate module: the champion's code runs
    in the fit subprocess only, and the holdout is scored in an evaluator
    subprocess. `records` supports synthetic constitution tests only; syndrome
    finalization always reads dataset bytes bound to its frozen manifest.
    `spawn` and `spawn_holdout` let tests observe the child boundaries.
    `task` selects the data/scoring adapter and unlock ledger."""
    from research.frozen import TIME_BUDGET_SECONDS
    from research.spawn import (
        run_sandboxed_candidate,
        sandbox_image_identity,
        sandbox_worker_process_timeout,
        stage_candidate_source,
    )
    from research.tasks import (
        adapter_for,
        validate_manifest_contract,
        write_synthetic_preflight_inputs,
    )
    from research.train_worker import (
        MAX_CANDIDATE_SOURCE_BYTES,
        _candidate_path,
        _read_regular_file_limited,
    )

    adapter = adapter_for(task)
    if adapter.name != task:
        raise ValueError(f"task adapter mismatch for {task!r}")
    if task == "syndrome" and records is not None:
        raise ValueError(
            "syndrome finalization requires dataset bytes verified against the "
            "frozen manifest; in-memory records are not accepted"
        )
    editable = adapter.resolve_editable(editable)
    worktree = Path(worktree).resolve(strict=True)
    out_dir = Path(out_dir)
    budget = TIME_BUDGET_SECONDS if budget_seconds is None else float(budget_seconds)
    candidate_path = _candidate_path({
        "worktree_path": str(worktree), "editable": editable,
    })
    candidate_source = _read_regular_file_limited(
        candidate_path,
        max_bytes=MAX_CANDIDATE_SOURCE_BYTES,
        label="candidate source",
    )
    expected_sha = hashlib.sha256(candidate_source).hexdigest()
    _assert_git_binding({
        "worktree_path": str(worktree),
        "editable": editable,
        "git_commit": commit,
    })
    _assert_candidate_commit_binding(worktree, editable, commit, expected_sha)
    frozen = adapter.load_manifest(manifest_path)
    validate_manifest_contract(task, frozen)

    # Check available budget before running candidate code or reading records.
    _ledger_file, max_unlocks = ledger_params(task)
    ledger = global_ledger_path(worktree, task)
    _assert_unlock_budget(ledger, max_unlocks=max_unlocks)
    # Preflight executes the candidate only in Docker on generated synthetic
    # data. One immutable source snapshot is reused for the actual pool fit.
    sandbox_temp = tempfile.TemporaryDirectory(prefix="research-finalize-sandbox-")
    sandbox_root = Path(sandbox_temp.name)
    source_stage = stage_candidate_source(
        worktree, editable, expected_sha, sandbox_root / "source",
    )
    preflight_manifest, preflight_records = write_synthetic_preflight_inputs(
        task, frozen, sandbox_root / "preflight-input",
    )
    preflight_manifest_sha256 = _sha256(preflight_manifest)
    # Preflight gives the trusted worker time to start, build and export, but
    # its candidate.train call still receives exactly zero training seconds.
    preflight_budget = TIME_BUDGET_SECONDS
    preflight_config = {
        "worktree_path": str(worktree),
        "editable": editable,
        "task": task,
        "candidate_sha256": expected_sha,
        "seed": seed,
        "time_budget_seconds": preflight_budget,
        "train_batch_size": 8,
        "fit_manifest_path": str(preflight_manifest),
        "fit_records_path": str(preflight_records),
        "output_dir": str(sandbox_root / "preflight-output"),
        "output_stem": "contract",
    }
    try:
        preflight_result = run_sandboxed_candidate(
            preflight_config,
            preflight=True,
            timeout=sandbox_worker_process_timeout(preflight_budget),
            source_root=source_stage,
        )
        preflight = _validate_preflight_artifacts(
            preflight_result, task, expected_sha, preflight_config,
        )
        image_id = sandbox_image_identity()
        if image_id != preflight.get("sandbox_image_id"):
            raise RuntimeError("candidate sandbox image changed after preflight")
    except BaseException:
        sandbox_temp.cleanup()
        raise


    dataset_records = None if records is None else list(records)
    if records is None:
        if task == "syndrome":
            records = adapter.load_records_verified(
                dataset_path, frozen["dataset_sha256"]
            )
        else:
            records = adapter.load_records(dataset_path)
    pool = adapter.pool_manifest(
        frozen, records,
        notes=f"source-level champion {commit[:8]} refit: selection pool train+dev",
    ).data
    validate_manifest_contract(task, pool, selection_pool=True)
    champion_dir = out_dir / "champion"
    champion_dir.mkdir(parents=True, exist_ok=True)
    pool_path = champion_dir / "pool-manifest.json"
    pool_path.write_text(json.dumps(pool, indent=2, ensure_ascii=False))
    # the fit subprocess's only records: the pool rows (F-08 — no full-set
    # file path enters its config, and pool_manifest carries no holdout)
    pool_rows = set(pool["subject_row_by_id"].values())
    pool_records_path = champion_dir / "pool-records.json"
    pool_records_path.write_text(adapter.records_to_json(
        [r for r in records if r.subject_index in pool_rows]
    ))

    config = {
        "worktree_path": str(worktree),
        "editable": editable,
        "task": task,
        "candidate_sha256": expected_sha,
        "git_commit": commit,
        "sandbox_source_path": str(source_stage),
        "candidate_preflight": preflight,
        "data_parent_manifest_sha256": frozen["manifest_sha256"],
        "seed": seed,
        "time_budget_seconds": budget,
        "train_batch_size": 32,
        "data": {
            "pool_manifest_path": str(pool_path),
            "pool_records_path": str(pool_records_path),
        },
        "out_dir": str(out_dir),
    }
    config_path = out_dir / "finalize-worker-config.json"
    config_path.write_text(json.dumps(config, indent=2))
    try:
        fitted = (spawn or _spawn_worker)(config, config_path)
        fit_artifacts = _validate_fitted_artifacts(
            fitted,
            task=task,
            editable=editable,
            candidate_sha256=expected_sha,
            commit=commit,
            pool=pool,
            output_dir=champion_dir,
            preflight_config=preflight_config,
        )
    finally:
        sandbox_temp.cleanup()
    # Fit failure and malformed state_dict/architecture never spend an unlock.
    # Bind again after the worker and before opening the ledger for append.
    current_source = _read_regular_file_limited(
        candidate_path,
        max_bytes=MAX_CANDIDATE_SOURCE_BYTES,
        label="candidate source",
    )
    if hashlib.sha256(current_source).hexdigest() != expected_sha:
        raise RuntimeError("candidate changed during finalization; refusing unlock")
    _assert_git_binding({
        "worktree_path": str(worktree),
        "editable": editable,
        "git_commit": commit,
    })
    _assert_candidate_commit_binding(worktree, editable, commit, expected_sha)
    event = {
        "event_id": uuid.uuid4().hex,
        "event": "holdout_unlock_intent",
        "task": task,
        "unlocked_at": _iso(),
        "champion_commit": commit,
        "candidate_sha256": expected_sha,
        "candidate_preflight": {
            **preflight,
            "synthetic_manifest_sha256": preflight_manifest_sha256,
        },
        "fit_artifacts": fit_artifacts,
        "pool_manifest_sha256": pool["manifest_sha256"],
        "selection_parent_manifest_sha256": frozen["manifest_sha256"],
    }
    _charge_unlock_event(ledger, event, max_unlocks=max_unlocks)
    holdout_config = {
        "worktree_path": str(worktree),
        "task": task,
        "data": {
            "pool_manifest_path": str(pool_path),
            "frozen_manifest_path": str(manifest_path),
        },
        "unlock": {
            "ledger_path": str(ledger),
            "event_id": event["event_id"],
            "task": task,
            "candidate_sha256": expected_sha,
            "champion_commit": commit,
            "pool_manifest_sha256": pool["manifest_sha256"],
            "selection_parent_manifest_sha256": frozen["manifest_sha256"],
            "candidate_preflight": event["candidate_preflight"],
            "fit_artifacts": fit_artifacts,
        },
        "out_dir": str(out_dir),
    }
    if dataset_records:
        # records were injected (tests / smoke): pass them to the evaluator
        # subprocess as a file instead of a workbook path
        records_path = champion_dir / "holdout-input-records.json"
        records_path.write_text(adapter.records_to_json(dataset_records))
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
    from research.spawn import (
        apply_evaluator_resource_limits,
        bootstrap_command,
    )

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
                preexec_fn=apply_evaluator_resource_limits,
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
        return result.stdout.rstrip("\r\n")

    import re

    head = git("rev-parse", "HEAD")
    expected_commit = config.get("git_commit")
    if (
        not isinstance(expected_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", expected_commit) is None
        or head != expected_commit
    ):
        raise RuntimeError(
            f"worktree HEAD {head[:12]}… is not the full recorded champion "
            "commit; refusing to fit code from an unrecorded tree state"
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
    """Fit only the train+dev pool inside the candidate Docker sandbox.

    This worker does not receive the unlock ledger or any holdout path. The
    driver charges only after this fit and the trusted artifact round-trip.
    """
    from research.spawn import run_sandboxed_candidate, sandbox_worker_process_timeout
    from research.tasks import adapter_for, validate_manifest_contract
    from research.train_worker import (
        MAX_CANDIDATE_SOURCE_BYTES,
        _read_regular_file_limited,
    )

    task = config.get("task", "constitution")
    adapter = adapter_for(task)
    source_root = Path(config["sandbox_source_path"]).resolve(strict=True)
    parts = Path(config["editable"]).parts
    if (
        len(parts) != 3 or parts[0] != "src"
        or any(part in ("", ".", "..") for part in parts)
    ):
        raise RuntimeError(f"invalid editable path {config['editable']!r}")
    candidate_path = source_root.joinpath(*parts[1:])
    if candidate_path.is_symlink() or not candidate_path.is_file():
        raise RuntimeError("staged candidate source is missing or not a regular file")
    candidate_source = _read_regular_file_limited(
        candidate_path,
        max_bytes=MAX_CANDIDATE_SOURCE_BYTES,
        label="candidate source",
    )
    actual_sha = hashlib.sha256(candidate_source).hexdigest()
    expected_sha = config["candidate_sha256"]
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"staged candidate hashes to {actual_sha[:12]}… but the finalize "
            f"record binds {expected_sha[:12]}…; refusing to run"
        )
    _assert_git_binding(config)

    pool = adapter.load_manifest(config["data"]["pool_manifest_path"])
    validate_manifest_contract(task, pool, selection_pool=True)
    if pool["split"]["holdout"] or pool["split"]["dev"]:
        raise RuntimeError(
            "pool manifest must contain train rows only and no holdout/dev"
        )
    parent_sha = config.get("data_parent_manifest_sha256")
    if pool.get("selection_parent_manifest_sha256") != parent_sha:
        raise RuntimeError("pool manifest is not bound to the selected parent manifest")
    if records is None:
        records = adapter.records_from_json(
            Path(config["data"]["pool_records_path"]).read_text()
        )
    pool_rows = set(pool["subject_row_by_id"].values())
    leaked = sorted({record.subject_index for record in records} - pool_rows)
    if leaked:
        raise RuntimeError(
            f"pool records contain rows outside the pool manifest: {leaked[:3]} "
            "— refusing to fit on data the pool contract did not trim"
        )

    pool_records_path = config["data"]["pool_records_path"]
    temp_records = None
    injected_records = records is not None
    if injected_records:
        temp_records = tempfile.TemporaryDirectory(prefix="research-pool-input-")
        pool_records_path = str(Path(temp_records.name) / "pool-records.json")
        Path(pool_records_path).write_text(adapter.records_to_json(records))
    budget = float(config["time_budget_seconds"])
    train_config = {
        "worktree_path": str(config["worktree_path"]),
        "editable": config["editable"],
        "task": task,
        "candidate_sha256": actual_sha,
        "seed": int(config["seed"]),
        "time_budget_seconds": budget,
        "train_batch_size": int(config.get("train_batch_size", 32)),
        "fit_manifest_path": config["data"]["pool_manifest_path"],
        "fit_records_path": pool_records_path,
        "output_dir": str(Path(config["out_dir"]) / "champion"),
        "output_stem": "weights",
    }
    try:
        fitted = run_sandboxed_candidate(
            train_config,
            timeout=sandbox_worker_process_timeout(budget),
            source_root=source_root,
            expected_image_id=config["candidate_preflight"]["sandbox_image_id"],
        )
    finally:
        if temp_records is not None:
            temp_records.cleanup()
    worktree_commit = _assert_git_binding(config)
    finalize = {
        "candidate_file": config["editable"],
        "candidate_sha256": actual_sha,
        "worktree_commit": worktree_commit,
        "git_commit": config["git_commit"],
        "pool_manifest_sha256": pool["manifest_sha256"],
        "selection_parent_manifest_sha256": pool["selection_parent_manifest_sha256"],
        "refit_partition": "train+dev",
        "refit_rows": len(pool["split"]["train"]),
        "seed": int(config["seed"]),
        "time_budget_seconds": budget,
        "fit_seconds": fitted["fit_seconds"],
        "input_spec_version": fitted["input_spec_version"],
        "candidate_output": fitted["candidate_output"],
        "sandbox_image_id": fitted["sandbox_image_id"],
        "weights_sha256": fitted["weights_sha256"],
        "architecture_sha256": fitted["architecture_sha256"],
    }
    champion_dir = Path(config["out_dir"]) / "champion"
    finalize.update({
        "weights_path": fitted["weights_path"],
        "architecture_path": fitted["architecture_path"],
    })
    (champion_dir / "finalize.json").write_text(json.dumps(finalize, indent=2))
    return {"finalize": finalize, **fitted}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="research.finalize")
    parser.add_argument("--config", required=True,
                        help="JSON written by the trusted driver: worktree, "
                             "editable, candidate SHA, and trimmed pool paths")
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
