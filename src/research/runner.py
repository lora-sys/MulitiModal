"""Subprocess orchestrator: fit ONE candidate on the selection folds, one
one-shot training subprocess per fold, scoring in evaluator subprocesses.

Run as `python -m research.runner --config <json>`. Prints a JSON line:

    {"primary_value": float, "primary_std": float | None,
     "uncertainty_semantics": str, "folds": [...], "n_folds": int,
     "overran": bool, "fold_seconds": [...]}

This process imports NO candidate code — not the module, not a full pickled
model, nothing (F-02: candidate code ran here until the second hardening
round, and a monkeypatched `subprocess.run` could forge its own score). For
every (repeat, fold) it:

1. derives the fold's subject ids from the frozen manifest (the same
   derivation the evaluator uses) and writes the training subprocess's ONLY
   data inputs: a fit manifest (holdout emptied, dev dropped, row mapping
   pruned to the training rows) and the records for exactly those rows — the
   full-set workbook path never enters the training subprocess's config, and
   each fold starts from a fresh interpreter, so nothing accumulates across
   folds that a set-difference could turn back into holdout identity (F-08);
2. spawns `python -m research.train_worker` with a parent timeout that adds
   bounded Docker startup, handoff, and cleanup allowances to the compute
   budget, then reads back file paths and hashes from its JSON line;
3. spawns `research.evaluator`, which rebuilds the fold from the frozen
   manifest and scores the state_dict there. The runner's own integrity
   snapshot (manifest, dataset, candidate file) is taken before training and
   re-checked by the evaluator with its own pristine hashlib.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

PARENT_DOCKER_STARTUP_ALLOWANCE_SECONDS = 30.0
PARENT_DOCKER_CLEANUP_ALLOWANCE_SECONDS = 10.0
DEFAULT_EVALUATOR_TIMEOUT_SECONDS = 300.0
ROUND_FOLD_ORCHESTRATION_ALLOWANCE_SECONDS = 120.0
ROUND_LAUNCH_HANDOFF_ALLOWANCE_SECONDS = 120.0
CV_UNCERTAINTY_SEMANTICS = (
    "primary_std is the descriptive population standard deviation across the "
    "observed repeated-CV fold scores; folds overlap and are not independent, "
    "so no SEM or confidence interval is inferred"
)


def summarize_cv_scores(scores: list[float]) -> dict:
    """Summarize observed fold dispersion without an independence assumption."""
    values = [float(score) for score in scores]
    if not values:
        raise ValueError("at least one scored fold is required")
    mean = sum(values) / len(values)
    std = None
    if len(values) > 1:
        std = (sum((score - mean) ** 2 for score in values) / len(values)) ** 0.5
    return {
        "primary_value": mean,
        "primary_std": std,
        "uncertainty_semantics": CV_UNCERTAINTY_SEMANTICS,
    }


def _trusted_fit_seconds(result: dict, budget: float) -> float:
    from research.spawn import TRUSTED_FIT_TIMER

    if result.get("fit_timer") != TRUSTED_FIT_TIMER:
        raise RuntimeError("training worker did not provide trusted fit timing")
    try:
        reported_budget = float(result.get("budget_seconds"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(
            "training worker did not provide its trusted fit budget"
        ) from exc
    if not math.isfinite(reported_budget) or reported_budget != budget:
        raise RuntimeError("training worker timing does not match its fit budget")
    value = result.get("fit_seconds")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError("training worker provided invalid trusted fit timing")
    try:
        elapsed = float(value)
    except (OverflowError, ValueError) as exc:
        raise RuntimeError(
            "training worker provided invalid trusted fit timing"
        ) from exc
    if not math.isfinite(elapsed) or elapsed < 0:
        raise RuntimeError("training worker provided invalid trusted fit timing")
    if elapsed > budget:
        raise RuntimeError("training worker reported a trusted fit budget overrun")
    return elapsed


def run(config: dict, folds: int, repeats: int, records=None,
        spawn_training=None, spawn_evaluator=None) -> dict:
    from research.evaluator import integrity_snapshot, load_selection_inputs
    from research.frozen import TIME_BUDGET_SECONDS, selection_fold_ids
    from research.tasks import adapter_for

    task = config.get("task", "constitution")
    adapter = adapter_for(task)
    editable = config.get("editable")
    if editable is None:
        editable = adapter.default_editable
    if not isinstance(editable, str) or not editable:
        raise ValueError("editable must be a non-empty task candidate path")
    # Bind the task-specific candidate path once so hashing, worker execution,
    # and the trusted evaluator all refer to the same file.
    config = {**config, "task": task, "editable": editable}
    budget = float(config.get("time_budget_seconds", TIME_BUDGET_SECONDS))
    spawn_train = spawn_training or _spawn_training
    spawn = spawn_evaluator or _spawn_evaluator
    manifest, records, input_hashes = load_selection_inputs(
        config, adapter=adapter, records_override=records
    )
    candidate_sha = integrity_snapshot(
        config, input_hashes=input_hashes
    ).get("candidate_sha256")
    scores: list[float | None] = []
    fold_seconds: list[float] = []
    worker_compute_seconds: list[float] = []
    overran = False
    def progress(**payload) -> None:
        """One machine-readable line per fold, for the TRUSTED COORDINATOR.

        The dashboard's fold-level progress must not be written by candidate
        code, so the runner reports and the coordinator emits. This is the
        runner talking, not the candidate.
        """
        print(json.dumps({"progress": payload}, ensure_ascii=False), flush=True)

    with tempfile.TemporaryDirectory(prefix="research-eval-") as work_dir:
        work = Path(work_dir)
        for repeat in range(repeats):
            for fold in range(folds):
                progress(stage="TRAINING", repeat=repeat, fold=fold, folds=folds,
                         n_folds=folds * repeats)
                before = integrity_snapshot(config)
                train_ids, _dev_ids = selection_fold_ids(manifest, fold, folds, repeat)
                # the fit manifest is the fold manifest with dev dropped: holdout
                # emptied, scaler refit on the fold's train rows, row mapping
                # pruned to exactly those rows
                fit_manifest = adapter.fold_manifest(
                    manifest, records, train_ids, [],
                    notes=f"fit inputs for selection fold {fold} repeat {repeat}",
                ).data
                fit_rows = set(fit_manifest["subject_row_by_id"].values())
                fit_records = [r for r in records if r.subject_index in fit_rows]
                fit_manifest_path = work / f"fit-manifest-r{repeat}-f{fold}.json"
                fit_manifest_path.write_text(json.dumps(fit_manifest, ensure_ascii=False))
                fit_records_path = work / f"fit-records-r{repeat}-f{fold}.json"
                fit_records_path.write_text(adapter.records_to_json(fit_records))
                worker_config = {
                    "worktree_path": config["worktree_path"],
                    "editable": config["editable"],
                    "task": task,
                    "candidate_sha256": candidate_sha,
                    "seed": int(config.get("trainer", {}).get("seed", 0)),
                    "time_budget_seconds": budget,
                    "train_batch_size": int(
                        config.get("train_provider", {}).get("batch_size", 32)),
                    "fit_manifest_path": str(fit_manifest_path),
                    "fit_records_path": str(fit_records_path),
                    "output_dir": str(work),
                    "output_stem": f"model_r{repeat}_f{fold}",
                }
                worker_config_path = work / f"train-config-r{repeat}-f{fold}.json"
                worker_config_path.write_text(json.dumps(worker_config))
                started = time.monotonic()
                trained = spawn_train(worker_config, worker_config_path)
                elapsed = time.monotonic() - started
                fold_seconds.append(round(elapsed, 2))
                fit_seconds = _trusted_fit_seconds(trained, budget)
                worker_compute_seconds.append(fit_seconds)
                after = integrity_snapshot(config)
                if after != before:
                    changed = sorted(k for k in before if before[k] != after.get(k))
                    raise RuntimeError(
                        "training inputs were modified during training: "
                        f"{changed} — the fold is not scoreable"
                    )
                request = {
                    "fold": fold, "folds": folds, "repeat": repeat,
                    "weights_path": trained["weights_path"],
                    "architecture_path": trained["architecture_path"],
                    "weights_sha256": trained["weights_sha256"],
                    "architecture_sha256": trained["architecture_sha256"],
                    "integrity": before,
                }
                report = spawn(config, request)
                scores.append(report["primary_value"])
                progress(stage="EVALUATING", repeat=repeat, fold=fold, folds=folds,
                         n_folds=folds * repeats, score=report["primary_value"],
                         seconds=fold_seconds[-1])
    scored = [s for s in scores if s is not None]
    if not scored:
        raise RuntimeError("no fold produced a scoreable primary metric")
    summary = summarize_cv_scores(scored)
    return {
        **summary,
        "folds": scores,
        "n_folds": len(scored),
        "overran": False,
        "fold_seconds": fold_seconds,
        "worker_compute_seconds": worker_compute_seconds,
        "budget_seconds": budget,
        # H-08, recorded so a reader knows what the snapshot does NOT cover:
        # the pre-training snapshot hashes the manifest, the dataset and the
        # candidate file; the saved weights/architecture outputs are outside
        # it — the evaluator re-hashes those against the worker-recorded
        # hashes instead
        "integrity_snapshot_scope": (
            "pre-training snapshot: manifest_sha256, dataset_sha256, "
            "candidate_sha256; saved weights/architecture outputs are "
            "re-hashed by the evaluator against the worker-recorded hashes"
        ),
    }


def _child_env() -> dict:
    """Whitelist environment for every child: no inherited secrets, no other
    checkout's modules — see research.spawn (H-02: no PYTHONPATH at all, the
    isolated bootstrap injects the import path instead)."""
    from research.spawn import child_env

    return child_env()


def _spawn(
    config: dict,
    command: list[str],
    timeout: float,
    *,
    preexec_fn=None,
) -> dict:
    result = subprocess.run(
        command, cwd=config["worktree_path"], text=True, capture_output=True,
        timeout=timeout, env=_child_env(), preexec_fn=preexec_fn,
    )
    payload = result.stdout
    if result.returncode != 0:
        # stdout may carry candidate/library noise; the traceback
        # lives on stderr, so it wins when the child failed
        detail = (result.stderr or payload).strip().splitlines()
        last = detail[-1][:400] if detail else "no output"
        raise RuntimeError(f"subprocess failed ({result.returncode}): {last}")
    try:
        parsed = json.loads(payload.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        raise RuntimeError(
            f"subprocess produced no JSON result: {payload[-300:]!r}"
        ) from exc
    if not parsed.get("ok"):
        raise RuntimeError(f"subprocess refused: {parsed.get('error')}")
    return parsed


def _training_parent_timeout(compute_budget: float) -> float:
    """Add each bounded sandbox stage around the strict fit deadline."""
    from research.spawn import (
        ARTIFACT_VALIDATION_TIMEOUT_SECONDS,
        HOST_ARTIFACT_EXPORT_FILE_COUNT,
        HOST_ARTIFACT_EXPORT_TIMEOUT_SECONDS,
        HOST_CLEANUP_ALLOWANCE_SECONDS,
        HOST_HANDOFF_SECONDS,
        sandbox_worker_process_timeout,
    )

    return (
        PARENT_DOCKER_STARTUP_ALLOWANCE_SECONDS
        + sandbox_worker_process_timeout(compute_budget)
        + ARTIFACT_VALIDATION_TIMEOUT_SECONDS
        + HOST_ARTIFACT_EXPORT_TIMEOUT_SECONDS * HOST_ARTIFACT_EXPORT_FILE_COUNT
        + HOST_HANDOFF_SECONDS
        + HOST_CLEANUP_ALLOWANCE_SECONDS
        + PARENT_DOCKER_CLEANUP_ALLOWANCE_SECONDS
    )


def source_loop_round_timeout_seconds(
    folds: int,
    repeats: int,
    compute_budget_seconds: float,
    evaluator_timeout_seconds: float = DEFAULT_EVALUATOR_TIMEOUT_SECONDS,
) -> float:
    """Bound one source-loop round by every child deadline plus orchestration.

    The fit budget remains the strict candidate compute budget. This outer
    deadline only adds the maximum legal worker lifecycle, evaluator deadline,
    and bounded time for per-fold data preparation, hashing, JSON handoff, and
    one-time runner startup/cleanup.
    """
    compute_budget_seconds = float(compute_budget_seconds)
    evaluator_timeout_seconds = float(evaluator_timeout_seconds)
    if folds <= 0 or repeats <= 0:
        raise ValueError("folds and repeats must be positive")
    if not math.isfinite(compute_budget_seconds) or compute_budget_seconds <= 0:
        raise ValueError("compute budget must be finite and positive")
    if not math.isfinite(evaluator_timeout_seconds) or evaluator_timeout_seconds <= 0:
        raise ValueError("evaluator timeout must be finite and positive")

    per_fold_seconds = (
        _training_parent_timeout(compute_budget_seconds)
        + evaluator_timeout_seconds
        + ROUND_FOLD_ORCHESTRATION_ALLOWANCE_SECONDS
    )
    return (
        folds * repeats * per_fold_seconds
        + ROUND_LAUNCH_HANDOFF_ALLOWANCE_SECONDS
    )


def _spawn_training(config: dict, worker_config_path: Path) -> dict:
    """Run a fold with explicit setup, export, validation, handoff and cleanup bounds."""
    from research.spawn import bootstrap_command

    budget = float(config["time_budget_seconds"])
    return _spawn(
        config,
        bootstrap_command(
            config["worktree_path"], "research.train_worker",
            "--config", str(worker_config_path),
        ),
        timeout=_training_parent_timeout(budget),
    )


def _spawn_evaluator(config: dict, request: dict) -> dict:
    """Score one fold in a fresh, environment-restricted subprocess."""
    from research.spawn import apply_evaluator_resource_limits, bootstrap_command

    with tempfile.TemporaryDirectory(prefix="research-eval-io-") as io_dir:
        io = Path(io_dir)
        config_path = io / "config.json"
        config_path.write_text(json.dumps(config))
        request_path = io / "request.json"
        request_path.write_text(json.dumps(request))
        return _spawn(
            config,
            bootstrap_command(
                config["worktree_path"], "research.evaluator",
                "--config", str(config_path), "--request", str(request_path),
            ),
            timeout=float(config.get(
                "evaluator_timeout_s", DEFAULT_EVALUATOR_TIMEOUT_SECONDS
            )),
            preexec_fn=apply_evaluator_resource_limits,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="research.runner")
    parser.add_argument("--config", required=True,
                        help="JSON file with data paths, batch size and seed")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text())
    try:
        result = run(config, args.folds, args.repeats)
    except Exception as exc:  # noqa: BLE001 — the loop records the crash
        print(json.dumps({
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-1500:],
        }))
        return 1
    print(json.dumps({"ok": True, **result}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
