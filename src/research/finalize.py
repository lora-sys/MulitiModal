"""Finalise the source-level run: refit the champion on the selection pool in
its own subprocess, then score the holdout once.

What this must not do:

* read the holdout while choosing anything. The champion is already fixed by the
  manifest before this module runs, and the holdout is touched exactly once, at
  the end, to produce a number;
* refit on train alone. The champion was selected by cross-validation over
  train+dev, so both partitions belong to the selection and the deliverable is
  fitted on all of it — with the pool manifest's scaler refit on those rows,
  and its holdout partition intact and untouched;
* import the champion's code into the driver. The first implementation imported
  research.candidate into the process running research-src-finalize, with no
  binding between the code that fitted and the candidate file whose SHA-256 the
  finalize record shows. Now the driver only prepares the pool manifest and
  spawns `python -m research.finalize` with the worktree's src as the only
  import path (the same pattern as the loop's runner); the worker asserts the
  candidate file's hash matches the recorded one BEFORE importing it, so the
  fitted code is provably the recorded one;
* unlock the holdout a third time. Phase 1 allows two unlock events
  (AGENTS.md rule 11): every successful unlock appends one event to the
  out_dir's append-only ledger and a third attempt is refused before any
  fitting starts. The pool manifest handed to the worker carries no holdout
  row mapping (data.real_split.pool_manifest); the worker restores it from the
  frozen parent AFTER fitting, so the fitting code never had holdout identity
  in reach.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

#: AGENTS.md rule 11: at most two holdout unlock events in Phase 1.
MAX_UNLOCKS = 2
UNLOCK_LEDGER_FILE = "holdout-unlock-ledger.jsonl"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _ledger_path(out_dir: Path) -> Path:
    return Path(out_dir) / UNLOCK_LEDGER_FILE


def unlock_count(out_dir: Path) -> int:
    path = _ledger_path(out_dir)
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def _assert_unlock_budget(out_dir: Path) -> None:
    used = unlock_count(out_dir)
    if used >= MAX_UNLOCKS:
        raise RuntimeError(
            f"holdout unlock budget exhausted: {used} of {MAX_UNLOCKS} events "
            f"recorded in {_ledger_path(out_dir)}; a further unlock needs a new "
            "data milestone or explicit user authorisation (AGENTS.md rule 11)"
        )


def _record_unlock(out_dir: Path, event: dict) -> None:
    with _ledger_path(out_dir).open("a") as fh:
        fh.write(json.dumps(event, sort_keys=True) + "\n")


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
) -> dict:
    """Drive the finalize: pool manifest here, fitting + holdout in the worker.

    The driver never imports research.candidate: the champion's code runs in
    the worker subprocess only. `records` and `spawn` exist so tests can
    inject a parsed fixture and observe the spawn without a real workbook or
    interpreter.
    """
    from data.mendeley import load_mendeley_scores
    from data.real_split import load_real_manifest, pool_manifest
    from research.frozen import TIME_BUDGET_SECONDS

    worktree = Path(worktree)
    out_dir = Path(out_dir)
    budget = TIME_BUDGET_SECONDS if budget_seconds is None else budget_seconds
    _assert_unlock_budget(out_dir)

    candidate_path = worktree / editable
    expected_sha = _sha256(candidate_path)
    frozen = load_real_manifest(manifest_path)
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

    config = {
        "worktree_path": str(worktree),
        "editable": editable,
        "candidate_sha256": expected_sha,
        "git_commit": commit,
        "seed": seed,
        "time_budget_seconds": budget,
        "data": {
            "pool_manifest_path": str(pool_path),
            "frozen_manifest_path": str(manifest_path),
            "dataset_path": str(dataset_path),
        },
        "out_dir": str(out_dir),
    }
    config_path = out_dir / "finalize-worker-config.json"
    config_path.write_text(json.dumps(config, indent=2))
    result = (spawn or _spawn_worker)(config, config_path)
    _record_unlock(out_dir, {
        "unlocked_at": _iso(),
        "champion_commit": commit,
        "candidate_sha256": expected_sha,
        "pool_manifest_sha256": pool["manifest_sha256"],
        "selection_parent_manifest_sha256": frozen["manifest_sha256"],
    })
    return result


def _spawn_worker(config: dict, config_path: Path) -> dict:
    """Run the finalize worker with this worktree's src as its import path."""
    import os
    import subprocess

    worktree = config["worktree_path"]
    budget = float(config["time_budget_seconds"])
    timeout = budget + 600.0
    command = [
        sys.executable, "-m", "research.finalize", "--config", str(config_path),
    ]
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "PYTHONPATH": str(Path(worktree) / "src"),
    }
    try:
        result = subprocess.run(
            command, cwd=worktree, text=True, capture_output=True,
            timeout=timeout, env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"finalize worker exceeded {timeout:.0f}s"
        ) from exc
    payload = result.stdout
    if result.returncode != 0:
        detail = (payload or result.stderr).strip().splitlines()
        last = detail[-1][:400] if detail else "no output"
        raise RuntimeError(f"finalize worker failed ({result.returncode}): {last}")
    try:
        parsed = json.loads(payload.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        raise RuntimeError(
            f"finalize worker produced no JSON result: {payload[-300:]!r}"
        ) from exc
    if not parsed.get("ok"):
        raise RuntimeError(f"finalize worker failed: {parsed.get('error')}")
    return parsed


def _worker(config: dict, records=None) -> dict:
    """Fit the champion on the pool and read the holdout once (worker side)."""
    import torch

    from data.real_split import load_real_manifest
    from research.frozen import FoldData
    from runtime.real_data import RealBatchProvider, real_eval_manifest, real_input_spec
    from runtime.tcm_eval import TCMTrustedEvaluator

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
    import research.candidate as candidate

    pool = load_real_manifest(config["data"]["pool_manifest_path"])
    if records is None:
        from data.mendeley import load_mendeley_scores

        records = load_mendeley_scores(config["data"]["dataset_path"])
    seed = int(config["seed"])
    budget = float(config["time_budget_seconds"])
    spec = real_input_spec(pool)
    dims = {"modalities": {n: p["dim"] for n, p in pool["modalities"].items()},
            "targets": 9}
    # the pool manifest carries pool rows only; the records are pruned the
    # same way, so holdout scores are not reachable from the fitting code
    pool_rows = set(pool["subject_row_by_id"].values())
    pool_records = [r for r in records if r.subject_index in pool_rows]
    train_provider = RealBatchProvider(pool, pool_records, "train", batch_size=32)
    fold = FoldData(
        train=train_provider, dev=None, input_spec=spec, dims=dims,
        n_train_subjects=len(pool["split"]["train"]), n_dev_subjects=0, seed=seed,
    )
    started = time.monotonic()
    torch.manual_seed(seed)
    model = candidate.build_model(fold)
    candidate.train(model, fold, budget, seed)
    fit_seconds = round(time.monotonic() - started, 2)

    champion_dir = Path(config["out_dir"]) / "champion"
    champion_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), champion_dir / "weights.pt")
    finalize = {
        "candidate_file": config["editable"],
        "candidate_sha256": actual_sha,
        "git_commit": config["git_commit"],
        "pool_manifest_sha256": pool["manifest_sha256"],
        "selection_parent_manifest_sha256": pool["selection_parent_manifest_sha256"],
        "refit_partition": "train+dev",
        "refit_rows": len(pool["split"]["train"]),
        "holdout_rows_untouched": len(pool["split"]["holdout"]),
        "seed": seed,
        "time_budget_seconds": budget,
        "fit_seconds": fit_seconds,
        "input_spec_version": spec.version,
    }
    (champion_dir / "finalize.json").write_text(json.dumps(finalize, indent=2))

    # ---- the single holdout read -----------------------------------------
    # restore the holdout row mapping from the frozen parent NOW, after
    # fitting: the pool manifest deliberately does not carry it
    frozen = load_real_manifest(config["data"]["frozen_manifest_path"])
    holdout_view = json.loads(json.dumps(pool))
    holdout_view["subject_row_by_id"] = {
        **holdout_view["subject_row_by_id"],
        **{sid: frozen["subject_row_by_id"][sid]
           for sid in frozen["split"]["holdout"]},
    }
    holdout = RealBatchProvider(holdout_view, records, "holdout", batch_size=64)
    report = TCMTrustedEvaluator(real_eval_manifest(holdout_view)).evaluate(
        model, holdout, partition="holdout"
    )
    from runtime.baselines import evaluate_baselines

    baselines = evaluate_baselines(holdout_view, records)
    return {
        "finalize": finalize,
        "holdout": report.to_dict(),
        "baselines": {
            name: {
                "macro_ap": vals["holdout"]["macro_ap"],
                "macro_f1": vals["holdout"]["macro_f1"],
                "subject_mae": vals["holdout"]["subject_mae"],
            }
            for name, vals in baselines.items()
            if isinstance(vals, dict) and "holdout" in vals
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="research.finalize")
    parser.add_argument("--config", required=True,
                        help="JSON written by the driver: worktree, editable, "
                             "candidate SHA, data paths, seed and budget")
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
