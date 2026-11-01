"""Subprocess entry point: fit ONE candidate on the selection folds.

Run as `python -m research.runner --config <json>`. Prints a JSON line:

    {"primary_value": float, "primary_std": float, "primary_sem": float,
     "folds": [...], "n_folds": int, "overran": bool, "fold_seconds": [...]}

The split of responsibility is the point, and it spans TWO process
boundaries:

- this process builds every fold from the frozen manifest and imports the
  candidate: `build_model` + `train` with the fixed budget. The model is
  then saved to a file and the fold data is discarded — the candidate
  trained against them in place, so nothing it held may be trusted after
  train() returns;
- the score comes from a fresh `research.evaluator` subprocess that rebuilds
  the fold from the frozen manifest and loads the saved model there. The
  candidate never sees a score, cannot poison the FoldData the evaluator
  uses (the evaluator builds its own), and cannot rebind the evaluator
  in-process (its code ran in a different interpreter). This process records
  an integrity snapshot of the manifest, the dataset and the candidate file
  before training and refuses to score if any of them changed during
  training; the evaluator re-checks the snapshot with its own pristine
  hashlib.

A candidate that hangs is killed by the PARENT (the loop passes a hard
timeout), because a candidate can always write `while True`. That is why the
budget is enforced on both sides: the runner stops the loop it controls, and
the parent kills the process the candidate controls.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import traceback
from pathlib import Path


def run(config: dict, folds: int, repeats: int, records=None,
        spawn_evaluator=None) -> dict:
    import research.candidate as candidate
    from research.evaluator import integrity_snapshot
    from research.frozen import TIME_BUDGET_SECONDS, build_fold

    manifest_path = config["data"]["manifest_path"]
    dataset_path = config["data"]["dataset_path"]
    budget = float(config.get("time_budget_seconds", TIME_BUDGET_SECONDS))
    spawn = spawn_evaluator or _spawn_evaluator
    scores: list[float | None] = []
    fold_seconds: list[float] = []
    overran = False
    with tempfile.TemporaryDirectory(prefix="research-eval-") as eval_dir:
        for repeat in range(repeats):
            for fold in range(folds):
                before = integrity_snapshot(config)
                fold_data, _ = build_fold(
                    config, manifest_path, dataset_path, fold, folds, repeat,
                    records=records,
                )
                model = candidate.build_model(fold_data)
                started = time.monotonic()
                candidate.train(model, fold_data, budget, fold_data.seed)
                elapsed = time.monotonic() - started
                fold_seconds.append(round(elapsed, 2))
                if elapsed > budget * 1.5:
                    # the runner cannot kill the candidate's code, but it can
                    # refuse to report a score that was bought with more than
                    # the budget — the parent's timeout is the real
                    # enforcement, this is the honest record
                    overran = True
                after = integrity_snapshot(config)
                if after != before:
                    changed = sorted(k for k in before if before[k] != after.get(k))
                    raise RuntimeError(
                        "training inputs were modified during training: "
                        f"{changed} — the fold is not scoreable"
                    )
                # hand the evaluator a file and a request, never the fold:
                # the objects the candidate trained against are dead here
                model_path = Path(eval_dir) / f"model-r{repeat}-f{fold}.pt"
                import torch

                torch.save(model, model_path)
                del model, fold_data
                request = {
                    "fold": fold, "folds": folds, "repeat": repeat,
                    "model_path": str(model_path),
                    "integrity": before,
                }
                report = spawn(config, request)
                scores.append(report["primary_value"])
    scored = [s for s in scores if s is not None]
    if not scored:
        raise RuntimeError("no fold produced a scoreable primary metric")
    mean = sum(scored) / len(scored)
    std = (sum((s - mean) ** 2 for s in scored) / len(scored)) ** 0.5
    return {
        "primary_value": mean,
        "primary_std": std,
        "primary_sem": std / (len(scored) ** 0.5),
        "folds": scores,
        "n_folds": len(scored),
        "overran": overran,
        "fold_seconds": fold_seconds,
        "budget_seconds": budget,
    }


def _spawn_evaluator(config: dict, request: dict) -> dict:
    """Score one fold in a fresh, environment-restricted subprocess.

    Same pattern as the loop's _run_runner: only this worktree's src on the
    import path, no inherited environment, so the evaluator cannot reach the
    loop's modules or another checkout's code.
    """
    import os
    import subprocess

    worktree = config["worktree_path"]
    with tempfile.TemporaryDirectory(prefix="research-eval-io-") as io_dir:
        io = Path(io_dir)
        config_path = io / "config.json"
        config_path.write_text(json.dumps(config))
        request_path = io / "request.json"
        request_path.write_text(json.dumps(request))
        command = [
            sys.executable, "-m", "research.evaluator",
            "--config", str(config_path), "--request", str(request_path),
        ]
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            "PYTHONPATH": str(Path(worktree) / "src"),
        }
        try:
            result = subprocess.run(
                command, cwd=worktree, text=True, capture_output=True,
                timeout=float(config.get("evaluator_timeout_s", 300.0)),
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("evaluator subprocess timed out") from exc
        payload = result.stdout
    if result.returncode != 0:
        detail = (payload or result.stderr).strip().splitlines()
        last = detail[-1][:400] if detail else "no output"
        raise RuntimeError(f"evaluator failed ({result.returncode}): {last}")
    try:
        parsed = json.loads(payload.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        raise RuntimeError(
            f"evaluator produced no JSON result: {payload[-300:]!r}"
        ) from exc
    if not parsed.get("ok"):
        raise RuntimeError(f"evaluator refused to score: {parsed.get('error')}")
    return parsed


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
