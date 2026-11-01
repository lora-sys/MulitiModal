"""THE ONLY PROCESS THAT RUNS CANDIDATE CODE: one fit, one shot, then exit.

Run as `python -m research.train_worker --config <json>`. Prints a JSON line:

    {"ok": true, "weights_path": ..., "architecture_path": ...,
     "weights_sha256": ..., "architecture_sha256": ..., "fit_seconds": float}

The structural rule of this loop (2026-11-02 hardening round): candidate code
— `research.candidate.train` and any deserialization hook it could install —
never executes in the runner, the loop driver, the finalize worker, or the
evaluator. It executes HERE and nowhere else: a one-shot subprocess whose

* environment is a whitelist (PATH/HOME/PYTHONPATH only — no inherited
  credentials, no other checkout's modules);
* data inputs are files the frozen driver wrote: the fit manifest (holdout
  emptied, dev partition dropped, subject_row_by_id pruned to the training
  rows) and the records for exactly those rows — the full-set workbook path
  is not in this process's config at all;
* candidate-facing FoldData has `dev=None`: there is no dev partition in
  reach to fit against (F-06);
* outputs are files on disk: a state_dict-only weights file (loadable with
  `torch.load(weights_only=True)`) and the frozen-architecture JSON. Nothing
  the candidate trained against is passed in memory to any other process.

The candidate file's SHA-256 is checked BEFORE the module is imported, so
what runs is provably the code the record binds. A worker that overruns its
budget is killed by its parent (the runner passes budget + grace as the
subprocess timeout); that wall clock is the whole compute the candidate gets,
including any background thread it tries to leave behind.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

from data.mendeley import records_from_json
from data.real_split import load_real_manifest
from research.frozen import TIME_BUDGET_SECONDS, training_fold_data


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assert_candidate_binding(config: dict) -> str:
    """Refuse to import unless the file hashes to the recorded value."""
    candidate_path = Path(config["worktree_path"]) / config["editable"]
    actual = _sha256(candidate_path)
    expected = config.get("candidate_sha256")
    if expected and actual != expected:
        raise RuntimeError(
            f"candidate file {candidate_path} hashes to {actual[:12]}… but the "
            f"record binds {expected[:12]}…; refusing to run code that is not "
            "the recorded one"
        )
    return actual


def fit(config: dict, records=None) -> dict:
    """Fit one candidate on the trimmed inputs; write weights + architecture."""
    import research.candidate as candidate
    from research.model_export import save_eval_model

    actual_sha = assert_candidate_binding(config)

    fit_manifest = load_real_manifest(config["fit_manifest_path"])
    if records is None:
        records = records_from_json(Path(config["fit_records_path"]).read_text())
    fit_rows = set(fit_manifest["subject_row_by_id"].values())
    leaked = sorted({r.subject_index for r in records} - fit_rows)
    if leaked:
        raise RuntimeError(
            f"fit records contain rows outside the fit manifest: {leaked[:3]} — "
            "refusing to train on data the fold contract did not trim"
        )
    seed = int(config.get("seed", 0))
    budget = float(config.get("time_budget_seconds", TIME_BUDGET_SECONDS))
    import torch

    torch.manual_seed(seed)  # before training_fold_data and build_model (H-06)
    fold = training_fold_data(fit_manifest, records, config)

    started = time.monotonic()
    model = candidate.build_model(fold)
    candidate.train(model, fold, budget, seed)
    fit_seconds = round(time.monotonic() - started, 2)
    saved = save_eval_model(model, config["output_dir"], config["output_stem"])
    return {
        "candidate_sha256": actual_sha,
        "fit_seconds": fit_seconds,
        "seed": seed,
        "budget_seconds": budget,
        "input_spec_version": fold.input_spec.version,
        **saved,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="research.train_worker")
    parser.add_argument("--config", required=True,
                        help="JSON written by the runner/driver: worktree, "
                             "editable, candidate SHA, trimmed data paths")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text())
    try:
        result = fit(config)
    except Exception as exc:  # noqa: BLE001 — the parent records the crash
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
