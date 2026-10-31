"""Subprocess entry point: fit ONE candidate on the selection folds and score it.

Run as `python -m research.runner --config <json>`. Prints a JSON line:

    {"primary_value": float, "primary_std": float, "primary_sem": float,
     "folds": [...], "n_folds": int, "overran": bool, "fold_seconds": [...]}

The split of responsibility is the point:

- the runner builds every fold from the frozen manifest, imports the candidate,
  calls `build_model` + `train` with the fixed budget, and then scores the
  returned model with the frozen evaluator;
- the candidate never sees a score, and never touches the evaluator.

A candidate that hangs is killed by the PARENT (the loop passes a hard timeout),
because a candidate can always write `while True`. That is why the budget is
enforced on both sides: the runner stops the loop it controls, and the parent
kills the process the candidate controls.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path


def run(config: dict, folds: int, repeats: int) -> dict:
    import research.candidate as candidate
    from research.frozen import TIME_BUDGET_SECONDS, build_fold, evaluate_model

    manifest_path = config["data"]["manifest_path"]
    dataset_path = config["data"]["dataset_path"]
    budget = float(config.get("time_budget_seconds", TIME_BUDGET_SECONDS))
    scores: list[float | None] = []
    fold_seconds: list[float] = []
    overran = False
    for repeat in range(repeats):
        for fold in range(folds):
            fold_data, fold_manifest = build_fold(
                config, manifest_path, dataset_path, fold, folds, repeat
            )
            model = candidate.build_model(fold_data)
            started = time.monotonic()
            candidate.train(model, fold_data, budget, fold_data.seed)
            elapsed = time.monotonic() - started
            fold_seconds.append(round(elapsed, 2))
            if elapsed > budget * 1.5:
                # the runner cannot kill the candidate's code, but it can refuse
                # to report a score that was bought with more than the budget —
                # the parent's timeout is the real enforcement, this is the
                # honest record
                overran = True
            report = evaluate_model(model, fold_data, fold_manifest)
            scores.append(report.primary_value)
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