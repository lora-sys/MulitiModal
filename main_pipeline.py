#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Main training pipeline")
    parser.add_argument(
        "--force-optuna",
        action="store_true",
        help="Force rerun Optuna even if checkpoints/optuna_best_params.json exists.",
    )
    return parser.parse_args()


def has_valid_optuna_params(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    required = ("lr", "weight_decay", "batch_size")
    return all(k in data for k in required)


def run_step(cmd: list[str], name: str) -> None:
    print(f"[{ts()}] >>> Starting {name} ...", flush=True)
    subprocess.run(cmd, check=True)
    print(f"[{ts()}] >>> Finished {name}", flush=True)


def main() -> None:
    args = parse_args()

    # Hard gate: run dry-run checks in-process before any subprocess call.
    try:
        from run_dry_run import run_dry_run

        print(f"[{ts()}] >>> Running in-process Dry Run gate ...", flush=True)
        run_dry_run()
        print(f"[{ts()}] >>> Dry Run gate passed", flush=True)
    except Exception as exc:
        print(f"[{ts()}] >>> Dry Run gate failed: {exc}", flush=True)
        sys.exit(1)

    optuna_file = Path("checkpoints") / "optuna_best_params.json"
    if args.force_optuna:
        run_step(["python3", "-u", "run_optuna.py"], "Optuna")
    elif has_valid_optuna_params(optuna_file):
        print(f"[{ts()}] >>> Reusing existing Optuna params at {optuna_file}, skipping Optuna.", flush=True)
    else:
        run_step(["python3", "-u", "run_optuna.py"], "Optuna")

    run_step(["python3", "-u", "run_experiments.py"], "Experiments")
    run_step(["python3", "-u", "run_cross_domain.py"], "Cross-domain Validation")


if __name__ == "__main__":
    main()
