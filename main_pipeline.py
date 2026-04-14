#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from datetime import datetime

from run_dry_run import run_dry_run


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_step(cmd: list[str], name: str) -> None:
    print(f"[{ts()}] >>> Starting {name} ...", flush=True)
    subprocess.run(cmd, check=True)
    print(f"[{ts()}] >>> Finished {name}", flush=True)


def main() -> None:
    # Hard gate: run dry-run checks in-process before any subprocess call.
    try:
        print(f"[{ts()}] >>> Running in-process Dry Run gate ...", flush=True)
        run_dry_run()
        print(f"[{ts()}] >>> Dry Run gate passed", flush=True)
    except Exception as exc:
        print(f"[{ts()}] >>> Dry Run gate failed: {exc}", flush=True)
        sys.exit(1)

    run_step(["python3", "-u", "run_optuna.py"], "Optuna")
    run_step(["python3", "-u", "run_experiments.py"], "Experiments")
    run_step(["python3", "-u", "run_cross_domain.py"], "Cross-domain Validation")


if __name__ == "__main__":
    main()
