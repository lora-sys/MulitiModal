"""Benchmark entry for the realtime engineering path (#95); SMOKE_ONLY.

Run with the source tree on the path:

    PYTHONPATH=src python -m scripts.physio_benchmark --pretty
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from physio.benchmark import BenchmarkConfig, run_benchmark


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=240)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    report = run_benchmark(
        BenchmarkConfig(n_events=args.events, repeats=args.repeats)
    )
    print(json.dumps(report, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
