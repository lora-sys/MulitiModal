"""Windows-friendly CLI; all private runtime artifacts stay under ignored runs/."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import ResearchError, read_json
from .harness import DEFAULT_STORE, attempt, ingest, round_status, run_round, safe_store


def make_synthetic(path: Path, subjects: int, sessions: int = 1):
    if subjects < 1 or sessions < 1:
        raise ResearchError("BAD_SAMPLE_SIZE", "positive subject/session counts required")
    rng = np.random.default_rng(42)
    rows = []
    for subject in range(subjects):
        age, base_hr = int(rng.integers(24, 65)), float(rng.uniform(60, 95))
        for session in range(sessions):
            pre_hr, hrv = base_hr + rng.normal(0, 2), float(rng.uniform(20, 65))
            rows.append({"record_id": f"synthetic-{subject:03d}-{session:02d}", "subject_id": f"synthetic-{subject:03d}",
                         "age": age, "pre_hr": pre_hr, "pre_hrv": None if subject == 1 else hrv,
                         "post_hr": 0.8 * pre_hr + 8 + 0.03 * age + rng.normal(0, 1)})
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Bounded offline research; no Git commits or model API calls")
    subs = parser.add_subparsers(dest="operation", required=True)
    demo = subs.add_parser("demo")
    demo.add_argument("--subjects", type=int, default=5)
    demo.add_argument("--sessions", type=int, default=1)
    demo.add_argument("--store", type=Path, default=DEFAULT_STORE)
    ingestion = subs.add_parser("ingest")
    ingestion.add_argument("data", type=Path)
    ingestion.add_argument("--contract", type=Path, required=True)
    ingestion.add_argument("--store", type=Path, default=DEFAULT_STORE)
    ingestion.add_argument("--run", action="store_true", help="QC, version the explicit data and automatically run its bounded round")
    ingestion.add_argument("--review-code-upgrade", action="store_true", help="human-reviewed code upgrade; preserves existing data-version budget and historical artifacts")
    for operation in ("run", "status", "attempt"):
        sub = subs.add_parser(operation)
        sub.add_argument("round", type=Path)
        if operation == "attempt":
            sub.add_argument("--candidate-json", required=True, help="declarative JSON, not executable code")
        if operation == "run":
            sub.add_argument("--plan", type=Path)
    bench = subs.add_parser("bench")
    bench.add_argument("--output", type=Path, default=DEFAULT_STORE / "harness-benchmark.json")
    prediction = subs.add_parser("predict")
    prediction.add_argument("round", type=Path)
    prediction.add_argument("inputs", type=Path)
    prediction.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.operation == "demo":
            store = safe_store(args.store)
            path = store / "synthetic-input" / f"subjects-{args.subjects}-sessions-{args.sessions}.csv"
            make_synthetic(path, args.subjects, args.sessions)
            round_path = ingest(path, Path(__file__).parent / "constitution.synthetic.yaml", store)
            output = run_round(round_path)
        elif args.operation == "ingest":
            round_path = ingest(args.data, args.contract, args.store, args.review_code_upgrade)
            output = run_round(round_path) if args.run else round_status(round_path)
        elif args.operation == "run":
            output = run_round(args.round, read_json(args.plan) if args.plan else None)
        elif args.operation == "status":
            output = round_status(args.round)
        elif args.operation == "attempt":
            output = attempt(args.round, json.loads(args.candidate_json))
        elif args.operation == "predict":
            from .inference import predict
            output = predict(args.round, args.inputs, args.output)
        else:
            from .benchmark import benchmark
            output = benchmark(args.output)
        print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    except (ResearchError, OSError, ValueError) as exc:
        print(json.dumps({"error": getattr(exc, "code", "IO_ERROR"), "detail": str(exc)}))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
