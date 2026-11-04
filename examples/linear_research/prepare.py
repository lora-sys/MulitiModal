"""Fixed evaluator for a synthetic linear-regression research example."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import sys
import time
import uuid
from pathlib import Path

import research.history
import research.sandbox
from research.history import (
    append_history,
    committed_source,
    open_regular,
    read_regular,
    write_new_atomic,
)
from research.sandbox import resolve_image, run_candidate

TRAIN = [{"x": x, "y": 3 * x + 1} for x in (-3, -2, 0, 1, 2)]
INPUTS = [-4, 3, 4]
LABELS = [-11, 10, 13]
EDITABLE = "examples/linear_research/train.py"
METRIC = "synthetic/mean_absolute_error"


def protocol(image: str, budget: int, seed: int) -> str:
    fixed = {"train": TRAIN, "inputs": INPUTS, "labels": LABELS,
             "metric": METRIC, "direction": "minimize", "image": image,
             "budget": budget, "seed": seed, "promotion": "strict",
             "code_sha256": {name: hashlib.sha256(
                 read_regular(Path(path), 2 * 1024 * 1024)).hexdigest()
                             for name, path in (("evaluator", __file__),
                                                ("history", research.history.__file__),
                                                ("sandbox", research.sandbox.__file__))}}
    return hashlib.sha256(json.dumps(fixed, sort_keys=True).encode()).hexdigest()


def status(root: Path, state: str, **extra) -> None:
    payload = {"state": state, "task": "synthetic-linear", "evidence": "SMOKE_ONLY",
               "updated_at": time.time(), **extra}
    temporary = root / (".status-" + uuid.uuid4().hex)
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream)
        os.replace(temporary, root / "status.json")
    finally:
        temporary.unlink(missing_ok=True)


def record(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Record one synthetic experiment")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--decision", choices=("KEEP", "REVERT", "CRASH"), required=True)
    parser.add_argument("--description", required=True)
    args = parser.parse_args(argv)
    candidate = committed_source(args.repo, args.commit, EDITABLE)
    value, identity = None, ""
    if args.decision != "CRASH":
        if args.result is None:
            parser.error("a scored decision requires --result")
        result = json.loads(read_regular(args.result, 1024 * 1024))
        if result.get("candidate_sha256") != hashlib.sha256(candidate).hexdigest():
            raise ValueError("result does not match the committed candidate")
        if result.get("evidence") != "SMOKE_ONLY" or result.get("metric") != METRIC:
            raise ValueError("result is not from this synthetic task")
        if (result.get("direction") != "minimize"
                or not isinstance(result.get("image"), str)
                or not re.fullmatch(r"sha256:[0-9a-f]{64}", result["image"])
                or isinstance(result.get("budget_seconds"), bool)
                or not isinstance(result.get("budget_seconds"), int)
                or not 1 <= result["budget_seconds"] <= 3600
                or isinstance(result.get("seed"), bool)
                or not isinstance(result.get("seed"), int)):
            raise ValueError("result has invalid protocol parameters")
        identity = protocol(result["image"], result["budget_seconds"], result["seed"])
        if result.get("protocol_sha256") != identity:
            raise ValueError("result protocol does not match the fixed evaluator")
        value = result["value"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError("mean absolute error must be nonnegative")
    append_history(args.results, commit=args.commit, metric=METRIC, value=value,
                   direction="minimize", decision=args.decision, description=args.description,
                   protocol=identity, evidence="SMOKE_ONLY")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "record":
        return record(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--image", default="multimodal-research-worker:2.7.0-cpu")
    args = parser.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open_regular(args.out.parent / ".evaluation.lock") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.out.exists():
            raise FileExistsError("result already exists")
        try:
            status(args.out.parent, "PREPARING", deadline=time.time() + 60)
            candidate = read_regular(args.candidate, 2 * 1024 * 1024)
            image = resolve_image(args.image)
            identity = protocol(image, args.budget, args.seed)
            status(args.out.parent, "TRAINING", deadline=time.time() + args.budget + 45)
            predictions = run_candidate(candidate, TRAIN, INPUTS, image=image,
                                        budget_seconds=args.budget, seed=args.seed)
            if (len(predictions) != len(LABELS) or any(
                    isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) for value in predictions)):
                raise ValueError("candidate returned invalid predictions")
            value = sum(abs(actual - predicted) for actual, predicted in
                        zip(LABELS, predictions, strict=True)) / len(LABELS)
            if not math.isfinite(value):
                raise ValueError("mean absolute error is not finite")
            if protocol(image, args.budget, args.seed) != identity:
                raise RuntimeError("fixed evaluator code changed during the run")
            result = {"metric": METRIC, "direction": "minimize", "value": value,
                      "candidate_sha256": hashlib.sha256(candidate).hexdigest(),
                      "protocol_sha256": identity,
                      "image": image, "budget_seconds": args.budget, "seed": args.seed,
                      "evidence": "SMOKE_ONLY"}
            payload = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()
            write_new_atomic(args.out, payload)
            status(args.out.parent, "COMPLETE")
        except Exception as exc:
            status(args.out.parent, "FAILED", reason=type(exc).__name__)
            raise
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
