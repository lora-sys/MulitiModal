"""Compare live integrity-check overhead with the PR20 baseline, without training."""

import argparse
import ast
import hashlib
import json
import statistics
import subprocess
import time
import tracemalloc

from physio_autoresearch import harness
from physio_autoresearch.__main__ import make_synthetic
from physio_autoresearch.contracts import digest, write_json

BASELINE = "d6abaf1065d6e2359c37a80b96d5902e5baae4b8"


def baseline_loader():
    source = subprocess.check_output(["git", "show", f"{BASELINE}:physio_autoresearch/harness.py"], cwd=harness.ROOT)
    names = {"environment", "load_round"}
    functions = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = vars(harness).copy()
    exec(compile(ast.Module(body=functions, type_ignores=[]), "pr20_harness", "exec"), namespace)
    return namespace["load_round"], hashlib.sha256(source).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-code-upgrade", action="store_true")
    args = parser.parse_args()
    before, baseline_hash = baseline_loader()
    store = harness.ROOT / "runs/optimization/round-validation"
    data = store / "synthetic-6.csv"
    make_synthetic(data, 6, 2)
    path = harness.ingest(data, harness.ROOT / "physio_autoresearch/constitution.synthetic.yaml", store,
                          review_code_upgrade=args.review_code_upgrade)
    expected = before(path)
    assert harness.load_round(path) == expected
    samples = {"before": [], "after": []}
    calls = {}
    original_run, original_hashes = harness.subprocess.run, harness.core_hashes
    # Count the same live checks; nothing is cached between requests.
    for label, loader in (("before", before), ("after", harness.load_round)):
        counts = {"git_processes": 0, "source_scans": 0}
        def counted_run(*args, **kwargs):
            counts["git_processes"] += 1
            return original_run(*args, **kwargs)
        def counted_hashes():
            counts["source_scans"] += 1
            return original_hashes()
        harness.subprocess.run = counted_run
        if label == "before":
            loader.__globals__["core_hashes"] = counted_hashes
        else:
            harness.core_hashes = counted_hashes
        try:
            assert loader(path) == expected
        finally:
            harness.subprocess.run = original_run
            harness.core_hashes = original_hashes
            before.__globals__["core_hashes"] = original_hashes
        calls[label] = counts
    for block in range(9):
        order = [("before", before), ("after", harness.load_round)]
        if block % 2:
            order.reverse()
        for label, loader in order:
            started = time.perf_counter()
            for _ in range(10):
                assert loader(path) == expected
            samples[label].append((time.perf_counter() - started) / 10)
    peaks = {}
    for label, loader in (("before", before), ("after", harness.load_round)):
        tracemalloc.start()
        loader(path)
        peaks[label] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    times = {key: statistics.median(values) for key, values in samples.items()}
    report = {"baseline_commit": BASELINE, "baseline_harness_sha256": baseline_hash,
              "environment": harness.environment(), "data_sha256": expected[0]["data_sha256"],
              "split_sha256": expected[0]["qc"]["split_sha256"], "contract_sha256": digest(expected[1]),
              "source": "synthetic", "subjects": 6, "sessions_per_subject": 2, "seed": expected[1]["seed"],
              "output_equivalent": True, "median_seconds": times, "timing_samples_seconds": samples,
              "speedup": times["before"] / times["after"], "peak_python_bytes": peaks, "calls_per_read": calls,
              "scope": "warm round integrity check only; excludes Python startup/training; tracemalloc peak is not process RSS"}
    output = store / "comparison.json"
    write_json(output, report)
    print(json.dumps({"report": str(output), "median_seconds": times, "speedup": report["speedup"],
                      "peak_python_bytes": peaks, "calls_per_read": calls}))


if __name__ == "__main__":
    main()
