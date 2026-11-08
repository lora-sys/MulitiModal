"""#95 benchmark: measured numbers only, no invented acceptance threshold."""

from __future__ import annotations

import pytest

from physio.benchmark import BENCHMARK_SCHEMA_VERSION, BenchmarkConfig, run_benchmark

pytestmark = pytest.mark.synthetic


def test_benchmark_reports_measured_latency_and_memory():
    report = run_benchmark(BenchmarkConfig(n_events=30, repeats=3))
    assert report["schema_version"] == BENCHMARK_SCHEMA_VERSION
    assert report["evidence_level"] == "LOCAL_EXPERIMENTAL"
    assert report["import"]["events"] == 30
    assert report["import"]["throughput_events_per_s"] > 0
    for key in ("p50", "p95", "p99", "max"):
        assert report["window_build_ms"][key] >= 0
        assert report["inference_ms"][key] >= 0
    percentiles = report["inference_ms"]
    assert percentiles["p50"] <= percentiles["p95"] <= percentiles["p99"] <= percentiles["max"]
    assert report["memory"]["traced_peak_bytes"] >= 0


def test_benchmark_keeps_unvalidated_model_out_of_scoring():
    report = run_benchmark(BenchmarkConfig(n_events=12, repeats=2))
    assert "OK" not in report["inference_statuses"]
    assert set(report["inference_statuses"]) <= {"NOT_AVAILABLE", "ABSTAIN"}
    assert report["stale_status"] == "ABSTAIN"
    assert report["dataset"]["session_events"] == 12
    assert report["dataset"]["duplicates"] == 0


def test_benchmark_declares_no_medical_threshold():
    report = run_benchmark(BenchmarkConfig(n_events=8, repeats=1))
    joined = " ".join(report["notes"])
    assert "no latency or medical acceptance threshold" in joined
    assert "synthetic events only" in joined


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_events": 1},
        {"n_events": 10, "window_s": 0},
        {"n_events": 10, "min_events": 0},
        {"n_events": 10, "repeats": 0},
    ],
)
def test_invalid_benchmark_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        BenchmarkConfig(**kwargs)
