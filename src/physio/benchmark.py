"""Reproducible latency/throughput benchmark for the realtime engineering path (#95).

The benchmark reports measured numbers only. It deliberately defines no medical
or product acceptance threshold: without a device latency budget, a threshold
would be invented. Results are ``LOCAL_EXPERIMENTAL`` and machine-specific.
"""

from __future__ import annotations

import gc
import os
import platform
import statistics
import time
import tracemalloc
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from .adapters import DeviceEventAdapter
from .inference import InferenceSession
from .preprocessing import WindowScaler
from .stream import OnlineIngestor
from .windows import PhysioWindowBuilder, WindowConfig

BENCHMARK_SCHEMA_VERSION = "physio-benchmark-v1"
BASE = datetime(2026, 11, 7, 10, 0, tzinfo=UTC)


@dataclass(frozen=True)
class BenchmarkConfig:
    n_events: int = 240
    window_s: int = 60
    min_events: int = 2
    repeats: int = 5

    def __post_init__(self) -> None:
        if self.n_events < 2:
            raise ValueError("n_events must be at least two")
        if self.window_s <= 0 or self.min_events <= 0:
            raise ValueError("window_s and min_events must be positive")
        if self.repeats < 1:
            raise ValueError("repeats must be positive")


def _payload(sequence: int) -> dict[str, Any]:
    measured = BASE + timedelta(seconds=sequence * 5)
    return {
        "session_id": "bench-session",
        "subject_ref": "bench-subject",
        "device_ref": "bench-chair",
        "phase": "during",
        "measured_at": measured.isoformat(),
        "received_at": (measured + timedelta(seconds=1)).isoformat(),
        "sequence": sequence,
        "measurements": {
            "hr": 68 + sequence % 11,
            "spo2": 0.96 + (sequence % 4) * 0.005,
            "skin_temp": 33.0 + (sequence % 5) * 0.1,
            "rr": 15 + sequence % 3,
            "sbp": 115 + sequence % 9,
            "dbp": 75 + sequence % 7,
        },
        "units": {
            "hr": "bpm",
            "spo2": "fraction",
            "skin_temp": "°C",
            "rr": "breaths/min",
            "sbp": "mmHg",
            "dbp": "mmHg",
        },
    }


def _percentiles(samples_ms: list[float]) -> dict[str, float]:
    ordered = sorted(samples_ms)
    if not ordered:
        raise ValueError("no samples")

    def pick(fraction: float) -> float:
        index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
        return ordered[index]

    return {
        "mean": statistics.fmean(ordered),
        "p50": pick(0.50),
        "p95": pick(0.95),
        "p99": pick(0.99),
        "max": ordered[-1],
    }


def run_benchmark(config: BenchmarkConfig | None = None) -> dict[str, Any]:
    """Run import, window and inference timing on synthetic events."""
    config = config or BenchmarkConfig()
    gc.collect()
    tracemalloc.start()

    started = time.perf_counter()
    adapter = DeviceEventAdapter()
    events = [adapter.parse_event(_payload(sequence)) for sequence in range(config.n_events)]
    import_elapsed = time.perf_counter() - started

    ingestor = OnlineIngestor("bench-session", "bench-subject", "bench-chair")
    for event in events:
        ingestor.push(event)
    session = ingestor.session()
    builder = PhysioWindowBuilder(
        WindowConfig(window_s=config.window_s, min_events=config.min_events)
    )
    b = session.events[-1].measured_at
    cutoffs = [
        b - timedelta(seconds=config.window_s) + timedelta(seconds=index)
        for index in range(config.repeats)
    ]

    window_samples: list[float] = []
    for cutoff in cutoffs:
        start = time.perf_counter()
        builder.build(session, cutoff)
        window_samples.append((time.perf_counter() - start) * 1000.0)

    windows = [builder.build(session, cutoff) for cutoff in cutoffs]
    scaler = WindowScaler.fit(windows)

    inference_samples: list[float] = []
    statuses: list[str] = []
    for cutoff in cutoffs:
        inference = InferenceSession(session, builder, scaler=scaler)
        start = time.perf_counter()
        result = inference.update(cutoff)
        inference_samples.append((time.perf_counter() - start) * 1000.0)
        statuses.append(result.status)

    # One deliberately stale cutoff exercises the refusal path.
    stale = InferenceSession(session, builder).update(b + timedelta(hours=2))
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    payload = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "evidence_level": "LOCAL_EXPERIMENTAL",
        "isolation": "AGENT_ISOLATION_UNVERIFIED",
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
        },
        "config": {
            "n_events": config.n_events,
            "window_s": config.window_s,
            "min_events": config.min_events,
            "repeats": config.repeats,
        },
        "import": {
            "events": len(events),
            "elapsed_s": import_elapsed,
            "throughput_events_per_s": len(events) / import_elapsed if import_elapsed else None,
        },
        "window_build_ms": _percentiles(window_samples),
        "inference_ms": _percentiles(inference_samples),
        "inference_statuses": sorted(set(statuses)),
        "stale_status": stale.status,
        "cache_entries": len(getattr(inference, "_cache", {})),
        "memory": {
            "traced_current_bytes": current,
            "traced_peak_bytes": peak,
        },
        "dataset": {
            "session_events": len(session),
            "gaps": [round(gap.total_seconds(), 1) for gap in ingestor.gaps],
            "duplicates": ingestor.duplicates,
        },
        "notes": [
            "synthetic events only; no real device or subject",
            "no latency or medical acceptance threshold is defined",
            "inference uses the NOT_AVAILABLE gate because no validated model is supplied",
        ],
    }
    return payload


__all__ = ["BENCHMARK_SCHEMA_VERSION", "BenchmarkConfig", "run_benchmark"]
