"""Causal sliding-window features for the realtime fatigue-state task (#97).

Only events that had already happened at ``cutoff`` may enter a window. Post
events, future events and whole-session aggregates are rejected by construction:
the builder never receives anything past the cutoff and it filters phases and
quality explicitly.

The window length is an engineering configuration, not a medical sampling
claim. Features are computed per canonical metric so that a partially missing
field (for example SpO2) never blanks the whole physiological modality.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from .metrics import METRIC_SPECS
from .schema import (
    DATA_TYPE_STRING,
    PHASE_DURING,
    PHASE_PRE,
    QUALITY_DEGRADED,
    QUALITY_OK,
    QUALITY_REJECTED,
    QUALITY_STALE,
    QUALITY_VALUES,
    PhysioEventV1,
    PhysioSessionV1,
)

WINDOW_FEATURE_VERSION = "physio-window-v1"

STAT_MEAN = "mean"
STAT_TREND = "trend"
STAT_VARIANCE = "variance"
STAT_BASELINE_DELTA = "baseline_delta"
STAT_COVERAGE = "coverage"
STATS = (STAT_MEAN, STAT_TREND, STAT_VARIANCE, STAT_BASELINE_DELTA, STAT_COVERAGE)

NUMERIC_METRICS: tuple[str, ...] = tuple(
    name for name, spec in METRIC_SPECS.items() if spec.data_type != DATA_TYPE_STRING
)

WINDOW_FEATURE_NAMES: tuple[str, ...] = tuple(
    f"{metric}.{stat}" for metric in NUMERIC_METRICS for stat in STATS
)

_USABLE_QUALITIES = (QUALITY_OK, QUALITY_DEGRADED)
_ALLOWED_PHASES = (PHASE_PRE, PHASE_DURING)


@dataclass(frozen=True)
class WindowConfig:
    """Engineering window settings; none of these are clinical thresholds."""

    window_s: float = 60.0
    min_events: int = 2
    max_age_s: float = 120.0
    trend_epsilon: float = 0.5

    def __post_init__(self) -> None:
        if not math.isfinite(self.window_s) or self.window_s <= 0:
            raise ValueError("window_s must be a positive finite number")
        if isinstance(self.min_events, bool) or not isinstance(self.min_events, int):
            raise ValueError("min_events must be an integer >= 1")
        if self.min_events < 1:
            raise ValueError("min_events must be an integer >= 1")
        if not math.isfinite(self.max_age_s) or self.max_age_s <= 0:
            raise ValueError("max_age_s must be a positive finite number")
        if not math.isfinite(self.trend_epsilon) or self.trend_epsilon < 0:
            raise ValueError("trend_epsilon must be a non-negative finite number")


def _observed_numeric(event: PhysioEventV1) -> tuple[tuple[str, float], ...]:
    observed: list[tuple[str, float]] = []
    for measurement in event.measurements:
        if not measurement.valid:
            continue
        spec = METRIC_SPECS.get(measurement.metric)
        if spec is None or spec.data_type == DATA_TYPE_STRING:
            continue
        value = measurement.value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        number = float(value)
        if math.isfinite(number):
            observed.append((measurement.metric, number))
    return tuple(observed)


def event_has_valid_numeric(event: PhysioEventV1) -> bool:
    """True when an event carries at least one usable numeric field."""
    return bool(_observed_numeric(event))


def _quality_of(events: tuple[PhysioEventV1, ...]) -> str:
    if not events:
        return QUALITY_REJECTED
    if any(event.quality == QUALITY_DEGRADED for event in events):
        return QUALITY_DEGRADED
    return QUALITY_OK


@dataclass(frozen=True)
class WindowFeatures:
    """Fixed-order, mask-aware feature vector for one cutoff.

    ``values`` always has the same length as ``feature_names``. Entries whose
    ``mask`` is False are placeholders (``0.0``) and must never be consumed as
    observations. `feature_map()` only exposes observed entries.
    """

    usable: bool
    feature_names: tuple[str, ...]
    values: tuple[float, ...]
    mask: tuple[bool, ...]
    coverage: float
    n_events: int
    start: datetime | None
    end: datetime | None
    quality: str
    metrics_present: tuple[str, ...]
    phase_counts: dict[str, int] = field(default_factory=dict)
    reason: str = ""

    def __post_init__(self) -> None:
        if len(self.feature_names) != len(self.values) or len(self.values) != len(self.mask):
            raise ValueError("feature_names, values and mask must have equal lengths")
        if any(not isinstance(name, str) or not name for name in self.feature_names):
            raise ValueError("feature_names must be non-empty strings")
        if any(not math.isfinite(value) for value in self.values):
            raise ValueError("window feature values must be finite")
        if not 0.0 <= self.coverage <= 1.0:
            raise ValueError("coverage must be within 0..1")
        if self.quality not in QUALITY_VALUES:
            raise ValueError(f"unknown window quality: {self.quality!r}")
        if self.usable and not any(self.mask):
            raise ValueError("a usable window must expose at least one observed feature")
        object.__setattr__(self, "phase_counts", dict(self.phase_counts))

    def feature_map(self) -> dict[str, float]:
        """Observed name -> value only; masked entries are not reported."""
        pairs = zip(self.feature_names, self.values, self.mask, strict=True)
        return {name: value for name, value, observed in pairs if observed}

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_version": WINDOW_FEATURE_VERSION,
            "usable": self.usable,
            "reason": self.reason,
            "quality": self.quality,
            "coverage": self.coverage,
            "n_events": self.n_events,
            "start": self.start.isoformat() if self.start is not None else None,
            "end": self.end.isoformat() if self.end is not None else None,
            "metrics_present": list(self.metrics_present),
            "phase_counts": dict(self.phase_counts),
            "feature_names": list(self.feature_names),
            "values": list(self.values),
            "mask": list(self.mask),
        }


def _empty_window(
    *,
    n_events: int,
    start: datetime | None,
    end: datetime | None,
    quality: str,
    phase_counts: dict[str, int],
    reason: str,
) -> WindowFeatures:
    names = WINDOW_FEATURE_NAMES
    return WindowFeatures(
        usable=False,
        feature_names=names,
        values=tuple(0.0 for _ in names),
        mask=tuple(False for _ in names),
        coverage=0.0,
        n_events=n_events,
        start=start,
        end=end,
        quality=quality,
        metrics_present=(),
        phase_counts=phase_counts,
        reason=reason,
    )


def _slope(points: list[tuple[float, float]], epsilon: float) -> float | None:
    """Least-squares slope in value units per second, or None if not estimable.

    The x origin is the first sample of that metric; a slope is translation
    invariant, so identical samples always produce the identical slope.
    """
    if len(points) < 2:
        return None
    origin = points[0][0]
    xs = [seconds - origin for seconds, _ in points]
    span = max(xs) - min(xs)
    if span < epsilon:
        return None
    count = float(len(points))
    mean_x = sum(xs) / count
    mean_y = sum(value for _, value in points) / count
    numerator = sum((x - mean_x) * (y - mean_y) for x, (_, y) in zip(xs, points, strict=True))
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        return None
    return numerator / denominator


class PhysioWindowBuilder:
    """Build one causal window at ``cutoff`` from a session's events."""

    def __init__(self, config: WindowConfig | None = None) -> None:
        self.config = config or WindowConfig()

    def baseline(
        self, session: PhysioSessionV1, cutoff: datetime
    ) -> dict[str, float]:
        """Per-metric Pre-only baseline at or before ``cutoff``.

        Post and future Pre events are excluded. This is a reference level, not
        a fatigue label.
        """
        self._check_cutoff(cutoff)
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for event in session.events:
            if event.phase != PHASE_PRE or event.measured_at > cutoff:
                continue
            if event.quality not in _USABLE_QUALITIES:
                continue
            for metric, value in _observed_numeric(event):
                totals[metric] = totals.get(metric, 0.0) + value
                counts[metric] = counts.get(metric, 0) + 1
        return {
            metric: totals[metric] / counts[metric]
            for metric in NUMERIC_METRICS
            if counts.get(metric, 0) > 0
        }

    def build(
        self,
        session: PhysioSessionV1,
        cutoff: datetime,
        baseline: Mapping[str, float] | None = None,
    ) -> WindowFeatures:
        self._check_cutoff(cutoff)
        effective_s = min(self.config.window_s, self.config.max_age_s)
        lower = cutoff - timedelta(seconds=effective_s)

        in_range = tuple(
            event
            for event in session.events
            if event.phase in _ALLOWED_PHASES
            and lower <= event.measured_at <= cutoff
        )
        usable_events = tuple(
            event for event in in_range if event.quality in _USABLE_QUALITIES and event.is_valid
        )
        phase_counts = {
            PHASE_PRE: sum(1 for event in usable_events if event.phase == PHASE_PRE),
            PHASE_DURING: sum(1 for event in usable_events if event.phase == PHASE_DURING),
        }

        if not in_range:
            return _empty_window(
                n_events=0,
                start=None,
                end=None,
                quality=QUALITY_REJECTED,
                phase_counts=phase_counts,
                reason="empty_window",
            )
        if not usable_events:
            quality = (
                QUALITY_STALE
                if all(event.quality == QUALITY_STALE for event in in_range)
                else QUALITY_REJECTED
            )
            return _empty_window(
                n_events=0,
                start=None,
                end=None,
                quality=quality,
                phase_counts=phase_counts,
                reason="no_usable_events",
            )

        points: dict[str, list[tuple[float, float]]] = {metric: [] for metric in NUMERIC_METRICS}
        total_samples = sum(len(event.measurements) for event in in_range)
        window_event_count = len(in_range)
        valid_numeric = 0
        for event in usable_events:
            for measurement in event.measurements:
                if not measurement.valid:
                    continue
                spec = METRIC_SPECS.get(measurement.metric)
                if spec is None or spec.data_type == DATA_TYPE_STRING:
                    continue
                value = measurement.value
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                number = float(value)
                if not math.isfinite(number) or measurement.metric not in points:
                    continue
                seconds = (event.measured_at - cutoff).total_seconds()
                points[measurement.metric].append((seconds, number))
                valid_numeric += 1

        n_events = len(usable_events)
        start = min(event.measured_at for event in usable_events)
        end = max(event.measured_at for event in usable_events)
        quality = _quality_of(usable_events)
        if valid_numeric == 0:
            return _empty_window(
                n_events=n_events,
                start=start,
                end=end,
                quality=quality,
                phase_counts=phase_counts,
                reason="no_numeric_values",
            )

        baseline_values = self._clean_baseline(baseline)
        values: list[float] = []
        mask: list[bool] = []
        metrics_present: list[str] = []
        for metric in NUMERIC_METRICS:
            observed = points[metric]
            samples = [value for _, value in observed]
            if samples:
                metrics_present.append(metric)
                mean = sum(samples) / len(samples)
                if len(samples) >= 2:
                    variance = sum((value - mean) ** 2 for value in samples) / len(samples)
                else:
                    variance = 0.0
            else:
                mean = 0.0
                variance = 0.0
            trend = _slope(observed, self.config.trend_epsilon) if observed else None
            delta = None
            if samples and metric in baseline_values:
                delta = mean - baseline_values[metric]

            stat_values = {
                STAT_MEAN: (mean if samples else 0.0, bool(samples)),
                STAT_TREND: (trend if trend is not None else 0.0, trend is not None),
                STAT_VARIANCE: (variance, len(samples) >= 2),
                STAT_BASELINE_DELTA: (delta if delta is not None else 0.0, delta is not None),
                STAT_COVERAGE: (
                    len(samples) / window_event_count if samples else 0.0,
                    bool(samples),
                ),
            }
            for stat in STATS:
                value, observed_flag = stat_values[stat]
                values.append(float(value))
                mask.append(observed_flag)

        coverage = valid_numeric / total_samples if total_samples else 0.0
        usable = n_events >= self.config.min_events
        reason = "ok" if usable else "too_few_events"
        return WindowFeatures(
            usable=usable,
            feature_names=WINDOW_FEATURE_NAMES,
            values=tuple(values),
            mask=tuple(mask),
            coverage=coverage,
            n_events=n_events,
            start=start,
            end=end,
            quality=quality,
            metrics_present=tuple(metrics_present),
            phase_counts=phase_counts,
            reason=reason,
        )

    @staticmethod
    def _check_cutoff(cutoff: datetime) -> None:
        if not isinstance(cutoff, datetime):
            raise TypeError("cutoff must be a datetime")
        if cutoff.tzinfo is None:
            raise ValueError("cutoff must be timezone-aware")

    @staticmethod
    def _clean_baseline(baseline: Mapping[str, float] | None) -> dict[str, float]:
        if baseline is None:
            return {}
        cleaned: dict[str, float] = {}
        for metric, value in baseline.items():
            if metric not in NUMERIC_METRICS:
                continue
            number = float(value)
            if math.isfinite(number):
                cleaned[metric] = number
        return cleaned


__all__ = [
    "NUMERIC_METRICS",
    "STATS",
    "STAT_BASELINE_DELTA",
    "STAT_COVERAGE",
    "STAT_MEAN",
    "STAT_TREND",
    "STAT_VARIANCE",
    "WINDOW_FEATURE_NAMES",
    "WINDOW_FEATURE_VERSION",
    "PhysioWindowBuilder",
    "WindowConfig",
    "WindowFeatures",
    "event_has_valid_numeric",
]
