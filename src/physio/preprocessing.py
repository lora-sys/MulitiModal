"""Mask-aware normalization of window features (#97).

The scaler is fit on training rows only. Missing entries stay masked: the
transform emits a placeholder ``0.0`` only so that a fixed-length tensor can be
built, while the accompanying mask remains False and downstream code must not
read that value. Nothing is imputed with a mean or a zero observation.

Artifacts are plain JSON with an explicit schema, feature-name contract and
sha256 digest, so a wrong or hand-edited artifact fails loudly.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .windows import WINDOW_FEATURE_NAMES, WINDOW_FEATURE_VERSION, WindowFeatures

PHYSIO_WINDOW_SCALER_SCHEMA = "physio-window-scaler-v1"
DEFAULT_MIN_STD = 1e-8

_FLOAT_TOLERANCE = 1e-12


class PreprocessingError(ValueError):
    """Raised when normalization cannot be applied honestly."""


@dataclass(frozen=True)
class ScaledWindow:
    """Normalized single-row window plus its explicit observation mask."""

    feature_names: tuple[str, ...]
    values: tuple[float, ...]
    mask: tuple[bool, ...]

    def __post_init__(self) -> None:
        if len(self.feature_names) != len(self.values) or len(self.values) != len(self.mask):
            raise ValueError("feature_names, values and mask must have equal lengths")
        if any(not math.isfinite(value) for value in self.values):
            raise ValueError("scaled values must be finite")


def _canonical_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class WindowScaler:
    """Per-feature standardization fitted on observed training entries only."""

    feature_names: tuple[str, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    observed_counts: tuple[int, ...]
    min_std: float = DEFAULT_MIN_STD
    feature_version: str = WINDOW_FEATURE_VERSION

    def __post_init__(self) -> None:
        size = len(self.feature_names)
        if size == 0:
            raise ValueError("scaler needs at least one feature")
        if len(set(self.feature_names)) != size:
            raise ValueError("scaler feature names must be unique")
        if len(self.mean) != size or len(self.std) != size or len(self.observed_counts) != size:
            raise ValueError("scaler arrays must match feature_names")
        if any(not math.isfinite(value) for value in self.mean):
            raise ValueError("scaler means must be finite")
        if any(not math.isfinite(value) or value <= 0 for value in self.std):
            raise ValueError("scaler std values must be positive and finite")
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in self.observed_counts
        ):
            raise ValueError("observed_counts must be non-negative integers")
        if not math.isfinite(self.min_std) or self.min_std <= 0:
            raise ValueError("min_std must be positive and finite")

    @classmethod
    def fit(
        cls,
        windows: Iterable[WindowFeatures],
        *,
        feature_names: Sequence[str] | None = None,
        min_std: float = DEFAULT_MIN_STD,
    ) -> WindowScaler:
        """Fit on training rows only; unobserved entries never enter statistics."""
        rows = list(windows)
        if not rows:
            raise PreprocessingError("cannot fit WindowScaler without training windows")
        names = tuple(feature_names) if feature_names is not None else rows[0].feature_names
        if not names or len(set(names)) != len(names):
            raise PreprocessingError("feature_names must be non-empty and unique")
        for row in rows:
            if row.feature_names != names:
                raise PreprocessingError(
                    "training window feature names do not match the scaler contract"
                )

        size = len(names)
        sums = [0.0] * size
        squares = [0.0] * size
        counts = [0] * size
        for row in rows:
            for index, (value, observed) in enumerate(
                zip(row.values, row.mask, strict=True)
            ):
                if not observed:
                    continue
                number = float(value)
                if not math.isfinite(number):
                    raise PreprocessingError("training window contains a non-finite value")
                sums[index] += number
                squares[index] += number * number
                counts[index] += 1

        means: list[float] = []
        stds: list[float] = []
        for index in range(size):
            if counts[index] == 0:
                means.append(0.0)
                stds.append(1.0)
                continue
            mean = sums[index] / counts[index]
            variance = squares[index] / counts[index] - mean * mean
            if variance < 0.0 and variance > -_FLOAT_TOLERANCE:
                variance = 0.0
            std = math.sqrt(max(variance, 0.0))
            means.append(mean)
            stds.append(std if std >= min_std else 1.0)
        return cls(
            feature_names=names,
            mean=tuple(means),
            std=tuple(stds),
            observed_counts=tuple(counts),
            min_std=min_std,
        )

    def transform(self, window: WindowFeatures) -> ScaledWindow:
        """Normalize one window; masked entries remain masked."""
        if not isinstance(window, WindowFeatures):
            raise PreprocessingError("transform expects a WindowFeatures instance")
        if tuple(window.feature_names) != self.feature_names:
            raise PreprocessingError(
                "window feature_names do not match the scaler feature contract"
            )
        values: list[float] = []
        mask: list[bool] = []
        for index, name in enumerate(self.feature_names):
            observed = bool(window.mask[index])
            raw = float(window.values[index])
            if not observed:
                values.append(0.0)
                mask.append(False)
                continue
            if not math.isfinite(raw):
                raise PreprocessingError(f"observed feature {name!r} is not finite")
            if self.observed_counts[index] == 0:
                raise PreprocessingError(
                    f"feature {name!r} was never observed during fitting; "
                    "refusing to fabricate a normalization"
                )
            values.append((raw - self.mean[index]) / self.std[index])
            mask.append(True)
        return ScaledWindow(
            feature_names=self.feature_names,
            values=tuple(values),
            mask=tuple(mask),
        )

    def payload(self) -> dict[str, Any]:
        return {
            "schema": PHYSIO_WINDOW_SCALER_SCHEMA,
            "feature_version": self.feature_version,
            "feature_names": list(self.feature_names),
            "mean": list(self.mean),
            "std": list(self.std),
            "observed_counts": list(self.observed_counts),
            "min_std": self.min_std,
        }

    def digest(self) -> str:
        """Stable sha256 of the scaler payload, independent of JSON formatting."""
        return _canonical_digest(self.payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.payload()
        return {**payload, "sha256": self.digest()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WindowScaler:
        if not isinstance(data, dict):
            raise PreprocessingError("scaler artifact must be a JSON object")
        if data.get("schema") != PHYSIO_WINDOW_SCALER_SCHEMA:
            raise PreprocessingError(
                f"unsupported scaler schema: {data.get('schema')!r}"
            )
        digest = data.get("sha256")
        payload = {key: value for key, value in data.items() if key != "sha256"}
        if not isinstance(digest, str) or digest != _canonical_digest(payload):
            raise PreprocessingError("scaler artifact sha256 mismatch (corrupt or edited)")
        if payload.get("feature_version") != WINDOW_FEATURE_VERSION:
            raise PreprocessingError(
                f"unsupported scaler feature_version: {payload.get('feature_version')!r}"
            )
        try:
            names = tuple(str(name) for name in payload["feature_names"])
            mean = tuple(float(value) for value in payload["mean"])
            std = tuple(float(value) for value in payload["std"])
            raw_counts = payload["observed_counts"]
            if any(
                isinstance(count, bool) or not isinstance(count, int)
                for count in raw_counts
            ):
                raise TypeError("observed_counts must be integers")
            counts = tuple(raw_counts)
            min_std = float(payload["min_std"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PreprocessingError(f"malformed scaler artifact: {exc}") from exc
        if names != WINDOW_FEATURE_NAMES:
            raise PreprocessingError(
                "scaler feature_names do not match the current window feature contract"
            )
        try:
            return cls(
                feature_names=names,
                mean=mean,
                std=std,
                observed_counts=counts,
                min_std=min_std,
            )
        except ValueError as exc:
            raise PreprocessingError(f"invalid scaler artifact: {exc}") from exc

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        try:
            destination.write_text(
                json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            raise PreprocessingError(
                f"cannot write scaler artifact {destination}: {exc}"
            ) from exc

    @classmethod
    def load(cls, path: str | Path) -> WindowScaler:
        source = Path(path)
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except OSError as exc:
            raise PreprocessingError(f"cannot read scaler artifact {source}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise PreprocessingError(f"scaler artifact is not valid JSON: {exc}") from exc
        return cls.from_dict(data)


__all__ = [
    "DEFAULT_MIN_STD",
    "PHYSIO_WINDOW_SCALER_SCHEMA",
    "PreprocessingError",
    "ScaledWindow",
    "WindowScaler",
]
