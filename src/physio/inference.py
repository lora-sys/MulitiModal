"""Causal realtime inference session and output gate (#97).

The session estimates the current fatigue-related state at ``t`` from data that
had already happened by ``t``. It never predicts end-of-massage efficacy and it
never renders a score without a validated model artifact: absent, unvalidated
or mismatched artifacts produce ``NOT_AVAILABLE``; empty, stale or too-thin
windows produce ``ABSTAIN``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .artifacts import ArtifactError, ModelArtifact
from .metrics import METRIC_SPECS
from .model import (
    REALTIME_TARGET_VERSION,
    VALIDATION_VALIDATED,
    WindowModelBatch,
)
from .preprocessing import PreprocessingError, ScaledWindow, WindowScaler
from .schema import (
    PHASE_DURING,
    PHASE_PRE,
    QUALITY_DEGRADED,
    QUALITY_OK,
    PhysioSessionV1,
)
from .windows import (
    WINDOW_FEATURE_VERSION,
    PhysioWindowBuilder,
    WindowConfig,
    WindowFeatures,
    event_has_valid_numeric,
)

STATUS_OK = "OK"
STATUS_ABSTAIN = "ABSTAIN"
STATUS_NOT_AVAILABLE = "NOT_AVAILABLE"
STATUSES = (STATUS_OK, STATUS_ABSTAIN, STATUS_NOT_AVAILABLE)

_MEASURED_CLOCK = "measured"
_USABLE_QUALITIES = (QUALITY_OK, QUALITY_DEGRADED)


@dataclass(frozen=True)
class InferenceResult:
    """One gated inference output. ``prediction`` is only legal for OK."""

    status: str
    reason: str
    target_version: str
    feature_version: str
    model_version: str
    input_cutoff: datetime
    quality: str
    coverage: float
    measured: tuple[dict[str, Any], ...]
    prediction: float | None
    window: dict[str, Any] = field(default_factory=dict)
    artifact_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"unknown inference status: {self.status!r}")
        if self.input_cutoff.tzinfo is None:
            raise ValueError("input_cutoff must be timezone-aware")
        if not 0.0 <= self.coverage <= 1.0:
            raise ValueError("coverage must be within 0..1")
        if self.prediction is not None:
            if self.status != STATUS_OK:
                raise ValueError("a prediction may only appear when status is OK")
            if not math.isfinite(self.prediction):
                raise ValueError("prediction must be finite")
        if self.status == STATUS_OK and self.prediction is None:
            raise ValueError("status OK requires a prediction")
        if self.artifact_sha256 is not None:
            if (
                len(self.artifact_sha256) != 64
                or set(self.artifact_sha256) - set("0123456789abcdef")
            ):
                raise ValueError("artifact_sha256 must be a lowercase hex sha256 digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "target_version": self.target_version,
            "feature_version": self.feature_version,
            "model_version": self.model_version,
            "input_cutoff": self.input_cutoff.isoformat(),
            "quality": self.quality,
            "coverage": self.coverage,
            "measured": [dict(item) for item in self.measured],
            "prediction": self.prediction,
            "window": dict(self.window),
            "artifact_sha256": self.artifact_sha256,
        }


def _first_scalar(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "reshape"):
        value = value.reshape(-1)[0]
    while isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("model returned an empty prediction")
        value = value[0]
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("model returned a non-finite prediction")
    return number


class InferenceSession:
    """Stateful, cached, cutoff-driven inference over a ``PhysioSessionV1``."""

    def __init__(
        self,
        session: PhysioSessionV1,
        builder: PhysioWindowBuilder,
        *,
        model: Any = None,
        scaler: WindowScaler | None = None,
        artifact: ModelArtifact | None = None,
        update_period_s: float = 30.0,
        stale_after_s: float = 120.0,
        clock: str = "measured",
    ) -> None:
        if not isinstance(session, PhysioSessionV1):
            raise TypeError("session must be a PhysioSessionV1")
        if not isinstance(builder, PhysioWindowBuilder):
            raise TypeError("builder must be a PhysioWindowBuilder")
        if artifact is not None and not isinstance(artifact, ModelArtifact):
            raise TypeError("artifact must be a ModelArtifact")
        if clock != _MEASURED_CLOCK:
            raise ValueError(
                "InferenceSession supports the measured clock only; received-arrival "
                "ordering cannot stand in for the scientific cutoff"
            )
        if not math.isfinite(update_period_s) or update_period_s <= 0:
            raise ValueError("update_period_s must be a positive finite number")
        if not math.isfinite(stale_after_s) or stale_after_s < 0:
            raise ValueError("stale_after_s must be a non-negative finite number")
        self.session = session
        self.builder = builder
        self.model = model
        self.scaler = scaler
        self.artifact = artifact
        self.model_version = artifact.model_version if artifact is not None else "NOT_AVAILABLE"
        self.update_period_s = update_period_s
        self.stale_after_s = stale_after_s
        self.clock = clock
        self._cache: dict[datetime, InferenceResult] = {}
        self._last_update_at: datetime | None = None

    def reset(self) -> None:
        """Drop cached results, for example after late data changes the session."""
        self._cache.clear()
        self._last_update_at = None

    def due(self, now: datetime) -> bool:
        """Whether a configured periodic update is due at ``now``."""
        self._check_now(now)
        if self._last_update_at is None:
            return True
        return (now - self._last_update_at).total_seconds() >= self.update_period_s

    def update(self, now: datetime) -> InferenceResult:
        """Estimate the state at ``now`` from events measured at or before it."""
        self._check_now(now)
        cached = self._cache.get(now)
        if cached is not None:
            return cached

        window = self.builder.build(self.session, now)
        measured = self._measured(window)
        latest = self._latest_valid_event(now)
        if latest is None:
            result = self._result(
                STATUS_ABSTAIN, "no_valid_measurements", window, measured, None, now
            )
        elif (now - latest.measured_at).total_seconds() > self.stale_after_s:
            result = self._result(
                STATUS_ABSTAIN,
                f"stale_measurements: latest valid event is older than {self.stale_after_s:g}s",
                window,
                measured,
                None,
                now,
            )
        elif not window.usable:
            result = self._result(
                STATUS_ABSTAIN,
                window.reason or "window_not_usable",
                window,
                measured,
                None,
                now,
            )
        elif self.model is None:
            result = self._result(
                STATUS_NOT_AVAILABLE, "model_not_available", window, measured, None, now
            )
        elif self.scaler is None:
            result = self._result(
                STATUS_NOT_AVAILABLE, "scaler_not_available", window, measured, None, now
            )
        elif self.artifact is None:
            result = self._result(
                STATUS_NOT_AVAILABLE, "artifact_not_bound", window, measured, None, now
            )
        elif self.artifact.validation_status != VALIDATION_VALIDATED:
            result = self._result(
                STATUS_NOT_AVAILABLE,
                "model_not_validated",
                window,
                measured,
                None,
                now,
            )
        else:
            try:
                self.artifact.verify(self.scaler, self.model)
            except ArtifactError as exc:
                result = self._result(
                    STATUS_NOT_AVAILABLE,
                    f"artifact_error: {exc}",
                    window,
                    measured,
                    None,
                    now,
                )
            else:
                result = self._validated_result(window, measured, now)

        self._cache[now] = result
        self._last_update_at = now
        return result

    # -- prediction -----------------------------------------------------

    def _validated_result(
        self,
        window: WindowFeatures,
        measured: tuple[dict[str, Any], ...],
        now: datetime,
    ) -> InferenceResult:
        assert self.scaler is not None
        try:
            scaled = self.scaler.transform(window)
            if not isinstance(scaled, ScaledWindow):
                raise PreprocessingError("scaler.transform returned an unexpected type")
            self._check_model_dim(scaled)
            prediction = self._predict(scaled)
        except PreprocessingError as exc:
            return self._result(
                STATUS_NOT_AVAILABLE,
                f"artifact_error: {exc}",
                window,
                measured,
                None,
                now,
            )
        except Exception as exc:  # noqa: BLE001 - never fabricate a score on failure
            return self._result(
                STATUS_NOT_AVAILABLE,
                f"prediction_error: {type(exc).__name__}",
                window,
                measured,
                None,
                now,
            )
        return self._result(STATUS_OK, "ok", window, measured, prediction, now)

    def _check_model_dim(self, scaled: ScaledWindow) -> None:
        spec = getattr(self.model, "input_spec", None)
        if spec is None:
            return
        try:
            dim = spec.get("window_features").dim
        except (AttributeError, KeyError) as exc:
            raise PreprocessingError(
                "model input spec does not declare the window_features modality"
            ) from exc
        if dim != len(scaled.feature_names):
            raise PreprocessingError(
                f"model expects {dim} window features, scaler provides "
                f"{len(scaled.feature_names)}"
            )

    def _predict(self, scaled: ScaledWindow) -> float:
        batch = WindowModelBatch(
            feature_names=scaled.feature_names,
            values=scaled.values,
            mask=scaled.mask,
        )
        try:
            import torch  # noqa: F401
        except ImportError:
            pass
        else:
            input_spec = getattr(self.model, "input_spec", None)
            batch = batch.to_model_batch(input_spec)
        output = self.model(batch, feature_mask=scaled.mask)
        predictions = getattr(output, "predictions", None)
        if not isinstance(predictions, dict) or "fatigue_state" not in predictions:
            raise ValueError("model output is missing the fatigue_state prediction")
        return _first_scalar(predictions["fatigue_state"])

    # -- helpers --------------------------------------------------------

    @staticmethod
    def _check_now(now: datetime) -> None:
        if not isinstance(now, datetime):
            raise TypeError("now must be a datetime")
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")

    def _latest_valid_event(self, now: datetime) -> Any:
        for event in reversed(self.session.events):
            if event.phase not in (PHASE_PRE, PHASE_DURING):
                continue
            if event.measured_at > now:
                continue
            if event.quality in _USABLE_QUALITIES and event_has_valid_numeric(event):
                return event
        return None

    @staticmethod
    def _measured(window: WindowFeatures) -> tuple[dict[str, Any], ...]:
        observed = window.feature_map()
        items: list[dict[str, Any]] = []
        for metric in window.metrics_present:
            value = observed.get(f"{metric}.mean")
            if value is None:
                continue
            spec = METRIC_SPECS.get(metric)
            if spec is None:
                continue
            items.append(
                {
                    "metric": metric,
                    "value": float(value),
                    "unit": spec.canonical_unit,
                    "valid": True,
                }
            )
        return tuple(items)

    def _result(
        self,
        status: str,
        reason: str,
        window: WindowFeatures,
        measured: tuple[dict[str, Any], ...],
        prediction: float | None,
        now: datetime,
    ) -> InferenceResult:
        config = getattr(self.builder, "config", WindowConfig())
        metadata = {
            "start": window.start.isoformat() if window.start is not None else None,
            "end": window.end.isoformat() if window.end is not None else None,
            "window_s": config.window_s,
            "min_events": config.min_events,
            "n_events": window.n_events,
            "phase_counts": dict(window.phase_counts),
            "reason": window.reason,
        }
        return InferenceResult(
            status=status,
            reason=reason,
            target_version=REALTIME_TARGET_VERSION,
            feature_version=WINDOW_FEATURE_VERSION,
            model_version=self.model_version,
            input_cutoff=now,
            quality=window.quality,
            coverage=window.coverage,
            measured=measured,
            prediction=prediction,
            window=metadata,
            artifact_sha256=self.artifact.digest() if self.artifact is not None else None,
        )


__all__ = [
    "STATUSES",
    "STATUS_ABSTAIN",
    "STATUS_NOT_AVAILABLE",
    "STATUS_OK",
    "InferenceResult",
    "InferenceSession",
]
