"""Field registry, alias routing, unit normalization and per-field QC (#96).

Unknown fields and units are rejected, never guessed. Zero semantics are
explicit per metric: a zero respiration rate or blood pressure is missing /
invalid engineering input, not a normal reading. Ranges are plausibility
guardrails, not clinical thresholds.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .schema import (
    DATA_TYPE_FLOAT,
    DATA_TYPE_INT,
    DATA_TYPE_STRING,
    MEASUREMENT_ACCEPTED,
    MeasurementV1,
)

STATUS_MISSING = "missing"
STATUS_UNKNOWN_METRIC = "unknown_metric"
STATUS_UNKNOWN_UNIT = "unknown_unit"
STATUS_INVALID_TYPE = "invalid_type"
STATUS_INVALID_VALUE = "invalid_value"
STATUS_OUT_OF_RANGE = "out_of_range"
STATUS_ZERO_UNKNOWN = "zero_unknown"

ZERO_INVALID = "invalid"
ZERO_MISSING = "missing"
ZERO_VALID = "valid"

_UNIT_ALIASES = {
    "%": "%",
    "percent": "%",
    "percentage": "%",
    "fraction": "fraction",
    "ratio": "fraction",
    "0-1": "fraction",
    "0~1": "fraction",
    "bpm": "bpm",
    "1/min": "1/min",
    "beats/min": "bpm",
    "breaths/min": "breaths/min",
    "rpm": "breaths/min",
    "ms": "ms",
    "c": "°c",
    "°c": "°c",
    "celsius": "°c",
    "f": "°f",
    "°f": "°f",
    "fahrenheit": "°f",
    "mmhg": "mmhg",
    "kpa": "kpa",
    "kcal/day": "kcal/day",
    "kcal": "kcal/day",
    "kj/day": "kj/day",
    "cm": "cm",
    "m": "m",
    "kg": "kg",
    "lb": "lb",
    "lbs": "lb",
    "years": "years",
    "year": "years",
    "y": "years",
    "score": "score",
    "index": "index",
}


def normalize_unit(unit: str | None) -> str | None:
    if unit is None:
        return None
    text = str(unit).strip().lower().replace(" ", "")
    if not text:
        return None
    return _UNIT_ALIASES.get(text, text)


@dataclass(frozen=True)
class MetricSpec:
    """One canonical metric and its engineering contract."""

    name: str
    canonical_unit: str
    data_type: str
    plausible_range: tuple[float, float] | None
    zero_policy: str
    aliases: tuple[str, ...] = ()
    allowed_values: tuple[str, ...] = ()
    semantic: str = ""

    def __post_init__(self) -> None:
        if self.zero_policy not in (ZERO_INVALID, ZERO_MISSING, ZERO_VALID):
            raise ValueError(f"unknown zero policy: {self.zero_policy!r}")
        if self.data_type not in (DATA_TYPE_FLOAT, DATA_TYPE_INT, DATA_TYPE_STRING):
            raise ValueError(f"unsupported metric data type: {self.data_type!r}")
        if self.data_type == DATA_TYPE_STRING and self.plausible_range is not None:
            raise ValueError("string metrics cannot declare a numeric range")


def _spec(
    name: str,
    unit: str,
    data_type: str,
    low: float | None,
    high: float | None,
    zero_policy: str,
    aliases: tuple[str, ...],
    semantic: str = "",
    allowed_values: tuple[str, ...] = (),
) -> MetricSpec:
    rng = None if low is None or high is None else (float(low), float(high))
    return MetricSpec(name, unit, data_type, rng, zero_policy, aliases,
                      allowed_values, semantic)


METRIC_SPECS: dict[str, MetricSpec] = {
    "heart_rate": _spec(
        "heart_rate", "bpm", DATA_TYPE_FLOAT, 20, 250, ZERO_INVALID,
        ("hr", "heartrate", "heart_rate", "pulse", "心率"),
        "instantaneous heart rate; engineering guardrail only",
    ),
    "spo2": _spec(
        "spo2", "%", DATA_TYPE_FLOAT, 50, 100, ZERO_INVALID,
        ("spO2", "spo2", "oxygen_saturation", "blood_oxygen", "血氧", "血氧饱和度"),
        "peripheral oxygen saturation; canonical percent, fraction input is scaled",
    ),
    "skin_temperature": _spec(
        "skin_temperature", "°C", DATA_TYPE_FLOAT, 25, 45, ZERO_INVALID,
        ("skin_temp", "skin_temperature", "temperature", "temp", "皮温", "皮肤温度"),
        "skin surface temperature; not core temperature",
    ),
    "body_temperature": _spec(
        "body_temperature", "°C", DATA_TYPE_FLOAT, 30, 45, ZERO_INVALID,
        ("body_temp", "body_temperature", "core_temperature", "体温"),
        "device-reported body temperature",
    ),
    "respiration_rate": _spec(
        "respiration_rate", "breaths/min", DATA_TYPE_FLOAT, 4, 60, ZERO_MISSING,
        ("rr", "resp_rate", "respiration", "respiratory_rate", "呼吸率", "呼吸频率"),
        "respiration rate; zero means no usable signal",
    ),
    "systolic_bp": _spec(
        "systolic_bp", "mmHg", DATA_TYPE_FLOAT, 50, 260, ZERO_MISSING,
        ("sbp", "sys", "systolic", "systolic_bp", "收缩压"),
        "systolic blood pressure; zero means no reading",
    ),
    "diastolic_bp": _spec(
        "diastolic_bp", "mmHg", DATA_TYPE_FLOAT, 30, 200, ZERO_MISSING,
        ("dbp", "dia", "diastolic", "diastolic_bp", "舒张压"),
        "diastolic blood pressure; zero means no reading",
    ),
    "hrv_rmssd": _spec(
        "hrv_rmssd", "ms", DATA_TYPE_FLOAT, 2, 500, ZERO_MISSING,
        ("hrv", "rmssd", "hrv_rmssd", "心率变异性"),
        "HRV RMSSD when the device reports it",
    ),
    "fatigue_index": _spec(
        "fatigue_index", "index", DATA_TYPE_FLOAT, 0, 100, ZERO_VALID,
        ("fatigue", "fatigue_index", "vendor_fatigue", "疲劳指数"),
        "vendor-provided proxy score; NOT a clinical or independent fatigue label",
    ),
    "stress_index": _spec(
        "stress_index", "index", DATA_TYPE_FLOAT, 0, 100, ZERO_VALID,
        ("stress", "stress_index", "压力指数"),
        "vendor-provided proxy score; not independently validated",
    ),
    "microcirculation_index": _spec(
        "microcirculation_index", "index", DATA_TYPE_FLOAT, 1, 100, ZERO_INVALID,
        ("micro_hrv", "microcirculation", "micro", "微循环", "末梢微循环"),
        "vendor microcirculation proxy index; not independently validated",
    ),
    "vendor_health_index": _spec(
        "vendor_health_index", "index", DATA_TYPE_FLOAT, 0, 100, ZERO_VALID,
        ("vendor_score", "vendor_health", "health_score", "健康评分", "健康指数"),
        "vendor-provided composite health score; NOT a clinical or independent label",
    ),
    "bmr": _spec(
        "bmr", "kcal/day", DATA_TYPE_FLOAT, 500, 5000, ZERO_MISSING,
        ("bmr", "basal_metabolic_rate", "基础代谢", "基础代谢率"),
        "basal metabolic rate baseline; zero means no reading",
    ),
    "body_fat_percentage": _spec(
        "body_fat_percentage", "%", DATA_TYPE_FLOAT, 2, 70, ZERO_INVALID,
        ("body_fat", "body_fat_pct", "body_fat_percentage", "体脂率"),
        "body fat percentage baseline",
    ),
    "bmi": _spec(
        "bmi", "kg/m2", DATA_TYPE_FLOAT, 10, 70, ZERO_INVALID,
        ("bmi", "BMI", "身体质量指数"),
        "body mass index baseline",
    ),
    "height_cm": _spec(
        "height_cm", "cm", DATA_TYPE_FLOAT, 50, 250, ZERO_INVALID,
        ("height", "height_cm", "stature", "身高"),
        "height baseline",
    ),
    "weight_kg": _spec(
        "weight_kg", "kg", DATA_TYPE_FLOAT, 2, 300, ZERO_INVALID,
        ("weight", "weight_kg", "body_weight", "体重"),
        "weight baseline",
    ),
    "age": _spec(
        "age", "years", DATA_TYPE_INT, 1, 120, ZERO_INVALID,
        ("age", "年龄"),
        "age baseline",
    ),
    "sex": _spec(
        "sex", "", DATA_TYPE_STRING, None, None, ZERO_VALID,
        ("sex", "gender", "性别"),
        "self-reported or device-provided sex category",
        ("female", "male", "other", "unknown"),
    ),
}

ALIAS_TO_METRIC: dict[str, str] = {}
for _canonical, _metric in METRIC_SPECS.items():
    for _alias in (_canonical, *_metric.aliases):
        key = str(_alias).strip().lower()
        ALIAS_TO_METRIC.setdefault(key, _canonical)

#: Multiplicative/additive conversion into the canonical unit.
#: factor * value + offset.
_UNIT_CONVERSIONS: dict[str, dict[str, tuple[float, float]]] = {
    "spo2": {"%": (1.0, 0.0), "fraction": (100.0, 0.0)},
    "skin_temperature": {"°c": (1.0, 0.0), "°f": (5.0 / 9.0, -32.0 * 5.0 / 9.0)},
    "body_temperature": {"°c": (1.0, 0.0), "°f": (5.0 / 9.0, -32.0 * 5.0 / 9.0)},
    "heart_rate": {"bpm": (1.0, 0.0), "1/min": (1.0, 0.0)},
    "respiration_rate": {"breaths/min": (1.0, 0.0), "1/min": (1.0, 0.0)},
    "systolic_bp": {"mmhg": (1.0, 0.0), "kpa": (7.50062, 0.0)},
    "diastolic_bp": {"mmhg": (1.0, 0.0), "kpa": (7.50062, 0.0)},
    "hrv_rmssd": {"ms": (1.0, 0.0)},
    "fatigue_index": {"index": (1.0, 0.0), "score": (1.0, 0.0)},
    "stress_index": {"index": (1.0, 0.0), "score": (1.0, 0.0)},
    "microcirculation_index": {"index": (1.0, 0.0)},
    "vendor_health_index": {"index": (1.0, 0.0), "score": (1.0, 0.0)},
    "bmr": {"kcal/day": (1.0, 0.0), "kj/day": (0.239006, 0.0)},
    "body_fat_percentage": {"%": (1.0, 0.0)},
    "bmi": {"kg/m2": (1.0, 0.0)},
    "height_cm": {"cm": (1.0, 0.0), "m": (100.0, 0.0)},
    "weight_kg": {"kg": (1.0, 0.0), "lb": (0.45359237, 0.0)},
    "age": {"years": (1.0, 0.0)},
    "sex": {"": (1.0, 0.0)},
}


def canonical_metric(raw_metric: str | None) -> str | None:
    if raw_metric is None:
        return None
    return ALIAS_TO_METRIC.get(str(raw_metric).strip().lower())


def _rejected(
    raw_metric: str,
    raw_value: Any,
    raw_unit: str | None,
    data_type: str,
    status: str,
    note: str,
    *,
    metric: str | None = None,
    unit: str = "",
) -> MeasurementV1:
    return MeasurementV1(
        metric=metric or str(raw_metric),
        value=None,
        unit=unit or (str(raw_unit) if raw_unit is not None else ""),
        data_type=data_type,
        valid=False,
        status=status,
        raw_metric=str(raw_metric),
        raw_value=raw_value,
        raw_unit=raw_unit,
        notes=(note,),
    )


def _coerce_number(raw_value: Any) -> float | None:
    if isinstance(raw_value, bool) or raw_value is None:
        return None
    if isinstance(raw_value, (int, float)):
        return float(raw_value)
    if isinstance(raw_value, str):
        text = raw_value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _infer_data_type(raw_value: Any, fallback: str | None) -> str:
    if fallback in (DATA_TYPE_FLOAT, DATA_TYPE_INT, DATA_TYPE_STRING):
        return fallback
    if isinstance(raw_value, bool):
        return DATA_TYPE_INT
    if isinstance(raw_value, int):
        return DATA_TYPE_INT
    if isinstance(raw_value, float):
        return DATA_TYPE_FLOAT
    if isinstance(raw_value, str):
        return DATA_TYPE_STRING
    return DATA_TYPE_FLOAT


def route_measurement(
    raw_metric: str,
    raw_value: Any,
    raw_unit: str | None = None,
    data_type: str | None = None,
) -> MeasurementV1:
    """Route one raw field to the canonical contract without guessing."""
    metric = canonical_metric(raw_metric)
    dtype = _infer_data_type(raw_value, data_type)
    if metric is None:
        return _rejected(
            raw_metric, raw_value, raw_unit, dtype,
            STATUS_UNKNOWN_METRIC,
            f"unknown field {raw_metric!r}; preserved as unavailable, not mapped",
        )
    spec = METRIC_SPECS[metric]

    if raw_value is None or (isinstance(raw_value, str) and not raw_value.strip()):
        return _rejected(
            raw_metric, raw_value, raw_unit, spec.data_type, STATUS_MISSING,
            "field is absent/blank", metric=metric, unit=spec.canonical_unit,
        )

    if spec.data_type == DATA_TYPE_STRING:
        text = str(raw_value).strip().lower()
        if spec.allowed_values and text not in spec.allowed_values:
            return _rejected(
                raw_metric, raw_value, raw_unit, spec.data_type,
                STATUS_INVALID_VALUE,
                f"value {raw_value!r} is not one of {spec.allowed_values}",
                metric=metric,
            )
        return MeasurementV1(
            metric=metric, value=text, unit=spec.canonical_unit,
            data_type=spec.data_type, valid=True, status=MEASUREMENT_ACCEPTED,
            raw_metric=str(raw_metric), raw_value=raw_value, raw_unit=raw_unit,
        )

    number = _coerce_number(raw_value)
    if number is None:
        return _rejected(
            raw_metric, raw_value, raw_unit, spec.data_type, STATUS_INVALID_TYPE,
            f"value {raw_value!r} is not a finite number", metric=metric,
            unit=spec.canonical_unit,
        )
    if not math.isfinite(number):
        return _rejected(
            raw_metric, raw_value, raw_unit, spec.data_type, STATUS_INVALID_VALUE,
            "value is NaN/Inf", metric=metric, unit=spec.canonical_unit,
        )

    unit = normalize_unit(raw_unit)
    if unit is None and spec.canonical_unit == "":
        unit = ""
    conversions = _UNIT_CONVERSIONS[metric]
    if unit not in conversions:
        return _rejected(
            raw_metric, raw_value, raw_unit, spec.data_type, STATUS_UNKNOWN_UNIT,
            f"unknown unit {raw_unit!r} for {metric}; expected {spec.canonical_unit!r}",
            metric=metric, unit=spec.canonical_unit,
        )
    factor, offset = conversions[unit]
    canonical_value = number * factor + offset

    if canonical_value == 0 and spec.zero_policy != ZERO_VALID:
        status = STATUS_ZERO_UNKNOWN if spec.zero_policy == ZERO_MISSING else STATUS_INVALID_VALUE
        return _rejected(
            raw_metric, raw_value, raw_unit, spec.data_type, status,
            "zero has no usable engineering meaning for this field",
            metric=metric, unit=spec.canonical_unit,
        )
    if spec.plausible_range is not None:
        low, high = spec.plausible_range
        if not low <= canonical_value <= high:
            return _rejected(
                raw_metric, raw_value, raw_unit, spec.data_type, STATUS_OUT_OF_RANGE,
                f"value {canonical_value:g} outside engineering guardrail [{low:g}, {high:g}]",
                metric=metric, unit=spec.canonical_unit,
            )
    if spec.data_type == DATA_TYPE_INT:
        rounded = int(round(canonical_value))
        if abs(canonical_value - rounded) > 0.5:
            return _rejected(
                raw_metric, raw_value, raw_unit, spec.data_type, STATUS_INVALID_VALUE,
                "non-integral value for an integer field", metric=metric,
                unit=spec.canonical_unit,
            )
        canonical_value = rounded
    return MeasurementV1(
        metric=metric, value=canonical_value, unit=spec.canonical_unit,
        data_type=spec.data_type, valid=True, status=MEASUREMENT_ACCEPTED,
        raw_metric=str(raw_metric), raw_value=raw_value, raw_unit=raw_unit,
    )


class FieldRouter:
    """Route a raw payload (device/service/offline) into measurements."""

    def __init__(self, field_aliases: Mapping[str, str] | None = None) -> None:
        self.field_aliases = {
            str(k).strip().lower(): str(v) for k, v in (field_aliases or {}).items()
        }

    def resolve_field(self, raw_metric: str) -> str:
        key = str(raw_metric).strip().lower()
        return self.field_aliases.get(key, raw_metric)

    def route(
        self,
        raw_metric: str,
        raw_value: Any,
        raw_unit: str | None = None,
        data_type: str | None = None,
    ) -> MeasurementV1:
        return route_measurement(
            self.resolve_field(raw_metric), raw_value, raw_unit, data_type
        )

    def route_payload(
        self, payload: Mapping[str, Any], *, units: Mapping[str, str] | None = None
    ) -> tuple[MeasurementV1, ...]:
        """Route a flat ``{field: value}`` payload.

        ``units`` may supply per-field units. A value that is itself a mapping
        ``{"value": ..., "unit": ...}`` is accepted too; this is the shape
        vendor services commonly use.
        """
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        units = units or {}
        routed: list[MeasurementV1] = []
        for raw_metric, raw in payload.items():
            unit = units.get(raw_metric)
            data_type = None
            value = raw
            if isinstance(raw, Mapping):
                value = raw.get("value")
                unit = raw.get("unit", unit)
                data_type = raw.get("data_type")
            routed.append(self.route(str(raw_metric), value, unit, data_type))
        return tuple(routed)


DEFAULT_ROUTER = FieldRouter()


def metric_names() -> tuple[str, ...]:
    return tuple(METRIC_SPECS)


def is_known_metric(metric: str) -> bool:
    return canonical_metric(metric) is not None


__all__ = [
    "ALIAS_TO_METRIC",
    "DEFAULT_ROUTER",
    "METRIC_SPECS",
    "STATUS_INVALID_TYPE",
    "STATUS_INVALID_VALUE",
    "STATUS_MISSING",
    "STATUS_OUT_OF_RANGE",
    "STATUS_UNKNOWN_METRIC",
    "STATUS_UNKNOWN_UNIT",
    "STATUS_ZERO_UNKNOWN",
    "ZERO_INVALID",
    "ZERO_MISSING",
    "ZERO_VALID",
    "FieldRouter",
    "MetricSpec",
    "canonical_metric",
    "is_known_metric",
    "metric_names",
    "normalize_unit",
    "route_measurement",
]
