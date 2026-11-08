"""#96 field routing: aliases, units, zero semantics and invalid values."""

from __future__ import annotations

import pytest

from physio.metrics import (
    STATUS_INVALID_TYPE,
    STATUS_INVALID_VALUE,
    STATUS_MISSING,
    STATUS_OUT_OF_RANGE,
    STATUS_UNKNOWN_METRIC,
    STATUS_UNKNOWN_UNIT,
    STATUS_ZERO_UNKNOWN,
    FieldRouter,
    route_measurement,
)

pytestmark = pytest.mark.synthetic


def test_alias_and_canonical_metric():
    routed = route_measurement("HR", "72", "bpm")
    assert routed.valid
    assert routed.metric == "heart_rate"
    assert routed.value == 72.0
    assert routed.raw_metric == "HR"


def test_spo2_fraction_is_scaled_to_percent():
    fraction = route_measurement("spO2", 0.97, "fraction")
    percent = route_measurement("spo2", 97, "%")
    assert fraction.valid and percent.valid
    assert fraction.value == pytest.approx(97.0)
    assert fraction.unit == percent.unit == "%"


def test_spo2_zero_is_invalid_not_normal():
    routed = route_measurement("spo2", 0, "%")
    assert not routed.valid
    assert routed.status == STATUS_INVALID_VALUE


def test_zero_semantics_for_rr_bp_and_bmr():
    for metric, unit, status in (
        ("rr", "breaths/min", STATUS_ZERO_UNKNOWN),
        ("sbp", "mmHg", STATUS_ZERO_UNKNOWN),
        ("dbp", "mmHg", STATUS_ZERO_UNKNOWN),
        ("bmr", "kcal/day", STATUS_ZERO_UNKNOWN),
        ("heart_rate", "bpm", STATUS_INVALID_VALUE),
    ):
        routed = route_measurement(metric, 0, unit)
        assert not routed.valid
        assert routed.status == status


def test_unknown_field_is_preserved_but_unavailable():
    routed = route_measurement("vendor_magic", 42, "score")
    assert not routed.valid
    assert routed.status == STATUS_UNKNOWN_METRIC
    assert routed.metric == "vendor_magic"
    assert routed.raw_value == 42
    assert routed.value is None


def test_unknown_unit_is_rejected_not_guessed():
    routed = route_measurement("heart_rate", 72, "furlongs")
    assert not routed.valid
    assert routed.status == STATUS_UNKNOWN_UNIT


def test_missing_unit_is_rejected_for_unit_metrics():
    routed = route_measurement("heart_rate", 72, None)
    assert not routed.valid
    assert routed.status == STATUS_UNKNOWN_UNIT


def test_non_numeric_and_nan_rejected():
    assert route_measurement("heart_rate", "abc", "bpm").status == STATUS_INVALID_TYPE
    assert route_measurement("heart_rate", True, "bpm").status == STATUS_INVALID_TYPE
    assert route_measurement("heart_rate", float("nan"), "bpm").status == STATUS_INVALID_VALUE
    assert route_measurement("heart_rate", float("inf"), "bpm").status == STATUS_INVALID_VALUE


def test_out_of_guardrail_rejected():
    routed = route_measurement("heart_rate", 400, "bpm")
    assert not routed.valid
    assert routed.status == STATUS_OUT_OF_RANGE


def test_missing_value_is_explicit():
    for raw in (None, "", "   "):
        routed = route_measurement("heart_rate", raw, "bpm")
        assert not routed.valid
        assert routed.status == STATUS_MISSING


def test_temperature_unit_conversion():
    fahrenheit = route_measurement("skin_temp", 92.3, "F")
    celsius = route_measurement("skin_temp", 33.5, "°C")
    assert fahrenheit.valid and celsius.valid
    assert fahrenheit.value == pytest.approx(celsius.value, abs=0.03)


def test_pressure_mmhg_passthrough_and_kpa_conversion():
    mmhg = route_measurement("sbp", 120, "mmHg")
    kpa = route_measurement("sbp", 16, "kPa")
    assert mmhg.valid and kpa.valid
    assert kpa.value == pytest.approx(120.01, abs=0.02)


def test_integer_age_and_sex_category():
    age = route_measurement("age", "35", "years")
    assert age.valid and age.data_type == "int" and age.value == 35
    sex = route_measurement("gender", "FEMALE", None)
    assert sex.valid and sex.value == "female"
    bad = route_measurement("gender", "robot", None)
    assert not bad.valid and bad.status == STATUS_INVALID_VALUE


def test_router_payload_supports_value_unit_objects_and_aliases():
    router = FieldRouter(field_aliases={"氧饱和度": "spo2"})
    routed = router.route_payload(
        {
            "心率": {"value": 71, "unit": "bpm"},
            "氧饱和度": {"value": 0.98, "unit": "fraction"},
            "mystery": 3,
        },
    )
    by_metric = {item.metric: item for item in routed}
    assert by_metric["heart_rate"].valid
    assert by_metric["spo2"].value == pytest.approx(98.0)
    assert by_metric["mystery"].status == STATUS_UNKNOWN_METRIC


def test_partial_payload_does_not_drop_the_whole_event_semantics():
    router = FieldRouter()
    measurements = router.route_payload(
        {"hr": 70, "spo2": 0, "rr": 16},
        units={"hr": "bpm", "spo2": "%", "rr": "breaths/min"},
    )
    valid = [m for m in measurements if m.valid]
    assert len(valid) == 2
    assert {m.metric for m in valid} == {"heart_rate", "respiration_rate"}
