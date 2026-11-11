"""P0-03: target provenance, label mask and research gate regressions."""
from __future__ import annotations

from pathlib import Path

import pytest

from physio.admission_report import (
    ENGINEERING_ONLY,
    GATE_NOT_GRANTED,
    TIER_UNAVAILABLE,
    TIER_VENDOR_PROXY,
    inspect_session_admission,
)
from physio.real_export import build_real_sessions, load_real_records

pytestmark = pytest.mark.synthetic
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "physio-legacy-sessions-synthetic.csv"


def _session(session_id: str):
    records = [record for record in load_real_records(FIXTURE)
               if record.session_id == session_id]
    return build_real_sessions(records)[0]


def test_vendor_delta_can_be_measured_but_is_never_an_independent_fatigue_reference():
    report = inspect_session_admission(_session("SYNTH-A"))
    proxy = report.labels["y_delta_vendor_fatigue"]
    assert proxy["available_for_engineering_smoke"] is True
    assert proxy["tier"] == TIER_VENDOR_PROXY
    assert proxy["independent_fatigue_reference"] is False
    assert report.real_training_gate == GATE_NOT_GRANTED
    assert report.research_claims_gate == GATE_NOT_GRANTED
    assert report.engineering_status == ENGINEERING_ONLY
    assert "independent_fatigue_reference_not_collected" in report.blockers


def test_missing_body_features_are_not_misreported_as_real_observations():
    report = inspect_session_admission(_session("SYNTH-B"))
    assert "weight_kg" in report.input_missing
    assert "bmi" in report.input_missing
    assert "weight_kg" not in report.input_observed
    assert report.collection_status == "ENDPOINTS_PRESENT"


def test_invalid_delta_has_a_null_value_and_reason_in_manifest():
    report = inspect_session_admission(_session("SYNTH-B"))
    target = report.labels["y_delta_sbp"]
    assert target["tier"] == TIER_UNAVAILABLE
    assert target["available_for_engineering_smoke"] is False
    assert target["value"] is None
    assert "bp_delta_anomaly" in target["reason"]


def test_report_serializes_without_relabeling_proxy_as_clinical_truth():
    report = inspect_session_admission(_session("SYNTH-C"))
    payload = report.to_dict()
    assert payload["schema_version"] == "physio-session-admission-v1"
    assert payload["labels"]["y_delta_vendor_fatigue"]["independent_fatigue_reference"] is False
    assert payload["real_training_gate"] == "NOT_GRANTED"
