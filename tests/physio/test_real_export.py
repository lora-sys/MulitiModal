"""#102 real 94-field Session export: endpoint merge, labels and QC."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from physio.real_export import (
    QC_BP_DELTA_ANOMALY,
    QC_BRAIN_SELECTION_NOT_RECOVERABLE,
    QC_CODED_FIELD_SEMANTICS_UNVERIFIED,
    QC_DECLARED_UNUSABLE,
    QC_IDENTITY_SHARED,
    QC_INVALID_VALUE,
    QC_MISSING_MODULE,
    QC_POINT_STREAM_EXCLUDED,
    QC_SNAPSHOT_ENDPOINT,
    QC_STATIC_PROFILE_COLLISION,
    REAL_EXPORT_SCHEMA_VERSION,
    RealExportError,
    build_real_sessions,
    contract_coverage,
    import_real_export,
    load_real_records,
    reconcile_reference_contract,
    session_feature_row,
    session_labels,
)

pytestmark = pytest.mark.synthetic

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "physio-legacy-sessions-synthetic.csv"


def _sessions():
    return import_real_export(FIXTURE)


def _session(session_id: str):
    for session in _sessions():
        if session.session_id == session_id:
            return session
    raise AssertionError(f"missing session {session_id}")


def test_import_builds_one_session_per_export_key():
    sessions = _sessions()
    assert len(sessions) == 5
    assert [session.session_id for session in sessions] == [
        "SYNTH-A",
        "SYNTH-B",
        "SYNTH-C",
        "SYNTH-D",
        "SYNTH-E",
    ]
    assert sum(len(session.records) for session in sessions) == 36


def test_endpoint_merge_takes_first_valid_value_per_metric():
    """rr=0 in the snapshot record is unusable, so the completed record supplies it."""
    session = _session("SYNTH-A")
    pre = session.endpoints["health"]["pre"]
    values = {item.metric: item.value for item in pre.measurements}
    assert values["heart_rate"] == 70.0
    assert values["fatigue_index"] == 30.0
    assert values["vendor_health_index"] == 90.0
    assert values["microcirculation_index"] == 8.0
    assert values["respiration_rate"] == 17.0
    assert pre.anchor_record == "9003"
    assert set(pre.record_ids) == {"9003", "9004"}


def test_spo2_is_scaled_to_the_reference_fraction():
    """The endpoint keeps canonical percent; the reference fraction is applied downstream."""
    session = _session("SYNTH-A")
    pre = session.endpoints["health"]["pre"]
    spo2 = pre.measurement("spo2")
    assert spo2 is not None and spo2.value == 97.0
    assert spo2.raw_value == "97"
    labels = session_labels(session)
    assert labels.values["y_delta_spo2"] == 0.0
    row = session_feature_row(session)
    assert row.values[row.feature_names.index("spo2")] == 0.97


def test_labels_are_rebuilt_from_raw_endpoints():
    labels = session_labels(_session("SYNTH-A"))
    assert labels.values["y_delta_sbp"] == -5.0
    assert labels.values["y_delta_dbp"] == -2.0
    assert labels.values["y_delta_hr"] == 1.0
    assert labels.values["y_delta_spo2"] == 0.0
    assert labels.values["y_delta_micro"] == 0.5
    assert labels.values["y_delta_vendor_fatigue"] == -2.0
    assert labels.values["y_delta_vendor_health"] == -2.0


def test_bp_delta_anomaly_excludes_blood_pressure_labels():
    session = _session("SYNTH-B")
    labels = session_labels(session)
    assert labels.mask["y_delta_sbp"] is False
    assert labels.mask["y_delta_dbp"] is False
    assert labels.values["y_delta_sbp"] is None
    assert "bp_delta_anomaly" in labels.reasons["y_delta_sbp"]
    assert labels.excluded_bp_labels == ("y_delta_dbp", "y_delta_sbp")
    assert QC_BP_DELTA_ANOMALY in {issue.code for issue in labels.issues}


def test_respiration_rate_delta_is_rejected_when_both_endpoints_are_zero():
    labels = session_labels(_session("SYNTH-C"))
    assert labels.mask["y_delta_rr"] is False
    assert labels.values["y_delta_rr"] is None
    assert "delta_invalid" in labels.reasons["y_delta_rr"]
    assert QC_INVALID_VALUE in _session("SYNTH-C").issue_codes()


def test_shared_account_identity_is_flagged_and_never_trusted():
    sessions = _sessions()
    members = {session.member_ref for session in sessions}
    assert members == {"member:m1hash", "member:m2hash"}
    for session in sessions:
        assert QC_IDENTITY_SHARED in session.issue_codes()
        assert session.identity_trust == "SHARED_ACCOUNT_UNKNOWN"


def test_identical_static_profiles_are_flagged():
    flagged = {
        session.session_id
        for session in _sessions()
        if QC_STATIC_PROFILE_COLLISION in session.issue_codes()
    }
    assert flagged == {"SYNTH-A", "SYNTH-D"}


def test_point_stream_rows_and_brain_selection_are_reported():
    session = _session("SYNTH-A")
    assert QC_POINT_STREAM_EXCLUDED in session.issue_codes()
    assert QC_BRAIN_SELECTION_NOT_RECOVERABLE in session.issue_codes()
    # Two 脑波 records and no declared selection rule: nothing is emitted.
    assert session.eeg_record_id == ""
    assert session.eeg_values == {}
    assert session.module_present("eeg") is True


def test_single_brain_record_is_emitted_and_auditable():
    assert _session("SYNTH-E").eeg_record_id == "9405"
    assert _session("SYNTH-E").eeg_values["brain_overall"] == 28.0
    assert _session("SYNTH-E").eeg_values["sample_points"] == 400.0
    assert _session("SYNTH-E").static_values == {}
    assert _session("SYNTH-B").eeg_record_id == ""


def test_missing_module_families_are_reported_at_session_level():
    session = _session("SYNTH-B")
    assert QC_MISSING_MODULE in session.issue_codes()
    for module in ("body", "tcm", "eeg"):
        assert session.module_present(module) is False
    full = _session("SYNTH-A")
    assert QC_MISSING_MODULE not in full.issue_codes()
    for module in ("bp", "health", "body", "tcm", "eeg"):
        assert full.module_present(module) is True


def test_columns_with_no_recoverable_meaning_are_reported():
    for session in _sessions():
        assert QC_DECLARED_UNUSABLE in session.issue_codes()
        details = [i.detail for i in session.issues if i.code == QC_DECLARED_UNUSABLE]
        assert any("fatigue_level" in detail for detail in details)


def test_contract_reconciliation_abstains_when_no_rule_is_declared():
    """A derivable label the reference excludes must not be claimed admissible."""
    labels = session_labels(_session("SYNTH-C"))
    assert labels.mask["y_delta_vendor_health"] is True
    assert labels.values["y_delta_vendor_health"] == -1.0
    admitted = (
        "y_delta_sbp",
        "y_delta_dbp",
        "y_delta_hr",
        "y_delta_spo2",
        "y_delta_micro",
        "y_delta_vendor_fatigue",
    )
    reconciled = reconcile_reference_contract(labels, admitted)
    assert reconciled.mask["y_delta_vendor_health"] is False
    assert reconciled.abstained == ("y_delta_vendor_health",)
    assert set(reconciled.available) == set(admitted)
    assert "exclusion_rule_not_recoverable" in reconciled.reasons["y_delta_vendor_health"]
    # The derived value survives for audit; only admissibility changes.
    assert reconciled.values["y_delta_vendor_health"] == -1.0


def test_contract_reconciliation_keeps_labels_the_reference_admits():
    labels = session_labels(_session("SYNTH-A"))
    admitted = ("y_delta_sbp", "y_delta_vendor_health")
    reconciled = reconcile_reference_contract(labels, admitted)
    assert set(reconciled.available) >= set(admitted)
    assert reconciled.mask["y_delta_vendor_health"] is True


def test_coded_gender_is_preserved_without_a_semantic_claim():
    session = _session("SYNTH-B")
    assert session.coded_values["gender"] == "1"
    assert QC_CODED_FIELD_SEMANTICS_UNVERIFIED in session.issue_codes()
    row = session_feature_row(session)
    assert "sex" not in row.feature_names


def test_snapshot_endpoints_are_flagged():
    for session in _sessions():
        assert QC_SNAPSHOT_ENDPOINT in session.issue_codes()


def test_feature_row_masks_missing_body_composition():
    row = session_feature_row(_session("SYNTH-B"))
    values = dict(zip(row.feature_names, row.values, strict=True))
    valid = dict(zip(row.feature_names, row.valid_mask, strict=True))
    assert valid["weight_kg"] is False
    assert valid["bmi"] is False
    assert values["weight_kg"] == 0.0
    assert valid["systolic_bp"] is True


def test_contract_coverage_reports_excluded_body_feature():
    coverage = contract_coverage(_session("SYNTH-A"))
    assert coverage["body_pre"]["present"] == ["weight_kg", "bmi"]
    assert coverage["physio_pre"]["missing"] == []
    assert coverage["body_pre"]["excluded"] == [
        {"feature": "sex", "reason": QC_CODED_FIELD_SEMANTICS_UNVERIFIED}
    ]


def test_import_is_deterministic():
    first = import_real_export(FIXTURE)
    second = import_real_export(FIXTURE)
    assert [session_labels(session).values for session in first] == [
        session_labels(session).values for session in second
    ]
    assert [session.issue_codes() for session in first] == [
        session.issue_codes() for session in second
    ]


def test_events_satisfy_the_session_contract():
    session = _session("SYNTH-A")
    contract = session.to_session()
    assert contract.schema_version == "physio-session-v1"
    phases = sorted({event.phase for event in contract.events})
    assert phases == ["post", "pre"]
    for event in contract.events:
        assert event.provenance.adapter == "physio.real_export"
        assert event.provenance.adapter_version == REAL_EXPORT_SCHEMA_VERSION


def test_unknown_source_type_is_refused(tmp_path):
    path = tmp_path / "unknown.csv"
    rows = list(csv.DictReader(FIXTURE.open(encoding="utf-8")))
    rows[0]["source_type"] = "未知模块"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(RealExportError, match="unknown source_type"):
        load_real_records(path)


def test_missing_required_column_is_refused(tmp_path):
    path = tmp_path / "short.csv"
    rows = list(csv.DictReader(FIXTURE.open(encoding="utf-8")))
    for row in rows:
        row.pop("subject_key")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(RealExportError, match="missing required columns"):
        load_real_records(path)


def test_missing_file_is_refused(tmp_path):
    with pytest.raises(RealExportError, match="not found"):
        load_real_records(tmp_path / "absent.csv")


def test_empty_export_is_refused(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text(
        "session_id,source_type,record_id,member_id_hash,subject_key,measured_at\n",
        encoding="utf-8",
    )
    with pytest.raises(RealExportError, match="no rows"):
        load_real_records(path)


def test_build_sessions_without_records_is_refused():
    with pytest.raises(RealExportError, match="without records"):
        build_real_sessions([])
