"""#102 real data audit: every reference field is match, explained or unrecoverable."""

from __future__ import annotations

from pathlib import Path

import pytest

from physio.real_data_audit import (
    STATUS_FAILED,
    VERDICT_EXPLAINED,
    VERDICT_MATCH,
    VERDICT_MISMATCH,
    VERDICT_NOT_RECOVERABLE,
    VERDICT_OBSERVATION,
    audit_real_export,
    load_reference_rows,
    summarize_findings,
)
from physio.real_export import import_real_export

pytestmark = pytest.mark.synthetic

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
RAW = FIXTURES / "physio-legacy-sessions-synthetic.csv"
REFERENCE = FIXTURES / "physio-legacy-reference-synthetic.csv"
QC = FIXTURES / "physio-legacy-qc-synthetic.csv"


def _audit():
    sessions = import_real_export(RAW)
    return audit_real_export(
        sessions, load_reference_rows(REFERENCE), load_reference_rows(QC)
    )


def test_every_reference_field_is_accounted_for():
    audit = _audit()
    assert audit.status != STATUS_FAILED
    assert audit.passed is True
    counts = audit.verdict_counts()
    assert counts[VERDICT_MISMATCH] == 0
    assert counts[VERDICT_MATCH] > 0
    assert counts[VERDICT_EXPLAINED] > 0
    assert counts[VERDICT_NOT_RECOVERABLE] > 0
    assert audit.sessions == 5
    assert audit.records == 37


def test_provenance_covers_exactly_the_provided_records():
    audit = _audit()
    findings = [f for f in audit.findings if f.field == "source_record_ids"]
    assert findings
    assert all(f.verdict == VERDICT_MATCH for f in findings)
    assert all("exactly" in f.note for f in findings)


def test_all_comparable_qc_cross_checks_match_the_reference_report():
    audit = _audit()
    assert audit.qc_cross_checks
    comparable = [c for c in audit.qc_cross_checks if c.verdict != VERDICT_OBSERVATION]
    assert comparable
    assert all(c.verdict == VERDICT_MATCH for c in comparable)
    assert {c.item for c in comparable} == {
        "health_completion_count",
        "session_window",
        "missing_field_body",
        "missing_field_tcm",
        "missing_field_eeg",
        "missing_field_rr",
        "bp_anomaly",
        "reference_label_mask",
    }


def test_observations_are_never_counted_as_agreement():
    audit = _audit()
    observations = [c for c in audit.qc_cross_checks if c.verdict == VERDICT_OBSERVATION]
    assert observations
    assert {c.item for c in observations} == {
        "shared_account_identity",
        "snapshot_endpoints",
        "point_stream_excluded",
        "hr_cross_module",
        "module_family_coverage",
    }
    assert all(c.reference and c.reconstructed for c in observations)


def test_extract_absent_columns_are_not_recoverable():
    audit = _audit()
    ages = [f for f in audit.findings if f.field == "age"]
    assert ages
    assert all(f.verdict == VERDICT_NOT_RECOVERABLE for f in ages)


def test_imputed_reference_values_are_explained_not_matched():
    audit = _audit()
    rr = [f for f in audit.findings if f.field in {"rr_pre", "rr_post"}]
    assert rr
    verdicts = {f.verdict for f in rr}
    assert verdicts <= {VERDICT_MATCH, VERDICT_EXPLAINED}
    assert any(f.verdict == VERDICT_EXPLAINED for f in rr)


def test_qc_status_is_reconstructed_from_module_presence():
    audit = _audit()
    statuses = {f.session_id: f.reconstructed for f in audit.findings if f.field == "qc_status"}
    assert statuses["SYNTH-A"] == "COMPLETE"
    assert statuses["SYNTH-B"] == "INCOMPLETE"
    assert statuses["SYNTH-E"] == "COMPLETE"
    assert all(f.verdict == VERDICT_MATCH for f in audit.findings if f.field == "qc_status")


def test_health_completion_count_compares_the_leading_integer():
    from physio.real_data_audit import _leading_int

    assert _leading_int("2") == 2
    assert _leading_int("0(仅快照)") == 0
    assert _leading_int(" 3 ") == 3
    assert _leading_int("") is None
    assert _leading_int("仅快照") is None


def test_health_completion_count_fails_when_the_reference_is_wrong():
    sessions = import_real_export(RAW)
    qc_rows = [dict(row) for row in load_reference_rows(QC)]
    qc_rows[0]["健康完成态数"] = "9"
    audit = audit_real_export(sessions, load_reference_rows(REFERENCE), qc_rows)
    check = [c for c in audit.qc_cross_checks if c.item == "health_completion_count"]
    assert check
    assert check[0].verdict == VERDICT_MISMATCH


def test_reference_response_hash_is_not_recoverable():
    audit = _audit()
    hashes = [f for f in audit.findings if f.field == "response_hash"]
    assert hashes
    assert all(f.verdict == VERDICT_NOT_RECOVERABLE for f in hashes)
    assert all("not declared" in f.note for f in hashes)


def test_every_reference_column_is_accounted_for():
    audit = _audit()
    audited = {f.field for f in audit.findings}
    columns = set(load_reference_rows(REFERENCE)[0])
    assert columns - audited == set()


def test_a_tampered_reference_value_fails_the_audit():
    sessions = import_real_export(RAW)
    rows = [dict(row) for row in load_reference_rows(REFERENCE)]
    rows[0]["y_delta_sbp"] = "-99"
    audit = audit_real_export(sessions, rows, load_reference_rows(QC))
    assert audit.status == STATUS_FAILED
    assert audit.passed is False
    assert any(
        f.field == "y_delta_sbp" and f.verdict == VERDICT_MISMATCH for f in audit.findings
    )


def test_a_reference_only_session_is_reported():
    sessions = import_real_export(RAW)
    rows = [dict(row) for row in load_reference_rows(REFERENCE)]
    rows.append(dict(rows[0], session_id="SYNTH-Z"))
    audit = audit_real_export(sessions, rows, load_reference_rows(QC))
    finding = [f for f in audit.findings if f.field == "reference_only_sessions"]
    assert finding
    assert finding[0].verdict == VERDICT_NOT_RECOVERABLE


def test_a_session_without_a_reference_row_is_reported():
    sessions = import_real_export(RAW)
    rows = [dict(row) for row in load_reference_rows(REFERENCE)]
    rows = [row for row in rows if row["session_id"] != "SYNTH-C"]
    audit = audit_real_export(sessions, rows, load_reference_rows(QC))
    finding = [f for f in audit.findings if f.field == "reference_row"]
    assert finding
    assert finding[0].verdict == VERDICT_NOT_RECOVERABLE


def test_missing_reference_file_is_refused():
    sessions = import_real_export(RAW)
    with pytest.raises(FileNotFoundError):
        load_reference_rows(FIXTURES / "does-not-exist.csv")
    assert sessions


def test_audit_requires_at_least_one_session():
    with pytest.raises(ValueError, match="at least one session"):
        audit_real_export([], [])


def test_summarize_findings_groups_by_verdict():
    audit = _audit()
    grouped = summarize_findings(audit.findings)
    assert VERDICT_MISMATCH not in grouped
    assert grouped[VERDICT_MATCH]
    assert all(item.count(":") == 1 for items in grouped.values() for item in items)
