"""#102 real data audit: every reference field is match, explained or unrecoverable."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from physio_reference_columns import EXPECTED_REFERENCE_COLUMNS

from physio.dataset import FeatureRow
from physio.preprocessing import PreprocessingError, WindowScaler
from physio.real_data_audit import (
    STATUS_FAILED,
    STATUS_MAPPING_NOT_VERIFIED,
    VERDICT_EXPLAINED,
    VERDICT_MATCH,
    VERDICT_MISMATCH,
    VERDICT_NOT_RECOVERABLE,
    VERDICT_OBSERVATION,
    audit_real_export,
    load_reference_rows,
    summarize_findings,
)
from physio.real_export import import_real_export, session_feature_row
from physio.real_training import RealTrainingError, _fit_scaler, _scaled_row
from physio.windows import WindowFeatures

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
    assert audit.records == 36


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
    assert statuses["SYNTH-E"] == "INCOMPLETE"
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


def test_synthetic_fixture_carries_the_full_94_column_shape():
    """The fixture must exercise the real reference shape, not a reduced one."""
    columns = list(load_reference_rows(REFERENCE)[0])
    real_shape = {
        "y_delta_sbp", "brain_record_id", "feature_route", "response_hash",
        "brain_neutral_pct", "sublingual_vein", "impedance", "age_group",
    }
    assert len(columns) == 94
    assert len(set(columns)) == 94
    assert real_shape <= set(columns)


def test_synthetic_fixture_column_order_matches_the_committed_expectation():
    """Column order is asserted against a committed constant, so it is sensitive
    on every host, including one without the private extract."""
    actual = list(load_reference_rows(REFERENCE)[0])
    assert actual == list(EXPECTED_REFERENCE_COLUMNS)


def test_synthetic_fixture_column_order_matches_the_reference():
    """When the private reference is present, the committed constant is checked
    against it too, so a drift in either fails loudly instead of silently."""
    private = FIXTURES.parent.parent / "data" / "local" / "physio-legacy" / (
        "reference_train94_5sessions.csv"
    )
    if not private.is_file():
        pytest.skip("private reference export is not present on this host")
    expected = list(load_reference_rows(private)[0])
    assert list(EXPECTED_REFERENCE_COLUMNS) == expected
    assert list(load_reference_rows(REFERENCE)[0]) == expected


def test_mapping_status_is_not_verified_while_blockers_remain():
    audit = _audit()
    assert audit.mapping_status == STATUS_MAPPING_NOT_VERIFIED
    assert audit.mapping_verified is False
    assert audit.mapping_blockers
    # Engineering PASS and mapping verdict are deliberately independent.
    assert audit.passed is True
    assert audit.mapping_verified is False


def test_mapping_blockers_name_the_unrecoverable_columns():
    audit = _audit()
    summary = [b for b in audit.mapping_blockers if "not recoverable" in b]
    assert summary
    assert "age" in summary[0] and "height_cm" in summary[0]


def test_response_hash_is_a_mapping_blocker():
    audit = _audit()
    hash_blockers = [b for b in audit.mapping_blockers if "response_hash" in b]
    assert hash_blockers
    assert "not declared" in hash_blockers[0]


def test_report_hash_fragments_match_the_private_inputs():
    """The report quotes SHA-256 fragments; a transcription error must not pass.

    A wrong fragment makes the report's provenance claim unverifiable, so the
    fragments are checked against the real files whenever they are present. On a
    host without the private extract the check skips rather than passing on an
    unchecked assumption.
    """
    private = FIXTURES.parent.parent / "data" / "local" / "physio-legacy"
    report = FIXTURES.parents[1] / "docs" / "reports" / "issue-102-real-data-access.md"
    if not private.is_dir() or not report.is_file():
        pytest.skip("private extract or report is not present on this host")

    text = report.read_text(encoding="utf-8")
    for name in (
        "raw_records_5sessions.csv",
        "reference_train94_5sessions.csv",
        "reference_qc_5sessions.csv",
        "reprocessed_5sessions.csv",
    ):
        path = private / name
        if not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest[:8] in text, f"{name}: fragment {digest[:8]} is not quoted in the report"
        assert digest[-4:] in text, f"{name}: suffix {digest[-4:]} is not quoted in the report"


def test_to_dict_carries_the_independent_mapping_verdict():
    """The persisted audit.json must not read as unqualified verification.

    R4-F3 was that to_dict() dropped the mapping verdict while the script writes
    exactly that dict to runs/physio-real-data/audit.json.
    """
    audit = _audit()
    payload = audit.to_dict()
    for key in (
        "mapping_status",
        "mapping_verified",
        "mapping_blockers",
        "passed",
        "status",
        "verdicts",
        "qc_verdicts",
        "open_items",
    ):
        assert key in payload, key
    assert payload["mapping_status"] == audit.mapping_status
    assert payload["mapping_verified"] == audit.mapping_verified
    assert payload["mapping_blockers"] == list(audit.mapping_blockers)
    assert payload["passed"] == audit.passed
    # Engineering PASS and mapping sign-off must stay independently readable.
    assert payload["passed"] is True
    assert payload["mapping_verified"] is False


def test_a_refused_fold_never_reaches_the_models():
    """R4-F1: a feature with no training observations must be refused, not scaled.

    ``WindowScaler.transform`` refuses it; ``_scaled_row`` must match, and the fold
    must be recorded as refused rather than degraded.
    """
    from datetime import UTC, datetime

    from physio.dataset import FoldImputer

    names = session_feature_row(import_real_export(RAW)[0]).feature_names
    size = len(names)
    cutoff = datetime(2026, 11, 1, tzinfo=UTC)

    def make_row(session: str, subject: str, values, observed):
        return FeatureRow(
            feature_names=names,
            values=values,
            observed_mask=observed,
            valid_mask=observed,
            imputed_mask=(False,) * size,
            imputation=("none",) * size,
            subject_ref=subject,
            session_id=session,
            cutoff=cutoff,
        )

    # Two training rows that never observe feature 0, and one held-out row that
    # does. That is the real-extract shape: a fold whose training partition has zero
    # observations for a feature the held-out row measured.
    training = [
        make_row("T1", "subject:t1", (0.0,) + (5.0,) * (size - 1), (False,) + (True,) * (size - 1)),
        make_row("T2", "subject:t2", (0.0,) + (6.0,) * (size - 1), (False,) + (True,) * (size - 1)),
    ]
    held_out = make_row(
        "E1", "subject:e1", (1.0,) + (7.0,) * (size - 1), (True,) * size
    )
    scaler = _fit_scaler(training)
    assert scaler.observed_counts[0] == 0
    assert scaler.mean[0] == 0.0 and scaler.std[0] == 1.0

    # The shared contract refuses such a value; so must _scaled_row.
    window = WindowFeatures(
        usable=True,
        feature_names=names,
        values=held_out.values,
        mask=held_out.valid_mask,
        coverage=1.0,
        n_events=1,
        start=cutoff,
        end=cutoff,
        quality="ok",
        metrics_present=tuple(names),
    )
    with pytest.raises(PreprocessingError, match="never observed"):
        WindowScaler.transform(scaler, window)
    with pytest.raises(RealTrainingError, match="refusing to fabricate a normalization"):
        _scaled_row(held_out, scaler)

    # The imputer cannot invent a median either, so it leaves the position as the
    # row supplied it; the refusal has to come from the scaler, not the imputer.
    imputer = FoldImputer.fit(training)
    assert imputer.transform(held_out).valid_mask[0] is True
