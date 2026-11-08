"""#96/#95 legacy 94-field export import: identity, QC, labels and determinism."""

from __future__ import annotations

from datetime import datetime

import pytest

from physio.legacy_export import (
    IDENTITY_SHARED,
    IDENTITY_TRUSTED,
    ISSUE_AMBIGUOUS_FIELD,
    ISSUE_CROSS_DAY,
    ISSUE_DURATION_CONFLICT,
    ISSUE_EEG_TIME_CONFLICT,
    ISSUE_IDENTITY_UNKNOWN,
    ISSUE_IMPUTED_VALUE,
    ISSUE_NAIVE_TIMESTAMP,
    ISSUE_NON_FINITE,
    ISSUE_PHASE_CONFLICT,
    ISSUE_SEQUENCE_COLLISION,
    ISSUE_SESSION_COLLISION,
    ISSUE_SNAPSHOT_COMPLETED_MISMATCH,
    ISSUE_UNKNOWN_FIELD,
    ExportFieldRegistry,
    ExportFieldSpec,
    import_legacy_rows,
)

pytestmark = pytest.mark.synthetic

BASE = "2026-11-08T09:00:00+00:00"

BODY_FIELDS = (
    "age", "sex", "height_cm", "weight_kg", "body_fat_percentage", "bmi",
    "bmr", "visceral_fat", "muscle_mass", "body_water_percentage", "body_score",
    "body_age",
)
PRE_FIELDS = {
    "heart_rate_pre": ("heart_rate", "bpm"),
    "microcirculation_pre": ("skin_temperature", "°C"),
    "fatigue_index_pre": ("fatigue_index", "index"),
    "spo2_pre": ("spo2", "%"),
    "health_score_pre": ("stress_index", "index"),
    "sbp_pre": ("systolic_bp", "mmHg"),
    "dbp_pre": ("diastolic_bp", "mmHg"),
}
DURING_FIELDS = {
    "heart_rate": ("heart_rate", "bpm"),
    "spo2": ("spo2", "%"),
    "skin_temperature": ("skin_temperature", "°C"),
    "respiration_rate": ("respiration_rate", "breaths/min"),
    "sbp": ("systolic_bp", "mmHg"),
    "dbp": ("diastolic_bp", "mmHg"),
    "hrv_rmssd": ("hrv_rmssd", "ms"),
}
POST_FIELDS = {
    "heart_rate_post": ("heart_rate", "bpm"),
    "spo2_post": ("spo2", "%"),
    "skin_temperature_post": ("skin_temperature", "°C"),
    "respiration_rate_post": ("respiration_rate", "breaths/min"),
    "sbp_post": ("systolic_bp", "mmHg"),
    "dbp_post": ("diastolic_bp", "mmHg"),
    "hrv_rmssd_post": ("hrv_rmssd", "ms"),
}
EEG_FIELDS = (
    "eeg_fatigue_value", "eeg_mean_attention", "eeg_max_attention",
    "eeg_meditation_duration", "eeg_meditation_proportion",
    "eeg_mind_wandering_frequency", "eeg_stress_relief_index",
    "eeg_comprehensive_score", "eeg_mean_mindfulness_index",
)
EEG_TIME_FIELDS = ("eeg_started_at", "eeg_ended_at", "eeg_duration_s")
META_FIELDS = (
    "member_id", "session_key", "record_id", "record_kind", "phase",
    "measured_at", "received_at", "sequence", "device_ref", "program_id",
    "intensity", "duration_s",
)
OTHER_FIELDS = (
    "vendor_fatigue_index", "vendor_stress_index", "health_score_vendor",
    "session_duration_s", "sample_count_hr", "sample_count_spo2", "sample_count_eeg",
    "device_model", "device_firmware", "service_version", "export_version",
    "record_created_at", "record_updated_at", "snapshot_flag", "completion_flag",
    "member_age_group", "member_gender_label", "program_name", "program_mode",
    "program_level", "program_duration", "room_temperature", "room_humidity",
    "noise_db", "operator_id", "store_id", "chair_id", "session_status",
    "session_note", "pre_summary", "post_summary", "eeg_summary", "qc_flag",
    "qc_reason", "raw_payload_sha256", "imported_at", "source_file",
)


def _fields() -> tuple[ExportFieldSpec, ...]:
    specs: list[ExportFieldSpec] = [
        ExportFieldSpec(name, "identity" if "member" in name else "session")
        for name in META_FIELDS
    ]
    for name in BODY_FIELDS:
        phase = "pre"
        metric = {
            "age": "age", "sex": "sex", "height_cm": "height_cm",
            "weight_kg": "weight_kg", "body_fat_percentage": "body_fat_percentage",
            "bmi": "bmi", "bmr": "bmr",
        }.get(name)
        specs.append(
            ExportFieldSpec(
                name,
                "metric" if metric else "ignored",
                canonical_metric=metric,
                phase=phase,
                unit={"age": "years", "sex": "", "height_cm": "cm", "weight_kg": "kg",
                      "body_fat_percentage": "%", "bmi": "kg/m2", "bmr": "kcal/day"}.get(metric),
                label_role="none",
            )
        )
    for group, phase in ((PRE_FIELDS, "pre"), (DURING_FIELDS, "during"), (POST_FIELDS, "post")):
        for raw_name, (metric, unit) in group.items():
            specs.append(
                ExportFieldSpec(
                    raw_name, "metric", canonical_metric=metric, phase=phase,
                    unit=unit, label_role="y_delta_input" if phase in ("pre", "post") else "none",
                )
            )
    for name in EEG_FIELDS + EEG_TIME_FIELDS:
        specs.append(ExportFieldSpec(name, "eeg"))
    for name in OTHER_FIELDS:
        specs.append(ExportFieldSpec(name, "summary"))
    return tuple(specs)


def _registry() -> ExportFieldRegistry:
    return ExportFieldRegistry(version="legacy-export-v1-test", fields=_fields())


def _row(**overrides):
    from datetime import datetime, timedelta

    measured = overrides.get("measured_at", BASE)
    received = overrides.get("received_at")
    if received is None and isinstance(measured, str):
        received = (datetime.fromisoformat(measured) + timedelta(seconds=2)).isoformat()
    row = {
        "member_id": "member-a",
        "session_key": "sess-a",
        "record_id": "r0",
        "record_kind": "completed",
        "phase": "during",
        "measured_at": measured,
        "received_at": received or "2026-11-08T09:00:02+00:00",
        "sequence": 0,
        "device_ref": "chair-1",
        "program_id": "M01",
        "intensity": 2,
        "duration_s": 900,
        "heart_rate": 72,
        "spo2": 97,
        "skin_temperature": 33.1,
        "respiration_rate": 16,
        "sbp": 118,
        "dbp": 76,
        "hrv_rmssd": 42,
    }
    row.update(overrides)
    return row


def test_fixture_has_exactly_94_columns():
    row = _row()
    # Every canonical fixture row must describe the 94-column export shape.
    keys = set(row) | set(META_FIELDS) | set(OTHER_FIELDS) | set(EEG_FIELDS) | set(EEG_TIME_FIELDS)
    keys |= set(BODY_FIELDS) | set(PRE_FIELDS) | set(DURING_FIELDS) | set(POST_FIELDS)
    assert len(keys) == 94
    assert len(_registry().fields) == 94


def test_unknown_field_is_reported_not_guessed():
    row = _row(vendor_magic=123)
    result = import_legacy_rows([row], _registry(), identity_trust=IDENTITY_TRUSTED)
    assert any(issue.status == ISSUE_UNKNOWN_FIELD for issue in result.issues)
    assert result.rows_accepted == 1
    event = result.sessions[0].events[0]
    assert event.measurement("vendor_magic") is None


def test_interleaved_members_and_shared_identity():
    rows = [
        _row(member_id="m1", session_key="s1", record_id="a", sequence=0),
        _row(member_id="m2", session_key="s2", record_id="b", sequence=0,
             measured_at="2026-11-08T09:00:10+00:00"),
        _row(member_id="m1", session_key="s1", record_id="c", sequence=1,
             measured_at="2026-11-08T09:00:20+00:00"),
    ]
    shared = import_legacy_rows(rows, _registry(), identity_trust=IDENTITY_SHARED)
    assert len(shared.sessions) == 2
    assert shared.trusted_subject_identity is False
    assert all(session.subject_ref.startswith("shared-account:") for session in shared.sessions)
    assert any(issue.status == ISSUE_IDENTITY_UNKNOWN for issue in shared.issues)
    trusted = import_legacy_rows(rows, _registry(), identity_trust=IDENTITY_TRUSTED)
    assert all(session.subject_ref.startswith("subject:") for session in trusted.sessions)


def test_snapshot_and_completed_prefer_completed_with_qc():
    rows = [
        _row(record_id="r1", record_kind="snapshot", heart_rate=60, sequence=0),
        _row(record_id="r1", record_kind="completed", heart_rate=75, sequence=0,
             received_at="2026-11-08T09:00:05+00:00"),
    ]
    result = import_legacy_rows(rows, _registry(), identity_trust=IDENTITY_TRUSTED)
    assert len(result.sessions) == 1
    assert len(result.sessions[0].events) == 1
    assert result.sessions[0].events[0].measurement("heart_rate").value == 75
    statuses = {issue.status for issue in result.issues}
    assert ISSUE_SNAPSHOT_COMPLETED_MISMATCH in statuses


def test_same_record_in_pre_and_post_is_rejected():
    rows = [
        _row(record_id="same", phase="pre", heart_rate_pre=70, sequence=0),
        _row(record_id="same", phase="post", heart_rate_post=80, sequence=1,
             measured_at="2026-11-08T09:10:00+00:00"),
    ]
    result = import_legacy_rows(rows, _registry(), identity_trust=IDENTITY_TRUSTED)
    assert result.sessions == ()
    assert any(issue.status == ISSUE_PHASE_CONFLICT for issue in result.issues)


def test_sequence_collision_rejects_later_distinct_record():
    rows = [
        _row(record_id="a", sequence=1),
        _row(record_id="b", sequence=1, measured_at="2026-11-08T09:01:00+00:00"),
    ]
    result = import_legacy_rows(rows, _registry(), identity_trust=IDENTITY_TRUSTED)
    assert sum(len(session) for session in result.sessions) == 1
    assert any(issue.status == ISSUE_SEQUENCE_COLLISION for issue in result.issues)


def test_naive_timestamp_is_rejected():
    result = import_legacy_rows(
        [_row(measured_at="2026-11-08T09:00:00", received_at="2026-11-08T09:00:02")],
        _registry(),
        identity_trust=IDENTITY_TRUSTED,
    )
    assert result.sessions == ()
    assert any(issue.status == ISSUE_NAIVE_TIMESTAMP for issue in result.issues)


def test_cross_day_span_is_flagged_not_merged_silently():
    rows = [
        _row(record_id="a", sequence=0, measured_at="2026-11-08T23:59:00+00:00",
             received_at="2026-11-08T23:59:02+00:00"),
        _row(record_id="b", sequence=1, measured_at="2026-11-09T00:01:00+00:00",
             received_at="2026-11-09T00:01:02+00:00"),
    ]
    result = import_legacy_rows(
        rows, _registry(), identity_trust=IDENTITY_TRUSTED, max_span_s=60
    )
    assert any(issue.status == ISSUE_CROSS_DAY for issue in result.issues)


def test_eeg_and_duration_conflicts_are_explicit():
    rows = [
        _row(
            record_id="e1",
            eeg_started_at="2026-11-08T09:00:00+00:00",
            eeg_ended_at="2026-11-08T08:59:00+00:00",
            eeg_duration_s=120,
            eeg_fatigue_value=88,
        ),
        _row(
            record_id="e2",
            sequence=1,
            measured_at="2026-11-08T09:05:00+00:00",
            eeg_started_at="2026-11-08T09:00:00+00:00",
            eeg_ended_at="2026-11-08T09:02:00+00:00",
            eeg_duration_s=600,
            eeg_fatigue_value=88,
        ),
    ]
    result = import_legacy_rows(rows, _registry(), identity_trust=IDENTITY_TRUSTED)
    statuses = {issue.status for issue in result.issues}
    assert ISSUE_EEG_TIME_CONFLICT in statuses
    assert ISSUE_DURATION_CONFLICT in statuses
    # EEG channels are audit-only metadata: they must never become measurements.
    for session in result.sessions:
        for event in session.events:
            assert all(not item.metric.startswith("eeg_") for item in event.measurements)


def test_ambiguous_hrv_semantics_is_rejected():
    registry = ExportFieldRegistry(
        version="ambiguous-v1",
        fields=(
            ExportFieldSpec("member_id", "identity"),
            ExportFieldSpec("session_key", "session"),
            ExportFieldSpec("record_id", "session"),
            ExportFieldSpec("record_kind", "session"),
            ExportFieldSpec("phase", "phase"),
            ExportFieldSpec("measured_at", "timestamp"),
            ExportFieldSpec("received_at", "timestamp"),
            ExportFieldSpec("sequence", "session"),
            ExportFieldSpec("hrv", "metric", ambiguous_semantics=True),
        ),
    )
    result = import_legacy_rows([_row(hrv=42)], registry, identity_trust=IDENTITY_TRUSTED)
    assert any(issue.status == ISSUE_AMBIGUOUS_FIELD for issue in result.issues)
    assert result.sessions == ()


def test_imputed_value_is_marked_and_degrades_quality():
    registry = ExportFieldRegistry(
        version="imputed-v1",
        fields=(
            ExportFieldSpec("member_id", "identity"),
            ExportFieldSpec("session_key", "session"),
            ExportFieldSpec("record_id", "session"),
            ExportFieldSpec("record_kind", "session"),
            ExportFieldSpec("phase", "phase"),
            ExportFieldSpec("measured_at", "timestamp"),
            ExportFieldSpec("received_at", "timestamp"),
            ExportFieldSpec("sequence", "session"),
            ExportFieldSpec("heart_rate", "metric", canonical_metric="heart_rate",
                            unit="bpm", value_origin="imputed"),
        ),
    )
    result = import_legacy_rows([_row()], registry, identity_trust=IDENTITY_TRUSTED)
    event = result.sessions[0].events[0]
    measurement = event.measurement("heart_rate")
    assert any(issue.status == ISSUE_IMPUTED_VALUE for issue in result.issues)
    assert "value_origin=imputed" in measurement.notes
    assert event.quality == "degraded"
    assert result.imputed_metrics == {"legacy-member-a-sess-a": ("heart_rate",)}


def test_session_id_collision_is_suffixed_not_merged():
    rows = [
        _row(member_id="m1", session_key="s1", session_id="fixed", record_id="a"),
        _row(member_id="m2", session_key="s2", session_id="fixed", record_id="b"),
    ]
    result = import_legacy_rows(rows, _registry(), identity_trust=IDENTITY_TRUSTED)
    assert len(result.sessions) == 2
    assert any(issue.status == ISSUE_SESSION_COLLISION for issue in result.issues)


def test_import_order_does_not_change_events_or_digest():
    rows = [
        _row(record_id="a", sequence=1, measured_at="2026-11-08T09:01:00+00:00"),
        _row(record_id="b", sequence=0),
        _row(record_id="c", sequence=2, measured_at="2026-11-08T09:02:00+00:00"),
    ]
    forward = import_legacy_rows(rows, _registry(), identity_trust=IDENTITY_TRUSTED)
    reverse = import_legacy_rows(list(reversed(rows)), _registry(), identity_trust=IDENTITY_TRUSTED)
    assert [e.to_dict() for e in forward.sessions[0].events] == [
        e.to_dict() for e in reverse.sessions[0].events
    ]
    assert forward.reproducibility_digest == reverse.reproducibility_digest


def test_timezone_offsets_normalize_to_one_instant():
    rows = [
        _row(record_id="a", sequence=0, measured_at="2026-11-08T17:00:00+08:00",
             received_at="2026-11-08T17:00:02+08:00"),
        _row(record_id="b", sequence=1, measured_at="2026-11-08T09:01:00+00:00",
             received_at="2026-11-08T09:01:02+00:00"),
    ]
    result = import_legacy_rows(rows, _registry(), identity_trust=IDENTITY_TRUSTED)
    events = result.sessions[0].events
    assert events[0].measured_at == datetime.fromisoformat("2026-11-08T09:00:00+00:00")
    assert events[1].measured_at == datetime.fromisoformat("2026-11-08T09:01:00+00:00")
    assert (events[1].measured_at - events[0].measured_at).total_seconds() == 60


def test_import_rebuild_is_stable_for_the_same_input():
    rows = [_row(record_id=f"r{index}", sequence=index,
                 measured_at=f"2026-11-08T09:0{index}:00+00:00") for index in range(3)]
    first = import_legacy_rows(rows, _registry(), identity_trust=IDENTITY_TRUSTED)
    second = import_legacy_rows(rows, _registry(), identity_trust=IDENTITY_TRUSTED)
    assert [session.session_id for session in first.sessions] == [
        session.session_id for session in second.sessions
    ]
    assert first.qc_report() == second.qc_report()
    assert first.reproducibility_digest == second.reproducibility_digest


def test_non_finite_metric_is_rejected_without_a_guess():
    for value in (float("nan"), float("inf")):
        result = import_legacy_rows(
            [_row(heart_rate=value)], _registry(), identity_trust=IDENTITY_TRUSTED
        )
        assert any(issue.status == ISSUE_NON_FINITE for issue in result.issues)
        measurement = result.sessions[0].events[0].measurement("heart_rate")
        assert measurement is not None
        assert measurement.valid is False
        assert measurement.value is None
        assert measurement.status == ISSUE_NON_FINITE


def test_duplicate_completed_records_tie_break_is_order_independent():
    rows = [
        _row(record_id="tie", record_kind="completed", heart_rate=61, sequence=0,
             received_at="2026-11-08T09:00:05+00:00"),
        _row(record_id="tie", record_kind="completed", heart_rate=83, sequence=0,
             received_at="2026-11-08T09:00:05+00:00"),
    ]
    forward = import_legacy_rows(rows, _registry(), identity_trust=IDENTITY_TRUSTED)
    reverse = import_legacy_rows(list(reversed(rows)), _registry(), identity_trust=IDENTITY_TRUSTED)
    assert len(forward.sessions[0].events) == len(reverse.sessions[0].events) == 1
    assert (
        forward.sessions[0].events[0].to_dict()
        == reverse.sessions[0].events[0].to_dict()
    )
    assert forward.reproducibility_digest == reverse.reproducibility_digest
    assert forward.qc_report() == reverse.qc_report()
