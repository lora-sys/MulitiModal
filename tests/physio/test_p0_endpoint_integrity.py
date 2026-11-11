"""P0-01 synthetic regressions for independently sourced Pre/Post measurements.

No participant records are used. These scenarios intentionally cover cases
that a legacy adjacent-blood-pressure pairing can wrongly accept.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

from physio.real_export import (
    QC_OVERLAPPING_ENDPOINTS,
    QC_RECORD_REUSED,
    QC_SESSION_OVERLAP,
    build_real_sessions,
    load_real_records,
    session_labels,
    session_post_state,
)

pytestmark = pytest.mark.synthetic
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "physio-legacy-sessions-synthetic.csv"


def _records(session_id: str):
    return [r for r in load_real_records(FIXTURE) if r.session_id == session_id]


def _record_clone(records, new_session: str, *, subject: str, time_offset_s: int = 0):
    return [
        replace(
            row,
            session_id=new_session,
            subject_key=subject,
            record_id=f"clone-{new_session}-{row.record_id}",
            measured_at=row.measured_at + timedelta(seconds=time_offset_s),
        )
        for row in records
    ]


def test_endpoint_provenance_names_the_value_record_not_just_anchor():
    session = build_real_sessions(_records("SYNTH-A"))[0]
    pre = session.endpoints["health"]["pre"]
    assert "value_source_record=9003" in pre.measurement("fatigue_index").notes
    # rr is only valid in the completed record, not the anchor snapshot.
    assert "value_source_record=9004" in pre.measurement("respiration_rate").notes
    assert session_labels(session).values["y_delta_vendor_fatigue"] == -2.0


def test_single_health_record_must_not_become_both_endpoints():
    # Keep two independent blood-pressure events, but only one valid health record.
    records = [
        r for r in _records("SYNTH-A")
        if r.module != "health" or r.record_id == "9003"
    ]
    session = build_real_sessions(records)[0]
    assert QC_OVERLAPPING_ENDPOINTS in session.issue_codes()
    labels = session_labels(session)
    assert all(value is None for value in labels.values.values())
    assert not any(labels.mask.values())
    assert session_post_state(session) is None


def test_two_health_events_within_one_minute_are_not_distinct_pre_post_groups():
    records = [
        r for r in _records("SYNTH-A")
        if r.module != "health" or r.record_id in {"9003", "9004"}
    ]
    session = build_real_sessions(records)[0]
    assert QC_OVERLAPPING_ENDPOINTS in session.issue_codes()
    assert session_labels(session).available == ()


def test_cross_session_record_reuse_blocks_both_sessions():
    original = _records("SYNTH-A")
    duplicated = [
        replace(original[0], session_id="SECOND", subject_key="s-other"),
        * _record_clone(original[1:], "SECOND", subject="s-other"),
    ]
    sessions = build_real_sessions([*original, *duplicated])
    assert len(sessions) == 2
    for session in sessions:
        assert QC_RECORD_REUSED in session.issue_codes()
        assert not any(session_labels(session).mask.values())
        assert session_post_state(session) is None


def test_same_subject_overlapping_windows_block_all_labels():
    original = _records("SYNTH-A")
    overlapping = _record_clone(
        original, "SECOND", subject="s1", time_offset_s=300
    )
    sessions = build_real_sessions([*original, *overlapping])
    for session in sessions:
        assert QC_SESSION_OVERLAP in session.issue_codes()
        assert not any(session_labels(session).mask.values())


def test_shared_member_with_distinct_subjects_does_not_imply_time_overlap():
    original = _records("SYNTH-A")
    parallel = _record_clone(original, "SECOND", subject="s2")
    sessions = build_real_sessions([*original, *parallel])
    assert len(sessions) == 2
    for session in sessions:
        assert QC_SESSION_OVERLAP not in session.issue_codes()
        assert QC_RECORD_REUSED not in session.issue_codes()
        # This still does not establish physical-subject isolation.
        assert session.identity_trust == "SHARED_ACCOUNT_UNKNOWN"
