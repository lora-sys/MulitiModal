"""#99 recommendation service and safety state machine tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from physio.programs import ProgramCapabilityCatalog
from physio.recommendation import (
    CONTROL_STATUS_NOT_CONNECTED,
    STATE_IDLE,
    STATE_MONITORING,
    STATE_SAFE_STOP,
    STATE_USER_CONFIRM,
    STATUS_ABSTAIN,
    STATUS_OK,
    STATUS_SAFE_STOP,
    ControlNotAuthorized,
    DeviceControlPort,
    MassageRecommendationService,
    RecommendationStateMachine,
    StateTransitionError,
    UserPreference,
)
from physio.schema import ProtocolContextV1

pytestmark = pytest.mark.synthetic


def catalog_dict(*, reviewed=True):
    return {
        "catalog_version": "test-catalog-v1",
        "reviewed": reviewed,
        "programs": [
            {
                "program_id": "M01", "name": "舒缓", "intensity_min": 1,
                "intensity_max": 4, "duration_min_s": 300, "duration_max_s": 1800,
                "actions": ["knead"], "selectable": True, "reviewed": True,
                "device_compat": ["chair-mock-01"],
            },
            {
                "program_id": "M02", "name": "活力", "intensity_min": 2,
                "intensity_max": 6, "duration_min_s": 600, "duration_max_s": 1800,
                "actions": ["vibrate"], "selectable": True, "reviewed": True,
                "device_compat": ["chair-mock-01"],
            },
            {
                "program_id": "M03", "name": "隐藏", "intensity_min": 1,
                "intensity_max": 3, "duration_min_s": 300, "duration_max_s": 900,
                "actions": ["knead"], "selectable": False, "reviewed": True,
            },
        ],
    }


def live_inference(status="NOT_AVAILABLE", quality="ok"):
    return SimpleNamespace(
        status=status, quality=quality, model_version="NOT_AVAILABLE",
        prediction=None,
    )


def service(**kwargs):
    return MassageRecommendationService(
        ProgramCapabilityCatalog.from_dict(catalog_dict(**kwargs))
    )


def test_state_machine_accepts_only_declared_transitions():
    machine = RecommendationStateMachine()
    assert machine.state == STATE_IDLE
    assert machine.trigger("baseline_ready") == "BASELINE"
    assert machine.trigger("monitoring_started") == STATE_MONITORING
    assert machine.trigger("recommend") == "RECOMMENDING"
    assert machine.trigger("request_confirmation") == STATE_USER_CONFIRM
    assert machine.trigger("user_confirmed") == STATE_MONITORING
    with pytest.raises(StateTransitionError):
        machine.trigger("user_confirmed")


def test_user_stop_and_safety_alert_have_highest_priority():
    svc = service()
    stopped = svc.user_stop()
    assert stopped.status == STATUS_SAFE_STOP
    assert stopped.state == STATE_SAFE_STOP
    assert stopped.candidates == []
    assert stopped.control_command is None

    alert = svc.safety_alert("sensor fault")
    assert alert.status == STATUS_SAFE_STOP
    assert "sensor fault" in alert.reason
    assert alert.control_command is None


def test_no_live_telemetry_refuses_and_stops_automatic_changes():
    svc = service()
    result = svc.update(None)
    assert result.status == STATUS_ABSTAIN
    assert result.candidates == []
    assert result.control_command is None
    assert result.state == STATE_MONITORING


def test_stale_or_rejected_quality_refuses_candidates():
    svc = service()
    for quality in ("stale", "rejected"):
        result = svc.update(live_inference(quality=quality))
        assert result.status == STATUS_ABSTAIN
        assert result.candidates == []
        assert result.control_command is None


def test_unreviewed_or_missing_catalog_refuses():
    unreviewed = service(reviewed=False)
    result = unreviewed.update(live_inference())
    assert result.status == STATUS_ABSTAIN
    assert "unreviewed" in result.reason


def test_static_candidates_work_without_a_validated_model():
    result = service().update(live_inference())
    assert result.status == STATUS_OK
    assert [c.program_id for c in result.candidates] == ["M01", "M02"]
    assert all(c.expected_effect is None for c in result.candidates)
    assert result.evidence["model_effect_ranking"] is False
    assert result.evidence["fatigue_prediction_used"] is False
    assert result.model_version == "NOT_AVAILABLE"
    assert result.control_command is None
    assert all(c.intensity is None for c in result.candidates)


def test_non_selectable_program_is_never_offered():
    result = service().update(live_inference())
    assert "M03" not in {c.program_id for c in result.candidates}


def test_unknown_current_program_is_rejected_not_mapped():
    svc = service()
    result = svc.update(
        live_inference(),
        current_program=ProtocolContextV1(program_id="M99", intensity=1, duration_s=600),
    )
    assert result.rejected and result.rejected[0]["program_id"] == "M99"
    assert "unknown program" in result.rejected[0]["reason"].lower()
    assert all(c.program_id != "M99" for c in result.candidates)


def test_user_preference_is_validated_and_never_out_of_range():
    svc = service()
    result = svc.update(
        live_inference(),
        user_preference=UserPreference(max_intensity=3, max_duration_s=900),
    )
    by_id = {c.program_id: c for c in result.candidates}
    assert by_id["M01"].intensity == 3
    assert by_id["M02"].intensity == 3
    assert by_id["M01"].duration_s == 900

    restrictive = service().update(
        live_inference(), user_preference=UserPreference(max_intensity=0)
    )
    assert restrictive.candidates == []
    assert restrictive.status == STATUS_ABSTAIN


def test_pending_recommendation_is_not_silently_replaced():
    svc = service()
    first = svc.update(live_inference())
    assert first.state == STATE_USER_CONFIRM
    again = svc.update(live_inference())
    assert again is first


def test_confirm_requires_an_offered_candidate_and_logs_privacy_safe_data():
    svc = service()
    svc.update(live_inference())
    assert svc.machine.state == STATE_USER_CONFIRM
    with pytest.raises(ValueError):
        svc.confirm("M99")
    entry = svc.confirm("M01")
    assert entry["control_command"] is None
    assert svc.machine.state == STATE_MONITORING
    log = json.dumps(svc.audit_log, ensure_ascii=False)
    assert "heart_rate" not in log
    assert "subj-" not in log
    assert "M01" in log


def test_control_port_is_not_connected_and_never_sends():
    port = DeviceControlPort()
    assert port.status == CONTROL_STATUS_NOT_CONNECTED
    with pytest.raises(ControlNotAuthorized):
        port.send({"program_id": "M01"})


def test_candidates_are_deterministic_not_effect_ranked():
    first = service().update(live_inference())
    second = service().update(live_inference())
    assert [c.program_id for c in first.candidates] == [
        c.program_id for c in second.candidates
    ]
    assert [c.rank for c in first.candidates] == [1, 2]
