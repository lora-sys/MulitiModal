"""#110 thin unified inference gateway: contracts, isolation, fail-closed."""

from __future__ import annotations

import pytest

from lifecycle.contracts import (
    CONTRACTS_VERSION,
    EVIDENCE_LOCAL,
    EVIDENCE_SMOKE,
    EVIDENCE_SYNTHETIC,
    GATE_ENGINEERING,
    GATE_GRANTED,
    GATE_NOT_GRANTED,
    GATE_REAL_DATA_TRAINING,
    GATE_RESEARCH_CLAIMS,
    STATUS_ABSTAIN,
    STATUS_BLOCKED,
    STATUS_NOT_AVAILABLE,
    STATUS_OK,
    STATUS_REJECTED,
    STATUS_SAFE_STOP,
    STATUS_STALE,
    Envelope,
    canonical_json,
    digest,
    gate_bundle,
)
from lifecycle.gateway import (
    GATEWAY_SCHEMA_VERSION,
    MODE_A_PRE_ONLY,
    MODE_B_REALTIME,
    MODE_C_TCM,
    Gateway,
    ModeIsolationError,
    envelope_from_dict,
    make_envelope,
)

pytestmark = pytest.mark.synthetic


def _sample_envelope(**kwargs) -> Envelope:
    defaults = {
        "schema_version": CONTRACTS_VERSION,
        "request_id": "req-001",
        "session_id": "sess-001",
        "trace_id": "tr-001",
        "mode": MODE_A_PRE_ONLY,
        "event_time": "",
        "feature_version": "feat-v1",
        "model_version": "model-v1",
        "target_version": "target-v1",
        "quality": "ok",
        "status": STATUS_OK,
        "provenance": "synthetic-test",
        "license": "CC-BY-4.0",
        "evidence_level": EVIDENCE_SYNTHETIC,
        "display_gate": "RESEARCH_ONLY",
        "payload": {"features": {"pre_heart_rate": 70}},
    }
    defaults.update(kwargs)
    return make_envelope(**defaults)


def test_versioned_envelope_round_trips_with_all_required_fields():
    env = _sample_envelope()
    assert env.schema_version == GATEWAY_SCHEMA_VERSION
    as_dict = env.to_dict()

    required_fields = [
        "schema_version",
        "request_id",
        "session_id",
        "trace_id",
        "mode",
        "event_time",
        "feature_version",
        "model_version",
        "target_version",
        "quality",
        "status",
        "provenance",
        "license",
        "evidence_level",
        "display_gate",
        "payload",
    ]
    for key in required_fields:
        assert key in as_dict

    restored = envelope_from_dict(as_dict)
    assert restored == env
    assert restored.to_dict() == as_dict
    assert digest(canonical_json(as_dict)) == digest(canonical_json(restored.to_dict()))


def test_batch_and_single_calls_produce_identical_results_and_idempotency():
    gateway = Gateway()
    req_1 = _sample_envelope(request_id="req-101", payload={"features": {"pre_hr": 68}})
    req_2 = _sample_envelope(request_id="req-102", payload={"features": {"pre_hr": 72}})
    req_3 = _sample_envelope(request_id="req-103", payload={"features": {"pre_hr": 75}})

    batch_responses = gateway.handle_batch([req_1, req_2, req_3])
    assert len(batch_responses) == 3

    gateway_single = Gateway()
    single_1 = gateway_single.handle(req_1)
    single_2 = gateway_single.handle(req_2)
    single_3 = gateway_single.handle(req_3)

    assert batch_responses[0] == single_1
    assert batch_responses[1] == single_2
    assert batch_responses[2] == single_3

    replayed = gateway.handle(req_1)
    assert replayed == batch_responses[0]
    assert gateway.idempotency_counter["req-101"] == 2
    assert gateway.idempotency_hits >= 1


def test_mode_a_rejects_during_and_post_features():
    gateway = Gateway()

    clean_req = _sample_envelope(
        request_id="req-clean",
        mode=MODE_A_PRE_ONLY,
        evidence_level=EVIDENCE_SMOKE,
        payload={"features": {"pre_hr": 70, "static_age": 30}},
    )
    clean_res = gateway.handle(clean_req)
    assert clean_res.status == STATUS_OK
    assert clean_res.payload["control_command"] is None

    during_req = _sample_envelope(
        request_id="req-during",
        mode=MODE_A_PRE_ONLY,
        payload={"features": {"pre_hr": 70, "during_heart_rate": 80}},
    )
    during_res = gateway.handle(during_req)
    assert during_res.status == STATUS_REJECTED
    assert "during/post features" in during_res.payload["reason"]
    assert "during_heart_rate" in str(during_res.payload["rejected_features"])

    post_req = _sample_envelope(
        request_id="req-post",
        mode=MODE_A_PRE_ONLY,
        payload={"feature_names": ["post_blood_pressure"]},
    )
    post_res = gateway.handle(post_req)
    assert post_res.status == STATUS_REJECTED
    assert "post_blood_pressure" in str(post_res.payload["rejected_features"])

    strict_gateway = Gateway(strict_mode=True)
    with pytest.raises(ModeIsolationError):
        strict_gateway.handle(during_req)


def test_mode_c_rejects_inferring_syndrome_from_physiological_numbers():
    gateway = Gateway()

    leak_req_1 = _sample_envelope(
        request_id="req-c-leak-1",
        mode=MODE_C_TCM,
        payload={"heart_rate": 78, "task": "syndrome"},
    )
    res_1 = gateway.handle(leak_req_1)
    assert res_1.status == STATUS_REJECTED
    assert "forbids inferring syndrome or constitution" in res_1.payload["reason"]

    leak_req_2 = _sample_envelope(
        request_id="req-c-leak-2",
        mode=MODE_C_TCM,
        payload={"measurements": [{"metric": "spo2", "value": 94}], "infer_syndrome": True},
    )
    res_2 = gateway.handle(leak_req_2)
    assert res_2.status == STATUS_REJECTED

    strict_gateway = Gateway(strict_mode=True)
    with pytest.raises(ModeIsolationError):
        strict_gateway.handle(leak_req_1)

    clean_c_req = _sample_envelope(
        request_id="req-c-clean",
        mode=MODE_C_TCM,
        payload={"symptom_text": "头晕目眩，少气懒言"},
    )
    clean_c_res = gateway.handle(clean_c_req)
    assert clean_c_res.status == STATUS_OK
    assert clean_c_res.payload["syndrome_judgment"] is None
    assert clean_c_res.payload["classifier_invoked"] is False
    assert clean_c_res.payload["control_command"] is None


def test_mode_b_defaults_not_available_without_validated_safety_clearance():
    gateway = Gateway()
    req = _sample_envelope(
        request_id="req-b-default",
        mode=MODE_B_REALTIME,
        payload={"window_events": 15},
    )
    res = gateway.handle(req)
    assert res.status == STATUS_NOT_AVAILABLE
    assert res.payload["prediction"] is None
    assert res.payload["control_command"] is None
    assert "unvalidated realtime model" in res.payload["reason"]


def test_fail_closed_stale_signal():
    gateway = Gateway(stale_after_s=60.0)

    explicit_stale = _sample_envelope(
        request_id="req-stale-1",
        status=STATUS_STALE,
    )
    res_1 = gateway.handle(explicit_stale)
    assert res_1.status == STATUS_STALE
    assert res_1.payload.get("prediction") is None

    old_event_req = _sample_envelope(
        request_id="req-stale-2",
        event_time="2020-01-01T00:00:00+00:00",
    )
    res_2 = gateway.handle(old_event_req)
    assert res_2.status == STATUS_STALE
    assert "older than" in res_2.payload["reason"]


def test_fail_closed_unknown_or_mismatched_versions():
    gateway = Gateway(
        known_feature_versions=["feat-v1"],
        known_model_versions=["model-v1"],
        known_target_versions=["target-v1"],
    )

    bad_feature = _sample_envelope(request_id="req-v1", feature_version="feat-v999")
    res_f = gateway.handle(bad_feature)
    assert res_f.status == STATUS_NOT_AVAILABLE
    assert "unknown feature version" in res_f.payload["reason"]

    bad_model = _sample_envelope(request_id="req-v2", model_version="model-v999")
    res_m = gateway.handle(bad_model)
    assert res_m.status == STATUS_NOT_AVAILABLE
    assert "unknown model version" in res_m.payload["reason"]

    bad_target = _sample_envelope(request_id="req-v3", target_version="target-v999")
    res_t = gateway.handle(bad_target)
    assert res_t.status == STATUS_NOT_AVAILABLE
    assert "unknown target version" in res_t.payload["reason"]


def test_fail_closed_tampered_and_bad_artifact_hashes():
    valid_hash = "a" * 64
    tampered_hash = "b" * 64
    gateway = Gateway(known_artifacts={"model-v1": valid_hash})

    bad_hex_req = _sample_envelope(
        request_id="req-hash-bad",
        payload={"artifact_sha256": "not-a-sha256"},
    )
    res_bad = gateway.handle(bad_hex_req)
    assert res_bad.status == STATUS_BLOCKED
    assert "bad artifact hash format" in res_bad.payload["reason"]

    tampered_req = _sample_envelope(
        request_id="req-hash-tampered",
        payload={"artifact_sha256": tampered_hash},
    )
    res_tampered = gateway.handle(tampered_req)
    assert res_tampered.status == STATUS_BLOCKED
    assert "tampered artifact hash" in res_tampered.payload["reason"]

    unknown_hash_gateway = Gateway(known_artifacts={"other-model": "c" * 64})
    unknown_req = _sample_envelope(
        request_id="req-hash-unknown",
        model_version="unbound-model",
        payload={"artifact_sha256": valid_hash},
    )
    res_unknown = unknown_hash_gateway.handle(unknown_req)
    assert res_unknown.status == STATUS_BLOCKED
    assert "unknown artifact hash" in res_unknown.payload["reason"]


def test_fail_closed_unknown_mode_refusal():
    gateway = Gateway()
    raw = {
        "request_id": "req-unknown-mode",
        "mode": "D_future_mode",
        "payload": {},
    }
    res = gateway.handle(raw)
    assert res.status == STATUS_REJECTED
    assert "unknown mode" in res.payload["reason"]
    assert res.payload["control_command"] is None


def test_safe_stop_overrides_everything_and_control_command_is_always_null():
    valid_hash = "a" * 64
    gateway = Gateway(known_artifacts={"model-v1": valid_hash})

    # SAFE_STOP sent alongside tampered artifact and during features
    req = _sample_envelope(
        request_id="req-stop-override",
        mode=MODE_A_PRE_ONLY,
        quality="safe_stop",
        payload={
            "signal": "SAFE_STOP",
            "artifact_sha256": "f" * 64,  # tampered
            "during_hr": 99,  # illegal feature in mode A
            "user_stop": True,
        },
    )
    res = gateway.handle(req)
    assert res.status == STATUS_SAFE_STOP
    assert res.payload["control_command"] is None
    assert res.payload["safe_stop"] is True
    assert "SAFE_STOP" in res.payload["reason"]


def test_throttle_overflow_refused_with_reason():
    gateway = Gateway(concurrency_limits={MODE_A_PRE_ONLY: 1})
    gateway._in_flight[MODE_A_PRE_ONLY] = 1

    overflow_req = _sample_envelope(request_id="req-overflow", mode=MODE_A_PRE_ONLY)
    res = gateway.handle(overflow_req)
    assert res.status == STATUS_REJECTED
    assert "throttle overflow" in res.payload["reason"]
    assert res.payload["control_command"] is None

    gateway._in_flight[MODE_A_PRE_ONLY] = 0
    accepted_res = gateway.handle(overflow_req)
    assert accepted_res.status == STATUS_OK


def test_timeout_yields_stale_or_abstain_without_fabricated_score():
    gateway = Gateway(timeout_s=0.5)

    timed_out_req = _sample_envelope(
        request_id="req-timeout",
        payload={"simulate_timeout": True, "budget_s": 0.1},
    )
    res = gateway.handle(timed_out_req)
    assert res.status in (STATUS_STALE, STATUS_ABSTAIN)
    assert "timeout" in res.payload["reason"]
    assert res.payload.get("prediction") is None
    assert res.payload["control_command"] is None


def test_session_rebuild_reproduces_same_state_without_invented_measurements():
    gateway = Gateway()
    session_id = "sess-rebuild-test"

    event_1 = {"measured_at": "2026-11-10T10:00:00+00:00", "metric": "heart_rate", "value": 72}
    event_2 = {"measured_at": "2026-11-10T10:05:00+00:00", "metric": "spo2", "value": 97}
    event_3 = {"measured_at": "2026-11-10T10:10:00+00:00", "metric": "heart_rate", "value": 75}

    gateway.record_event(session_id, event_2)
    gateway.record_event(session_id, event_1)
    gateway.record_event(session_id, event_3)

    state_1 = gateway.rebuild_session(session_id)
    state_2 = gateway.rebuild_session(session_id)

    assert state_1 == state_2
    assert state_1.state_hash == state_2.state_hash
    assert state_1.event_count == 3
    assert state_1.invented_measurements == 0
    assert state_1.events[0]["metric"] == "heart_rate"
    assert state_1.events[0]["measured_at"] == "2026-11-10T10:00:00+00:00"
    assert state_1.events[-1]["measured_at"] == "2026-11-10T10:10:00+00:00"
    assert sorted(state_1.metrics_present) == ["heart_rate", "spo2"]

    empty_state = gateway.rebuild_session("unseen-session")
    assert empty_state.event_count == 0
    assert empty_state.invented_measurements == 0
    assert empty_state.last_event_time is None


def test_transport_failure_emits_no_score():
    gateway = Gateway()
    res = gateway.handle_transport_failure("sess-transport-test", reason="socket disconnected")
    assert res.status == STATUS_NOT_AVAILABLE
    assert res.payload["prediction"] is None
    assert res.payload["control_command"] is None
    assert "transport failure" in res.payload["reason"]

    transport_req = _sample_envelope(
        request_id="req-transport",
        payload={"transport_failure": True},
    )
    handled_res = gateway.handle(transport_req)
    assert handled_res.status == STATUS_NOT_AVAILABLE
    assert handled_res.payload["prediction"] is None
    assert handled_res.payload["control_command"] is None


def test_synthetic_fake_model_labelled_and_cannot_answer_product_request():
    gateway = Gateway()

    synth_req = _sample_envelope(
        request_id="req-synth-ok",
        model_version="synthetic-smoke-v1",
        evidence_level=EVIDENCE_SYNTHETIC,
        payload={"model_kind": "synthetic"},
    )
    synth_res = gateway.handle(synth_req)
    assert synth_res.evidence_level == EVIDENCE_SYNTHETIC

    product_req_1 = _sample_envelope(
        request_id="req-prod-1",
        model_version="synthetic-smoke-v1",
        payload={"is_product": True},
    )
    res_p1 = gateway.handle(product_req_1)
    assert res_p1.status == STATUS_REJECTED
    assert "synthetic model cannot answer a real product request" in res_p1.payload["reason"]

    product_req_2 = _sample_envelope(
        request_id="req-prod-2",
        model_version="synthetic-smoke-v1",
        display_gate="PRODUCT_APPROVED",
    )
    res_p2 = gateway.handle(product_req_2)
    assert res_p2.status == STATUS_REJECTED

    product_req_3 = _sample_envelope(
        request_id="req-prod-3",
        model_version="synthetic-smoke-v1",
        evidence_level=EVIDENCE_LOCAL,
    )
    res_p3 = gateway.handle(product_req_3)
    assert res_p3.status == STATUS_REJECTED


def test_gate_bundle_independence():
    passed_engineering = gate_bundle(True)
    assert passed_engineering[GATE_ENGINEERING] == GATE_GRANTED
    assert passed_engineering[GATE_REAL_DATA_TRAINING] == GATE_NOT_GRANTED
    assert passed_engineering[GATE_RESEARCH_CLAIMS] == GATE_NOT_GRANTED

    failed_engineering = gate_bundle(False)
    assert failed_engineering[GATE_ENGINEERING] == GATE_NOT_GRANTED
    assert failed_engineering[GATE_REAL_DATA_TRAINING] == GATE_NOT_GRANTED
    assert failed_engineering[GATE_RESEARCH_CLAIMS] == GATE_NOT_GRANTED
