"""#111 stage observability: privacy, tamper evidence, measured reports, gates."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from lifecycle.contracts import (
    EVIDENCE_SYNTHETIC,
    GATE_ENGINEERING,
    GATE_GRANTED,
    GATE_NOT_GRANTED,
    GATE_REAL_DATA_TRAINING,
    GATE_RESEARCH_CLAIMS,
    QCReport,
    StageEvidence,
    canonical_json,
    digest,
)
from lifecycle.observe import (
    DECLARED,
    GENESIS_SHA256,
    HUMAN_APPROVAL_REQUIRED,
    MEASURED,
    OBSERVE_VERSION,
    ObservabilityStore,
    ObserveError,
    StageEvent,
    drift_check,
    experiment_record,
    on_new_data,
    readiness_report,
)
from lifecycle.training import DECISION_CRASH, DECISION_KEEP, DECISION_REVERT

pytestmark = pytest.mark.synthetic

EVENT_TIME = "2026-01-01T00:00:00+00:00"
MEASURED_AT = "2026-01-01T02:00:00+00:00"


def _event(
    stage: str = "ingest",
    status: str = "OK",
    counts: dict[str, float] | None = None,
    reason: str = "synthetic aggregate observation",
    event_time: str = EVENT_TIME,
) -> StageEvent:
    return StageEvent(
        stage=stage,
        run_id="synthetic-run-1",
        event_time=event_time,
        status=status,
        counts={"n_rows": 2} if counts is None else counts,
        reason=reason,
        evidence_level=EVIDENCE_SYNTHETIC,
    )


def _evidence(stage: str, go: bool, gate: str = GATE_GRANTED) -> StageEvidence:
    return StageEvidence(
        stage=stage,
        gate=gate,
        go=go,
        evidence_level=EVIDENCE_SYNTHETIC,
        reason="synthetic stage evidence",
    )


def _qc(n_input: int, n_rejected: int, n_duplicate: int, drift: tuple[str, ...]) -> QCReport:
    return QCReport(
        n_input=n_input,
        n_accepted=n_input - n_rejected,
        n_rejected=n_rejected,
        n_duplicate=n_duplicate,
        n_imputed=0,
        drift=drift,
    )


def test_stage_event_refuses_raw_identifiers_and_per_row_values():
    aggregate = _event(counts={"n_rows": 12, "n_refused": 2, "latency_ms_p95": 3.5})
    assert aggregate.validate() is None
    assert aggregate.to_dict()["counts_source"] == MEASURED
    assert aggregate.to_dict()["event_time_source"] == DECLARED
    assert aggregate.to_dict()["schema_version"] == OBSERVE_VERSION

    rejected = (
        {"subject_id": 1},
        {"patient_count": 1},
        {"session_key": "abc"},
        {"raw_heart_rate": 70},
        {"heart_rate": 70},
        {"physiology_value": 1.2},
        {"n_rows_per_subject": [1, 2, 3]},
        {"n_rows": {"row_1": 1, "row_2": 2}},
        {"n_rows": "12"},
        {"n_rows": -1},
        {"count": 1},
    )
    for counts in rejected:
        with pytest.raises(ObserveError):
            _event(counts=counts)

    with pytest.raises(ObserveError):
        replace(_event(), run_id="subject 42")
    with pytest.raises(ObserveError):
        replace(_event(), reason="raw physiology for one subject")
    with pytest.raises(ObserveError):
        replace(_event(), counts_source=DECLARED)


def test_append_only_chain_verifies_and_detects_tampering_of_every_past_line(tmp_path):
    path = tmp_path / "events.jsonl"
    store = ObservabilityStore(path)
    hashes = [
        store.append(_event(stage="ingest", counts={"n_rows": 1})),
        store.append(_event(stage="validate", counts={"n_rows": 2})),
        store.append(_event(stage="observe", counts={"n_rows": 3})),
    ]
    assert len(set(hashes)) == 3
    assert store.verify_chain() == {
        "verified": True,
        "n_events": 3,
        "head_sha256": hashes[-1],
    }
    lines = path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    assert [record["previous_sha256"] for record in records[1:]] == hashes[:2]
    assert all(
        record["sha256"]
        == digest({"event": record["event"], "previous_sha256": record["previous_sha256"]})
        for record in records
    )

    for index in range(len(lines)):
        mutated = list(records)
        changed = json.loads(json.dumps(mutated[index]))
        changed["event"]["counts"]["n_rows"] += 100
        mutated[index] = changed
        path.write_text(
            "".join(canonical_json(record) + "\n" for record in mutated), encoding="utf-8"
        )
        with pytest.raises(ObserveError, match="hash chain broken at line"):
            ObservabilityStore(path).verify_chain()

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    replaced = [json.loads(line) for line in lines]
    replaced[0]["sha256"] = GENESIS_SHA256
    path.write_text(
        "".join(canonical_json(record) + "\n" for record in replaced),
        encoding="utf-8",
    )
    with pytest.raises(ObserveError, match="hash chain broken at line 1"):
        ObservabilityStore(path).verify_chain()


def test_summary_counts_rates_rollback_and_staleness_are_measured_from_events(tmp_path):
    store = ObservabilityStore(tmp_path / "events.jsonl")
    store.append(
        _event(
            counts={"n_rows": 4, "n_refused": 1, "n_version_drift": 1, "n_rollback": 2},
            reason="first measured observation",
        )
    )
    store.append(
        _event(
            status="ABSTAIN",
            counts={"n_rows": 3, "n_abstain": 1},
            reason="second measured observation",
            event_time="2026-01-01T00:30:00+00:00",
        )
    )
    store.append(_event(status="REJECTED", counts={"n_rows": 2}))
    store.append(
        _event(stage="serve", status="NOT_AVAILABLE", event_time="2026-01-01T01:00:00+00:00")
    )

    first = store.summary(measured_at=MEASURED_AT)
    ingest = first["stages"]["ingest"]
    assert first["n_events"] == 4
    assert first["measurement"] == MEASURED
    assert ingest["n_events"] == 3
    assert ingest["availability"] == {"n_available": 1, "n_unavailable": 2}
    assert ingest["n_refused"] == 2
    assert ingest["refusal_rate"] == 2 / 3
    assert ingest["n_abstain"] == 2
    assert ingest["abstain_rate"] == 2 / 3
    assert ingest["n_version_drift"] == 1
    assert ingest["version_drift_rate"] == 1 / 3
    assert ingest["rollback_count"] == 2
    assert ingest["staleness"] == {
        "measurement": MEASURED,
        "latest_event_time": "2026-01-01T00:30:00+00:00",
        "measured_at": MEASURED_AT,
        "lag_s": 5400.0,
        "clock_skew": False,
        "marker": "MEASURED_LAG_FROM_DECLARED_EVENT_TIME",
    }
    assert "threshold" in first["note"]

    store.append(_event(status="ABSTAIN", counts={"n_refused": 1, "n_rollback": 1}))
    second = store.summary(measured_at=MEASURED_AT)["stages"]["ingest"]
    assert second["n_events"] == 4
    assert second["availability"] == {"n_available": 1, "n_unavailable": 3}
    assert second["n_refused"] == 3
    assert second["refusal_rate"] == 3 / 4
    assert second["n_abstain"] == 3
    assert second["abstain_rate"] == 3 / 4
    assert second["rollback_count"] == 3
    assert second["staleness"]["lag_s"] == 5400.0

    empty = ObservabilityStore(tmp_path / "empty.jsonl").summary()
    assert empty["n_events"] == 0
    assert empty["stages"] == {}
    assert "threshold" in empty["note"]


def test_staleness_is_a_measured_marker_without_a_verdict(tmp_path):
    store = ObservabilityStore(tmp_path / "events.jsonl")
    no_events = store.summary()["stages"]
    assert no_events == {}
    store.append(_event())
    declared = store.summary()["stages"]["ingest"]["staleness"]
    assert declared == {
        "measurement": MEASURED,
        "latest_event_time": EVENT_TIME,
        "measured_at": None,
        "lag_s": None,
        "marker": "NO_MEASUREMENT_TIME_DECLARED",
    }
    measured = store.summary(measured_at=MEASURED_AT)["stages"]["ingest"]["staleness"]
    assert measured["marker"] == "MEASURED_LAG_FROM_DECLARED_EVENT_TIME"
    assert measured["lag_s"] == 7200.0
    assert "PASS" not in canonical_json(measured)
    assert "FAIL" not in canonical_json(measured)


def test_readiness_report_separates_engineering_from_training_and_claims():
    engineering = readiness_report(
        [_evidence("ingest", True), _evidence("validate", True), _evidence("features", True)]
    )
    assert engineering["decision"] == "GO"
    assert engineering["gates"] == {
        GATE_ENGINEERING: GATE_GRANTED,
        GATE_REAL_DATA_TRAINING: GATE_NOT_GRANTED,
        GATE_RESEARCH_CLAIMS: GATE_NOT_GRANTED,
    }
    assert any(GATE_REAL_DATA_TRAINING in reason for reason in engineering["reasons"])
    assert any(GATE_RESEARCH_CLAIMS in reason for reason in engineering["reasons"])

    blocked = readiness_report([_evidence("ingest", False)])
    assert blocked["decision"] == "NO-GO"
    assert blocked["gates"][GATE_ENGINEERING] == GATE_NOT_GRANTED
    assert blocked["gates"][GATE_REAL_DATA_TRAINING] == GATE_NOT_GRANTED
    assert blocked["gates"][GATE_RESEARCH_CLAIMS] == GATE_NOT_GRANTED
    assert "ingest: NO-GO" in blocked["reasons"][0]

    missing_gate = readiness_report([_evidence("features", True, gate="BLOCKED_BY_DATA_OR_DEVICE")])
    assert missing_gate["decision"] == "NO-GO"
    assert missing_gate["gates"][GATE_ENGINEERING] == GATE_NOT_GRANTED
    assert readiness_report(())["decision"] == "NO-GO"


def test_on_new_data_is_a_plan_with_no_training_promotion_or_publishing(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    before = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    plan = on_new_data(
        batch_ids=["synthetic-batch-a", "synthetic-batch-b"],
        qc=_qc(100, 10, 5, ()),
        previous_qc=_qc(80, 8, 4, ()),
        evidence=[_evidence("ingest", True)],
    )
    after = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    assert before == after == set()
    assert plan["trigger"] == "DATA_UPDATE"
    assert plan["auto_train"] is False
    assert plan["auto_promote"] is False
    assert plan["auto_publish"] is False
    assert plan["candidate_preparation"]["auto_train"] is False
    assert plan["reason"] == HUMAN_APPROVAL_REQUIRED
    assert "human approval" in plan["reason"]
    assert set(plan["allowed_stages"]) == {"ingest", "validate", "features"}
    assert "readiness report" in plan["refresh_plan"]
    assert plan["executed"] is False
    assert "train" not in plan["allowed_stages"]
    assert "promote" not in plan["allowed_stages"]
    assert "deploy" not in plan["allowed_stages"]
    assert plan["gates"][GATE_REAL_DATA_TRAINING] == GATE_NOT_GRANTED
    assert plan["gates"][GATE_RESEARCH_CLAIMS] == GATE_NOT_GRANTED
    assert plan["candidate_preparation"]["allowed"] is False
    assert plan["drift"]["observation"] == "DRIFT_OBSERVED"

    approved = on_new_data(
        batch_ids=["synthetic-batch-a"],
        dataset_approval="APPROVED",
        frozen=True,
    )
    assert approved["candidate_preparation"]["allowed"] is True
    assert approved["auto_train"] is False
    assert approved["auto_promote"] is False
    assert approved["auto_publish"] is False
    assert approved["candidate_preparation"]["reason"] == HUMAN_APPROVAL_REQUIRED


def test_on_new_data_writes_no_log_or_lifecycle_artifacts(tmp_path):
    first = on_new_data(batch_ids=["synthetic-batch-a"], evidence=[_evidence("ingest", True)])
    second = on_new_data(batch_ids=["synthetic-batch-a"], evidence=[_evidence("ingest", True)])
    assert first == second
    assert list(tmp_path.iterdir()) == []
    assert first["executed"] is False
    assert first["readiness"]["gates"][GATE_ENGINEERING] == GATE_GRANTED


def test_drift_check_reports_measured_deltas_without_inventing_a_verdict():
    previous = _qc(100, 10, 5, ("synthetic-batch-a: unknown columns ['age']",))
    current = _qc(
        120,
        18,
        12,
        ("synthetic-batch-b: unknown columns ['age', 'bmi']",),
    )
    report = drift_check(previous, current)
    assert report["observation"] == "DRIFT_OBSERVED"
    assert report["measurement"] == MEASURED
    assert report["input_volume"] == {"baseline": 100, "candidate": 120, "delta": 20}
    assert report["reject_rate"]["baseline"] == 0.1
    assert report["reject_rate"]["candidate"] == 0.15
    assert report["reject_rate"]["delta"] == pytest.approx(0.05)
    assert report["duplicate_rate"]["baseline"] == 0.05
    assert report["duplicate_rate"]["candidate"] == 0.1
    assert report["duplicate_rate"]["delta"] == pytest.approx(0.05)
    assert report["unknown_columns"] == {
        "baseline": ["age"],
        "candidate": ["age", "bmi"],
        "delta": 1,
        "new": ["bmi"],
    }
    assert report["verdict"] == "NOT_EVALUATED"
    assert "no validated drift threshold" in report["note"]
    assert "PASS" not in canonical_json(report)
    assert "FAIL" not in canonical_json(report)
    with pytest.raises(ObserveError):
        drift_check({}, current)


def test_experiment_records_are_durable_reasoned_and_never_silently_retried(tmp_path):
    path = tmp_path / "results.tsv"
    for decision, run_id, reason in (
        (DECISION_KEEP, "synthetic-run-1", "measured synthetic lift"),
        (DECISION_REVERT, "synthetic-run-2", "no measured synthetic lift"),
        (DECISION_CRASH, "synthetic-run-3", "synthetic evaluator exception"),
    ):
        record = experiment_record(decision, run_id=run_id, reason=reason, path=path)
        assert record["decision"] == decision
        assert record["run_id"] == run_id
        assert record["reason"] == reason
        assert record["executed"] is False
        assert record["retried"] is False
        recomputed = digest({key: value for key, value in record.items() if key != "sha256"})
        assert record["sha256"] == recomputed

    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [line["decision"] for line in lines] == [
        DECISION_KEEP,
        DECISION_REVERT,
        DECISION_CRASH,
    ]
    assert all(line["executed"] is False for line in lines)
    assert all(line["retried"] is False for line in lines)
    assert all(line["reason"].strip() for line in lines)
    crash_lines = [line for line in lines if line["decision"] == DECISION_CRASH]
    assert len(crash_lines) == 1
    assert crash_lines[0]["run_id"] == "synthetic-run-3"
    with pytest.raises(ObserveError):
        experiment_record("RETRY", run_id="synthetic-run-3", reason="retry", path=path)
    with pytest.raises(ObserveError):
        experiment_record(DECISION_KEEP, run_id="", reason="missing run", path=path)
    with pytest.raises(ObserveError):
        experiment_record(DECISION_KEEP, run_id="synthetic-run-4", reason=" ", path=path)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 3


def test_negative_lag_is_reported_as_clock_skew(tmp_path):
    """G4: a client clock ahead of the measurement clock is not a negative age."""
    from lifecycle.observe import _staleness

    report = {"declared_event_times": ["2026-01-01T03:00:00+00:00"]}
    result = _staleness(report, "2026-01-01T02:00:00+00:00")
    assert result["lag_s"] == 0.0
    assert result["clock_skew"] is True
    assert result["marker"] == "CLOCK_SKEW_DETECTED"
