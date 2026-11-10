"""#105 incremental idempotent pipeline: determinism, resume, drift, gates."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from lifecycle.contracts import (
    APPROVAL_APPROVED,
    STATUS_ABSTAIN,
    STATUS_BLOCKED,
    STATUS_OK,
    TRUST_TRUSTED,
    canonical_json,
)
from lifecycle.pipeline import (
    PIPELINE_VERSION,
    QC_FILE,
    QUARANTINE_FILE,
    READY_FILE,
    SNAPSHOT_FILE,
    STATE_FILE,
    PipelineConfig,
    PipelineError,
    dataset_manifest_from_run,
    declare_batch,
    declare_source,
    run_pipeline,
)

pytestmark = pytest.mark.synthetic

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "lifecycle"
BATCH_A = FIXTURES / "batch_a.csv"
BATCH_B = FIXTURES / "batch_b.csv"


def _write_batch(path: Path, rows: list[dict]) -> None:
    headers = [
        "batch_id", "record_id", "session_id", "subject_ref", "phase", "metric",
        "value", "unit", "quality", "source_record", "event_time", "received_at", "origin",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _row(
    record_id: str,
    session: str,
    subject: str,
    metric: str,
    value: str,
    unit: str,
    *,
    phase: str = "pre",
    quality: str = "ok",
    origin: str = "measured",
    event_time: str = "2026-11-01T09:00:00+00:00",
) -> dict:
    return {
        "batch_id": "A",
        "record_id": record_id,
        "session_id": session,
        "subject_ref": subject,
        "phase": phase,
        "metric": metric,
        "value": value,
        "unit": unit,
        "quality": quality,
        "source_record": f"src-{record_id}",
        "event_time": event_time,
        "received_at": event_time,
        "origin": origin,
    }


def _base_rows() -> list[dict]:
    return [
        _row("1", "S1", "P1", "heart_rate", "70", "bpm"),
        _row("2", "S1", "P1", "spo2", "97", "%"),
        _row("3", "S2", "P2", "heart_rate", "72", "bpm"),
        _row("4", "S2", "P2", "spo2", "96", "%"),
    ]


def _run(tmp_path, batches, **kwargs):
    return run_pipeline(batches, tmp_path / "run", **kwargs)


def test_fixtures_exist_and_are_read_only_sources():
    assert BATCH_A.is_file() and BATCH_B.is_file()
    for path in (BATCH_A, BATCH_B):
        batch = declare_batch(path)
        source = declare_source(batch)
        assert source.kind == "offline"
        assert source.capability == "UNAVAILABLE"
        assert source.sha256 == batch.sha256


def test_declare_batch_hashes_the_raw_bytes(tmp_path):
    path = tmp_path / "b.csv"
    _write_batch(path, _base_rows())
    batch = declare_batch(path)
    assert len(batch.sha256) == 64
    raw = path.read_bytes()
    path.write_bytes(raw)
    assert declare_batch(path).sha256 == batch.sha256


def test_missing_batch_is_refused(tmp_path):
    with pytest.raises(PipelineError, match="not found"):
        declare_batch(tmp_path / "absent.csv")


def test_repeated_run_is_byte_identical_and_has_no_duplicates(tmp_path):
    batch = declare_batch(BATCH_A)
    first = _run(tmp_path, [batch])
    out = tmp_path / "run"
    first_state = (out / STATE_FILE).read_bytes()
    first_snapshot = (out / SNAPSHOT_FILE).read_bytes()
    first_qc = (out / QC_FILE).read_bytes()

    second = _run(tmp_path, [batch])
    assert second.reused_snapshot is True
    assert second.qc.n_duplicate == 0
    assert second.qc.sha256 == first.qc.sha256
    assert second.snapshot.sha256 == first.snapshot.sha256
    assert (out / STATE_FILE).read_bytes() == first_state
    assert (out / SNAPSHOT_FILE).read_bytes() == first_snapshot
    assert (out / QC_FILE).read_bytes() == first_qc


def test_resend_of_the_same_batch_is_idempotent(tmp_path):
    batch = declare_batch(BATCH_A)
    first = _run(tmp_path, [batch])
    # Explicit resend with a fresh run_id: still no new rows.
    second = run_pipeline([batch], tmp_path / "run", run_id="resend", resume=False)
    assert second.qc.n_accepted == first.qc.n_accepted
    assert second.qc.n_input == first.qc.n_input
    assert second.snapshot.sha256 == first.snapshot.sha256


def test_out_of_order_and_incremental_batches_converge(tmp_path):
    a, b = declare_batch(BATCH_A), declare_batch(BATCH_B)
    forward = _run(tmp_path / "fwd", [a, b])
    reverse = _run(tmp_path / "rev", [b, a])
    assert forward.snapshot.sha256 == reverse.snapshot.sha256
    assert forward.qc.sha256 == reverse.qc.sha256
    assert forward.snapshot.rows == reverse.snapshot.rows


def test_late_batch_updates_the_snapshot_without_recomputing_earlier_rows(tmp_path):
    a, b = declare_batch(BATCH_A), declare_batch(BATCH_B)
    first = run_pipeline([a], tmp_path / "inc")
    assert not first.reused_snapshot
    second = run_pipeline([a, b], tmp_path / "inc")
    assert second.reused_snapshot is False
    assert second.snapshot.sha256 != first.snapshot.sha256
    assert second.qc.n_accepted > first.qc.n_accepted
    assert "batch_b" in second.batches_new


def test_config_or_code_change_forces_recompute(tmp_path):
    batch = declare_batch(BATCH_A)
    run_pipeline([batch], tmp_path / "cache")
    again = run_pipeline([batch], tmp_path / "cache")
    assert again.reused_snapshot is True
    changed = run_pipeline(
        [batch], tmp_path / "cache", config=PipelineConfig(min_subjects=3)
    )
    assert changed.reused_snapshot is False
    code_changed = run_pipeline([batch], tmp_path / "cache", code_version="v2")
    assert code_changed.reused_snapshot is False


def test_source_hash_change_forces_recompute(tmp_path):
    path = tmp_path / "mutable.csv"
    _write_batch(path, _base_rows())
    first = run_pipeline([declare_batch(path)], tmp_path / "hash")
    _write_batch(path, _base_rows() + [_row("5", "S3", "P3", "heart_rate", "75", "bpm")])
    second = run_pipeline([declare_batch(path)], tmp_path / "hash")
    assert second.reused_snapshot is False
    assert second.qc.n_accepted == first.qc.n_accepted + 1


def test_failure_injection_records_a_failed_state_and_can_resume(tmp_path):
    batch = declare_batch(BATCH_A)
    failed = run_pipeline([batch], tmp_path / "fail", fail_stage="validate")
    assert failed.status == STATUS_BLOCKED
    assert failed.error and "injected" in failed.error
    assert failed.state.status == "FAILED"
    assert failed.snapshot is None

    recovered = run_pipeline([batch], tmp_path / "fail")
    assert recovered.status != STATUS_BLOCKED
    assert recovered.snapshot is not None
    assert recovered.state.status in ("DONE", "PAUSED")
    assert recovered.error == ""


def test_bad_units_values_and_quality_are_quarantined_with_reasons(tmp_path):
    path = tmp_path / "bad.csv"
    _write_batch(
        path,
        [
            _row("1", "S1", "P1", "heart_rate", "70", "bpm"),
            _row("2", "S1", "P1", "heart_rate", "70", "mmHg"),  # wrong unit
            _row("3", "S1", "P1", "heart_rate", "999", "bpm"),  # out of range
            _row("4", "S1", "P1", "heart_rate", "abc", "bpm"),  # non-numeric
            _row("5", "S1", "P1", "unknown_metric", "1", "x"),  # unknown metric
            _row("6", "S1", "P1", "spo2", "97", "%", quality="stale"),
        ],
    )
    result = run_pipeline([declare_batch(path)], tmp_path / "bad")
    assert result.qc.n_rejected == 5
    assert result.qc.n_accepted == 1
    assert len(result.quarantine) == 5
    assert all(q["reason"] for q in result.quarantine)
    quarantine = (tmp_path / "bad" / QUARANTINE_FILE).read_text(encoding="utf-8")
    assert "unit mismatch" in quarantine
    assert "plausible range" in quarantine


def test_schema_drift_is_blocked_not_silently_dropped(tmp_path):
    path = tmp_path / "drift.csv"
    path.write_text(
        "batch_id,record_id,session_id,subject_ref,phase,metric,value,unit,quality\n"
        "A,1,S1,P1,pre,heart_rate,70,bpm,ok\n",
        encoding="utf-8",
    )
    result = run_pipeline([declare_batch(path)], tmp_path / "drift")
    assert result.status == STATUS_BLOCKED
    assert any("missing required columns" in b for b in result.qc.blocked)
    assert "source_record" in " ".join(result.qc.blocked)


def test_unknown_extra_column_is_drift(tmp_path):
    path = tmp_path / "extra.csv"
    rows = _base_rows()
    headers = list(rows[0]) + ["mystery_field"]
    for row in rows:
        row["mystery_field"] = "1"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    result = run_pipeline([declare_batch(path)], tmp_path / "extra")
    assert any("unknown columns" in d for d in result.qc.drift)


def test_cross_subject_session_mixing_is_blocked(tmp_path):
    path = tmp_path / "mixed.csv"
    _write_batch(
        path,
        [
            _row("1", "S1", "P1", "heart_rate", "70", "bpm"),
            _row("2", "S1", "P2", "heart_rate", "71", "bpm"),
        ],
    )
    result = run_pipeline([declare_batch(path)], tmp_path / "mixed")
    assert any("mixes subjects" in b for b in result.qc.blocked)
    assert result.readiness["train_ready"] is False


def test_ready_gate_never_auto_admits_training(tmp_path):
    batch = declare_batch(BATCH_A)
    result = _run(tmp_path, [batch])
    assert result.readiness["auto_train"] is False
    assert result.readiness["train_ready"] is False
    assert result.status == STATUS_ABSTAIN
    assert any("approval" in r for r in result.readiness["reasons"])
    assert any("trust" in r or "identity" in r for r in result.readiness["reasons"])


def test_ready_gate_still_refuses_smoke_and_unfrozen_snapshots(tmp_path):
    batch = declare_batch(BATCH_A)
    result = run_pipeline(
        [batch], tmp_path / "ready",
        approval=APPROVAL_APPROVED,
        identity_trust=TRUST_TRUSTED,
        frozen=False,
        evidence_level="SMOKE_ONLY",
    )
    assert result.readiness["train_ready"] is False
    assert any("not frozen" in r for r in result.readiness["reasons"])
    assert any("SMOKE_ONLY" in r for r in result.readiness["reasons"])


def test_ready_gate_can_admit_only_an_approved_frozen_trusted_snapshot(tmp_path):
    batch = declare_batch(BATCH_A)
    result = run_pipeline(
        [batch], tmp_path / "admit",
        approval=APPROVAL_APPROVED,
        identity_trust=TRUST_TRUSTED,
        frozen=True,
        evidence_level="SYNTHETIC",
    )
    assert result.readiness["train_ready"] is True
    assert result.readiness["status"] == STATUS_OK
    assert result.status == STATUS_OK


def test_dataset_manifest_trainable_gate(tmp_path):
    batch = declare_batch(BATCH_A)
    result = _run(tmp_path, [batch])
    manifest = dataset_manifest_from_run(
        result, dataset_id="ds-1", label_version="l1", feature_version="f1"
    )
    ok, reason = manifest.trainable()
    assert ok is False
    assert "approval" in reason or "identity" in reason
    approved = dataset_manifest_from_run(
        result, dataset_id="ds-1", label_version="l1", feature_version="f1",
        approval=APPROVAL_APPROVED, identity_trust=TRUST_TRUSTED, frozen=True,
        evidence_level="SYNTHETIC",
    )
    assert approved.trainable() == (True, "admitted")
    assert approved.sha256 != manifest.sha256


def test_state_cache_key_depends_on_all_declared_inputs(tmp_path):
    batch = declare_batch(BATCH_A)
    base = run_pipeline([batch], tmp_path / "key1")
    changed_rules = run_pipeline(
        [batch], tmp_path / "key2", rules_version="rules-v2"
    )
    assert base.state.cache_key != changed_rules.state.cache_key
    assert base.state.dataset_sha256 == changed_rules.state.dataset_sha256


def test_raw_input_is_never_modified(tmp_path):
    before = BATCH_A.read_bytes()
    result = _run(tmp_path, [declare_batch(BATCH_A)])
    assert BATCH_A.read_bytes() == before
    assert result.snapshot.source_sha256 == (declare_batch(BATCH_A).sha256,)


def test_run_result_serialises_to_canonical_json(tmp_path):
    result = _run(tmp_path, [declare_batch(BATCH_A)])
    payload = json.loads(canonical_json(result.to_dict()))
    assert payload["schema"] == PIPELINE_VERSION
    assert payload["reused_snapshot"] is False
    assert payload["readiness"]["auto_train"] is False
    out = tmp_path / "run"
    for name in (STATE_FILE, SNAPSHOT_FILE, QC_FILE, READY_FILE, QUARANTINE_FILE):
        assert (out / name).is_file(), name


def test_batch_id_override_is_stable(tmp_path):
    explicit = declare_batch(BATCH_A, batch_id="explicit")
    assert explicit.batch_id == "explicit"
    assert explicit.sha256 == declare_batch(BATCH_A).sha256


def test_config_change_never_empties_a_good_snapshot(tmp_path):
    """S2: a changed dependency must re-ingest, not skip every batch.

    Reverting the resume guard to a dataset-only check makes this fail with
    0 accepted rows and an empty snapshot overwriting the good one.
    """
    a, b = declare_batch(BATCH_A), declare_batch(BATCH_B)
    first = run_pipeline([a, b], tmp_path / "deps")
    changed = run_pipeline(
        [a, b], tmp_path / "deps", config=PipelineConfig(min_subjects=3)
    )
    assert changed.qc.n_accepted == first.qc.n_accepted
    assert changed.snapshot.rows == first.snapshot.rows
    # The snapshot binds its config, so a changed dependency legitimately
    # produces a different hash; what must NOT change is the row content.
    assert changed.snapshot.sha256 != first.snapshot.sha256
    assert changed.batches_new == ("batch_a", "batch_b")


def test_code_version_change_never_empties_a_good_snapshot(tmp_path):
    a = declare_batch(BATCH_A)
    first = run_pipeline([a], tmp_path / "code")
    changed = run_pipeline([a], tmp_path / "code", code_version="v2")
    assert changed.qc.n_accepted == first.qc.n_accepted
    assert changed.snapshot.rows == first.snapshot.rows


def test_reissued_batch_with_different_bytes_is_not_silently_dropped(tmp_path):
    """S3: same batch_id, different content is a correction, not a duplicate."""
    path = tmp_path / "reissued.csv"
    original = BATCH_A.read_text(encoding="utf-8").rstrip("\n")
    extra = (
        "A,99,S9,P9,pre,spo2,98,%,ok,src-99,"
        "2026-11-01T09:00:00+00:00,2026-11-01T09:00:00+00:00,measured"
    )
    path.write_text(original + "\n" + extra + "\n", encoding="utf-8")
    first = declare_batch(BATCH_A, batch_id="same")
    second = declare_batch(path, batch_id="same")
    result = run_pipeline([first, second], tmp_path / "reissue")
    assert first.sha256 != second.sha256
    assert result.qc.n_accepted == 5
    assert result.qc.n_input == result.qc.n_accepted + result.qc.n_rejected
    assert any("id collision with different content" in q["reason"] for q in result.quarantine)
    assert any("re-issued with a different sha256" in d for d in result.qc.drift)


def test_corrupt_cached_qc_falls_back_to_full_recompute(tmp_path):
    """S9: a corrupt QC file must not crash the pipeline."""
    batch = declare_batch(BATCH_A)
    first = run_pipeline([batch], tmp_path / "qc")
    (tmp_path / "qc" / QC_FILE).write_text("not json", encoding="utf-8")
    recovered = run_pipeline([batch], tmp_path / "qc")
    assert recovered.snapshot is not None
    assert recovered.snapshot.rows == first.snapshot.rows
    assert recovered.qc.n_accepted == first.qc.n_accepted
    assert any("cached QC unreadable" in d for d in recovered.qc.drift)


def test_state_file_is_deterministic_without_an_explicit_run_id(tmp_path):
    """S10: no wall clock in the persisted state on a resumed run."""
    batch = declare_batch(BATCH_A)
    first = run_pipeline([batch], tmp_path / "clock")
    state = (tmp_path / "clock" / STATE_FILE).read_bytes()
    second = run_pipeline([batch], tmp_path / "clock")
    assert second.state.run_id == first.state.run_id
    assert second.state.updated_at == first.state.updated_at
    assert (tmp_path / "clock" / STATE_FILE).read_bytes() == state


def test_subject_spanning_sessions_is_a_declared_deferral_not_contamination(tmp_path):
    """S11: one subject with several sessions is valid but identity is unverified."""
    path = tmp_path / "multi.csv"
    rows = _base_rows()
    rows[2]["session_id"] = "S1-extra"  # P2 appears in a second session
    _write_batch(path, rows)
    result = run_pipeline([declare_batch(path)], tmp_path / "multi")
    assert any("span multiple sessions" in d for d in result.qc.drift)
    assert not any("mixes subjects" in b for b in result.qc.blocked)
