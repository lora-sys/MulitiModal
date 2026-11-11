"""Regressions for the post-merge Codex findings on PR #116 and PR #117.

Each test reproduces the reported failure mode and asserts the fixed behaviour.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from lifecycle import gateway as gw
from lifecycle import observe as ob
from lifecycle import pipeline as pl
from lifecycle.contracts import QCReport, digest
from lifecycle.pipeline import declare_batch, run_pipeline

pytestmark = pytest.mark.synthetic
LEGACY_PIPELINE_VERSION = "ml-lifecycle-pipeline-v1"

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "lifecycle"
BATCH_A = FIXTURES / "batch_a.csv"


def _qc(drift):
    return QCReport(
        n_input=10, n_accepted=10, n_rejected=0, n_duplicate=0, n_imputed=0,
        rejections=(), drift=tuple(drift), blocked=(),
    )


def _equivalent_offset_batch(path: Path) -> object:
    rows = list(csv.DictReader(BATCH_A.open(encoding="utf-8")))
    resent = dict(rows[0])
    resent["event_time"] = "2026-11-01T10:00:00+01:00"
    resent["record_id"] = "5"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows + [resent])
    return declare_batch(path, batch_id="A")


def test_equivalent_utc_offsets_share_one_dedup_key(tmp_path):
    """F1: `09:00Z` and `10:00+01:00` are the same instant; one row must survive.

    Regresses pipeline.py `_row_key`: raw timestamp strings used to produce two
    keys, double-counting the observation and flipping `train_ready`.
    """
    batch = _equivalent_offset_batch(tmp_path / "resent.csv")
    result = run_pipeline([batch], tmp_path / "out")
    assert result.qc.n_input == 5
    assert result.qc.n_accepted == 4
    assert result.qc.n_duplicate == 1
    assert result.readiness["train_ready"] is False


def test_legacy_snapshot_is_invalidated_after_timestamp_identity_change(
    tmp_path,
):
    """P1: a v1 snapshot must not survive the changed row identity semantics.

    The old pipeline keyed `09:00Z` and `10:00+01:00` as separate rows. A
    default v2 run has to recompute that cache, not reuse the double-counted
    snapshot.
    """
    batch = _equivalent_offset_batch(tmp_path / "resent.csv")
    with patch.object(pl, "_canonical_event_time", lambda value: value):
        legacy = run_pipeline(
            [batch], tmp_path / "out", code_version=LEGACY_PIPELINE_VERSION
        )
    assert legacy.qc.n_accepted == 5
    assert legacy.qc.n_duplicate == 0

    upgraded = run_pipeline([batch], tmp_path / "out")
    assert upgraded.reused_snapshot is False
    assert upgraded.qc.n_accepted == 4
    assert upgraded.qc.n_duplicate == 1


def test_distinct_instants_still_get_distinct_keys(tmp_path):
    """F1 guard: normalization must not merge genuinely different timestamps."""
    rows = list(csv.DictReader(BATCH_A.open(encoding="utf-8")))
    extra = dict(rows[0])
    extra["event_time"] = "2026-11-01T11:00:00+00:00"
    extra["record_id"] = "5"
    path = tmp_path / "later.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows + [extra])
    result = run_pipeline([declare_batch(path, batch_id="A")], tmp_path / "out")
    assert result.qc.n_accepted == 5
    assert result.qc.n_duplicate == 0


def test_malformed_event_time_is_not_deduped_silently(tmp_path):
    """F1 guard: unparseable timestamps stay distinct instead of being merged."""
    rows = list(csv.DictReader(BATCH_A.open(encoding="utf-8")))
    bad = dict(rows[0])
    bad["event_time"] = "not-a-time"
    bad["record_id"] = "5"
    path = tmp_path / "bad.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows + [bad])
    result = run_pipeline([declare_batch(path, batch_id="A")], tmp_path / "out")
    assert result.qc.n_accepted == 5
    assert result.qc.n_duplicate == 0


def test_event_time_is_part_of_cached_request_identity():
    """F2: a changed measurement timestamp must not reuse a cached OK.

    Regresses gateway.py `_request_identity`: without event_time, an old
    timestamp collided with the cache and bypassed the staleness check.
    """
    common = dict(request_id="same", session_id="S1", trace_id="T1",
                  mode=gw.MODE_C_TCM, payload={"task": "knowledge"})
    fresh = gw.make_envelope(**common)
    old = gw.make_envelope(**common, event_time="2000-01-01T00:00:00+00:00")
    fresh_old = gw.make_envelope(**{**common, "request_id": "other"},
                                 event_time="2000-01-01T00:00:00+00:00")
    gateway = gw.Gateway()
    assert gateway.handle(fresh).status == gw.STATUS_OK
    assert gateway.handle(old).status == gw.STATUS_REJECTED
    assert gateway.handle(fresh_old).status == gw.STATUS_STALE


def test_malformed_observation_timestamp_is_refused_before_persistence(tmp_path):
    """F3: a bad event_time must be rejected at append, not break summary later.

    Regresses observe.py `StageEvent.validate`.
    """
    store = ob.ObservabilityStore(tmp_path / "events.jsonl")

    def event(when):
        return ob.StageEvent(
            stage="ingest", run_id="run-x", event_time=when, status="OK",
            counts={"n_rows": 1}, reason="ok", evidence_level="SYNTHETIC",
        )

    with pytest.raises(ob.ObserveError, match="invalid event_time"):
        store.append(event("not-a-time"))
    assert not (tmp_path / "events.jsonl").exists()

    store.append(event("2026-11-01T00:00:00+00:00"))
    assert store.verify_chain()["n_events"] == 1
    summary = store.summary(measured_at="2026-11-01T01:00:00+00:00")
    assert summary["n_events"] == 1


def test_equivalent_observation_offsets_are_accepted(tmp_path):
    """F3 guard: valid timestamps with different offsets must still work."""
    store = ob.ObservabilityStore(tmp_path / "events.jsonl")
    for when in ("2026-11-01T09:00:00+00:00", "2026-11-01T11:00:00+02:00"):
        store.append(ob.StageEvent(
            stage="ingest", run_id="run-x", event_time=when, status="OK",
            counts={"n_rows": 1}, reason="ok", evidence_level="SYNTHETIC",
        ))
    assert store.verify_chain()["n_events"] == 2


def test_unknown_columns_preserve_exact_identifiers():
    """F4: `sensor_x` / `sensor_y` must not be reduced to `sensor`.

    Regresses observe.py `_unknown_columns` word-token reconstruction.
    """
    baseline = _qc(["A: unknown columns ['sensor_x']"])
    candidate = _qc(["B: unknown columns ['sensor_x', 'sensor_y']"])
    report = ob.drift_check(baseline, candidate)["unknown_columns"]
    assert report["baseline"] == ["sensor_x"]
    assert report["candidate"] == ["sensor_x", "sensor_y"]
    assert report["new"] == ["sensor_y"]
    assert report["delta"] == 1


def test_unknown_columns_keep_punctuation_and_non_ascii_identifiers():
    """F4 guard: punctuation, digits and non-ASCII names survive intact."""
    report = ob.drift_check(
        _qc(["A: unknown columns ['a']"]),
        _qc(["B: unknown columns ['bp-2', '心率_x', 'col.v2']"]),
    )["unknown_columns"]
    assert report["new"] == ["bp-2", "心率_x", "col.v2"]
    assert report["delta"] == 2


def test_unknown_columns_do_not_discard_short_identifiers():
    """F4 guard: short names like `x`/`y`/`z` are real columns, not noise."""
    report = ob.drift_check(
        _qc(["A: unknown columns ['x']"]),
        _qc(["B: unknown columns ['x', 'y', 'z']"]),
    )["unknown_columns"]
    assert report["baseline"] == ["x"]
    assert report["new"] == ["y", "z"]
    assert report["delta"] == 2


def test_unknown_columns_tolerate_legacy_free_text():
    """F4 guard: non-list drift notes are kept verbatim, not tokenized away."""
    report = ob.drift_check(_qc(()), _qc(["A: unknown columns legacy_note"]))[
        "unknown_columns"
    ]
    assert report["candidate"] == ["legacy_note"]


def test_empty_evidence_iterator_does_not_grant_engineering_go():
    """R-01: a truthy but empty iterator cannot stand in for stage evidence."""
    report = ob.readiness_report(iter([]))
    assert report["decision"] == "NO-GO"
    assert report["gates"]["GO_ENGINEERING"] == "NOT_GRANTED"
    assert report["stages"] == {}


def test_submicrosecond_timestamps_are_not_truncated_into_one_key():
    """R-02: datetime parsing must not collapse distinct submicrosecond instants."""
    early = "2026-11-01T09:00:00.1234561Z"
    late = "2026-11-01T09:00:00.1234569Z"
    assert pl._canonical_event_time(early) == "2026-11-01T09:00:00.1234561+00:00"
    assert pl._canonical_event_time(late) == "2026-11-01T09:00:00.1234569+00:00"
    assert pl._row_key({"event_time": early, "source_record": "source-a"}) != (
        pl._row_key({"event_time": late, "source_record": "source-a"})
    )
    assert pl._canonical_event_time(early) == pl._canonical_event_time(
        "2026-11-01T10:00:00.1234561+01:00"
    )
    assert pl._canonical_event_time("2026-11-01T09:00:00,1234561Z") != (
        pl._canonical_event_time("2026-11-01T09:00:00,1234569Z")
    )


def test_rows_without_physical_identity_are_not_merged():
    """R-02: blank event/source identity cannot discard a distinct record."""
    first = {"event_time": "", "source_record": "", "record_id": "a"}
    second = {"event_time": "", "source_record": "", "record_id": "b"}
    assert pl._row_key(first) != pl._row_key(second)


def test_stage_event_rejects_email_identifiers_and_unapproved_counters():
    """R-03: PII and arbitrary raw measurement keys fail at validation."""
    with pytest.raises(ob.ObserveError, match="raw subject identifier"):
        ob.StageEvent(
            stage="ingest",
            run_id="alice@example.test",
            event_time="2026-11-01T00:00:00+00:00",
            status="OK",
            reason="processed alice@example.test",
            counts={"n_rows": 1},
            evidence_level="SYNTHETIC",
        )
    with pytest.raises(ob.ObserveError, match="approved aggregate counter"):
        ob.StageEvent(
            stage="ingest",
            run_id="run-1",
            event_time="2026-11-01T00:00:00+00:00",
            status="OK",
            reason="aggregate counters only",
            counts={"hr": 70},
            evidence_level="SYNTHETIC",
        )


def test_no_unknown_columns_note_is_not_schema_drift():
    """R-04: a status message reporting zero drift produces zero new columns."""
    report = ob.drift_check(
        _qc(()), _qc(["no unknown columns detected in export"])
    )["unknown_columns"]
    assert report["candidate"] == []
    assert report["new"] == []
    assert report["delta"] == 0


def test_request_identity_includes_schema_version():
    """R-05: a changed contract version cannot replay an older response."""
    common = {
        "request_id": "same",
        "session_id": "s",
        "trace_id": "t",
        "mode": gw.MODE_A_PRE_ONLY,
        "event_time": "2026-11-01T00:00:00+00:00",
        "feature_version": "v1",
        "model_version": "v1",
        "target_version": "v1",
        "quality": "ok",
        "status": "OK",
        "provenance": "gateway",
        "license": "internal",
        "evidence_level": "SYNTHETIC",
        "display_gate": "DEMO_ONLY",
    }
    old = gw.Envelope(schema_version="contract-v1", **common)
    new = gw.Envelope(schema_version="contract-v2", **common)
    assert gw._request_identity(old) != gw._request_identity(new)


def test_empty_data_refresh_plan_does_not_grant_engineering_go():
    """R-01: plan gates must agree with its own fail-closed readiness report."""
    plan = ob.on_new_data(batch_ids=["synthetic-batch-a"])
    assert plan["readiness"]["decision"] == "NO-GO"
    assert plan["gates"]["GO_ENGINEERING"] == "NOT_GRANTED"


def test_duplicate_stage_evidence_is_rejected():
    """R-01: repeated evidence for one stage cannot masquerade as full coverage."""
    evidence = ob.StageEvidence(
        stage="ingest",
        go=True,
        gate="GRANTED",
        evidence_level="SYNTHETIC",
        reason="ok",
        artifacts=(),
    )
    with pytest.raises(ob.ObserveError, match="duplicate stage evidence"):
        ob.readiness_report([evidence, evidence])


def test_drift_batch_id_containing_no_is_not_suppressed():
    """R-04: ordinary batch IDs do not turn real schema drift into a zero note."""
    report = ob.drift_check(
        _qc(()), _qc(["batch no 1: unknown columns ['sensor_x']"])
    )["unknown_columns"]
    assert report["candidate"] == ["sensor_x"]
    assert report["new"] == ["sensor_x"]
    assert report["delta"] == 1


def test_repeated_unknown_columns_marker_uses_the_last_list():
    """R-04: a descriptive prefix cannot become a fabricated column name."""
    columns = ob._unknown_columns(
        _qc(["unknown columns audit: unknown columns ['sensor_x']"])
    )
    assert columns == ("sensor_x",)


def test_experiment_record_rejects_raw_identifiers_without_writing(tmp_path):
    """R-03: PII checks apply before any durable decision record is created."""
    path = tmp_path / "decisions.jsonl"
    with pytest.raises(ob.ObserveError, match="raw subject identifier"):
        ob.experiment_record(
            "KEEP",
            run_id="alice@example.test",
            reason="patient MRN 123456",
            path=path,
        )
    assert not path.exists()


def test_equivalent_fraction_widths_and_naive_high_precision_are_stable():
    """Final review: width, separator, offset and naive UTC never change identity."""
    assert pl._canonical_event_time("2026-11-01T09:00:00.100000Z") == (
        pl._canonical_event_time("2026-11-01T10:00:00,1+01:00")
    )
    assert pl._canonical_event_time("2026-11-01T09:00:00.12345610Z") == (
        pl._canonical_event_time("2026-11-01T10:00:00,1234561+01:00")
    )
    assert pl._canonical_event_time("2026-11-01T09:00:00.1234561") != (
        pl._canonical_event_time("2026-11-01T09:00:00.1234569")
    )
    assert pl._canonical_event_time("2026-11-01T09:00:00,1234561") != (
        pl._canonical_event_time("2026-11-01T09:00:00,1234569")
    )


@pytest.mark.parametrize(
    "identifier",
    ["555.123.4567", "(555) 123-4567", "123-456-789-012", "123 456 789 012"],
)
def test_grouped_identifier_formats_are_rejected(identifier):
    """Final review: separator variants cannot bypass identifier checks."""
    with pytest.raises(ob.ObserveError, match="raw subject identifier"):
        ob._check_text(identifier, "reason")


def test_v1_observation_chain_remains_verifiable(tmp_path):
    """F1: verify existing append-only v1 logs after the schema bump."""
    legacy = ob.StageEvent(
        stage="ingest",
        run_id="run-1",
        event_time="2026-11-01T00:00:00+00:00",
        status="OK",
        reason="ok",
        counts={"n_rows": 1},
        evidence_level="SYNTHETIC",
        version=ob.LEGACY_OBSERVE_VERSION,
    )
    record = {"event": legacy.to_dict(), "previous_sha256": ob.GENESIS_SHA256}
    record["sha256"] = digest(record)
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    store = ob.ObservabilityStore(path)
    assert store.verify_chain()["verified"] is True
    assert store.append(legacy)


def test_metric_counter_key_rejects_raw_identifiers():
    """F2: a duration-shaped key cannot smuggle a subject identifier."""
    with pytest.raises(ob.ObserveError, match="approved aggregate counter"):
        ob._check_counts({"subject_123456789_duration": 42})


def test_general_lifecycle_stage_can_support_engineering_readiness():
    """F3: readiness_report is not restricted to data-refresh stages."""
    evidence = ob.StageEvidence(
        stage="train",
        go=True,
        gate="GRANTED",
        evidence_level="SYNTHETIC",
        reason="synthetic training smoke",
        artifacts=(),
    )
    report = ob.readiness_report([evidence])
    assert report["decision"] == "GO"
    assert report["gates"]["GO_ENGINEERING"] == "GRANTED"


def test_basic_iso_timestamps_preserve_submicrosecond_precision():
    """F4: colon-less ISO timestamps cannot lose precision or merge instants."""
    early = "20261101T090000.1234561Z"
    late = "20261101T090000.1234569Z"
    assert pl._canonical_event_time(early) == "2026-11-01T09:00:00.1234561+00:00"
    assert pl._canonical_event_time(late) == "2026-11-01T09:00:00.1234569+00:00"
    assert pl._row_key({"event_time": early, "source_record": "source-a"}) != (
        pl._row_key({"event_time": late, "source_record": "source-a"})
    )
    assert pl._canonical_event_time(early) == pl._canonical_event_time(
        "20261101T100000,12345610+01:00"
    )
