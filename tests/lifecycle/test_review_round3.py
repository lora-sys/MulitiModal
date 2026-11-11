"""Round-3 review regressions: 15 Codex findings + Claude/Step 5 P1/P2 items.

Each test is a negative case: it fails if the corresponding guard is removed.
Only synthetic fixtures are used; no real data, network or device access.
"""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from lifecycle import features as ft
from lifecycle import gateway as gw
from lifecycle import observe as ob
from lifecycle import release as rl
from lifecycle import sources as src
from lifecycle.contracts import (
    APPROVAL_APPROVED,
    EVIDENCE_LOCAL,
    EVIDENCE_SYNTHETIC,
    STATUS_BLOCKED,
    STATUS_NOT_AVAILABLE,
    STATUS_OK,
    STATUS_REJECTED,
    STATUS_SAFE_STOP,
    TRUST_TRUSTED,
    ArtifactManifest,
    FeatureSpec,
    FieldSpec,
    canonical_json,
    digest,
)
from lifecycle.pipeline import (
    QC_FILE,
    SNAPSHOT_FILE,
    PipelineConfig,
    declare_batch,
    run_pipeline,
)

pytestmark = pytest.mark.synthetic

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "lifecycle"
BATCH_A = FIXTURES / "batch_a.csv"


def _row(record_id: str, session: str, subject: str, metric: str, value: str, unit: str) -> dict:
    return {
        "batch_id": "A",
        "record_id": record_id,
        "session_id": session,
        "subject_ref": subject,
        "phase": "pre",
        "metric": metric,
        "value": value,
        "unit": unit,
        "quality": "ok",
        "source_record": f"src-{record_id}",
        "event_time": "2026-11-01T09:00:00+00:00",
        "received_at": "2026-11-01T09:00:00+00:00",
        "origin": "measured",
    }


def _write_batch(path: Path, rows: list[dict]) -> None:
    headers = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


# --- pipeline: crash recovery, raw-byte binding, cache integrity -------------


def test_failed_run_without_snapshot_re_ingests_on_recovery(tmp_path):
    """P1 (Codex 802): a cursor without a snapshot must not skip every batch."""
    batch = declare_batch(BATCH_A, batch_id="A")
    failed = run_pipeline([batch], tmp_path / "out", fail_stage="validate")
    assert failed.error
    assert failed.snapshot is None or failed.qc.n_accepted == 4
    recovered = run_pipeline([batch], tmp_path / "out")
    assert recovered.qc.n_accepted == 4
    assert recovered.snapshot is not None


def test_mutated_raw_bytes_after_declaration_are_rejected(tmp_path):
    """P2 (Codex 317): declared hash is verified against the bytes actually read."""
    path = tmp_path / "copy.csv"
    path.write_text(BATCH_A.read_text(encoding="utf-8"), encoding="utf-8")
    batch = declare_batch(path, batch_id="A")
    path.write_text(
        BATCH_A.read_text(encoding="utf-8").replace(",70,bpm", ",90,bpm"),
        encoding="utf-8",
    )
    result = run_pipeline([batch], tmp_path / "out")
    assert result.status == STATUS_BLOCKED
    assert any("raw bytes changed" in reason for reason in result.qc.blocked)
    assert all(row.value != 90.0 for row in (result.snapshot.rows if result.snapshot else ()))


def test_tampered_cached_snapshot_rows_force_full_recompute(tmp_path):
    """P2 (Codex 861): rows_sha256 binds the cached snapshot to its content."""
    out = tmp_path / "out"
    batch = declare_batch(BATCH_A, batch_id="A")
    run_pipeline([batch], out)
    snapshot_path = out / SNAPSHOT_FILE
    data = json.loads(snapshot_path.read_text(encoding="utf-8"))
    data["rows"][0]["value"] = 9999.0
    snapshot_path.write_text(canonical_json(data) + "\n", encoding="utf-8")
    result = run_pipeline([batch], out)
    assert result.reused_snapshot is False
    assert any("unreadable" in d for d in result.qc.drift)
    assert all(row.value != 9999.0 for row in result.snapshot.rows)


def test_cached_snapshot_preserves_qc_blockers(tmp_path):
    """P1 (Codex 861): a cache hit must not silently clear QC blockers."""
    path = tmp_path / "mixed.csv"
    _write_batch(
        path,
        [
            _row("1", "S1", "P1", "heart_rate", "70", "bpm"),
            _row("2", "S1", "P2", "heart_rate", "72", "bpm"),
            _row("3", "S2", "P2", "spo2", "96", "%"),
            _row("4", "S2", "P1", "spo2", "97", "%"),
        ],
    )
    batch = declare_batch(path, batch_id="A")
    kwargs = dict(
        approval=APPROVAL_APPROVED,
        identity_trust=TRUST_TRUSTED,
        frozen=True,
        evidence_level=EVIDENCE_LOCAL,
    )
    out = tmp_path / "out"
    first = run_pipeline([batch], out, **kwargs)
    second = run_pipeline([batch], out, **kwargs)
    assert first.qc.blocked
    assert second.qc.blocked
    assert first.readiness["train_ready"] is False
    assert second.readiness["train_ready"] is False


def _conflicting_batches(tmp_path: Path, tag: str):
    """Two byte-different copies of one logical row, for cache-binding checks."""
    directory = tmp_path / tag
    p1 = directory / "one.csv"
    p2 = directory / "two.csv"
    _write_batch(p1, [_row("1", "S1", "P1", "heart_rate", "70", "bpm")])
    _write_batch(p2, [_row("1", "S1", "P1", "heart_rate", "90", "bpm")])
    return directory, p1, p2


def test_dataset_cache_key_includes_batch_id_bindings(tmp_path):
    """P2 (Codex 772): swapping batch IDs must not reuse a stale winner."""
    directory, p1, p2 = _conflicting_batches(tmp_path, "bindings")
    a_first = declare_batch(p1, batch_id="A")
    b_first = declare_batch(p2, batch_id="B")
    run_pipeline([a_first, b_first], directory / "warm")
    a_swapped = declare_batch(p1, batch_id="B")
    b_swapped = declare_batch(p2, batch_id="A")
    warm = run_pipeline([a_swapped, b_swapped], directory / "warm")
    cold = run_pipeline([a_swapped, b_swapped], directory / "cold")
    assert [r.value for r in warm.snapshot.rows] == [r.value for r in cold.snapshot.rows]
    assert warm.snapshot.dataset_sha256 == cold.snapshot.dataset_sha256


def test_deterministic_duplicate_winner_is_byte_identical(tmp_path):
    """P1 (Claude): declaration order must not change the winning row."""
    directory, p1, p2 = _conflicting_batches(tmp_path, "winner")
    a = declare_batch(p1, batch_id="A")
    b = declare_batch(p2, batch_id="B")
    first = run_pipeline([a, b], directory / "ab")
    second = run_pipeline([b, a], directory / "ba")
    assert (directory / "ab" / SNAPSHOT_FILE).read_bytes() == (
        directory / "ba" / SNAPSHOT_FILE
    ).read_bytes()
    assert [r.batch_id for r in first.snapshot.rows] == [
        r.batch_id for r in second.snapshot.rows
    ]
    assert [r.value for r in first.snapshot.rows] == [
        r.value for r in second.snapshot.rows
    ]
    assert first.snapshot.sha256 == second.snapshot.sha256


def test_config_change_still_forces_recompute_with_bindings(tmp_path):
    """The binding-aware key must not break the declared config version gate."""
    batch = declare_batch(BATCH_A, batch_id="A")
    out = tmp_path / "out"
    first = run_pipeline([batch], out)
    reused = run_pipeline([batch], out)
    changed = run_pipeline([batch], out, config=PipelineConfig(min_subjects=3))
    assert first.reused_snapshot is False
    assert reused.reused_snapshot is True
    assert changed.reused_snapshot is False
    assert (out / QC_FILE).is_file()


# --- gateway: mode isolation, SAFE_STOP priority, structured phases ---------


def _envelope(mode: str, request_id: str, payload: dict):
    return gw.make_envelope(mode=mode, request_id=request_id, payload=payload)


def test_mode_b_never_emits_a_prediction_from_request_flags():
    """P1 (Codex 589): self-declared validation cannot authorize a score."""
    for strict in (False, True):
        gateway = gw.Gateway(strict_mode=strict)
        req = _envelope(
            gw.MODE_B_REALTIME,
            f"b-{strict}",
            {
                "validated": True,
                "safety_authorized": True,
                "prediction": 0.9,
            },
        )
        res = gateway.handle(req)
        assert res.status == STATUS_NOT_AVAILABLE
        assert res.payload["prediction"] is None
        assert res.payload["control_command"] is None


def test_safe_stop_beats_idempotency_cache_and_throttle():
    """P1 (Codex 340): stop signals are evaluated before cache and admission."""
    cached = gw.Gateway()
    normal = _envelope(gw.MODE_A_PRE_ONLY, "same-id", {"features": {"heart_rate_pre": 70}})
    assert cached.handle(normal).status == STATUS_OK
    stop = _envelope(gw.MODE_A_PRE_ONLY, "same-id", {"safe_stop": True})
    assert cached.handle(stop).status == STATUS_SAFE_STOP

    saturated = gw.Gateway(max_concurrency_per_mode=1)
    saturated._in_flight[gw.MODE_A_PRE_ONLY] = 1
    throttled_stop = gw.make_envelope(
        mode=gw.MODE_A_PRE_ONLY,
        request_id="throttled-stop",
        quality="safe_stop",
        payload={"signal": "SAFE_STOP"},
    )
    res = saturated.handle(throttled_stop)
    assert res.status == STATUS_SAFE_STOP
    assert res.payload["control_command"] is None


@pytest.mark.parametrize("phase_key", ["timepoint", "phase"])
def test_structured_during_post_timepoints_are_rejected(phase_key):
    """P2 (Codex 214): declared phase metadata cannot disguise a non-pre input."""
    payload = {
        "fields": [{"name": "heart_rate", phase_key: "post", "value": 80}],
    }
    gateway = gw.Gateway()
    res = gateway.handle(_envelope(gw.MODE_A_PRE_ONLY, f"post-{phase_key}", payload))
    assert res.status == STATUS_REJECTED
    assert any("post" in name for name in res.payload["rejected_features"])

    strict = gw.Gateway(strict_mode=True)
    with pytest.raises(gw.ModeIsolationError):
        strict.handle(_envelope(gw.MODE_A_PRE_ONLY, f"strict-{phase_key}", payload))


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), b"raw-bytes", {"a", "b"}],
    ids=["nan", "inf", "bytes", "set"],
)
@pytest.mark.parametrize("stop", [False, True], ids=["scored", "safe_stop"])
def test_non_json_payloads_are_refused_or_stopped_without_raising(value, stop):
    """Non-canonical payloads get a controlled refusal; SAFE_STOP is unconditional."""
    gateway = gw.Gateway()
    payload = {"value": value}
    if stop:
        payload["safe_stop"] = True
    req = gw.make_envelope(request_id="opaque", mode=gw.MODE_B_REALTIME, payload=payload)
    res = gateway.handle(req)
    expected = STATUS_SAFE_STOP if stop else STATUS_REJECTED
    assert res.status == expected
    assert res.payload.get("prediction") is None
    assert res.payload["control_command"] is None
    assert gateway._cache == {} and gateway._cache_identities == {}


def test_untrusted_request_objects_are_contained_at_the_boundary():
    """One boundary: exotic Python objects are refused, SAFE_STOP still wins."""
    class Meta(type):
        def __eq__(cls, other):
            raise RuntimeError("metaclass eq")

        def __getattribute__(cls, name):
            if name == "__name__":
                raise RuntimeError("metaclass name")
            return super().__getattribute__(name)

    class Weird(metaclass=Meta):
        pass

    class BadKey(str):
        def __eq__(self, other):
            raise RuntimeError("key eq")

        def __hash__(self):
            return hash(str(self))

    class BadStr(str):
        def __str__(self):
            raise RuntimeError("str hook")

        def upper(self):
            raise RuntimeError("upper hook")

    class HostileDict(dict):
        def items(self):
            raise RuntimeError("items hook")

    # Non-string keys, subclassed values, custom metaclass types, dict
    # subclasses, non-finite floats, bytes, sets, cycles: all refused, never
    # raising out of the gateway and never emitting a score.
    untrusted = (
        {"knowledge": [{1: "x"}]},
        {BadKey("a"): 1, BadKey("b"): 2},
        {"value": Weird()},
        {"value": BadStr("x")},
        {"value": float("nan")},
        {"value": b"bytes"},
        {"value": {1, 2}},
    )
    for payload in untrusted:
        gateway = gw.Gateway()
        res = gateway.handle(
            gw.make_envelope(request_id="bad", mode=gw.MODE_B_REALTIME, payload=payload)
        )
        assert res.status == STATUS_REJECTED, payload
        # The boundary refusal, not the generic totality wrapper, must fire.
        assert "strict canonical JSON" in res.payload["reason"], payload
        assert res.payload.get("prediction") is None
        assert gateway._cache == {} and gateway._cache_identities == {}

    # Envelope is a frozen dataclass: a dict subclass can reach the Python
    # object entry point directly, bypassing make_envelope's copy.
    direct = gw.Envelope(
        schema_version="ml-lifecycle-contracts-v1",
        request_id="sd", session_id="S1", trace_id="t", mode=gw.MODE_A_PRE_ONLY,
        event_time="", feature_version="v1", model_version="v1",
        target_version="v1", quality="ok", status=gw.STATUS_OK,
        provenance="test", license="test", evidence_level=gw.EVIDENCE_SYNTHETIC,
        display_gate="", payload=HostileDict({"value": 1}),
    )
    direct_res = gw.Gateway().handle(direct)
    assert direct_res.status == STATUS_REJECTED
    assert "strict canonical JSON" in direct_res.payload["reason"]

    # SAFE_STOP is decided before normalization and is unconditional.
    for raw in (
        gw.make_envelope(request_id="s", status=gw.STATUS_SAFE_STOP,
                         payload={"value": Weird()}),
        gw.make_envelope(request_id="s", payload={"safe_stop": True,
                                                 "signal": BadStr("SAFE_STOP")}),
        gw.make_envelope(request_id="s", payload={"user_stop": True,
                                                 "other": Weird()}),
        gw.Envelope(
            schema_version="ml-lifecycle-contracts-v1", request_id="s",
            session_id="S1", trace_id="t", mode=gw.MODE_A_PRE_ONLY,
            event_time="", feature_version="v1", model_version="v1",
            target_version="v1", quality="safe_stop", status=gw.STATUS_OK,
            provenance="test", license="test",
            evidence_level=gw.EVIDENCE_SYNTHETIC, display_gate="",
            payload=HostileDict({"value": 1}),
        ),
    ):
        res = gw.Gateway().handle(raw)
        assert res.status == STATUS_SAFE_STOP
        assert res.payload["control_command"] is None


def test_canonical_json_identity_is_exact_and_order_independent():
    """No truncation: distinct leaves stay distinct; dict insertion order is irrelevant."""
    first = {"a": 1, "b": 2, "c": 3}
    second = {"c": 3, "b": 2, "a": 1}
    fixed_time = datetime.now(UTC).isoformat()
    base = gw.make_envelope(
        request_id="ordered", session_id="S1", mode=gw.MODE_A_PRE_ONLY,
        event_time=fixed_time, payload={"features": first},
    )
    replay = gw.make_envelope(
        request_id="ordered", session_id="S1", mode=gw.MODE_A_PRE_ONLY,
        event_time=fixed_time, payload={"features": second},
    )
    gateway = gw.Gateway()
    assert gateway.handle(base).status == STATUS_OK
    assert gateway.handle(replay).status == STATUS_OK
    assert gateway.idempotency_hits == 1

    leaf_a, leaf_b = "first", "second"
    for _ in range(26):
        leaf_a, leaf_b = {"knowledge": leaf_a}, {"knowledge": leaf_b}
    deep_a = gw.make_envelope(
        request_id="deep", session_id="S1", mode=gw.MODE_C_TCM,
        event_time=fixed_time, payload={"knowledge": leaf_a},
    )
    deep_b = gw.make_envelope(
        request_id="deep", session_id="S1", mode=gw.MODE_C_TCM,
        event_time=fixed_time, payload={"knowledge": leaf_b},
    )
    gateway = gw.Gateway()
    assert gateway.handle(deep_a).status == STATUS_OK
    assert gateway.handle(deep_b).status == STATUS_REJECTED


def test_typed_canonical_json_does_not_collide():
    """1.0, 'float:1.0', 1 and True must be four different requests."""
    fixed_time = datetime.now(UTC).isoformat()

    def envelope(payload):
        return gw.make_envelope(
            request_id="typed", session_id="S1", mode=gw.MODE_A_PRE_ONLY,
            event_time=fixed_time, payload={"value": payload},
        )
    gateway = gw.Gateway()
    assert gateway.handle(envelope(1.0)).status == STATUS_OK
    for value in ("float:1.0", 1, True):
        assert gw.Gateway().handle(envelope(value)).status == STATUS_OK
        assert gateway.handle(envelope(value)).status == STATUS_REJECTED


def test_cyclic_deep_or_unrepresentable_payloads_never_raise():
    """Cycles, deep nesting, huge ints and hostile __repr__/__str__ are controlled."""
    cycle = []
    cycle.append(cycle)
    deep = "leaf"
    for _ in range(2000):
        deep = {"next": deep}

    class BadRepr:
        def __repr__(self):
            raise RuntimeError("boom")

        def __str__(self):
            raise RuntimeError("boom")

    for value in (cycle, deep, 10**5000, BadRepr()):
        for stop in (False, True):
            gateway = gw.Gateway()
            payload = {"value": value}
            if stop:
                payload["safe_stop"] = True
            req = gw.make_envelope(
                request_id="opaque", mode=gw.MODE_B_REALTIME, payload=payload
            )
            res = gateway.handle(req)
            if stop:
                assert res.status == STATUS_SAFE_STOP
            else:
                # Either a controlled refusal or the mode-B unavailable answer;
                # never an OK score, never an exception.
                assert res.status in (STATUS_REJECTED, STATUS_NOT_AVAILABLE)
            assert res.payload.get("prediction") is None
            assert len(gateway._cache) == len(gateway._cache_identities)


def test_safe_stop_detection_never_calls_arbitrary_object_methods():
    """A hostile signal object must not turn a SAFE_STOP into an exception."""
    class BadStr:
        def __str__(self):
            raise RuntimeError("boom")

    gateway = gw.Gateway()
    req = gw.make_envelope(
        request_id="stop", mode=gw.MODE_A_PRE_ONLY,
        payload={"safe_stop": True, "signal": BadStr()},
    )
    res = gateway.handle(req)
    assert res.status == STATUS_SAFE_STOP
    assert res.payload["control_command"] is None


def test_internal_failure_is_fail_closed(monkeypatch):
    """The totality wrapper converts any unexpected error into a refusal."""
    monkeypatch.setattr(
        gw, "_request_identity", lambda _req: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    gateway = gw.Gateway()
    res = gateway.handle(
        gw.make_envelope(request_id="boom", mode=gw.MODE_A_PRE_ONLY)
    )
    assert res.status == STATUS_REJECTED
    assert "could not be processed safely" in res.payload["reason"]
    assert res.payload["prediction"] is None
    assert gateway._cache == {} and gateway._cache_identities == {}


def test_refused_request_writes_no_partial_cache_state():
    """A boundary refusal must not leave a cache entry without its identity."""
    gateway = gw.Gateway()
    res = gateway.handle(
        gw.make_envelope(request_id="atomic", mode=gw.MODE_A_PRE_ONLY,
                         payload={"value": float("nan")})
    )
    assert res.status == STATUS_REJECTED
    assert "strict canonical JSON" in res.payload["reason"]
    assert gateway._cache == {}
    assert gateway._cache_identities == {}
    assert gateway.idempotency_counter == {}


def test_non_json_payload_still_distinguishes_conflicting_requests():
    """Controlled refusal does not silently cache a conflicting identity."""
    gateway = gw.Gateway()
    fixed_time = datetime.now(UTC).isoformat()
    first = gw.make_envelope(
        request_id="opaque", session_id="S1", mode=gw.MODE_A_PRE_ONLY,
        event_time=fixed_time, payload={"value": 7},
    )
    assert gateway.handle(first).status == STATUS_OK
    conflict = gw.make_envelope(
        request_id="opaque", session_id="S2", mode=gw.MODE_B_REALTIME,
        event_time=fixed_time, payload={"value": 7},
    )
    res = gateway.handle(conflict)
    assert res.status == STATUS_REJECTED
    assert "different request identity" in res.payload["reason"]
    assert gateway.handle(first).status == STATUS_OK


def test_idempotency_key_cannot_be_reused_across_modes_or_sessions():
    """P2 (round-4 Codex): a cached mode-A answer must not leak to mode B/S2."""
    gateway = gw.Gateway()
    fixed_time = datetime.now(UTC).isoformat()
    first = gw.make_envelope(
        request_id="shared-id", session_id="S1", mode=gw.MODE_A_PRE_ONLY,
        event_time=fixed_time, payload={"features": {"pre_hr": 70}},
    )
    assert gateway.handle(first).status == STATUS_OK
    conflicting = gw.make_envelope(
        request_id="shared-id", session_id="S2", mode=gw.MODE_B_REALTIME,
        event_time=fixed_time, payload={}
    )
    res = gateway.handle(conflicting)
    assert res.status == STATUS_REJECTED
    assert res.payload["prediction"] is None
    assert "different request identity" in res.payload["reason"]
    assert gateway.handle(first).status == STATUS_OK


def test_structured_pre_timepoint_is_still_admitted():
    """Pre declared phase metadata stays admissible (no blanket rejection)."""
    res = gw.Gateway().handle(
        _envelope(
            gw.MODE_A_PRE_ONLY,
            "pre-ok",
            {"fields": [{"name": "heart_rate", "timepoint": "pre", "value": 70}]},
        )
    )
    assert res.status == STATUS_OK


# --- sources: parsed timestamps, session isolation, capability honesty ------


def test_cutoff_and_order_compare_instants_not_strings():
    """P2 (Codex 398): UTC offsets must not expose future measurements."""
    window = src.SessionWindow("S1")
    late = src.make_event("late", event_time="2026-11-01T08:30:00-02:00")
    early = src.make_event("early", event_time="2026-11-01T09:30:00+00:00")
    window.push(late)  # 10:30 UTC
    window.push(early)  # 09:30 UTC
    assert [e.key for e in window.cutoff("2026-11-01T09:00:00+00:00")] == []
    assert [e.key for e in window.cutoff("2026-11-01T10:00:00+00:00")] == ["early"]
    ordered = src.replay([late, early])
    assert [e.key for e in ordered] == ["early", "late"]


def test_gap_detection_uses_normalized_instants():
    """P2 (Codex 398): gap spans compare instants across offsets."""
    window = src.SessionWindow("S1", gap_threshold_s=30.0)
    window.push(src.make_event("a", event_time="2026-11-01T09:00:00+00:00"))
    window.push(src.make_event("b", event_time="2026-11-01T09:05:00-02:00"))
    gaps = window.close()
    assert len(gaps) == 1
    assert gaps[0].seconds == pytest.approx(7500.0)


def test_session_window_rejects_foreign_session_with_a_reason():
    """P2 (Codex 390): a window must not accept another session's events."""
    window = src.SessionWindow("S1")
    assert window.push(src.make_event("x", session_id="S2")) is False
    assert window.accepted == ()
    assert any("S2" in reason for reason in window.rejections)


def _conformance():
    return src.ConformanceRecord(
        vendor="Acme", api_version="v1", protocol="USB",
        event_schema="physio-event-v1", supported_metrics=("heart_rate",),
        timestamp_semantics="measured_at UTC", replay_guarantee="sequence-ordered",
        verified_at="2026-11-01",
    )


def test_only_real_connection_kinds_may_declare_connected():
    """P2 (Claude): offline sources and unconformed services cannot be CONNECTED."""
    with pytest.raises(src.SourceError, match="offline"):
        src.SourceCapability("off", src.SOURCE_OFFLINE, src.CAP_CONNECTED)
    with pytest.raises(src.SourceError, match="conformance"):
        src.SourceCapability("svc", src.SOURCE_SERVICE, src.CAP_CONNECTED)
    connected = src.SourceCapability(
        "dev", src.SOURCE_DEVICE, src.CAP_CONNECTED, conformance=_conformance()
    )
    record = src.conformance_record(connected)
    assert record["capability"] == src.CAP_CONNECTED
    assert record["status"] == STATUS_OK


def test_same_id_revision_order_is_not_served_from_a_stale_cache(tmp_path):
    """P2 (round-4 Codex): correction order must be part of the cache identity."""
    directory = tmp_path / "revision"
    old = directory / "old.csv"
    new = directory / "new.csv"
    _write_batch(old, [_row("1", "S1", "P1", "heart_rate", "70", "bpm")])
    _write_batch(new, [_row("1", "S1", "P1", "heart_rate", "90", "bpm")])
    first = declare_batch(old, batch_id="A")
    second = declare_batch(new, batch_id="A")
    run_pipeline([first, second], directory / "warm")
    warm_reversed = run_pipeline([second, first], directory / "warm")
    cold_reversed = run_pipeline([second, first], directory / "cold")
    assert warm_reversed.reused_snapshot is False
    assert [r.value for r in warm_reversed.snapshot.rows] == [
        r.value for r in cold_reversed.snapshot.rows
    ]


# --- features: legal ranges -------------------------------------------------


def _legal_range_spec():
    return FeatureSpec(
        version="f-range",
        fields=(
            FieldSpec(
                "heart_rate", "bpm", "float", "pre", "none", "synthetic",
                "input", "train_fold", "observed", legal_range=(30.0, 220.0),
            ),
        ),
        transform_id="identity",
        evidence_level=EVIDENCE_SYNTHETIC,
    )


def test_out_of_range_observation_is_invalid_and_excluded_from_median():
    """P2 (Codex 810): legal_range gates observations and fold-internal medians."""
    rows = ft.build_feature_rows(
        [
            {"subject_ref": "P1", "session_id": "S1", "heart_rate": 9999.0},
            {"subject_ref": "P1", "session_id": "S1"},
        ],
        _legal_range_spec(),
        train_subjects={"P1"},
    )
    assert rows[0].observed_mask[0] is True
    assert rows[0].valid_mask[0] is False
    assert rows[0].imputed_mask[0] is False
    assert rows[0].values[0] == 0.0
    assert rows[1].valid_mask[0] is False


def test_in_range_observation_is_usable():
    rows = ft.build_feature_rows(
        [{"subject_ref": "P1", "session_id": "S1", "heart_rate": 70.0}],
        _legal_range_spec(),
        train_subjects={"P1"},
    )
    assert rows[0].observed_mask[0] is True
    assert rows[0].valid_mask[0] is True
    assert rows[0].values[0] == 70.0


# --- observe: durable hash chain tail ---------------------------------------


def _stage_event(i: int):
    return ob.StageEvent(
        stage="ingest",
        run_id="run-x",
        event_time="2026-11-01T00:00:00+00:00",
        status="OK",
        counts={"rows": i},
        reason="ok",
        evidence_level=EVIDENCE_SYNTHETIC,
    )


def test_two_stores_on_one_log_keep_a_valid_chain(tmp_path):
    """P2 (Codex 208): append re-syncs the durable tail under a shared lock."""
    path = tmp_path / "events.jsonl"
    first = ob.ObservabilityStore(path)
    second = ob.ObservabilityStore(path)
    first.append(_stage_event(1))
    second.append(_stage_event(2))
    assert ob.ObservabilityStore(path).verify_chain()["n_events"] == 2


# --- release: rollback verification and idempotent promotion ---------------


def _bundle(root: Path, artifact_id: str):
    root.mkdir(parents=True, exist_ok=True)
    (root / "weights.bin").write_bytes(artifact_id.encode())
    manifest = ArtifactManifest(
        artifact_id=artifact_id,
        model_sha256=digest({"m": artifact_id}),
        feature_sha256=digest({"f": 1}),
        target_sha256=digest({"t": 1}),
        scaler_sha256=digest({"s": 1}),
        dataset_sha256=digest({"d": 1}),
        split_sha256=digest({"sp": 1}),
        eval_sha256=digest({"e": 1}),
        commit="abc",
        dependencies=("numpy",),
        license="SMOKE_ONLY",
        approval=APPROVAL_APPROVED,
        validation_status="VALIDATED",
        evidence_level=EVIDENCE_LOCAL,
    )
    return rl.ArtifactBundle.from_directory(root, manifest, files=["weights.bin"])


def test_rollback_refuses_a_corrupted_previous_bundle(tmp_path):
    """P2 (Codex 455): rollback must verify files before restoring champion."""
    registry = rl.ReleaseRegistry(tmp_path / "registry.json")
    a_root = tmp_path / "a"
    a = _bundle(a_root, "A")
    b = _bundle(tmp_path / "b", "B")
    registry.register(a, "VALIDATED")
    registry.register(b, "VALIDATED")
    registry.promote("A", approver="x", reason="r")
    registry.promote("B", approver="x", reason="r")
    (a_root / "weights.bin").write_bytes(b"corrupt")
    result = registry.rollback("x", "r")
    assert result.accepted is False
    assert "verification" in result.reason
    assert result.state["champion"]["artifact_id"] == "B"


def test_repeated_promotion_preserves_rollback_target(tmp_path):
    """P2 (Codex 434): a retried promote must not erase `previous`."""
    registry = rl.ReleaseRegistry(tmp_path / "registry.json")
    a = _bundle(tmp_path / "a", "A")
    b = _bundle(tmp_path / "b", "B")
    registry.register(a, "VALIDATED")
    registry.register(b, "VALIDATED")
    registry.promote("A", approver="x", reason="r")
    registry.promote("B", approver="x", reason="r")
    registry.promote("B", approver="x", reason="r")
    result = registry.rollback("x", "r")
    assert result.accepted is True
    assert result.state["champion"]["artifact_id"] == "A"


# --- acceptance: private-path classifier ------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "data/raw/session.csv",
        "data/local/export.csv",
        "runs/exp/log.json",
        "artifacts/champion/weights.bin",
        "model.pt",
        "model.pth",
        "./runs/nested/log.jsonl",
    ],
)
def test_private_path_classifier_accepts_private_markers(path):
    from lifecycle.acceptance import _is_private_path

    assert _is_private_path(path) is True


@pytest.mark.parametrize(
    "path",
    [
        "src/lifecycle/pipeline.py",
        "docs/design/ml-lifecycle.md",
        "tests/lifecycle/test_pipeline.py",
        "README.md",
        "data/README.md",
    ],
)
def test_private_path_classifier_rejects_safe_paths(path):
    from lifecycle.acceptance import _is_private_path

    assert _is_private_path(path) is False
