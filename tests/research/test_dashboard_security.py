"""Dashboard security and robustness: the negative cases, in the test suite.

The end-to-end script exercises these against a live server; these tests keep
them enforceable in CI, on data crafted to be hostile:

* a hostile hypothesis/reason string must travel as DATA, never as markup;
* a torn JSONL tail and a huge log must not break a reader;
* the route whitelist must refuse everything else, including traversal-looking
  paths and any mutating method;
* the payload must contain no raw dataset content, subject ids or secrets.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from research import dashboard as dash
from research.session import ResearchSession

pytestmark = pytest.mark.synthetic

HOSTILE = "<script>alert('xss')</script> \u2028 \"'><img src=x onerror=alert(1)>"


def _session(tmp_path: Path) -> ResearchSession:
    return ResearchSession.create(
        tmp_path / "run", task="toy", editable="src/research/candidate_toy.py",
        worktree=str(tmp_path / "wt"), branch="autoresearch/sec",
        folds=2, repeats=1, max_rounds=3, budget_seconds=1.0,
        primary_metric="toy/macro_f1",
    )


def _entry(run_id: str, **extra) -> dict:
    base = {"run_id": run_id, "round_index": 0, "hypothesis": HOSTILE,
            "decision": "KEEP", "status": "completed", "reason": HOSTILE,
            "best_so_far": 0.5, "metrics": {"primary_value": 0.5,
                                            "primary_sem": 0.01},
            "outcome": "KEEP", "git_commit_sha": "deadbeef"}
    base.update(extra)
    return base


def test_hostile_text_round_trips_as_data_not_markup(tmp_path):
    session = _session(tmp_path)
    (session.root / "manifest.jsonl").write_text(
        json.dumps(_entry("toy-0000")) + "\n", encoding="utf-8")
    payload = dash._status_payload(session, after=0)
    # the payload carries the text verbatim as a JSON string; escaping is the
    # transport's job and the page's renderer uses textContent
    assert payload["history"][0]["hypothesis"] == HOSTILE
    page = dash.render_page()
    # structural guarantee: no data is ever assigned to innerHTML
    assert "innerHTML" not in page
    assert "textContent" in page


def test_torn_jsonl_and_oversized_log_are_contained(tmp_path):
    session = _session(tmp_path)
    (session.root / "manifest.jsonl").write_text(
        json.dumps(_entry("toy-0000")) + "\n", encoding="utf-8")
    with session.writer() as writer:
        for i in range(dash.MAX_EVENTS_PER_REQUEST + 50):
            writer.emit("TRAINING", "started", round_index=0, fold=0,
                        folds_total=4, message=f"event {i}")
    # a process killed mid-write
    with session.events_path.open("a", encoding="utf-8") as handle:
        handle.write('{"schema_version": "autoresearch-events-v1", "seq": 9999')
    payload = dash._status_payload(session, after=0)
    assert len(payload["events"]) == dash.MAX_EVENTS_PER_REQUEST
    assert payload["events"][0]["seq"] == 1
    # paging continues past the cap without losing order
    later = dash._status_payload(session, after=payload["events"][-1]["seq"])
    assert all(e["seq"] > payload["events"][-1]["seq"] for e in later["events"])


def test_payload_contains_no_raw_data_or_secrets(tmp_path):
    session = _session(tmp_path)
    (session.root / "manifest.jsonl").write_text(
        json.dumps(_entry("toy-0000")) + "\n", encoding="utf-8")
    with session.writer() as writer:
        writer.emit("DOCTOR", "finished", message="preflight doctor: 12 INFO")
    blob = json.dumps(dash._status_payload(session, after=0), ensure_ascii=False)
    for forbidden in ("user_id", "subject_index", "label_mask", "toy_records",
                      "scaler", "manifest_sha256", "TOKEN", "SECRET", "password"):
        assert forbidden not in blob, forbidden
    assert "toy_r" not in blob


def _server(session: ResearchSession):
    server = ThreadingHTTPServer(("127.0.0.1", 0), dash.make_handler(session))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def test_route_whitelist_and_read_only(tmp_path):
    session = _session(tmp_path)
    (session.root / "manifest.jsonl").write_text(
        json.dumps(_entry("toy-0000")) + "\n", encoding="utf-8")
    server, base = _server(session)
    try:
        for path in ("/", "/api/status", "/api/events?after=0", "/api/history"):
            with urllib.request.urlopen(f"{base}{path}", timeout=5) as response:
                assert response.status == 200
                body = response.read()
                assert len(body) <= dash.MAX_RESPONSE_BYTES
        for path in ("/etc/passwd", "/../frozen.py", "/api/status/../run.json",
                     "/api/unknown", "/api/run", "/api/stop"):
            try:
                urllib.request.urlopen(f"{base}{path}", timeout=5)
                raise AssertionError(f"{path} should not be served")
            except urllib.error.HTTPError as exc:
                assert exc.code in (400, 404), (path, exc.code)
        for method in ("POST", "PUT", "DELETE", "PATCH"):
            request = urllib.request.Request(f"{base}/api/status", method=method,
                                             data=b"{}")
            try:
                urllib.request.urlopen(request, timeout=5)
                raise AssertionError(f"{method} should not be allowed")
            except urllib.error.HTTPError as exc:
                assert exc.code == 405, (method, exc.code)
    finally:
        server.shutdown()
        server.server_close()


def test_finished_run_with_no_events_shows_history_and_no_fake_live_stage(tmp_path):
    session = _session(tmp_path)
    (session.root / "manifest.jsonl").write_text(
        json.dumps(_entry("toy-0000")) + "\n", encoding="utf-8")
    payload = dash._status_payload(session, after=0)
    assert payload["events"] == []
    assert payload["progress"] is None
    assert payload["stale"] is False, "no heartbeat is not the same as STALE"
    assert payload["history"][0]["outcome"] == "KEEP"
    assert payload["telemetry"]["loss_history"] == "NOT_AVAILABLE"
    assert payload["telemetry"]["gpu_memory_mb"] == "NOT_AVAILABLE"


def test_stale_flag_comes_from_the_heartbeat(tmp_path):
    session = _session(tmp_path)
    with session.writer() as writer:
        writer.emit("TRAINING", "started", round_index=0, fold=1, folds_total=4)
    fresh = dash._status_payload(session, after=0)
    assert fresh["stale"] is False and fresh["heartbeat_age_seconds"] is not None
    import time

    from research.events import STALE_AFTER_SECONDS

    stale = dash._status_payload(session, after=0)
    status_path = session.status_path
    raw = json.loads(status_path.read_text(encoding="utf-8"))
    raw["updated_monotonic"] = time.monotonic() - STALE_AFTER_SECONDS - 5
    status_path.write_text(json.dumps(raw), encoding="utf-8")
    stale = dash._status_payload(session, after=0)
    assert stale["stale"] is True
    assert stale["heartbeat_age_seconds"] > STALE_AFTER_SECONDS

def test_doctor_card_shows_the_latest_diagnosis_not_not_available(tmp_path):
    """A campaign's per-round postflight lives in the manifest; the dashboard
    must surface it. Otherwise every round with a WARNING renders as
    "NOT_AVAILABLE", which reads like nothing was ever checked."""
    session = _session(tmp_path)
    (session.root / "manifest.jsonl").write_text(
        json.dumps(_entry("toy-0000", doctor={
            "blocked": False,
            "counts": {"BLOCKED": 0, "WARNING": 1, "INFO": 3, "UNVERIFIED": 4},
            "findings": [{"check": "measurement_resolution", "severity": "WARNING",
                          "message": "the difference from the champion is smaller "
                                     "than one standard error"}],
        })) + "\n",
        encoding="utf-8",
    )
    payload = dash._status_payload(session, after=0)
    assert payload["doctor"]["counts"]["WARNING"] == 1
    assert payload["doctor"]["findings"][0]["check"] == "measurement_resolution"
    # and a preflight report, when present, still wins (it is the run's protocol)
    session.write_doctor({"blocked": True, "counts": {"BLOCKED": 1},
                          "findings": [{"check": "user_isolation",
                                        "severity": "BLOCKED", "message": "x"}]})
    payload = dash._status_payload(session, after=0)
    assert payload["doctor"]["findings"][0]["check"] == "user_isolation"
