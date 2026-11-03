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
    assert len(page) <= dash.MAX_PAGE_BYTES, "the page must fit the page cap"
    assert "echarts" in page, "the chart engine is inlined, not linked"
    assert "cdn.jsdelivr" not in page, "an offline page must not reference a CDN"
    # our renderer must never build markup from data. The vendored ECharts bundle
    # has innerHTML of its own inside it — that is the library's business and it is
    # pinned by hash in vendor/THIRD_PARTY_NOTICES.md — so the assertion is scoped
    # to OUR script, which is the block after the engine.
    ours = page.rsplit("<script>", 1)[-1]
    assert "innerHTML" not in ours, "our renderer must not write data as markup"
    assert "textContent" in ours


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
                limit = (dash.MAX_PAGE_BYTES if path == "/"
                         else dash.MAX_RESPONSE_BYTES)
                assert len(body) <= limit
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

def test_chart_follows_the_lieflat_g18_skeleton(tmp_path):
    """G18 · Draw-in + Counter is the upstream card for "one line + one big
    number", which is what this page draws. Its skeleton — cubicOut draw-in, a
    sparse label row, an area gradient, a counter in a graphic group, scroll-in
    reveal with click-to-replay and a reduced-motion escape — must be the real
    implementation, not a lookalike."""
    page = dash.render_page()
    for fragment in ("animationEasing: \"cubicOut\"", "type: \"line\"", "symbol: \"none\"",
                     "areaStyle", "id: \"kpi\"", "IntersectionObserver",
                     "prefers-reduced-motion", "click", "requestAnimationFrame"):
        assert fragment in page, fragment
    # the template draws a sparse axis label row, not one label per point
    assert "interval: Math.max(0, Math.ceil" in page


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


def test_absent_numbers_never_get_a_unit_glued_on(tmp_path):
    """`NOT_AVAILABLEs` is what you get when a heartbeat carries no elapsed time
    and the renderer appends the unit unconditionally. The page must say
    NOT_AVAILABLE, and a number must carry its unit."""
    page = dash.render_page()
    # the number path formats and then appends the unit
    assert "p.elapsed_seconds.toFixed(1)" in page
    # the absence path says NOT_AVAILABLE and adds no unit
    assert 'typeof p.elapsed_seconds === "number"' in page
    # and the buggy shape is gone
    assert 'fmt(p.elapsed_seconds, 1) + "s' not in page


def test_fixed_dashboard_follows_the_newest_run(tmp_path):
    """One dashboard, one port, following a runs root: starting a campaign must
    not need a new process, and a reader must be able to see which runs exist."""
    import time

    from research.dashboard import list_runs, resolve_run

    root = tmp_path / "runs"
    assert resolve_run(root) is None and list_runs(root) == []

    older = root / "run-a"
    newer = root / "run-b"
    ids = {}
    import os
    for path in (older, newer):
        ids[path] = ResearchSession.create(
            path, task="toy", editable="src/research/candidate_toy.py",
            worktree=str(tmp_path / "wt"), branch="autoresearch/x",
            folds=2, repeats=1, max_rounds=3, budget_seconds=1.0,
            primary_metric="toy/macro_f1",
        ).run_id
    # mtime decides; the ids a reader sees are the generated ones, not dir names
    os.utime(newer / "run.json", (2_000_000_000, 2_000_000_000))
    os.utime(older / "run.json", (1_000_000_000, 1_000_000_000))
    assert resolve_run(root) == newer
    runs = list_runs(root)
    assert [row["run_id"] for row in runs] == [ids[newer], ids[older]], runs
    assert all("rounds_completed" in row for row in runs)


def test_fixed_dashboard_routes_pin_a_run_and_serve_the_root(tmp_path):
    """`?run=<id>` replays a chosen run; without it, the newest is served."""
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer

    from research.dashboard import make_runs_handler

    root = tmp_path / "runs"
    created = [
        ResearchSession.create(
            root / f"run-{i}", task="toy", editable="src/research/candidate_toy.py",
            worktree=str(tmp_path / "wt"), branch="autoresearch/x",
            folds=2, repeats=1, max_rounds=3, budget_seconds=1.0,
            primary_metric="toy/macro_f1",
        ).run_id
        for i in range(2)
    ]
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_runs_handler(root))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        runs = json.loads(urllib.request.urlopen(f"{base}/api/runs", timeout=5).read())
        assert {row["run_id"] for row in runs["runs"]} == set(created)
        # pin by RUN ID — what the page and the reports show — not by directory
        pinned = json.loads(urllib.request.urlopen(
            f"{base}/api/status?run={created[0]}", timeout=5).read())
        assert pinned["run_id"] == created[0]
        # an unknown run is a stated 404 with the list, not an empty page
        try:
            urllib.request.urlopen(f"{base}/api/status?run=nope", timeout=5)
            raise AssertionError("unknown run must 404")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
            assert b"unknown run" in exc.read()
    finally:
        server.shutdown()
        server.server_close()


def test_empty_runs_root_says_so_instead_of_loading_forever(tmp_path):
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer

    from research.dashboard import make_runs_handler

    server = ThreadingHTTPServer(("127.0.0.1", 0),
                                 make_runs_handler(tmp_path / "empty"))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{server.server_address[1]}/",
                                   timeout=5)
            raise AssertionError("an empty root must not pretend to serve a run")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404 and b"no runs yet" in exc.read()
    finally:
        server.shutdown()
        server.server_close()
