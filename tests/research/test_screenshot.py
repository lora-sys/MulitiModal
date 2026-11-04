"""Screenshot helper: it must fail loudly, not report a page it never captured."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from scripts.autoresearch_screenshot import _NoBrowser, find_browser, screenshot

pytestmark = pytest.mark.synthetic


def _fake_browser(tmp_path: Path, *, write_png: bool, exit_code: int = 0) -> Path:
    """A stand-in browser: it either writes the file its argv asks for, or not."""
    script = tmp_path / "fake-browser"
    body = "#!/bin/sh\n"
    if write_png:
        body += 'for arg in "$@"; do case "$arg" in --screenshot=*) out="${arg#*=}";; esac; done\n'
        body += 'printf "\\211PNG\\r\\n" > "$out"\n'
    body += f"exit {exit_code}\n"
    script.write_text(body)
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


def test_captures_a_png_when_the_browser_writes_one(tmp_path):
    browser = _fake_browser(tmp_path, write_png=True)
    out = screenshot("http://127.0.0.1:1/static", tmp_path / "shots" / "run.png",
                     browser=str(browser))
    assert out.exists() and out.read_bytes().startswith(b"\x89PNG")
    assert out.parent.is_dir(), "the output directory is created when missing"


def test_a_browser_that_writes_nothing_is_an_error_not_a_success(tmp_path):
    browser = _fake_browser(tmp_path, write_png=False, exit_code=0)
    with pytest.raises(RuntimeError, match="without writing"):
        screenshot("http://127.0.0.1:1/static", tmp_path / "run.png",
                   browser=str(browser))


def test_a_failed_browser_is_an_error(tmp_path):
    browser = _fake_browser(tmp_path, write_png=False, exit_code=3)
    with pytest.raises(RuntimeError):
        screenshot("http://127.0.0.1:1/static", tmp_path / "run.png",
                   browser=str(browser))


def test_missing_browser_says_not_available_rather_than_pretending(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.autoresearch_screenshot.shutil.which", lambda _n: None)
    monkeypatch.setattr("scripts.autoresearch_screenshot.Path.exists", lambda _self: False)
    assert find_browser() is None
    with pytest.raises(_NoBrowser, match="no headless browser found"):
        screenshot("http://127.0.0.1:1/static", tmp_path / "run.png")

    # the CLI maps "no browser" to exit 2 and says NOT_AVAILABLE: a delivery that
    # has no picture must say so rather than report a page it never captured.
    # Called in-process: a real host may well have a browser, so the assertion is
    # about this code path, not about the machine running the test.
    from scripts.autoresearch_screenshot import main

    assert main(["--url", "http://127.0.0.1:1/static",
                 "--out", str(tmp_path / "x.png")]) == 2


def test_report_persists_the_chart_beyond_the_browser(tmp_path):
    """A live page is a view, not a record: the plotted series and a
    self-contained page must be written, so the curve survives without a browser
    and without this code."""
    import json

    from research.report import write_report
    from research.session import ResearchSession

    session = ResearchSession.create(
        tmp_path / "run", task="toy", editable="src/research/candidate_toy.py",
        worktree=str(tmp_path / "wt"), branch="autoresearch/chart",
        folds=2, repeats=1, max_rounds=3, budget_seconds=1.0,
        primary_metric="toy/macro_f1",
    )
    (session.root / "manifest.jsonl").write_text(
        json.dumps({"run_id": "toy-0000", "round_index": 0, "decision": "KEEP",
                    "status": "completed", "outcome": "KEEP", "reason": "baseline",
                    "hypothesis": "the baseline", "best_so_far": 0.5,
                    "metrics": {"primary_value": 0.5, "primary_sem": 0.01}}) + "\n",
        encoding="utf-8",
    )
    paths = write_report(session, out_dir=tmp_path / "out")
    series = json.loads(paths["chart_series"].read_text(encoding="utf-8"))
    assert series["points"][0]["value"] == 0.5
    assert series["points"][0]["outcome"] == "KEEP"
    page = paths["page_archive"].read_text(encoding="utf-8")
    assert "window.__STATIC_PAYLOAD__" in page, "the archive must carry its data"
    assert '"frozen": true' in page, "the archive must not poll"


def test_the_live_page_hydrates_before_it_polls(tmp_path):
    """The phone case: a first paint that waits for a request can hang forever on
    a roaming network. `/` must inline the payload (first paint from data) and
    only then start polling."""
    from research.dashboard import render_static
    from research.session import ResearchSession

    session = ResearchSession.create(
        tmp_path / "run", task="toy", editable="src/research/candidate_toy.py",
        worktree=str(tmp_path / "wt"), branch="autoresearch/chart",
        folds=2, repeats=1, max_rounds=3, budget_seconds=1.0,
        primary_metric="toy/macro_f1",
    )
    live = render_static(session, frozen=False)
    assert '"frozen": false' in live
    assert "if (window.__STATIC_PAYLOAD__)" in live
    assert "setInterval(tick, 3000)" in live, "the live page must still poll"
    assert "ctl.abort()" in live, "a hung request must surface, not hang"


def test_root_route_polls_and_static_route_does_not(tmp_path):
    """/`` used frozen=True by accident, so the 3s poll never started and
    the live page silently froze after first paint. /static must stay frozen
    so a screenshot can settle."""
    import urllib.request

    from research import dashboard as dash
    from research.session import ResearchSession

    session = ResearchSession.create(
        tmp_path / "run", task="toy", editable="src/research/candidate_toy.py",
        worktree=str(tmp_path / "wt"), branch="autoresearch/poll",
        folds=2, repeats=1, max_rounds=3, budget_seconds=1.0,
        primary_metric="toy/macro_f1",
    )
    import threading
    from http.server import ThreadingHTTPServer

    server = ThreadingHTTPServer(("127.0.0.1", 0), dash.make_handler(session))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        root = urllib.request.urlopen(base + "/", timeout=5).read().decode()
        static = urllib.request.urlopen(base + "/static", timeout=5).read().decode()
        assert '"frozen": false' in root
        assert '"frozen": true' in static
    finally:
        server.shutdown()
        server.server_close()


def test_history_view_has_round_and_stage_controls(tmp_path):
    """The history view is the selectable surface: round, stage and the
    unified trend grid must be real DOM, not just a string in a comment."""
    from research.dashboard import render_page

    page = render_page()
    for fragment in ('id="roundSel"', 'id="stageSel"', 'id="trends"',
                     'data-view="history"', 'id="sliceEvents"'):
        assert fragment in page, fragment
