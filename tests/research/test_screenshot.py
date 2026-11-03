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
