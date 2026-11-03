"""Screenshot a running dashboard with whatever browser the host already has.

Why this is a script and not a browser automation framework: the acceptance
requirement is "the page really paints, and here is the picture". That needs one
headless browser invocation, not a dependency. The script looks for a browser the
host already provides (chromium/chrome/firefox), runs it headless against a URL,
and writes a PNG — no install, no WebDriver, no Playwright.

Point it at `/static`, not at `/`: the live page repaints every three seconds, so
an automated capture of it may never settle. `/static` is the same page, same CSS
and same renderers, with the payload inlined and no polling — which is exactly why
that route exists.

    python scripts/autoresearch_screenshot.py --url http://127.0.0.1:8803/static \\
        --out shots/run.png --size 1280x1800

Exit codes: 0 written, 2 no browser found, 1 the browser failed. A delivery that
claims a working page without a picture should fail loudly here rather than
report success.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

#: first match wins; the command templates are the documented headless flags
CANDIDATES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("chromium", (
        "--headless=new", "--no-sandbox", "--disable-gpu",
        "--disable-dev-shm-usage", "--hide-scrollbars",
    )),
    ("chromium-browser", (
        "--headless=new", "--no-sandbox", "--disable-gpu",
        "--disable-dev-shm-usage", "--hide-scrollbars",
    )),
    ("google-chrome", (
        "--headless=new", "--no-sandbox", "--disable-gpu",
        "--disable-dev-shm-usage", "--hide-scrollbars",
    )),
    ("google-chrome-stable", (
        "--headless=new", "--no-sandbox", "--disable-gpu",
        "--disable-dev-shm-usage", "--hide-scrollbars",
    )),
)


def find_browser() -> tuple[str, tuple[str, ...]] | None:
    for name, flags in CANDIDATES:
        path = shutil.which(name)
        if path:
            return path, flags
    # snap installs land here and are not always on PATH
    for name, flags in CANDIDATES:
        snap = Path("/snap/bin") / name
        if snap.exists():
            return str(snap), flags
    return None


def screenshot(url: str, out: Path, *, size: str = "1280x1800",
               settle_ms: int = 4000, browser: str | None = None,
               timeout_s: float = 120.0) -> Path:
    """Capture `url` to `out`. Raises RuntimeError with the reason on failure."""
    if browser:
        command_path, flags = browser, CANDIDATES[0][1]
    else:
        found = find_browser()
        if found is None:
            raise _NoBrowser(
                "no headless browser found (tried chromium, chromium-browser, "
                "google-chrome, google-chrome-stable, and /snap/bin/*)"
            )
        command_path, flags = found
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    command = [
        command_path, *flags,
        f"--window-size={size}",
        # let the page's own renderers finish before the capture: the static
        # route is timer-free, but the JS still draws the SVG on one pass
        f"--virtual-time-budget={int(settle_ms)}",
        f"--screenshot={out}",
        url,
    ]
    try:
        result = subprocess.run(command, text=True, capture_output=True,
                                timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"browser timed out after {timeout_s:.0f}s: {command_path}") from exc
    if not out.exists():
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        tail = detail[-3:] if detail else []
        raise RuntimeError(
            f"browser exited {result.returncode} without writing {out}: {tail}"
        )
    return out


class _NoBrowser(RuntimeError):
    """No headless browser on this host."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="page URL, use /static for a "
                                                     "timer-free capture")
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", default="1280x1800", help="WIDTHxHEIGHT")
    parser.add_argument("--settle-ms", type=int, default=4000)
    parser.add_argument("--browser", default=None,
                        help="override the browser binary")
    args = parser.parse_args(argv)
    try:
        path = screenshot(args.url, Path(args.out), size=args.size,
                          settle_ms=args.settle_ms, browser=args.browser)
    except _NoBrowser as exc:
        print(f"NOT_AVAILABLE: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"screenshot failed: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())