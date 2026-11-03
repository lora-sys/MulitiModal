"""Expose an already-running loopback server on the tailnet — automatically.

Why this exists: the dashboard is loopback-only by design, so a human who wants to
watch a run had to set up a tunnel by hand every time. This makes that automatic:
whenever a dashboard starts, its port is published to the **tailnet** so any device
the user already has on their Tailscale can open it.

Three deliberate boundaries:

* **The server itself does not move.** It keeps binding 127.0.0.1; `tailscale serve`
  proxies to it from the sidecar. Nothing here widens the listen address, so the
  page is still unreachable outside the tailnet even if Tailscale is off.
* **`serve`, never `funnel`.** Serve is tailnet-only; funnel would publish to the
  public internet and this page has no authentication. The module refuses to call
  funnel and says why.
* **Best effort, and honest about it.** If Tailscale is missing, not connected, or
  the process lacks operator permission, the local URL still works and the result
  says `NOT_AVAILABLE` with the one command that would fix it. A convenience must
  never turn into a failure of the run.
"""

from __future__ import annotations

import json
import shutil
import subprocess

SERVE_FLAG = "--bg"
#: `tailscale serve` needs root unless the operator is set for this user; this is
#: the one-time command that removes the need for sudo on every start
OPERATOR_HINT = "sudo tailscale set --operator=$USER"


def _tailscale() -> str | None:
    return shutil.which("tailscale")


def tailnet_hostname(binary: str) -> str | None:
    """This node's tailnet DNS name, from `tailscale status --json`."""
    try:
        result = subprocess.run([binary, "status", "--json"], capture_output=True,
                                text=True, timeout=15)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    dns = ((payload.get("Self") or {}).get("DNSName") or "").rstrip(".")
    return dns or None


def _error_hint(text: str) -> str:
    if "operator" in text or "root" in text.lower() or "sudo" in text:
        return f"run once: {OPERATOR_HINT}"
    if "not running" in text.lower() or "stopped" in text.lower():
        return "tailscale is not connected on this host"
    return ""


def serve_local(port: int, *, https_port: int | None = None,
                binary: str | None = None) -> dict:
    """Publish `127.0.0.1:<port>` on the tailnet and return the URL.

    `https_port` defaults to the same number as the local port, so the URL mirrors
    the port in the logs and two runs never fight over one tailnet endpoint.
    """
    binary = binary or _tailscale()
    if not binary:
        return {"available": False, "reason": "tailscale is not installed",
                "local_url": f"http://127.0.0.1:{port}/"}
    https_port = int(https_port or port)
    target = f"http://127.0.0.1:{port}"
    command = [binary, "serve", SERVE_FLAG, f"--https={https_port}", target]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=45)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}",
                "local_url": f"http://127.0.0.1:{port}/"}
    output = ((result.stdout or "") + (result.stderr or "")).strip()
    if result.returncode != 0:
        return {"available": False,
                "reason": output.splitlines()[-1] if output else "serve failed",
                "hint": _error_hint(output),
                "local_url": f"http://127.0.0.1:{port}/"}
    hostname = tailnet_hostname(binary)
    return {
        "available": True,
        "url": f"https://{hostname}:{https_port}/" if hostname
               else f"https://<this-node>:{https_port}/",
        "https_port": https_port,
        # the boundary is stated in the result, not only in a docstring: a reader
        # of a run's output can see that nothing was published to the internet
        "scope": "tailnet only (serve, never funnel)",
        "local_url": f"http://127.0.0.1:{port}/",
    }