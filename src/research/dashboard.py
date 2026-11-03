"""Read-only live view of a run: stdlib HTTP + one page + 3-second polling.

Deliberate limits, because a training dashboard is an easy place to leak things
and an easy place to introduce a dependency:

* **Read-only.** The route whitelist is `GET /`, `/api/status`, `/api/events`,
  `/api/history`. There is no route that starts, stops, patches, scores or
  unlocks anything; those stay in the CLI.
* **localhost by default.** Binding elsewhere is an explicit operator choice and
  is logged, because the page is not authenticated.
* **No file serving.** Every response is built from the run directory through
  `ResearchSession`. There is no path parameter to traverse, and `/` returns a
  constant string — request paths are only matched, never opened.
* **No raw data.** The run's manifest and records are never exposed: only the
  trusted metrics, the research trail and the Doctor's findings. Subject ids,
  label values and file contents have no route.
* **Capped.** Events are paged (`after=<seq>`, max 500) and free-text fields are
  truncated, so one runaway round cannot flood a reader.
* **XSS-safe.** The page renders every value with `textContent`; no response
  field is ever interpolated into HTML.
* **Independent lifetime.** The server reads files; it never touches the
  training process, so closing the tab, refreshing or opening three tabs cannot
  affect a run.
* **STALE is shown, not hidden.** A run whose heartbeat is older than
  `STALE_AFTER_SECONDS` renders as STALE instead of a frozen "TRAINING".

Charts are drawn with inline SVG on the client; no chart library, no bundler,
no WebSocket (3-second polling is enough for a research loop whose rounds take
minutes).
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from research.session import ResearchSession

MAX_EVENTS_PER_REQUEST = 500
MAX_TEXT_CHARS = 400
MAX_RESPONSE_BYTES = 256 * 1024
PAGE_TITLE = "AutoResearch dashboard"


def _clip(value, limit: int = MAX_TEXT_CHARS):
    if isinstance(value, str):
        return value if len(value) <= limit else value[: limit - 3] + "..."
    if isinstance(value, dict):
        return {str(k)[:60]: _clip(v, limit) for k, v in list(value.items())[:40]}
    if isinstance(value, list):
        return [_clip(v, limit) for v in value[:40]]
    return value


def _status_payload(session: ResearchSession, after: int) -> dict:
    view = session.status(after_seq=after, events_limit=MAX_EVENTS_PER_REQUEST)
    champion = view.get("champion_run_id")
    history = session.history()["rounds"]
    live = _clip(view.get("progress"))
    return {
        "run_id": view["run_id"],
        "task": view["task"],
        "primary_metric": view["primary_metric"],
        "rounds_completed": view["rounds_completed"],
        "max_rounds": view["max_rounds"],
        "rounds_remaining": view["rounds_remaining"],
        "decisions": view["decisions"],
        "best_metric": view["best_metric"],
        "champion_run_id": champion,
        "champion_commit": (view.get("champion_commit") or None),
        "stop_requested": view["stop_requested"],
        "selection_policy": _clip(view.get("policy") or {}),
        "progress": live,
        "stale": bool((live or {}).get("stale")),
        "heartbeat_age_seconds": (live or {}).get("heartbeat_age_seconds"),
        "last_event_seq": view["last_event_seq"],
        "events": [_clip(e) for e in view["events"]],
        "history": [_clip(r) for r in history],
        "current_round": history[-1] if history else None,
        "telemetry": _telemetry(history),
    }


def _telemetry(history: list[dict]) -> dict:
    """Only what the host actually recorded; everything else says NOT_AVAILABLE."""
    losses = None
    gpu = None
    for row in reversed(history):
        if losses is None and isinstance(row.get("loss_history"), list):
            losses = row["loss_history"]
        if gpu is None and (row.get("resource_telemetry") or {}).get("gpu_memory_mb") is not None:
            gpu = row["resource_telemetry"]["gpu_memory_mb"]
    return {
        "loss_history": losses if losses is not None else "NOT_AVAILABLE",
        "gpu_memory_mb": gpu if gpu is not None else "NOT_AVAILABLE",
    }


def _history_payload(session: ResearchSession) -> dict:
    history = session.history()
    return {
        "run_id": history["run_id"],
        "rounds": [_clip(r) for r in history["rounds"]],
        "champion": _clip(history["champion"]) if history["champion"] else None,
        "decisions": history["decisions"],
        "doctor": _clip(session.read_doctor()),
    }


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#0f1115;--fg:#e6e8eb;--muted:#9aa3ad;--card:#171a21;--line:#2a2f3a;
--keep:#2a9d8f;--revert:#e76f51;--blocked:#e63946;--warn:#e9c46a;--info:#457b9d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{display:flex;gap:16px;flex-wrap:wrap;align-items:baseline;padding:14px 18px;
border-bottom:1px solid var(--line);background:var(--card)}
h1{font-size:16px;margin:0}.meta{color:var(--muted)}
.wrap{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px;padding:14px 18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.card h2{font-size:13px;margin:0 0 8px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.badge{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;border:1px solid var(--line)}
.stale{background:#3a2a2a;border-color:var(--blocked);color:#ffd9d9}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:4px 6px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:500}
.finding-BLOCKED{color:var(--blocked)}.finding-WARNING{color:var(--warn)}
.finding-INFO{color:var(--info)}.finding-UNVERIFIED{color:var(--muted)}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
.na{color:var(--muted);font-style:italic}
footer{color:var(--muted);padding:10px 18px;border-top:1px solid var(--line)}
</style></head>
<body>
<header>
  <h1 id="title">AutoResearch</h1>
  <span class="meta" id="idsub">loading…</span>
  <span class="badge" id="stage">…</span>
  <span class="meta" id="progress"></span>
  <span class="meta" id="heartbeat"></span>
</header>
<div class="wrap">
  <section class="card"><h2>Current hypothesis</h2><div id="hypothesis" class="na">NOT_AVAILABLE</div></section>
  <section class="card"><h2>Budget</h2><div id="budget"></div></section>
  <section class="card" style="grid-column:1/-1"><h2>Trusted metric per round</h2>
    <div id="chart"></div></section>
  <section class="card" style="grid-column:1/-1"><h2>Rounds</h2>
    <table id="rounds"><thead><tr><th>#</th><th>hypothesis</th><th>outcome</th>
    <th>value</th><th>best</th><th>next action</th></tr></thead><tbody></tbody></table></section>
  <section class="card"><h2>Doctor</h2><div id="doctor"></div></section>
  <section class="card"><h2>Telemetry</h2><div id="telemetry"></div></section>
  <section class="card" style="grid-column:1/-1"><h2>Recent events</h2>
    <div id="events" class="mono"></div></section>
</div>
<footer>Read-only view. Operations (run/stop/patch/finalize) live in the CLI.</footer>
<script>
const state = {lastSeq: 0, history: []};
function el(id){return document.getElementById(id);}
function text(node, value){ node.textContent = (value === null || value === undefined || value === "")
  ? "NOT_AVAILABLE" : String(value); }
function fmt(v, d){ return (v === null || v === undefined) ? "NOT_AVAILABLE"
  : (typeof v === "number" ? v.toFixed(d === undefined ? 4 : d) : String(v)); }
function renderStatus(s){
  text(el("title"), "AutoResearch · " + s.task);
  text(el("idsub"), s.run_id + " · " + s.primary_metric);
  const stage = (s.progress && s.progress.stage) || "IDLE";
  el("stage").textContent = s.stale ? "STALE" : stage;
  el("stage").className = "badge" + (s.stale ? " stale" : "");
  const p = s.progress || {};
  el("progress").textContent = "round " + s.rounds_completed + "/" + s.max_rounds
    + (p.fold !== null && p.fold !== undefined
        ? (" · fold " + (p.fold + 1) + "/" + (p.folds_total || "?")) : "")
    + (p.status ? (" · " + p.status) : "");
  el("heartbeat").textContent = s.heartbeat_age_seconds === null
    ? "last update NOT_AVAILABLE"
    : ("last update " + s.heartbeat_age_seconds + "s ago");
  const cur = s.current_round;
  if(cur){
    el("hypothesis").textContent = "";
    const h = document.createElement("div");
    h.textContent = cur.hypothesis || "NOT_AVAILABLE";
    const c = document.createElement("div"); c.className = "meta";
    c.textContent = "change: " + fmt(cur.controlled_change) + " · expect: " + fmt(cur.expected_observation);
    const n = document.createElement("div"); n.className = "meta";
    n.textContent = "next: " + fmt(cur.next_action);
    el("hypothesis").append(h, c, n);
  }
  el("budget").textContent = "rounds " + s.rounds_completed + "/" + s.max_rounds
    + " · decisions " + JSON.stringify(s.decisions)
    + " · best " + fmt(s.best_metric)
    + (p.budget_seconds ? (" · " + fmt(p.elapsed_seconds,1) + "s / " + p.budget_seconds
       + "s in current stage") : " · stage budget NOT_AVAILABLE");
  const tb = el("rounds").querySelector("tbody"); tb.textContent = "";
  for(const r of s.history){
    const tr = document.createElement("tr");
    for(const v of [r.round, r.hypothesis, r.outcome, fmt(r.primary_value), fmt(r.best_so_far), r.next_action]){
      const td = document.createElement("td"); td.textContent = (v === null || v === undefined) ? "NOT_AVAILABLE" : String(v); tr.appendChild(td);
    }
    tb.appendChild(tr);
  }
  const doc = el("doctor"); doc.textContent = "";
  const findings = (s.doctor && s.doctor.findings) || [];
  if(!findings.length){ doc.textContent = "No diagnosis recorded (NOT_AVAILABLE)"; }
  for(const f of findings){
    const d = document.createElement("div");
    d.className = "finding-" + (f.severity || "INFO");
    d.textContent = (f.severity || "") + " · " + (f.check || "") + ": " + (f.message || "");
    doc.appendChild(d);
  }
  const t = el("telemetry");
  t.textContent = "loss: " + fmt(s.telemetry.loss_history) + " · gpu_mb: " + fmt(s.telemetry.gpu_memory_mb);
  drawChart(s.history);
}
function drawChart(rows){
  const host = el("chart"); host.textContent = "";
  const scored = rows.filter(r => typeof r.primary_value === "number");
  if(!scored.length){ host.textContent = "No scored round yet (NOT_AVAILABLE)"; return; }
  const w = 640, h = 200, pad = 30;
  const vals = scored.map(r => r.primary_value);
  const best = rows.filter(r => typeof r.best_so_far === "number").map(r => r.best_so_far);
  const all = vals.concat(best.length ? best : vals);
  const lo = Math.min.apply(null, all), hi = Math.max.apply(null, all);
  const span = (hi - lo) || 1;
  const x = i => pad + (scored.length === 1 ? (w - 2*pad) / 2
      : i * (w - 2*pad) / (scored.length - 1));
  const y = v => h - pad - (v - lo) / span * (h - 2*pad);
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 " + w + " " + h);
  svg.setAttribute("width", "100%"); svg.setAttribute("height", h);
  const line = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
  line.setAttribute("fill", "none"); line.setAttribute("stroke", "#2a9d8f");
  line.setAttribute("stroke-width", "2");
  line.setAttribute("points", scored.map((r,i) => x(i) + "," + y(r.primary_value)).join(" "));
  svg.appendChild(line);
  if(best.length){
    const bl = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
    bl.setAttribute("fill", "none"); bl.setAttribute("stroke", "#9aa3ad");
    bl.setAttribute("stroke-dasharray", "4 3"); bl.setAttribute("stroke-width", "1.5");
    bl.setAttribute("points", best.map((v,i) => x(i) + "," + y(v)).join(" "));
    svg.appendChild(bl);
  }
  scored.forEach((r,i) => {
    const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    c.setAttribute("cx", x(i)); c.setAttribute("cy", y(r.primary_value)); c.setAttribute("r", "3.5");
    c.setAttribute("fill", r.outcome === "KEEP" ? "#2a9d8f" : "#e76f51");
    const t = document.createElementNS("http://www.w3.org/2000/svg", "title");
    t.textContent = (r.run_id || "") + " " + (r.outcome || "") + " " + r.primary_value;
    c.appendChild(t); svg.appendChild(c);
  });
  host.appendChild(svg);
}
async function tick(){
  try{
    const res = await fetch("/api/status?after=" + state.lastSeq, {cache: "no-store"});
    if(!res.ok) throw new Error("status " + res.status);
    const s = await res.json();
    state.history = s.history;
    state.lastSeq = s.last_event_seq || state.lastSeq;
    renderStatus(s);
    const list = el("events");
    for(const e of (s.events || [])){
      if(!e || e.seq <= (list.dataset.lastSeq || 0)) continue;
      const row = document.createElement("div");
      row.textContent = "#" + e.seq + " " + (e.time || "") + " " + (e.stage || "")
        + (e.fold !== null && e.fold !== undefined ? (" f" + e.fold) : "")
        + " " + (e.status || "") + " " + (e.message || "");
      list.prepend(row);
      list.dataset.lastSeq = e.seq;
    }
    while(list.childElementCount > 60) list.removeChild(list.lastChild);
  }catch(err){
    el("stage").textContent = "DISCONNECTED";
    el("stage").className = "badge stale";
    el("heartbeat").textContent = String(err);
  }
}
tick(); setInterval(tick, 3000);
</script>
</body></html>
"""


def render_page() -> str:
    return PAGE.replace("__TITLE__", PAGE_TITLE)


def make_handler(session: ResearchSession):
    class Handler(BaseHTTPRequestHandler):
        server_version = "AutoResearchDashboard/1"
        sys_version = ""

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            if len(body) > MAX_RESPONSE_BYTES:
                status, body = 413, b'{"error":"response too large"}'
                content_type = "application/json"
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: dict, status: int = 200) -> None:
            self._send(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            query = parse_qs(parsed.query)
            if ".." in parsed.path:
                # nothing is opened from a request path; this is belt and braces
                self._json({"error": "path not allowed"}, 400)
                return
            if path == "/":
                self._send(200, render_page().encode("utf-8"),
                           "text/html; charset=utf-8")
            elif path == "/api/status":
                after = int((query.get("after") or ["0"])[0] or 0)
                self._json(_status_payload(session, max(0, after)))
            elif path == "/api/events":
                after = int((query.get("after") or ["0"])[0] or 0)
                events = session.status(after_seq=max(0, after),
                                        events_limit=MAX_EVENTS_PER_REQUEST)["events"]
                self._json({"events": [_clip(e) for e in events],
                            "last_event_seq": events[-1]["seq"] if events else max(0, after)})
            elif path == "/api/history":
                self._json(_history_payload(session))
            else:
                self._json({"error": "not found", "allowed": ["/", "/api/status",
                                                              "/api/events", "/api/history"]}, 404)

        def do_HEAD(self) -> None:  # noqa: N802
            self._send(200, b"", "text/plain")

        def _reject_mutation(self) -> None:
            self._json({"error": "read-only dashboard: operations live in the CLI"}, 405)

        def do_POST(self) -> None:  # noqa: N802
            self._reject_mutation()

        # implemented rather than left to the base handler: the base answers 501
        # ("unsupported method"), which reads like a missing feature instead of
        # the deliberate refusal it is
        def do_PUT(self) -> None:  # noqa: N802
            self._reject_mutation()

        def do_PATCH(self) -> None:  # noqa: N802
            self._reject_mutation()

        def do_DELETE(self) -> None:  # noqa: N802
            self._reject_mutation()

        def log_message(self, fmt: str, *args) -> None:
            print(f"dashboard {self.address_string()} {fmt % args}", file=sys.stderr)

    return Handler


def serve(session: ResearchSession, *, host: str = "127.0.0.1", port: int = 8765,
          once: bool = False) -> int:
    if once:
        sys.stdout.write(render_page())
        return 0
    server = ThreadingHTTPServer((host, port), make_handler(session))
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(f"WARNING: dashboard bound to {host}, which is not loopback, and it "
              "has no authentication", file=sys.stderr)
    print(f"AutoResearch dashboard for {session.run_id} on http://{host}:{port}/ "
          "(read-only). Closing the browser does not affect training.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def spawn_dashboard(session: ResearchSession, *, port: int = 8765,
                    host: str = "127.0.0.1") -> dict:
    """Start the dashboard as its OWN process, so its lifetime is independent.

    The issue requires that closing the browser or the page cannot pause
    training. Running the server in a thread inside the training process would
    violate that; a separate process cannot.
    """
    command = [sys.executable, "-m", "research.cli", "dashboard",
               "--run", str(session.root), "--host", host, "--port", str(port)]
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, start_new_session=True,
                               cwd=str(Path(__file__).resolve().parents[1]))
    return {"pid": process.pid, "url": f"http://{host}:{port}/",
            "command": " ".join(shlex.quote(part) for part in command)}