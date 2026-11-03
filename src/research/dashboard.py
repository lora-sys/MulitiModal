"""Read-only live view of a run: stdlib HTTP + one page + 3-second polling.

Deliberate limits, because a training dashboard is an easy place to leak things
and an easy place to introduce a dependency:

* **Read-only.** The route whitelist is `GET /`, `/static`, `/api/status`,
  `/api/events`, `/api/history`. There is no route that starts, stops, patches, scores or
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
import os
import shlex
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from research.session import ResearchSession

MAX_EVENTS_PER_REQUEST = 500
MAX_TEXT_CHARS = 400
#: JSON endpoint cap: a status page is a summary, not a data export.
MAX_RESPONSE_BYTES = 256 * 1024
#: Page cap. The page inlines the vendored ECharts bundle (~1.1 MB) so it can render
#: the lieflat Glance charts offline; the JSON cap above still bounds every data
#: endpoint. Without this the page itself would be swapped for a 413 and the
#: dashboard would be unreachable.
MAX_PAGE_BYTES = 4 * 1024 * 1024
PAGE_TITLE = "AutoResearch dashboard"


_ECHARTS_CACHE: str | None = None


def _echarts_source() -> str:
    """The vendored ECharts bundle, read once and inlined into every page.

    Inlined, not linked: the dashboard runs on a loopback port with no network, and
    the upstream lieflat-charts Glance cards (G18 and friends) are ECharts-based, so
    the engine has to travel with the page. See vendor/THIRD_PARTY_NOTICES.md.
    """
    global _ECHARTS_CACHE
    if _ECHARTS_CACHE is None:
        path = Path(__file__).resolve().parent / "vendor" / "echarts.min.js"
        if not path.exists():
            raise RuntimeError(
                f"the vendored ECharts bundle is missing ({path}); the dashboard "
                "cannot render the lieflat Glance charts without it"
            )
        _ECHARTS_CACHE = path.read_text(encoding="utf-8")
    return _ECHARTS_CACHE


def _clip(value, limit: int = MAX_TEXT_CHARS):
    if isinstance(value, str):
        return value if len(value) <= limit else value[: limit - 3] + "..."
    if isinstance(value, dict):
        return {str(k)[:60]: _clip(v, limit) for k, v in list(value.items())[:40]}
    if isinstance(value, list):
        return [_clip(v, limit) for v in value[:40]]
    return value


def _latest_doctor(session: ResearchSession) -> dict | None:
    """The most recent diagnosis for this run.

    `doctor.json` holds the preflight and is what a finished campaign leaves
    behind; the per-round postflight lives in the manifest. Falling back to the
    last round matters: without it a campaign whose rounds all carry a WARNING
    renders as "No diagnosis recorded (NOT_AVAILABLE)", which reads like nothing
    was ever checked.
    """
    report = session.read_doctor()
    if (report or {}).get("findings"):
        return report
    entries = session.entries()
    if entries and (entries[-1].get("doctor") or {}).get("findings"):
        return entries[-1]["doctor"]
    return report


def resolve_run(runs_root: str | Path) -> Path | None:
    """The newest run under a runs root, by `run.json` mtime.

    A dashboard that follows a FIXED runs root is what makes one long-lived
    process enough: a new campaign appears the moment it is created, with no
    restart and no new port. `None` means nothing to show yet.
    """
    root = Path(runs_root)
    if not root.is_dir():
        return None
    candidates = [path.parent for path in root.glob("*/run.json") if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path / "run.json").stat().st_mtime)


def list_runs(runs_root: str | Path) -> list[dict]:
    """Every run under the root, newest first: id, dir, task, rounds, best."""
    root = Path(runs_root)
    out: list[dict] = []
    if not root.is_dir():
        return out
    for spec_path in root.glob("*/run.json"):
        try:
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        run_dir = spec_path.parent
        entry = {"run_id": spec.get("run_id"), "dir": str(run_dir),
                 "task": spec.get("task"),
                 "updated": (run_dir / "run.json").stat().st_mtime}
        try:
            session = ResearchSession.load(run_dir)
            entry["rounds_completed"] = session.rounds_completed()
            entry["best_metric"] = (session.champion() or {}).get("best_so_far")
        except (OSError, ValueError):
            pass
        out.append(entry)
    out.sort(key=lambda row: row.get("updated") or 0, reverse=True)
    return out


def _status_payload(session: ResearchSession, after: int) -> dict:
    view = session.status(after_seq=after, events_limit=MAX_EVENTS_PER_REQUEST)
    champion = view.get("champion_run_id")
    history = session.history()["rounds"]
    live = _clip(view.get("progress"))
    # the Doctor card must show the LATEST diagnosis. `doctor.json` holds the
    # preflight (and is what a finished campaign leaves behind); the per-round
    # postflight lives in the manifest, so fall back to the last round's own
    # findings — otherwise a run with a WARNING per round displays
    # "NOT_AVAILABLE", which reads like "nothing was checked".
    doctor = _latest_doctor(session)
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
        "doctor": _clip(doctor),
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
        "doctor": _clip(_latest_doctor(session)),
    }


#: The page follows the `lieflat-charts` skill's Mono grammar (see
#: skills/autoresearch/SKILL.md § Upstream visual skill). Values are lifted from
#: that skill's token file rather than eyeballed:
#:
#:   ink #1C1C1A · paper #F0EFEB · muted #8F8E88 · faint #C6C5BF · grid #DEDDD6
#:   grey ladder #1C1C1A #4A4944 #6A6963 #8F8E88 #B0AFA9 #C6C5BF #D8D7D1
#:
#: The skill's Glance family is the right family here, and its rule requires the
#: reason to be written down: Lupi Editorial and Lupi Basics are static
#: editorial charts, while this page is a live monitoring strip that must be
#: readable in three seconds at a glance, over a stream that repaints every
#: three seconds. That is exactly the case Glance exists for.
#:
#: Two deliberate departures, both because this page must run offline on a
#: loopback monitoring port with no network: the Inter webfont link is replaced
#: by a system-font stack that starts with Inter, and there is no entrance
#: animation library — only a CSS reveal that honours prefers-reduced-motion.
PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{
  --ink:#1C1C1A; --paper:#F0EFEB; --muted:#8F8E88; --faint:#C6C5BF; --grid:#DEDDD6;
  --l1:#1C1C1A; --l2:#4A4944; --l3:#6A6963; --l4:#8F8E88; --l5:#B0AFA9; --l6:#C6C5BF;
  --font:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --radius:24px;
}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--font);
     -webkit-font-smoothing:antialiased}
/* the run bar: conclusion first, then the facts, joined with · */
header{padding:28px 32px 4px}
h1{margin:0;font-size:26px;font-weight:700;letter-spacing:-.02em;line-height:1.15}
.sub{margin:8px 0 0;color:var(--muted);font-size:14px;font-weight:500}
.sub b{color:var(--ink);font-weight:700}
.pill{display:inline-block;margin-left:10px;padding:2px 10px;border-radius:999px;
      font-size:11px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;
      background:var(--ink);color:var(--paper)}
.pill.stale{background:var(--l3)}
.wrap{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:20px;
      padding:20px 32px 32px}
/* borderless cards, separated by whitespace; radius 24 */
.card{background:var(--paper);border-radius:var(--radius);padding:22px 24px 18px}
.card.wide{grid-column:1/-1}
h2{margin:0;font-size:16.5px;font-weight:700;letter-spacing:-.02em}
.csub{margin:6px 0 16px;color:var(--muted);font-size:12px;font-weight:500}
.src{margin:14px 0 0;color:var(--faint);font-size:10px;font-weight:600;
     letter-spacing:.12em;text-transform:uppercase}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:7px 6px;border-bottom:1px solid var(--grid);vertical-align:top}
th{color:var(--muted);font-weight:600;font-size:10px;letter-spacing:.1em;text-transform:uppercase}
td.num{font-variant-numeric:tabular-nums;font-weight:600}
.tag{font-size:10px;font-weight:700;letter-spacing:.06em;text-transform:uppercase}
.tag.KEEP{color:var(--ink)}
.tag.REVERT{color:var(--l4)}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11.5px;
      color:var(--l2);line-height:1.7}
.row{display:flex;gap:10px;align-items:baseline;color:var(--muted);font-size:12.5px}
.row b{color:var(--ink);font-weight:700}
.finding{font-size:12.5px;margin:6px 0}
.finding.BLOCKED{color:var(--ink);font-weight:700}
.finding.WARNING{color:var(--l2);font-weight:600}
.finding.INFO,.finding.UNVERIFIED{color:var(--muted)}
.na{color:var(--muted);font-style:italic}
.hyp{font-size:15px;font-weight:600;line-height:1.45}
.kv{margin-top:10px;color:var(--muted);font-size:12.5px}
.kv b{color:var(--ink)}
svg text{font-family:var(--font)}
footer{padding:18px 32px 28px;color:var(--faint);font-size:10px;font-weight:600;
       letter-spacing:.12em;text-transform:uppercase}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style></head>
<body>
<header>
  <h1 id="title">AutoResearch</h1>
  <p class="sub" id="idsub">loading…</p>
</header>
<div class="wrap">
  <section class="card wide">
    <h2 id="verdict">Waiting for the first scored round</h2>
    <p class="csub" id="verdictsub">the trusted metric decides; a candidate never reports its own score</p>
    <div class="row" id="progress"></div>
    <div class="kv" id="heartbeat"></div>
  </section>
  <section class="card wide">
    <h2>Trusted metric per round, against the best so far</h2>
    <p class="csub" id="chartsub">trusted metric per round · the line draws, the number counts · click to replay</p>
    <div id="chart"></div>
    <p class="src" id="chartsrc">source: run manifest · trusted evaluator (isolated process)</p>
  </section>
  <section class="card">
    <h2 id="rounds-title">What was tried, in order</h2>
    <p class="csub">hypothesis · decided by the trusted metric, not by the candidate</p>
    <table id="rounds"><thead><tr><th>#</th><th>hypothesis</th><th>decision</th>
      <th>value</th><th>best</th><th>next action</th></tr></thead><tbody></tbody></table>
    <p class="src">source: run manifest · append-only</p>
  </section>
  <section class="card">
    <h2 id="current-title">The hypothesis of this round</h2>
    <p class="csub">what it changed · what it expected to observe</p>
    <div id="hypothesis" class="na">NOT_AVAILABLE</div>
    <p class="src">source: proposer record in the manifest</p>
  </section>
  <section class="card">
    <h2>Research Doctor</h2>
    <p class="csub">BLOCKED stops the campaign · UNVERIFIED is not a pass</p>
    <div id="doctor"></div>
    <p class="src">source: doctor.json · preflight and postflight</p>
  </section>
  <section class="card">
    <h2>Resources and telemetry</h2>
    <p class="csub">only what the host measured; nothing is estimated</p>
    <div id="telemetry"></div>
    <p class="src">source: coordinator status + training child</p>
  </section>
  <section class="card wide">
    <h2>Recent coordinator events</h2>
    <p class="csub">written by the host that also runs the evaluator · never by candidate code</p>
    <div id="events" class="mono"></div>
    <p class="src">source: events.jsonl · single writer</p>
  </section>
</div>
<footer>read-only view · operations live in the cli · charts follow the lieflat-charts mono grammar</footer>
<script>__ECHARTS__</script>
<script>
const state = {lastSeq: 0, history: [], drawn: false};
const T = {ink:"#1C1C1A", paper:"#F0EFEB", muted:"#8F8E88", faint:"#C6C5BF", grid:"#DEDDD6",
           l1:"#1C1C1A", l2:"#4A4944", l3:"#6A6963", l4:"#8F8E88", l5:"#B0AFA9", l6:"#C6C5BF"};
function el(id){return document.getElementById(id);}
function text(node, value){ node.textContent = (value === null || value === undefined || value === "")
  ? "NOT_AVAILABLE" : String(value); }
function fmt(v, d){ return (v === null || v === undefined) ? "NOT_AVAILABLE"
  : (typeof v === "number" ? v.toFixed(d === undefined ? 4 : d) : String(v)); }
function svgEl(name){ return document.createElementNS("http://www.w3.org/2000/svg", name); }
function renderStatus(s){
  text(el("title"), "AutoResearch · " + s.task);
  const stage = (s.progress && s.progress.stage) || "IDLE";
  el("idsub").textContent = "";
  const b = document.createElement("b"); b.textContent = s.run_id;
  el("idsub").append(b, document.createTextNode(" · " + s.primary_metric
    + " · round " + s.rounds_completed + "/" + s.max_rounds
    + (s.champion_run_id ? (" · champion " + s.champion_run_id) : "")));
  const pill = document.createElement("span");
  pill.className = "pill" + (s.stale ? " stale" : "");
  pill.textContent = s.stale ? "STALE" : stage;
  el("idsub").append(pill);

  const best = s.best_metric;
  text(el("verdict"), best === null || best === undefined
    ? "No scored round yet"
    : "Best trusted " + s.primary_metric + " so far: " + fmt(best)
      + ", decided in " + s.rounds_completed + " round(s)");
  text(el("verdictsub"), "decisions " + JSON.stringify(s.decisions)
    + " · a KEEP needs to clear the noise margin, not just move the number");

  const p = s.progress || {};
  el("progress").textContent = "";
  const bits = [["stage", s.stale ? "STALE" : stage]];
  if (p.round !== null && p.round !== undefined) bits.push(["round", p.round]);
  if (p.fold !== null && p.fold !== undefined) bits.push(["fold", (p.fold + 1) + "/" + (p.folds_total || "?")]);
  // a heartbeat carries no elapsed time; render the absence instead of gluing a
  // unit onto the words NOT_AVAILABLE ("NOT_AVAILABLEs")
  if (p.budget_seconds) bits.push(["stage budget",
    (typeof p.elapsed_seconds === "number" ? p.elapsed_seconds.toFixed(1) : "NOT_AVAILABLE")
    + " / " + p.budget_seconds + "s"]);
  for (const [k, v] of bits) {
    const span = document.createElement("span");
    const kb = document.createElement("b"); kb.textContent = k + ": ";
    span.append(kb, document.createTextNode(String(v)));
    el("progress").append(span);
  }
  text(el("heartbeat"), s.heartbeat_age_seconds === null
    ? "last update: NOT_AVAILABLE (no event has been written yet)"
    : "last update: " + s.heartbeat_age_seconds + "s ago");

  const cur = s.current_round;
  el("hypothesis").textContent = "";
  if (cur) {
    const h = document.createElement("div"); h.className = "hyp"; text(h, cur.hypothesis);
    const c = document.createElement("div"); c.className = "kv";
    const cb = document.createElement("b"); cb.textContent = "changed: ";
    c.append(cb, document.createTextNode(fmt(cur.controlled_change)));
    const e2 = document.createElement("div"); e2.className = "kv";
    const eb = document.createElement("b"); eb.textContent = "expected: ";
    e2.append(eb, document.createTextNode(fmt(cur.expected_observation)));
    const n = document.createElement("div"); n.className = "kv";
    const nb = document.createElement("b"); nb.textContent = "next: ";
    n.append(nb, document.createTextNode(fmt(cur.next_action)));
    el("hypothesis").append(h, c, e2, n);
    text(el("current-title"), (cur.outcome ? cur.outcome + " · " : "") + "round " + cur.round);
  }

  const tb = el("rounds").querySelector("tbody"); tb.textContent = "";
  for (const r of s.history) {
    const tr = document.createElement("tr");
    const cells = [r.round, r.hypothesis, null, fmt(r.primary_value), fmt(r.best_so_far), r.next_action];
    cells.forEach((v, i) => {
      const td = document.createElement("td");
      if (i === 2) {
        const tag = document.createElement("span");
        tag.className = "tag " + (r.outcome || "");
        tag.textContent = r.outcome || "NOT_AVAILABLE";
        td.appendChild(tag);
      } else {
        td.textContent = (v === null || v === undefined) ? "NOT_AVAILABLE" : String(v);
        if (i === 3 || i === 4) td.className = "num";
      }
      tr.appendChild(td);
    });
    tb.appendChild(tr);
  }

  const doc = el("doctor"); doc.textContent = "";
  const findings = (s.doctor && s.doctor.findings) || [];
  if (!findings.length) {
    const span = document.createElement("div"); span.className = "na";
    span.textContent = "No diagnosis recorded (NOT_AVAILABLE)";
    doc.appendChild(span);
  }
  for (const f of findings) {
    const d = document.createElement("div");
    d.className = "finding " + (f.severity || "INFO");
    d.textContent = (f.severity || "") + " · " + (f.check || "") + ": " + (f.message || "");
    doc.appendChild(d);
  }

  const tel = el("telemetry"); tel.textContent = "";
  for (const [k, v] of [["training loss", s.telemetry.loss_history],
                        ["gpu memory (MB)", s.telemetry.gpu_memory_mb]]) {
    const d = document.createElement("div"); d.className = "kv";
    const kb = document.createElement("b"); kb.textContent = k + ": ";
    d.append(kb, document.createTextNode(fmt(v)));
    tel.appendChild(d);
  }
  drawChart(s.history);
}
/* ── G18 · Draw-in + Counter (lieflat-charts Glance family) ───────────────────
   Skeleton taken from the upstream gallery card "H1 revenue, drawn in one stroke":
   one line drawn in over DUR with cubicOut, a sparse axis label row, an area
   gradient, and a counter rendered as a graphic group beside the chart. Data,
   titles and annotations are ours; the structure, easing and spacing are the
   template's. A second line series carries the best-so-far reference — same chart
   family, not a second chart type.
   Reveal: scroll into view, replay on click, and no animation at all under
   prefers-reduced-motion (the upstream token's `obsReveal` contract).
   ───────────────────────────────────────────────────────────────────────────── */
const G18 = {DUR: 1400, INK: "#1C1C1A", MUTED: "#8F8E88", FAINT: "#C6C5BF"};
let g18Gen = 0;

function obsReveal(node, draw){
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let revealed = false;
  const run = () => { g18Gen += 1; draw(node); };
  if (node.dataset.replayBound !== "1") {
    node.dataset.replayBound = "1";
    node.style.cursor = "pointer";
    node.addEventListener("click", run);
  }
  if (reduce) { run(); return; }
  const io = new IntersectionObserver((entries) => {
    if (entries.some(e => e.isIntersecting)) { revealed = true; io.disconnect(); run(); }
  }, {threshold: 0.3});
  io.observe(node);
  if (revealed) run();
}

function drawChart(rows, replay){
  const host = el("chart");
  const scored = (rows || []).filter(r => typeof r.primary_value === "number");
  if (!scored.length){
    host.textContent = "";
    const span = document.createElement("div"); span.className = "na";
    span.textContent = "No scored round yet (NOT_AVAILABLE)";
    host.appendChild(span);
    delete host.dataset.sig;
    return;
  }
  // a 3s poll must not restart the draw-in: only new data (or a click) replays it
  const signature = JSON.stringify(scored.map(
    r => [r.round, r.primary_value, r.best_so_far, r.outcome]));
  if (host.dataset.sig === signature && !replay) return;
  host.dataset.sig = signature;
  host.textContent = "";
  obsReveal(host, (node) => {
    const g = echarts.getInstanceByDom(node) || echarts.init(node);
    g.clear();
    const gen = g18Gen;
    const labels = scored.map(r => "R" + r.round);
    const metric = scored.map(r => r.primary_value);
    const best = scored.map(r => (typeof r.best_so_far === "number" ? r.best_so_far : null));
    const champion = scored.reduce((acc, r) =>
      (typeof r.best_so_far === "number" && (acc === null || r.best_so_far > acc)) ? r.best_so_far : acc,
      null);
    const top = Math.max.apply(null, metric.concat(best.filter(v => v !== null))) || 1;
    const last = scored[scored.length - 1];
    g.setOption({
      animationDuration: G18.DUR, animationEasing: "cubicOut",
      textStyle: {fontFamily: "Inter"},
      tooltip: {show: false},
      grid: {left: 14, right: 20, top: 64, bottom: 24},
      xAxis: {type: "category", data: labels, boundaryGap: false,
        axisLine: {show: false}, axisTick: {show: false}, splitLine: {show: false},
        axisLabel: {color: G18.FAINT, fontFamily: "Inter", fontSize: 9, fontWeight: 600,
          interval: Math.max(0, Math.ceil(labels.length / 6) - 1)}},
      yAxis: {show: false, max: top * 1.06},
      series: [
        {name: "metric", type: "line", data: metric, smooth: 0.2, symbol: "none",
         lineStyle: {color: G18.INK, width: 2.6},
         areaStyle: {color: {type: "linear", x: 0, y: 0, x2: 0, y2: 1,
           colorStops: [{offset: 0, color: "rgba(28,28,26,.14)"},
                        {offset: 1, color: "rgba(28,28,26,0)"}]}}},
        {name: "best so far", type: "line", data: best, smooth: 0.2, symbol: "none",
         connectNulls: true, lineStyle: {color: G18.MUTED, width: 1.4, type: "dashed"}},
      ],
    });
    // the counter: one big number that counts up, exactly the template's shape
    const target = champion === null ? metric[metric.length - 1] : champion;
    const caption = (last.run_id || "") + " · " + last.outcome
      + " · round " + (scored.length - 1) + "/" + (rows.length - 1);
    const t0 = performance.now();
    const tick = () => {
      if (gen !== g18Gen) return;
      const p = Math.min(1, (performance.now() - t0) / G18.DUR);
      const eased = 1 - Math.pow(1 - p, 3);
      const value = (target * eased).toFixed(4);
      g.setOption({graphic: [{id: "kpi", type: "group", left: 16, top: 8, children: [
        {type: "text", left: 0, top: 0,
         style: {text: "best " + value, font: "800 32px Inter", fill: G18.INK}},
        {type: "text", left: 0, top: 38,
         style: {text: caption, font: "600 10px Inter", fill: G18.MUTED}},
      ]}]});
      if (p < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  });
}

addEventListener("resize", () => {
  const node = el("chart");
  const instance = node && echarts.getInstanceByDom(node);
  if (instance) instance.resize();
});

function renderEvents(s){
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
}
async function tick(){
  try{
    // without a deadline a hung request leaves the page silently stale, which
    // looks identical to "the run is idle" — say DISCONNECTED instead
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), 8000);
    let res;
    try {
      res = await fetch("/api/status?after=" + state.lastSeq,
                        {cache: "no-store", signal: ctl.signal});
    } finally {
      clearTimeout(timer);
    }
    if(!res.ok) throw new Error("status " + res.status);
    const s = await res.json();
    state.history = s.history;
    state.lastSeq = s.last_event_seq || state.lastSeq;
    renderStatus(s);
    renderEvents(s);
  }catch(err){
    el("idsub").append(" · ");
    const pill = document.createElement("span");
    pill.className = "pill stale"; pill.textContent = "DISCONNECTED";
    el("idsub").append(pill);
    text(el("heartbeat"), String(err));
  }
}
/* The first paint NEVER waits for a request. The server inlines the current
   payload, so the page shows the real state even if the client's first fetch is
   slow, blocked or never answered — a phone on a roaming network was left on
   "loading…" forever by the previous deferred-paint version. Polling then keeps
   it live; `frozen` payloads (the archive route) skip polling so a screenshot
   has a page that settles. */
if (window.__STATIC_PAYLOAD__) {
  renderStatus(window.__STATIC_PAYLOAD__);
  renderEvents(window.__STATIC_PAYLOAD__);
  if (window.__STATIC_PAYLOAD__.frozen) {
    state.lastSeq = window.__STATIC_PAYLOAD__.last_event_seq || 0;
  } else {
    tick(); setInterval(tick, 3000);
  }
} else {
  tick(); setInterval(tick, 3000);
}
</script>
</body></html>
"""


def render_static(session: ResearchSession, *, max_events: int = 60,
                  frozen: bool = True) -> str:
    """The page with its payload inlined and no polling.

    Same CSS, same markup, same renderers; only the transport is gone. Used for
    screenshots, archives and the acceptance check that the page really paints.
    """
    payload = _status_payload(session, after=0)
    payload["events"] = list(reversed(payload.get("events") or []))[:max_events]
    # `frozen` marks the archive route: the page hydrates and then stops, so a
    # screenshot has something that settles. `/` hydrates and keeps polling.
    payload["frozen"] = bool(frozen)
    blob = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    inject = f'<script>window.__STATIC_PAYLOAD__ = {blob};</script>\n'
    return render_page().replace("<script>\nconst state", inject + "<script>\nconst state", 1)


def render_page() -> str:
    return PAGE.replace("__TITLE__", PAGE_TITLE).replace(
        "__ECHARTS__", _echarts_source()
    )


def make_handler(session: ResearchSession):
    class Handler(BaseHTTPRequestHandler):
        server_version = "AutoResearchDashboard/1"
        sys_version = ""

        def _send(self, status: int, body: bytes, content_type: str,
                  limit: int = MAX_RESPONSE_BYTES) -> None:
            if len(body) > limit:
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
                # first paint from data, then poll: the page is useful even if
                # the client's first request never comes back
                self._send(200, render_static(session).encode("utf-8"),
                           "text/html; charset=utf-8", MAX_PAGE_BYTES)
            elif path == "/static":
                # same page, payload inlined, no polling: for archives and for
                # automated screenshots, which never settle on a polling page
                self._send(200, render_static(session, frozen=True).encode("utf-8"),
                           "text/html; charset=utf-8", MAX_PAGE_BYTES)
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
                self._json({"error": "not found",
                            "allowed": ["/", "/static", "/api/status",
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


def make_runs_handler(runs_root: str | Path):
    """Routes for the FIXED dashboard: the run is resolved per request.

    The route bodies are the single-run handler's, so there is one implementation
    of every route; only the session source differs. `?run=<id>` pins a run for
    replay, which is how a historical run is inspected without a second server.
    """
    def resolve(query: dict) -> ResearchSession:
        pinned = (query.get("run") or [None])[0]
        if pinned:
            candidate = Path(runs_root) / pinned
            if (candidate / "run.json").exists():
                return ResearchSession.load(candidate)
            for row in list_runs(runs_root):
                if row.get("run_id") == pinned and row.get("dir"):
                    return ResearchSession.load(row["dir"])
            raise FileNotFoundError(f"unknown run {pinned!r} under {runs_root}")
        newest = resolve_run(runs_root)
        if newest is None:
            raise FileNotFoundError(f"no runs yet under {runs_root}")
        return ResearchSession.load(newest)

    class Handler(BaseHTTPRequestHandler):
        server_version = "AutoResearchDashboard/1"
        sys_version = ""

        def _send(self, status: int, body: bytes, content_type: str,
                  limit: int = MAX_RESPONSE_BYTES) -> None:
            if len(body) > limit:
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


        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            query = parse_qs(parsed.query)
            if ".." in parsed.path:
                self._json({"error": "path not allowed"}, 400)
                return
            try:
                session = resolve(query)
            except FileNotFoundError as exc:
                self._json({"error": str(exc), "runs": list_runs(runs_root)}, 404)
                return
            if path == "/":
                self._send(200, render_static(session).encode("utf-8"),
                           "text/html; charset=utf-8", MAX_PAGE_BYTES)
            elif path == "/static":
                self._send(200, render_static(session, frozen=True).encode("utf-8"),
                           "text/html; charset=utf-8", MAX_PAGE_BYTES)
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
            elif path == "/api/runs":
                self._json({"runs": list_runs(runs_root)})
            else:
                self._json({"error": "not found",
                            "allowed": ["/", "/static", "/api/status", "/api/events",
                                        "/api/history", "/api/runs"]}, 404)

    return Handler


def serve_runs(runs_root: str | Path, *, host: str = "127.0.0.1",
               port: int = 8765, serve_tailnet: bool = True) -> int:
    """The FIXED dashboard: one process, one port, following a runs root."""
    if serve_tailnet:
        from research.tailnet import serve_local

        tailnet = serve_local(port)
        if tailnet.get("available"):
            print(f"tailnet (private, serve — not funnel): {tailnet['url']}", flush=True)
        else:
            print(f"tailnet: NOT_AVAILABLE ({tailnet.get('reason')})", flush=True)
    server = ThreadingHTTPServer((host, port), make_runs_handler(runs_root))
    current = resolve_run(runs_root)
    print(f"AutoResearch dashboard on http://{host}:{port}/ · runs root "
          f"{Path(runs_root).resolve()} · current run "
          f"{current.name if current else 'none yet'} (read-only)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def capture_static(session: ResearchSession, out_path, *, browser: str | None = None,
                   size: str = "1280x1760", settle_ms: int = 4000):
    """Screenshot this run's static page, without needing a running dashboard.

    The report should carry a picture of the page it describes, so this starts a
    throwaway server on an ephemeral loopback port, captures `/static`, and shuts
    it down. Nothing is exposed: the port is loopback-only, it lives for one
    capture, and no other route is reachable by the browser.

    The browser step lives in `scripts/autoresearch_screenshot.py` because it is
    also a CLI; the CLI is always run from the repository root, and this raises a
    clear error rather than guessing a path if that import cannot be resolved.
    """
    import threading
    from http.server import ThreadingHTTPServer
    from pathlib import Path

    try:
        from scripts.autoresearch_screenshot import screenshot
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "screenshot support needs the repository root on sys.path "
            "(run the CLI from the checkout, or from the repository root)"
        ) from exc

    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(session))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/static"
        return screenshot(url, Path(out_path), size=size, settle_ms=settle_ms,
                          browser=browser)
    finally:
        server.shutdown()
        server.server_close()


def serve(session: ResearchSession, *,  host: str = "127.0.0.1", port: int = 8765,
          once: bool = False) -> int:
    if once:
        sys.stdout.write(render_page())
        return 0
    # Publish this port on the tailnet automatically, so watching a run needs no
    # hand-built tunnel. The server still binds loopback and `tailscale serve` is
    # tailnet-only (never funnel); a failure here is printed, never fatal, and
    # `AUTORESEARCH_NO_SERVE=1` opts out.
    if not os.environ.get("AUTORESEARCH_NO_SERVE"):
        from research.tailnet import serve_local

        tailnet = serve_local(port)
        if tailnet.get("available"):
            print(f"tailnet (private, serve — not funnel): {tailnet['url']}",
                  flush=True)
        else:
            print(f"tailnet: NOT_AVAILABLE ({tailnet.get('reason')})"
                  + (f" — {tailnet['hint']}" if tailnet.get("hint") else ""),
                  flush=True)
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
    # the child's diagnostics go to the run directory, not /dev/null: if the
    # tailnet publish fails, that log is the only place the reason appears
    log = Path(session.root) / "dashboard.log"
    handle = log.open("a", encoding="utf-8")
    process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                               start_new_session=True,
                               cwd=str(Path(__file__).resolve().parents[1]))
    return {"pid": process.pid, "url": f"http://{host}:{port}/",
            "command": " ".join(shlex.quote(part) for part in command)}