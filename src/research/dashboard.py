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
    <p class="csub" id="chartsub">macro metric · each point is one AutoResearch round</p>
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
  if (p.budget_seconds) bits.push(["stage budget", fmt(p.elapsed_seconds, 1) + "s / " + p.budget_seconds + "s"]);
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
/* Mono grammar: brightness is importance — the metric is ink, the best-so-far is
   muted, a rolled-back round is faint. Hairline baseline only, no grid clutter,
   direct labels on the values that matter, axes labelled in words. */
function drawChart(rows){
  const host = el("chart"); host.textContent = "";
  const scored = rows.filter(r => typeof r.primary_value === "number");
  if (!scored.length){
    const span = document.createElement("div"); span.className = "na";
    span.textContent = "No scored round yet (NOT_AVAILABLE)";
    host.appendChild(span); return;
  }
  const best = rows.filter(r => typeof r.best_so_far === "number").map(r => r.best_so_far);
  const all = scored.map(r => r.primary_value).concat(best.length ? best : []);
  const lo = Math.min(...all), hi = Math.max(...all);
  const pad = (hi - lo) * 0.18 || 0.01;
  const y0 = lo - pad, y1 = hi + pad;
  const w = 720, h = 240, ml = 62, mr = 86, mt = 22, mb = 40;
  const x = i => ml + (scored.length === 1 ? (w - ml - mr) / 2
      : i * (w - ml - mr) / (scored.length - 1));
  const y = v => (h - mb) - (v - y0) / (y1 - y0) * (h - mt - mb);
  const svg = svgEl("svg");
  svg.setAttribute("viewBox", "0 0 " + w + " " + h);
  svg.setAttribute("width", "100%"); svg.setAttribute("height", h);
  svg.setAttribute("role", "img");
  const label = svgEl("title");
  label.textContent = "Trusted metric per round with the best-so-far line";
  svg.appendChild(label);

  const base = svgEl("line");
  base.setAttribute("x1", ml); base.setAttribute("x2", w - mr);
  base.setAttribute("y1", h - mb); base.setAttribute("y2", h - mb);
  base.setAttribute("stroke", T.grid); base.setAttribute("stroke-width", "1");
  svg.appendChild(base);

  const axis = (value, anchor) => {
    const t = svgEl("text");
    t.setAttribute("x", ml - 10); t.setAttribute("y", y(value) + 4);
    t.setAttribute("text-anchor", "end"); t.setAttribute("font-size", "6.5");
    t.setAttribute("font-weight", "600"); t.setAttribute("fill", T.muted);
    t.textContent = fmt(value, 4);
    if (anchor) t.setAttribute("y", y(value) + 10);
    svg.appendChild(t);
  };
  axis(hi); axis(lo);

  if (best.length > 1) {
    const bl = svgEl("polyline");
    bl.setAttribute("fill", "none"); bl.setAttribute("stroke", T.muted);
    bl.setAttribute("stroke-width", "1.25"); bl.setAttribute("stroke-dasharray", "4 4");
    bl.setAttribute("points", best.map((v, i) => x(i) + "," + y(v)).join(" "));
    svg.appendChild(bl);
  }
  const line = svgEl("polyline");
  line.setAttribute("fill", "none"); line.setAttribute("stroke", T.ink);
  line.setAttribute("stroke-width", "2"); line.setAttribute("stroke-linecap", "round");
  line.setAttribute("points", scored.map((r, i) => x(i) + "," + y(r.primary_value)).join(" "));
  svg.appendChild(line);

  scored.forEach((r, i) => {
    const c = svgEl("circle");
    c.setAttribute("cx", x(i)); c.setAttribute("cy", y(r.primary_value));
    c.setAttribute("r", "4");
    c.setAttribute("fill", r.outcome === "KEEP" ? T.ink : T.paper);
    c.setAttribute("stroke", r.outcome === "KEEP" ? T.ink : T.l4);
    c.setAttribute("stroke-width", "1.5");
    const t = svgEl("title");
    t.textContent = (r.run_id || "") + " · " + (r.outcome || "") + " · " + fmt(r.primary_value)
      + " · " + (r.hypothesis || "");
    c.appendChild(t); svg.appendChild(c);
  });

  const last = scored[scored.length - 1];
  const lastLabel = svgEl("text");
  lastLabel.setAttribute("x", x(scored.length - 1) + 10);
  lastLabel.setAttribute("y", y(last.primary_value) + 4);
  lastLabel.setAttribute("font-size", "6.5"); lastLabel.setAttribute("font-weight", "800");
  lastLabel.setAttribute("fill", T.ink);
  lastLabel.textContent = fmt(last.primary_value, 4);
  svg.appendChild(lastLabel);

  const xLabel = svgEl("text");
  xLabel.setAttribute("x", ml); xLabel.setAttribute("y", h - 14);
  xLabel.setAttribute("font-size", "6.5"); xLabel.setAttribute("font-weight", "600");
  xLabel.setAttribute("fill", T.muted);
  xLabel.textContent = "ROUND 0";
  const xLabel2 = svgEl("text");
  xLabel2.setAttribute("x", w - mr); xLabel2.setAttribute("y", h - 14);
  xLabel2.setAttribute("text-anchor", "end");
  xLabel2.setAttribute("font-size", "6.5"); xLabel2.setAttribute("font-weight", "600");
  xLabel2.setAttribute("fill", T.muted);
  xLabel2.textContent = "ROUND " + (scored.length - 1) + " · " + last.outcome;
  svg.appendChild(xLabel); svg.appendChild(xLabel2);

  const legend = svgEl("text");
  legend.setAttribute("x", ml); legend.setAttribute("y", mt - 6);
  legend.setAttribute("font-size", "6.5"); legend.setAttribute("font-weight", "600");
  legend.setAttribute("fill", T.muted);
  legend.textContent = "— filled dot: KEEP · hollow dot: rolled back · dashed: best so far";
  svg.appendChild(legend);
  host.appendChild(svg);
}
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
    const res = await fetch("/api/status?after=" + state.lastSeq, {cache: "no-store"});
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
/* Static mode renders the SAME page once, from an inlined payload, and never
   polls. It exists so the page can be archived, screenshotted and diffed — a
   live monitor that repaints every three seconds keeps an automated
   screenshotter from ever seeing the page settle. Same renderers, same tokens,
   same markup: only the transport differs. */
if (window.__STATIC_PAYLOAD__) {
  renderStatus(window.__STATIC_PAYLOAD__);
  renderEvents(window.__STATIC_PAYLOAD__);
} else {
  tick(); setInterval(tick, 3000);
}
</script>
</body></html>
"""


def render_static(session: ResearchSession, *, max_events: int = 60) -> str:
    """The page with its payload inlined and no polling.

    Same CSS, same markup, same renderers; only the transport is gone. Used for
    screenshots, archives and the acceptance check that the page really paints.
    """
    payload = _status_payload(session, after=0)
    payload["events"] = list(reversed(payload.get("events") or []))[:max_events]
    blob = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    inject = f'<script>window.__STATIC_PAYLOAD__ = {blob};</script>\n'
    return render_page().replace("<script>\nconst state", inject + "<script>\nconst state", 1)


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
            elif path == "/static":
                # same page, payload inlined, no polling: for archives and for
                # automated screenshots, which never settle on a polling page
                self._send(200, render_static(session).encode("utf-8"),
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