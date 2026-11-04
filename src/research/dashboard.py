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
import re
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


#: a run id is a single path component the dashboard looks up BY NAME. The
#: dashboard never joins a request value onto the runs root (`?run=` is
#: untrusted: an absolute value would discard the root, `..` would climb out),
#: so this only has to describe the names the engine can produce.
RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")


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
  /* lieflat-charts Mono grammar: unchanged tokens, lifted from mono-tokens.js */
  --ink:#1C1C1A; --paper:#F0EFEB; --muted:#8F8E88; --faint:#C6C5BF; --grid:#DEDDD6;
  --l1:#1C1C1A; --l2:#4A4944; --l3:#6A6963; --l4:#8F8E88; --l5:#B0AFA9; --l6:#C6C5BF;
  --card:#F7F6F2;
  --font:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  --radius:24px;
  color-scheme:light;
}
/* a monitoring strip is read around the clock; honour the host's theme.
   The same Mono ladder is inverted, so brightness still encodes importance. */
@media (prefers-color-scheme:dark){
  :root{
    --ink:#F0EFEB; --paper:#161614; --card:#1E1E1B;
    --muted:#9A9992; --faint:#6B6A64; --grid:#333330;
    --l1:#F0EFEB; --l2:#CAC9C2; --l3:#A4A39C; --l4:#8B8A83;
    --l5:#5F5E59; --l6:#45443F;
    color-scheme:dark;
  }
}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--font);
     -webkit-font-smoothing:antialiased;line-height:1.5}
.eyebrow{margin:0 0 10px;color:var(--l4);font-size:10px;font-weight:700;
         letter-spacing:.18em;text-transform:uppercase}
header{padding:36px 32px 22px;border-bottom:1px solid var(--grid)}
h1{margin:0;font-size:34px;font-weight:700;letter-spacing:-.035em;line-height:1.1}
.sub{margin:10px 0 0;color:var(--muted);font-size:13px;font-weight:500}
.sub b{color:var(--ink);font-weight:700}
.pill{display:inline-flex;align-items:center;gap:6px;margin-left:10px;padding:3px 11px;
      border-radius:999px;font-size:10px;font-weight:700;letter-spacing:.07em;
      text-transform:uppercase;background:var(--ink);color:var(--paper);
      vertical-align:middle}
.pill::before{content:"";width:5px;height:5px;border-radius:50%;background:currentColor;
              opacity:.85}
.pill.stale{background:var(--l3)}
.wrap{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,340px),1fr));
      gap:18px;padding:24px 32px 40px}
/* borderless cards separated by whitespace, radius 24 (Mono grammar).
   A one-step paper lift gives the card an edge without a border or shadow. */
/* No entrance animation: a monitoring page must be fully legible on first
   paint, including in headless captures and on slow links where a delayed
   opacity:0 frame can leave a card permanently dim. */
.card{background:var(--card);border-radius:var(--radius);padding:26px 28px 22px;
      min-width:0}
.card.wide{grid-column:1/-1}
h2{margin:0;font-size:18px;font-weight:700;letter-spacing:-.025em;line-height:1.3}
.csub{margin:7px 0 18px;color:var(--muted);font-size:12.5px;font-weight:500}
.src{margin:16px 0 0;color:var(--l3);font-size:10px;font-weight:600;
     letter-spacing:.11em;text-transform:uppercase}
.tablewrap{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{width:100%;border-collapse:collapse;font-size:13px;table-layout:fixed}
th,td{text-align:left;padding:9px 10px;border-bottom:1px solid var(--grid);
      vertical-align:top;overflow-wrap:anywhere}
th{color:var(--l4);font-weight:700;font-size:10px;letter-spacing:.1em;text-transform:uppercase}
th.num,td.num{text-align:right;font-variant-numeric:tabular-nums;font-weight:600}
tbody tr:hover{background:var(--paper)}
.tag{display:inline-flex;align-items:center;padding:3px 9px;border-radius:7px;
     font-size:10px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;
     background:var(--grid);color:var(--l3)}
.tag.KEEP{background:var(--ink);color:var(--paper)}
.tag.REVERT{background:transparent;background:var(--grid);color:var(--l3)}
.mono{font-family:var(--mono);font-size:11.5px;color:var(--l2);line-height:1.75}
.row{display:flex;gap:18px;align-items:baseline;flex-wrap:wrap;color:var(--muted);
     font-size:12.5px}
.row b{color:var(--ink);font-weight:700}
.finding{font-size:12.5px;margin:7px 0;padding-left:11px;position:relative}
.finding::before{content:"";position:absolute;left:0;top:.55em;width:4px;height:4px;
                 border-radius:50%;background:var(--l5)}
.finding.BLOCKED{color:var(--ink);font-weight:700}
.finding.BLOCKED::before{background:var(--ink)}
.finding.WARNING{color:var(--l2);font-weight:600}
.finding.WARNING::before{background:var(--l3)}
.finding.INFO,.finding.UNVERIFIED{color:var(--muted)}
.na{color:var(--muted);font-style:italic}
.hyp{font-size:17px;font-weight:600;line-height:1.45;letter-spacing:-.01em}
.kv{margin-top:11px;color:var(--muted);font-size:12.5px}
.kv b{color:var(--ink)}
svg text{font-family:var(--font)}
/* ECharts needs an explicit height; the KPI counter sits in the top band,
   so the plot gets more room than the old 250px card. */
#chart{height:320px;width:100%;cursor:pointer}
#chart:focus-visible{outline:2px solid var(--l3);outline-offset:4px;border-radius:8px}
.legend{display:flex;gap:18px;flex-wrap:wrap;align-items:center;margin:0 0 8px;
        color:var(--l4);font-size:11px;font-weight:600}
.legend i{display:inline-block;width:9px;height:9px;margin-right:6px;
          vertical-align:-1px;border-radius:50%;background:var(--ink)}
.legend i.hollow{background:transparent;border:1.5px solid var(--l4)}
.legend i.dash{width:18px;height:0;border-radius:0;background:transparent;
               border-top:1.5px dashed var(--l4);vertical-align:3px}
/* ── view controls ─────────────────────────────────────────────────────── */
[hidden]{display:none!important}
.controls{display:flex;gap:18px;align-items:center;flex-wrap:wrap;
          padding:14px 32px;border-bottom:1px solid var(--grid);
          position:sticky;top:0;z-index:20;background:var(--paper)}
.tabs{display:inline-flex;gap:4px;padding:4px;border-radius:999px;background:var(--card)}
.tab{appearance:none;border:0;background:transparent;color:var(--l4);cursor:pointer;
     padding:7px 18px;border-radius:999px;font:inherit;font-size:12px;font-weight:700;
     letter-spacing:.08em;text-transform:uppercase;transition:background .2s,color .2s}
.tab:hover{color:var(--ink)}
.tab.is-active{background:var(--ink);color:var(--paper)}
.tab:focus-visible,.ghost:focus-visible,select:focus-visible{outline:2px solid var(--l3);
     outline-offset:2px}
.selectors{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-left:auto}
.selectors label{display:inline-flex;align-items:center;gap:8px;color:var(--l4);
  font-size:10px;font-weight:700;letter-spacing:.11em;text-transform:uppercase}
select{appearance:none;background:var(--card);color:var(--ink);border:0;border-radius:9px;
  padding:8px 30px 8px 12px;font:inherit;font-size:12px;font-weight:600;cursor:pointer;
  background-image:linear-gradient(45deg,transparent 50%,var(--l4) 50%),
                   linear-gradient(135deg,var(--l4) 50%,transparent 50%);
  background-position:calc(100% - 15px) 52%,calc(100% - 10px) 52%;
  background-size:5px 5px,5px 5px;background-repeat:no-repeat}
.ghost{appearance:none;background:transparent;border:0;color:var(--l4);cursor:pointer;
  font:inherit;font-size:11px;font-weight:700;letter-spacing:.07em;text-transform:uppercase;
  padding:8px 10px;border-radius:9px}
.ghost:hover{background:var(--card);color:var(--ink)}
/* ── history: slice inspector ─────────────────────────────────────────── */
.slicehead{display:flex;gap:12px;align-items:baseline;flex-wrap:wrap}
.slicehead .chip{display:inline-block;padding:3px 10px;border-radius:7px;background:var(--ink);
  color:var(--paper);font-size:10px;font-weight:700;letter-spacing:.07em;text-transform:uppercase}
.slicegrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));
  gap:18px;margin-top:18px}
.sliceblock h3{margin:0 0 10px;color:var(--l4);font-size:10px;font-weight:700;
  letter-spacing:.12em;text-transform:uppercase}
.kvrow{display:flex;justify-content:space-between;gap:12px;padding:6px 0;
  border-bottom:1px solid var(--grid);font-size:12.5px;color:var(--muted)}
.kvrow b{color:var(--ink);font-weight:600;font-variant-numeric:tabular-nums;text-align:right}
.evt{display:grid;grid-template-columns:52px 1fr auto;gap:10px;padding:9px 0;
  border-bottom:1px solid var(--grid);font-size:12px;align-items:baseline}
.evt .seq{color:var(--l4);font-family:var(--mono);font-size:11px}
.evt .msg{color:var(--l2);overflow-wrap:anywhere}
.evt .num{color:var(--ink);font-variant-numeric:tabular-nums;font-weight:600;white-space:nowrap}
.evt .meta{grid-column:2/-1;color:var(--l4);font-size:10.5px;letter-spacing:.04em}
.stagebar{display:flex;align-items:center;gap:10px;padding:6px 0;font-size:12px;color:var(--muted)}
.stagebar .bar{height:6px;border-radius:3px;background:var(--ink);min-width:2px}
.stagebar .bar.ref{background:var(--l5)}
.stagebar b{color:var(--ink);font-variant-numeric:tabular-nums}
/* ── history: unified trend grid (Glance small multiples) ──────────────── */
.trends{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:18px}
.trend{background:var(--paper);border-radius:16px;padding:16px 18px 12px;min-width:0}
.trend h3{margin:0;font-size:12.5px;font-weight:700;letter-spacing:-.01em;color:var(--ink)}
.trend .tunit{margin:4px 0 8px;color:var(--l4);font-size:10.5px;font-weight:600}
.spark{height:104px;width:100%;cursor:pointer}
.sparkval{font-size:22px;font-weight:800;letter-spacing:-.03em;color:var(--ink);
  font-variant-numeric:tabular-nums;line-height:1.1;margin-top:2px}
.sparksub{color:var(--l4);font-size:10px;font-weight:600;letter-spacing:.09em;
  text-transform:uppercase;margin:4px 0 0}
.trend.is-na .spark{display:none}
.trend.is-na .na{margin:14px 0}
@media (max-width:640px){.controls{padding:12px 18px}.selectors{margin-left:0;width:100%}
  .trends,.slicegrid{grid-template-columns:1fr}}
footer{padding:20px 32px 32px;color:var(--l3);font-size:10px;font-weight:600;
       letter-spacing:.11em;text-transform:uppercase;border-top:1px solid var(--grid)}
@media (max-width:640px){
  header{padding:26px 18px 18px} h1{font-size:28px}
  .wrap{padding:18px 18px 28px;gap:14px} .card{padding:20px 18px}
  #chart{height:270px} footer{padding:18px}
  table{font-size:12px} th,td{padding:8px 6px}
}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style></head>
<body>
<header>
  <p class="eyebrow">autonomous research · read-only monitor · loopback</p>
  <h1 id="title">AutoResearch</h1>
  <p class="sub" id="idsub">loading…</p>
</header>
<div class="controls">
  <div class="tabs" role="tablist" aria-label="dashboard view">
    <button type="button" class="tab is-active" data-view="current"
            role="tab" aria-selected="true">Current</button>
    <button type="button" class="tab" data-view="history"
            role="tab" aria-selected="false">History</button>
  </div>
  <div class="selectors" id="selectors" hidden>
    <label for="roundSel">round <select id="roundSel"></select></label>
    <label for="stageSel">stage <select id="stageSel"></select></label>
    <button type="button" class="ghost" id="clearSel">clear</button>
  </div>
</div>

<div id="view-current">
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
    <div class="legend">
      <span><i></i> kept round</span><span><i class="hollow"></i> reverted round</span>
      <span><i class="dash"></i> best so far</span>
    </div>
    <div id="chart"></div>
    <p class="src" id="chartsrc">source: run manifest · trusted evaluator (isolated process)</p>
  </section>
  <section class="card wide">
    <h2 id="rounds-title">What was tried, in order</h2>
    <p class="csub">hypothesis · decided by the trusted metric, not by the candidate</p>
    <div class="tablewrap"><table id="rounds">
      <colgroup><col style="width:48px"><col style="width:30%"><col style="width:96px">
      <col style="width:88px"><col style="width:88px"><col></colgroup>
      <thead><tr><th>#</th><th>hypothesis</th><th>decision</th>
      <th class="num">value</th><th class="num">best</th><th>next action</th></tr></thead>
      <tbody></tbody></table></div>
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
</div>

<div id="view-history" hidden>
<div class="wrap">
  <section class="card wide">
    <div class="slicehead">
      <h2 id="sliceTitle">All rounds · all stages</h2>
      <span class="chip" id="sliceChip">overview</span>
    </div>
    <p class="csub" id="sliceSub">pick a round and a stage to inspect exactly what the coordinator recorded there</p>
    <div class="row" id="sliceStats"></div>
    <div class="slicegrid">
      <div class="sliceblock"><h3>Round record</h3><div id="sliceRound"></div></div>
      <div class="sliceblock"><h3>Stage breakdown</h3><div id="sliceStages"></div></div>
      <div class="sliceblock"><h3>Diagnostics &amp; doctor</h3><div id="sliceDiag"></div></div>
    </div>
    <p class="src">source: run manifest + coordinator event log · append-only</p>
  </section>
  <section class="card wide">
    <h2 id="sliceEventsTitle">Events in this slice</h2>
    <p class="csub" id="sliceEventsSub">coordinator-written only; a candidate never writes here</p>
    <div id="sliceEvents" class="mono"></div>
    <p class="src">source: events.jsonl · single trusted writer</p>
  </section>
  <section class="card wide">
    <h2>Every recorded number, as a trend</h2>
    <p class="csub">one line per metric · the same Mono grammar as the main chart · click a card to pin that round</p>
    <div class="trends" id="trends"></div>
    <p class="src">source: run manifest + coordinator events · NOT_AVAILABLE means nothing was recorded</p>
  </section>
</div>
</div>

<footer>read-only view · operations live in the cli · charts follow the lieflat-charts mono grammar</footer>
<script>__ECHARTS__</script>
<script>
const state = {lastSeq: 0, history: [], events: [], drawn: false,
               view: "current", round: "all", stage: "all", sparkSig: ""};
const cv = (n, f) => getComputedStyle(document.documentElement)
                        .getPropertyValue(n).trim() || f;
function pal(){
  return {ink: cv("--ink","#1C1C1A"), paper: cv("--paper","#F0EFEB"),
          card: cv("--card","#F7F6F2"), muted: cv("--muted","#8F8E88"),
          faint: cv("--faint","#C6C5BF"), grid: cv("--grid","#DEDDD6"),
          l2: cv("--l2","#4A4944"), l3: cv("--l3","#6A6963"),
          l4: cv("--l4","#8F8E88"), l5: cv("--l5","#B0AFA9")};
}
const T = pal();
const STAGES = ["DOCTOR","PROPOSING","TRAINING","EVALUATING","DECIDING","IDLE","DONE"];
function num(v){ return (typeof v === "number" && isFinite(v)) ? v : null; }
function mean(xs){ const a = xs.filter(v => v !== null);
  return a.length ? a.reduce((p,c)=>p+c,0)/a.length : null; }
function fmtN(v, d){ return num(v) === null ? "NOT_AVAILABLE"
  : Number(v).toFixed(d === undefined ? 4 : d); }
function mergeEvents(base, incoming){
  const bySeq = new Map();
  for (const e of base || []) if (e && typeof e.seq === "number") bySeq.set(e.seq, e);
  for (const e of incoming || []) if (e && typeof e.seq === "number") bySeq.set(e.seq, e);
  return Array.from(bySeq.values()).sort((a,b)=>a.seq-b.seq);
}
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
  state.events = mergeEvents(state.events, s.events || []);
  state.history = s.history || [];
  drawChart(s.history);
  if (state.view === "history") renderHistory();
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
  // A reveal mechanism that can leave the card permanently blank is a bug, not a
  // feature: headless capture, a print stylesheet and some in-app browsers never
  // deliver an IntersectionObserver callback. Draw anyway shortly after, and let
  // the observer only make it earlier.
  setTimeout(() => {
    if (!revealed) { revealed = true; io.disconnect(); run(); }
  }, 400);
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
    const C = pal();
    const labels = scored.map(r => "R" + r.round);
    const metric = scored.map(r => r.primary_value);
    const best = scored.map(r => (typeof r.best_so_far === "number" ? r.best_so_far : null));
    const champion = scored.reduce((acc, r) =>
      (typeof r.best_so_far === "number" && (acc === null || r.best_so_far > acc)) ? r.best_so_far : acc,
      null);
    const all = metric.concat(best.filter(v => v !== null));
    const lo = Math.min.apply(null, all), hi = Math.max.apply(null, all);
    // A metric that moves in the third decimal collapses if the axis starts at
    // zero. Show a labelled, tightly padded value axis instead of hiding it:
    // the reader can then see BOTH the shape and the actual magnitudes.
    const pad = Math.max((hi - lo) * 0.28, Math.abs(hi) * 0.002, 1e-6);
    const last = scored[scored.length - 1];
    // decision markers: filled ink = KEEP, hollow = REVERT (Mono grammar).
    // Two fixed-symbol series, because per-item `symbol` overrides are not
    // reliable across ECharts versions and silently dropped the hollow dots.
    // A category axis needs the category VALUE, not its numeric index.
    const kept = [], reverted = [];
    scored.forEach((r, i) => {
      (r.outcome === "KEEP" ? kept : reverted)
        .push({value: [labels[i], r.primary_value]});
    });
    g.setOption({
      animationDuration: G18.DUR, animationEasing: "cubicOut",
      textStyle: {fontFamily: "Inter, ui-sans-serif, system-ui, sans-serif"},
      tooltip: {show: true, trigger: "axis", renderMode: "richText",
        backgroundColor: C.card, borderColor: C.grid, borderWidth: 1,
        textStyle: {color: C.ink, fontSize: 12},
        axisPointer: {type: "line", lineStyle: {color: C.l4, type: "dashed"}},
        formatter: (ps) => {
          const i = ps && ps[0] ? ps[0].dataIndex : 0;
          const r = scored[i] || {};
          return "round " + (r.round === undefined ? "NOT_AVAILABLE" : r.round)
            + "  ·  " + (r.outcome || "NOT_AVAILABLE")
            + "\\nmetric  " + fmt(r.primary_value)
            + "\\nbest    " + fmt(r.best_so_far)
            + "\\nfold sd " + fmt(r.primary_std);
        }},
      grid: {left: 62, right: 24, top: 74, bottom: 34},
      xAxis: {type: "category", data: labels, boundaryGap: false,
        axisLine: {lineStyle: {color: C.grid}}, axisTick: {show: false},
        splitLine: {show: false},
        axisLabel: {color: C.l3, fontFamily: "Inter", fontSize: 11, fontWeight: 600,
          margin: 12, interval: Math.max(0, Math.ceil(labels.length / 8) - 1)}},
      yAxis: {type: "value", min: lo - pad, max: hi + pad, splitNumber: 4,
        axisLine: {show: false}, axisTick: {show: false},
        axisLabel: {color: C.l3, fontSize: 10, fontFamily: "Inter",
          formatter: (v) => Number(v).toFixed(3), margin: 10},
        splitLine: {lineStyle: {color: C.grid, width: 1}}},
      series: [
        {name: "metric", type: "line", data: metric, smooth: 0.2, symbol: "none",
         lineStyle: {color: C.ink, width: 2.6, cap: "round"}, z: 3,
         areaStyle: {color: {type: "linear", x: 0, y: 0, x2: 0, y2: 1,
           colorStops: [{offset: 0, color: hexA(C.ink, .15)},
                        {offset: 1, color: hexA(C.ink, 0)}]}}},
        {name: "best so far", type: "line", data: best, smooth: 0.2, symbol: "none",
         connectNulls: true, z: 2,
         lineStyle: {color: C.l4, width: 1.6, type: "dashed"}},
        {name: "kept", type: "scatter", data: kept, z: 5, symbol: "circle",
         symbolSize: 9, itemStyle: {color: C.ink}, tooltip: {show: false}},
        {name: "reverted", type: "scatter", data: reverted, z: 5,
         symbol: "circle", symbolSize: 10,
         itemStyle: {color: C.card, borderColor: C.ink, borderWidth: 2},
         tooltip: {show: false}},
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
         style: {text: "best " + value, font: "800 34px Inter", fill: C.ink}},
        {type: "text", left: 0, top: 42,
         style: {text: caption, font: "600 10px Inter", fill: C.muted}},
      ]}]});
      if (p < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  });
}

function hexA(hex, a){
  const h = String(hex || "").replace("#", "");
  if (h.length !== 6) return "rgba(28,28,26," + a + ")";
  const n = parseInt(h, 16);
  return "rgba(" + ((n >> 16) & 255) + "," + ((n >> 8) & 255) + ","
    + (n & 255) + "," + a + ")";
}

addEventListener("resize", () => {
  const node = el("chart");
  const instance = node && echarts.getInstanceByDom(node);
  if (instance) instance.resize();
  for (const n of document.querySelectorAll(".spark")) {
    const s = echarts.getInstanceByDom(n);
    if (s) s.resize();
  }
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
/* ── History view: round/stage picker, slice inspector, unified trend grid ──
   Everything here reads the same inlined/polled payload as the current view.
   Data is always written through textContent: no markup is ever built from a
   run's strings, so a hostile hypothesis cannot become HTML.               */
function setView(view){
  state.view = view;
  for (const b of document.querySelectorAll(".tab")) {
    const on = b.dataset.view === view;
    b.classList.toggle("is-active", on);
    b.setAttribute("aria-selected", on ? "true" : "false");
  }
  el("view-current").hidden = view !== "current";
  el("view-history").hidden = view !== "history";
  el("selectors").hidden = view !== "history";
  if (view === "history") { fillSelectors(); renderHistory(); }
  else {
    const g = echarts.getInstanceByDom(el("chart"));
    if (g) g.resize();
  }
}

function addOption(sel, value, label){
  const o = document.createElement("option");
  o.value = value; o.textContent = label; sel.appendChild(o);
}

function fillSelectors(){
  const rs = el("roundSel"), ss = el("stageSel");
  const rounds = (state.history || []).map(r => String(r.round));
  const seen = new Set();
  for (const e of state.events) if (e.stage) seen.add(e.stage);
  const stages = STAGES.filter(s => seen.has(s))
    .concat(Array.from(seen).filter(s => STAGES.indexOf(s) < 0).sort());
  const rSig = rounds.join(",");
  if (rs.dataset.sig !== rSig) {
    rs.dataset.sig = rSig; rs.textContent = "";
    addOption(rs, "all", "all rounds");
    for (const r of rounds) addOption(rs, r, "round " + r);
  }
  const sSig = stages.join(",");
  if (ss.dataset.sig !== sSig) {
    ss.dataset.sig = sSig; ss.textContent = "";
    addOption(ss, "all", "all stages");
    for (const s of stages) addOption(ss, s, s.toLowerCase());
  }
  rs.value = rounds.indexOf(state.round) >= 0 || state.round === "all" ? state.round : "all";
  ss.value = stages.indexOf(state.stage) >= 0 || state.stage === "all" ? state.stage : "all";
  state.round = rs.value; state.stage = ss.value;
}

function kvRow(host, k, v){
  const d = document.createElement("div"); d.className = "kvrow";
  const a = document.createElement("span"); a.textContent = String(k);
  const b = document.createElement("b");
  b.textContent = (v === null || v === undefined || v === "") ? "NOT_AVAILABLE" : String(v);
  d.append(a, b); host.appendChild(d);
}

function sliceEvents(){
  return state.events.filter(e =>
    (state.round === "all" || String(e.round) === state.round) &&
    (state.stage === "all" || e.stage === state.stage));
}

function renderSlice(){
  const rounds = state.history || [];
  const evts = sliceEvents();
  const picked = state.round !== "all"
    ? rounds.find(r => String(r.round) === state.round) : null;
  const where = [state.round === "all" ? "all rounds" : "round " + state.round,
                 state.stage === "all" ? "all stages" : state.stage.toLowerCase()];
  text(el("sliceTitle"), where.join(" · "));
  text(el("sliceChip"), picked ? (picked.outcome || "recorded") : "overview");
  const secs = evts.reduce((a,e) => a + (num(e.elapsed_seconds) || 0), 0);
  const folds = new Set(evts.filter(e => e.fold !== null && e.fold !== undefined)
    .map(e => String(e.repeat) + ":" + String(e.fold))).size;
  text(el("sliceSub"), evts.length + " coordinator events · " + secs.toFixed(1)
    + "s recorded · " + folds + " distinct fold runs · pick a round and a stage to narrow it");

  const st = el("sliceStats"); st.textContent = "";
  const counts = {};
  for (const e of evts) counts[e.status || "unknown"] = (counts[e.status || "unknown"] || 0) + 1;
  for (const [k,v] of Object.entries(counts).sort()) {
    const s = document.createElement("span");
    const b = document.createElement("b"); b.textContent = String(v);
    s.append(b, document.createTextNode(" " + k)); st.appendChild(s);
  }
  if (!Object.keys(counts).length) text(st, "no events in this slice (NOT_AVAILABLE)");

  const rec = el("sliceRound"); rec.textContent = "";
  if (!picked) {
    text(rec, "no single round selected — choose one above to see its record");
  } else {
    kvRow(rec, "run id", picked.run_id);
    kvRow(rec, "decision", picked.outcome || picked.decision);
    kvRow(rec, "hypothesis", picked.hypothesis);
    kvRow(rec, "controlled change", picked.controlled_change);
    kvRow(rec, "expected observation", picked.expected_observation);
    kvRow(rec, "reason", picked.reason);
    kvRow(rec, "trusted metric", fmtN(picked.primary_value));
    kvRow(rec, "best so far", fmtN(picked.best_so_far));
    kvRow(rec, "fold sd", fmtN(picked.primary_std));
    kvRow(rec, "wall clock", picked.elapsed_s === null || picked.elapsed_s === undefined
      ? "NOT_AVAILABLE" : fmtN(picked.elapsed_s, 1) + "s");
    kvRow(rec, "next action", picked.next_action);
  }

  const sb = el("sliceStages"); sb.textContent = "";
  const byStage = {};
  for (const e of evts) {
    const k = e.stage || "unknown";
    byStage[k] = byStage[k] || {n: 0, s: 0};
    byStage[k].n += 1; byStage[k].s += (num(e.elapsed_seconds) || 0);
  }
  const keys = Object.keys(byStage).sort();
  const maxS = Math.max.apply(null, keys.map(k => byStage[k].s).concat([1e-9]));
  if (!keys.length) text(sb, "no stage events in this slice (NOT_AVAILABLE)");
  for (const k of keys) {
    const row = document.createElement("div"); row.className = "stagebar";
    const name = document.createElement("span");
    name.style.minWidth = "88px"; name.textContent = k.toLowerCase();
    const bar = document.createElement("span"); bar.className = "bar";
    bar.style.width = Math.max(2, 90 * byStage[k].s / maxS) + "px";
    const v = document.createElement("b");
    v.textContent = byStage[k].n + " · " + byStage[k].s.toFixed(1) + "s";
    row.append(name, bar, v); sb.appendChild(row);
  }

  const dg = el("sliceDiag"); dg.textContent = "";
  const d = picked && picked.diagnostics ? picked.diagnostics : {};
  const dc = picked && picked.doctor_counts ? picked.doctor_counts : {};
  kvRow(dg, "zero-F1 classes", d.zero_f1_count === undefined ? "NOT_AVAILABLE" : d.zero_f1_count);
  kvRow(dg, "classes evaluated", d.total_evaluated_classes === undefined ? "NOT_AVAILABLE" : d.total_evaluated_classes);
  const worst = Array.isArray(d.worst_classes) ? d.worst_classes : [];
  kvRow(dg, "worst classes", worst.length ? worst.map(w =>
    String(w.class) + (w.f1 === undefined ? "" : " (" + Number(w.f1).toFixed(4) + ")")).join(", ") : "NOT_AVAILABLE");
  for (const k of ["BLOCKED","WARNING","INFO","UNVERIFIED"]) {
    kvRow(dg, "doctor " + k.toLowerCase(), dc[k] === undefined ? "NOT_AVAILABLE" : dc[k]);
  }
  const pc = picked && picked.per_class;
  kvRow(dg, "per-class report", pc && Object.keys(pc).length
    ? Object.keys(pc).length + " classes" : "NOT_AVAILABLE");

  const list = el("sliceEvents"); list.textContent = "";
  const shown = evts.slice(-200).reverse();
  if (!shown.length) text(list, "no events in this slice (NOT_AVAILABLE)");
  for (const e of shown) {
    const row = document.createElement("div"); row.className = "evt";
    const q = document.createElement("span"); q.className = "seq";
    q.textContent = "#" + (e.seq === undefined ? "—" : e.seq);
    const m = document.createElement("span"); m.className = "msg";
    m.textContent = (e.stage || "unknown") + " · " + (e.status || "unknown")
      + (e.message ? " · " + e.message : "");
    const n = document.createElement("span"); n.className = "num";
    n.textContent = num(e.elapsed_seconds) === null ? "—"
      : e.elapsed_seconds.toFixed(1) + "s" + (num(e.budget_seconds) === null
        ? "" : " / " + e.budget_seconds + "s");
    const meta = document.createElement("span"); meta.className = "meta";
    meta.textContent = (e.time || "NOT_AVAILABLE")
      + (e.round !== null && e.round !== undefined ? " · round " + e.round : "")
      + (e.fold !== null && e.fold !== undefined ? " · fold " + e.fold : "")
      + (e.repeat !== null && e.repeat !== undefined ? " · repeat " + e.repeat : "")
      + (e.metrics && Object.keys(e.metrics).length ? " · " + JSON.stringify(e.metrics).slice(0, 160) : "");
    row.append(q, m, n, meta); list.appendChild(row);
  }
  if (evts.length > shown.length) {
    const more = document.createElement("div"); more.className = "na";
    more.textContent = "showing the latest 200 of " + evts.length + " events";
    list.appendChild(more);
  }
  text(el("sliceEventsSub"), evts.length + " events match this slice · newest first");
}

/* One catalogue, one renderer: every number the manifest or the coordinator
   records gets the SAME Glance-family treatment (one stroke, area fade, sparse
   labels, a big last-value counter). A metric with no numeric observation is
   shown as NOT_AVAILABLE rather than silently dropped. */
function metricCatalogue(){
  const rounds = state.history || [];
  const evts = state.events || [];
  const labels = rounds.map(r => "R" + r.round);
  const at = (r, k) => num(r[k]);
  const nested = (r, group, k) => num((r && r[group] || {})[k]);
  const diag = (r, k) => num((r && r.diagnostics || {})[k]);
  const doc = (r, k) => num((r && r.doctor_counts || {})[k]);
  const folds = r => Array.isArray(r.fold_seconds)
    ? r.fold_seconds.filter(v => typeof v === "number" && isFinite(v)) : [];
  const stageSecs = (r, stage) => {
    const xs = evts.filter(e => String(e.round) === String(r.round) && e.stage === stage)
      .map(e => num(e.elapsed_seconds)).filter(v => v !== null);
    return xs.length ? xs.reduce((a,b)=>a+b,0) : null;
  };
  const evCount = r => evts.filter(e => String(e.round) === String(r.round)).length;
  let keeps = 0, reverts = 0;
  const cumKeep = [], cumRevert = [];
  for (const r of rounds) {
    if (r.outcome === "KEEP") keeps += 1;
    if (r.outcome === "REVERT") reverts += 1;
    cumKeep.push(keeps); cumRevert.push(reverts);
  }
  const defs = [
    ["Trusted metric", "primary metric · higher is better", r => at(r,"primary_value"), 4, "run manifest · trusted evaluator"],
    ["Best so far", "champion after each round", r => at(r,"best_so_far"), 4, "run manifest"],
    ["Fold dispersion", "primary_std across CV folds", r => at(r,"primary_std"), 4, "run manifest"],
    ["Fold SEM", "primary_sem where recorded", r => at(r,"primary_sem"), 4, "run manifest"],
    ["Round wall clock", "seconds", r => at(r,"elapsed_s"), 1, "coordinator timer"],
    ["Mean fold time", "seconds per fold", r => mean(folds(r)), 1, "runner fold timings"],
    ["Slowest fold", "seconds", r => folds(r).length ? Math.max.apply(null, folds(r)) : null, 1, "runner fold timings"],
    ["AST nodes after", "complexity of the editable file", r => nested(r,"complexity","nodes_after"), 0, "guard complexity metric"],
    ["Complexity growth", "nodes added this round", r => nested(r,"complexity","growth"), 0, "guard complexity metric"],
    ["Gain floor", "min gain required per added node", r => nested(r,"complexity","min_gain_required"), 5, "selection policy"],
    ["Zero-F1 classes", "diagnostic reflection slice", r => diag(r,"zero_f1_count"), 0, "evaluator diagnostics"],
    ["Classes evaluated", "diagnostic reflection slice", r => diag(r,"total_evaluated_classes"), 0, "evaluator diagnostics"],
    ["Doctor BLOCKED", "findings per round", r => doc(r,"BLOCKED"), 0, "doctor postflight"],
    ["Doctor WARNING", "findings per round", r => doc(r,"WARNING"), 0, "doctor postflight"],
    ["Doctor INFO", "findings per round", r => doc(r,"INFO"), 0, "doctor postflight"],
    ["Doctor UNVERIFIED", "findings per round", r => doc(r,"UNVERIFIED"), 0, "doctor postflight"],
    ["Cumulative KEEP", "rounds kept so far", r => cumKeep[rounds.indexOf(r)], 0, "run manifest"],
    ["Cumulative REVERT", "rounds reverted so far", r => cumRevert[rounds.indexOf(r)], 0, "run manifest"],
    ["Coordinator events", "events written for this round", evCount, 0, "events.jsonl"],
  ];
  for (const s of STAGES) {
    defs.push(["Stage · " + s.toLowerCase(), "recorded seconds in this stage",
      r => stageSecs(r, s), 1, "events.jsonl"]);
  }
  return {labels: labels, rounds: rounds, defs: defs.map(d => ({
    label: d[0], unit: d[1], get: d[2], digits: d[3], source: d[4],
    values: rounds.map(r => { const v = d[2](r);
      return (typeof v === "number" && isFinite(v)) ? v : null; })
  }))};
}

function disposeSparks(){
  for (const n of document.querySelectorAll(".spark")) {
    const g = echarts.getInstanceByDom(n);
    if (g) g.dispose();
  }
}

function renderTrends(){
  const cat = metricCatalogue();
  const host = el("trends");
  const C = pal();
  disposeSparks();
  host.textContent = "";
  const selected = state.round === "all" ? -1
    : cat.rounds.findIndex(r => String(r.round) === state.round);
  for (const m of cat.defs) {
    const observed = m.values.filter(v => v !== null);
    const card = document.createElement("section");
    card.className = "trend" + (observed.length ? "" : " is-na");
    const h = document.createElement("h3"); h.textContent = m.label;
    const u = document.createElement("p"); u.className = "tunit"; u.textContent = m.unit;
    const box = document.createElement("div"); box.className = "spark";
    const val = document.createElement("div"); val.className = "sparkval";
    const sub = document.createElement("p"); sub.className = "sparksub";
    sub.textContent = m.source;
    card.append(h, u, box, val, sub);
    host.appendChild(card);
    if (!observed.length) {
      const na = document.createElement("div"); na.className = "na";
      na.textContent = "NOT_AVAILABLE · nothing recorded";
      card.insertBefore(na, val);
      val.textContent = "—";
      continue;
    }
    const last = observed[observed.length - 1];
    val.textContent = last.toFixed(m.digits);
    const g = echarts.init(box);
    const lo = Math.min.apply(null, observed), hi = Math.max.apply(null, observed);
    const pad = Math.max((hi - lo) * 0.25, Math.abs(hi) * 0.001, 1e-6);
    const markLine = selected >= 0 && selected < cat.labels.length ? {
      silent: true, symbol: "none",
      lineStyle: {color: C.l4, width: 1, type: "dashed"},
      label: {show: false}, data: [{xAxis: selected}]
    } : {data: []};
    g.setOption({
      animationDuration: G18.DUR, animationEasing: "cubicOut",
      grid: {left: 4, right: 6, top: 8, bottom: 18},
      tooltip: {show: true, trigger: "axis", renderMode: "richText",
        backgroundColor: C.card, borderColor: C.grid, borderWidth: 1,
        textStyle: {color: C.ink, fontSize: 11},
        formatter: (ps) => {
          const i = ps && ps[0] ? ps[0].dataIndex : 0;
          return cat.labels[i] + "  " + (m.values[i] === null
            ? "NOT_AVAILABLE" : m.values[i].toFixed(m.digits));
        }},
      xAxis: {type: "category", data: cat.labels, boundaryGap: false,
        axisLine: {lineStyle: {color: C.grid}}, axisTick: {show: false},
        axisLabel: {color: C.l4, fontSize: 9, fontWeight: 600, margin: 6,
          interval: Math.max(0, Math.ceil(cat.labels.length / 4) - 1)}},
      yAxis: {type: "value", show: false, min: lo - pad, max: hi + pad},
      series: [{name: m.label, type: "line", data: m.values,
        connectNulls: true, smooth: 0.2, symbol: "none",
        lineStyle: {color: C.ink, width: 2}, z: 3,
        areaStyle: {color: {type: "linear", x: 0, y: 0, x2: 0, y2: 1,
          colorStops: [{offset: 0, color: hexA(C.ink, .13)},
                       {offset: 1, color: hexA(C.ink, 0)}]}},
        markLine: markLine}],
    });
    g.on("click", (p) => {
      const i = p.dataIndex;
      if (i === undefined || !cat.rounds[i]) return;
      state.round = String(cat.rounds[i].round);
      el("roundSel").value = state.round;
      renderHistory();
    });
  }
}

function renderHistory(){
  fillSelectors();
  renderSlice();
  // a 3s poll delivers the same payload again; re-initing every sparkline
  // would flicker and restart every draw-in. Only redraw on real changes.
  const sig = JSON.stringify([state.history, state.events, state.round, state.stage]);
  if (sig === state.sparkSig) return;
  state.sparkSig = sig;
  renderTrends();
}

for (const b of document.querySelectorAll(".tab")) {
  b.addEventListener("click", () => setView(b.dataset.view));
}
el("roundSel").addEventListener("change", (e) => {
  state.round = e.target.value; renderHistory();
});
el("stageSel").addEventListener("change", (e) => {
  state.stage = e.target.value; renderSlice();
});
el("clearSel").addEventListener("click", () => {
  state.round = "all"; state.stage = "all"; renderHistory();
});

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


def render_static(session: ResearchSession, *, max_events: int = MAX_EVENTS_PER_REQUEST,
                  frozen: bool = True) -> str:
    """The page with its payload inlined; `frozen` decides whether it polls.

    Same CSS, same markup, same renderers on every route. `/` passes
    ``frozen=False`` so the page hydrates from the inlined payload and then
    keeps polling; `/static` and the report archive pass ``frozen=True`` so a
    screenshot has a page that settles. Events are inlined in ascending seq
    order so the client's prepend renderer shows every row, newest first.
    """
    payload = _status_payload(session, after=0)
    # the history view needs every event to slice by round/stage, not just the
    # latest few. MAX_RESPONSE_BYTES still bounds the payload, and _clip has
    # already bounded every message and metric carried in each event.
    payload["events"] = list(payload.get("events") or [])[:max_events]
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
                # the client's first request never comes back. `frozen=False`
                # is what actually starts tick()/setInterval — the archive
                # route keeps frozen=True so a screenshot can settle.
                self._send(200, render_static(session, frozen=False).encode("utf-8"),
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
            # `?run=` is untrusted input. Resolve it BY NAME through the walk,
            # which only ever yields directories one level under the root, and
            # re-assert containment after following symlinks - never join the
            # raw value onto the root.
            if not RUN_ID_RE.match(pinned):
                raise FileNotFoundError(f"unknown run {pinned!r} under {runs_root}")
            root = Path(runs_root).resolve()
            for row in list_runs(runs_root):
                if row.get("run_id") == pinned and row.get("dir"):
                    run_dir = Path(row["dir"]).resolve()
                    if run_dir.parent != root:
                        raise FileNotFoundError(
                            f"run {pinned!r} resolves outside {runs_root}")
                    return ResearchSession.load(run_dir)
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
                self._send(200, render_static(session, frozen=False).encode("utf-8"),
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