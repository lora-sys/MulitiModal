"""Read-only monitor for an AutoResearch run directory.

One page, two views:

* **Current** — live status, the trusted metric per round against the champion,
  and a drill-down for the selected round (per-fold scores, device, budget,
  protocol hashes).
* **History** — every recorded number as a trend, plus a per-round picker.

Everything is rendered from `status.json`, `results.tsv` and `round-*.json`.
The page never writes to the run and never controls training. Charts are
hand-built SVG: no chart library, no network, no new dependency.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import stat
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

#: how many round-*.json files one page load may read
MAX_ROUNDS = 64
#: bound on one round file
MAX_ROUND_BYTES = 64 * 1024

PAGE = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AutoResearch 监控</title><style>
:root{
  --paper:#F0EFEB; --card:#F7F6F2; --ink:#1C1C1A; --muted:#8F8E88; --faint:#C6C5BF;
  --grid:#DEDDD6; --l2:#4A4944; --l3:#6A6963; --l4:#8F8E88; --l5:#B0AFA9;
  --ok:#2F6B4F; --bad:#9A3324; --warn:#A6601B;
  --font:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  --radius:20px; color-scheme:light;
}
@media (prefers-color-scheme:dark){
  :root{--paper:#161614; --card:#1E1E1B; --ink:#F0EFEB; --muted:#9A9992; --faint:#6B6A64;
        --grid:#333330; --l2:#CAC9C2; --l3:#A4A39C; --l4:#8B8A83; --l5:#5F5E59;
        --ok:#6DBE93; --bad:#E2826F; --warn:#E0A35C; color-scheme:dark;}
}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--font);
     -webkit-font-smoothing:antialiased;line-height:1.5}
.eyebrow{margin:0 0 10px;color:var(--l4);font-size:10px;font-weight:700;
         letter-spacing:.18em;text-transform:uppercase}
header{padding:36px 32px 20px;border-bottom:1px solid var(--grid)}
h1{margin:0;font-size:32px;font-weight:700;letter-spacing:-.035em;line-height:1.1}
.sub{margin:10px 0 0;color:var(--muted);font-size:13px;font-weight:500}
.sub b{color:var(--ink);font-weight:700}
.badge{display:inline-flex;align-items:center;gap:6px;margin-left:10px;padding:3px 11px;
  border-radius:999px;font-size:10px;font-weight:700;letter-spacing:.07em;
  text-transform:uppercase;background:var(--ink);color:var(--paper);vertical-align:middle}
.badge::before{content:"";width:5px;height:5px;border-radius:50%;background:currentColor;opacity:.85}
.badge.local{background:var(--warn)}
.badge.smoke{background:var(--l3)}
/* controls */
.controls{display:flex;gap:18px;align-items:center;flex-wrap:wrap;padding:14px 32px;
  border-bottom:1px solid var(--grid);position:sticky;top:0;z-index:20;background:var(--paper)}
.tabs{display:inline-flex;gap:4px;padding:4px;border-radius:999px;background:var(--card)}
.tab{appearance:none;border:0;background:transparent;color:var(--l4);cursor:pointer;
  padding:7px 18px;border-radius:999px;font:inherit;font-size:12px;font-weight:700;
  letter-spacing:.08em;text-transform:uppercase;transition:background .2s,color .2s}
.tab:hover{color:var(--ink)}
.tab.is-active{background:var(--ink);color:var(--paper)}
.tab:focus-visible,.chip:focus-visible,select:focus-visible{outline:2px solid var(--l3);outline-offset:2px}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin-left:auto}
.chip{appearance:none;border:0;cursor:pointer;font:inherit;font-size:11px;font-weight:700;
  padding:6px 13px;border-radius:999px;background:var(--card);color:var(--l4);
  font-variant-numeric:tabular-nums;transition:background .2s,color .2s}
.chip:hover{color:var(--ink)}
.chip.is-active{background:var(--ink);color:var(--paper)}
.chip.keep::after{content:"";display:inline-block;width:6px;height:6px;border-radius:50%;
  background:var(--ok);margin-left:7px;vertical-align:1px}
.chip.revert::after{content:"";display:inline-block;width:6px;height:6px;border-radius:50%;
  background:transparent;border:1.5px solid var(--l4);margin-left:7px;vertical-align:1px}
.wrap{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,320px),1fr));
  gap:18px;padding:24px 32px 40px}
.card{background:var(--card);border-radius:var(--radius);padding:26px 28px 22px;min-width:0}
.card.wide{grid-column:1/-1}
h2{margin:0;font-size:18px;font-weight:700;letter-spacing:-.025em;line-height:1.3}
.csub{margin:7px 0 18px;color:var(--muted);font-size:12.5px;font-weight:500}
.src{margin:16px 0 0;color:var(--l3);font-size:10px;font-weight:600;
  letter-spacing:.11em;text-transform:uppercase}
.kpi{font-size:38px;font-weight:800;letter-spacing:-.035em;line-height:1.05;
  font-variant-numeric:tabular-nums}
.kpi small{font-size:15px;font-weight:600;color:var(--muted);letter-spacing:0}
.stats{display:flex;gap:26px;flex-wrap:wrap;margin-top:16px}
.stat{min-width:96px}
.stat dt{color:var(--l4);font-size:10px;font-weight:700;letter-spacing:.1em;text-transform:uppercase}
.stat dd{margin:4px 0 0;font-size:17px;font-weight:700;font-variant-numeric:tabular-nums}
.chartbox{position:relative}
svg{display:block;width:100%;overflow:visible}
svg text{font-family:var(--font)}
.axis{font-size:10px;font-weight:600;fill:var(--l3)}
.gridline{stroke:var(--grid);stroke-width:1}
.refline{stroke:var(--l4);stroke-width:1.4;stroke-dasharray:5 4}
.metricline{fill:none;stroke:var(--ink);stroke-width:2.6;stroke-linecap:round;
  stroke-linejoin:round}
.area{fill:var(--ink);opacity:.07}
.dot{stroke:var(--card);stroke-width:2}
.dot.keep{fill:var(--ink)}
.dot.revert{fill:var(--card);stroke:var(--l3);stroke-width:2}
.bar{fill:var(--ink);opacity:.82}
.bar.best{fill:var(--ok)}
.hit{fill:transparent;cursor:pointer}
.hit:hover{fill:var(--ink);opacity:.05}
.tip{position:absolute;z-index:30;pointer-events:none;opacity:0;transition:opacity .12s;
  background:var(--card);border:1px solid var(--grid);border-radius:10px;padding:10px 12px;
  font-size:12px;box-shadow:0 6px 24px rgba(0,0,0,.14);max-width:280px}
.tip b{font-variant-numeric:tabular-nums}
.tip .t{font-weight:700;margin-bottom:4px}
.tip .r{color:var(--muted);font-variant-numeric:tabular-nums}
.legend{display:flex;gap:20px;flex-wrap:wrap;align-items:center;margin:0 0 10px;
  color:var(--l4);font-size:11px;font-weight:600}
.legend i{display:inline-block;width:9px;height:9px;margin-right:6px;vertical-align:-1px;
  border-radius:50%;background:var(--ink)}
.legend i.hollow{background:var(--card);border:1.5px solid var(--l3)}
.legend i.dash{width:18px;height:0;border-radius:0;background:transparent;
  border-top:1.5px dashed var(--l4);vertical-align:3px}
.kv{display:grid;grid-template-columns:auto 1fr;gap:7px 16px;font-size:12.5px}
.kv dt{color:var(--l4)}
.kv dd{margin:0;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}
.kv dd.mono{font-family:var(--mono);font-size:11px}
.two{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.1fr);gap:26px}
.tablewrap{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-size:13px;table-layout:fixed}
th,td{text-align:left;padding:9px 10px;border-bottom:1px solid var(--grid);
  vertical-align:top;overflow-wrap:anywhere}
th{color:var(--l4);font-weight:700;font-size:10px;letter-spacing:.1em;text-transform:uppercase}
td.num{text-align:right;font-variant-numeric:tabular-nums;font-weight:600}
td.mono{font-family:var(--mono);font-size:11px}
tbody tr{cursor:pointer}
tbody tr:hover{background:var(--paper)}
tbody tr.is-active{background:var(--paper);box-shadow:inset 3px 0 0 var(--ink)}
.tag{display:inline-flex;align-items:center;padding:3px 9px;border-radius:7px;font-size:10px;
  font-weight:700;letter-spacing:.06em;text-transform:uppercase;background:var(--grid);color:var(--l3)}
.tag.KEEP{background:var(--ok);color:#fff}
.tag.REVERT{background:transparent;border:1px solid var(--l4);color:var(--l3)}
.trends{display:grid;grid-template-columns:repeat(auto-fit,minmax(215px,1fr));gap:16px}
.trend{background:var(--paper);border-radius:16px;padding:16px 18px 12px;min-width:0}
.trend h3{margin:0;font-size:12.5px;font-weight:700;letter-spacing:-.01em}
.trend .u{margin:4px 0 6px;color:var(--l4);font-size:10.5px;font-weight:600}
.spark{height:76px}
.tv{font-size:21px;font-weight:800;letter-spacing:-.03em;font-variant-numeric:tabular-nums;line-height:1.1}
.ts{color:var(--l4);font-size:10px;font-weight:600;letter-spacing:.09em;
  text-transform:uppercase;margin:4px 0 0}
.na{color:var(--muted);font-style:italic}
.muted{color:var(--muted)}
.pill{display:inline-block;padding:2px 9px;border-radius:999px;background:var(--grid);
  color:var(--l3);font-size:10px;font-weight:700;letter-spacing:.06em;text-transform:uppercase}
footer{padding:20px 32px 32px;color:var(--l3);font-size:10px;font-weight:600;
  letter-spacing:.11em;text-transform:uppercase;border-top:1px solid var(--grid)}
@media (max-width:760px){
  header{padding:26px 18px 16px} h1{font-size:26px}
  .controls{padding:12px 18px} .chips{margin-left:0;width:100%}
  .wrap{padding:18px 18px 28px;gap:14px} .card{padding:20px 18px}
  .two{grid-template-columns:1fr;gap:20px} footer{padding:18px}
  .kpi{font-size:32px}
}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style></head><body>
<header>
  <p class="eyebrow">autonomous research · read-only monitor · loopback</p>
  <h1 id="title">AutoResearch</h1>
  <p class="sub" id="idsub">loading…</p>
</header>
<div class="controls">
  <div class="tabs" role="tablist" aria-label="dashboard view">
    <button type="button" class="tab is-active" data-view="current" role="tab" aria-selected="true">Current</button>
    <button type="button" class="tab" data-view="history" role="tab" aria-selected="false">History</button>
  </div>
  <div class="chips" id="chips" role="group" aria-label="round selector"></div>
</div>

<div id="view-current">
<div class="wrap">
  <section class="card wide">
    <h2 id="verdict">Waiting for the first scored round</h2>
    <p class="csub" id="verdictsub">the trusted evaluator scores; a candidate never reports its own number</p>
    <div class="kpi" id="kpi">—</div>
    <dl class="stats" id="stats"></dl>
    <p class="src" id="verdictsrc">source: results.tsv · run manifest</p>
  </section>

  <section class="card wide">
    <h2>Trusted metric per round, against the best so far</h2>
    <p class="csub" id="chartsub">one line, one dot per round · click a dot or a row to inspect that round</p>
    <div class="legend">
      <span><i></i> kept round</span><span><i class="hollow"></i> reverted round</span>
      <span><i class="dash"></i> best so far</span>
    </div>
    <div class="chartbox" id="chartbox">
      <svg id="chart" viewBox="0 0 860 300" role="img" aria-label="每轮可信指标与最佳值"></svg>
      <div class="tip" id="tip" role="status"></div>
    </div>
    <p class="src">source: round-*.json · trusted evaluator (isolated process)</p>
  </section>

  <section class="card wide">
    <h2 id="roundtitle">Round detail</h2>
    <p class="csub" id="roundsub">select a round from the chart, the chips or the table</p>
    <div class="two">
      <div>
        <h3 class="eyebrow">Protocol</h3>
        <dl class="kv" id="roundkv"></dl>
      </div>
      <div>
        <h3 class="eyebrow">Per-fold scores</h3>
        <div class="chartbox">
          <svg id="folds" viewBox="0 0 560 240" role="img" aria-label="每折指标"></svg>
        </div>
        <p class="src" id="foldsrc"></p>
      </div>
    </div>
    <p class="src">source: round-NNN.json</p>
  </section>

  <section class="card wide">
    <h2>Experiment record</h2>
    <p class="csub">append-only · one row per decided round</p>
    <div class="tablewrap"><table id="table">
      <colgroup><col style="width:96px"><col style="width:96px"><col style="width:88px">
      <col style="width:26%"><col><col style="width:120px"></colgroup>
      <thead><tr><th>Commit</th><th class="num">指标值</th><th>决定</th><th>假设</th>
      <th>证据 / 设备</th><th>Protocol</th></tr></thead><tbody id="rows"></tbody></table></div>
    <p class="src">source: results.tsv</p>
  </section>
</div>
</div>

<div id="view-history" hidden>
<div class="wrap">
  <section class="card wide">
    <h2 id="slicetitle">All rounds</h2>
    <p class="csub" id="slicesub">every recorded number, as a trend</p>
    <div class="trends" id="trends"></div>
    <p class="src">source: round-*.json · NOT_AVAILABLE means nothing was recorded</p>
  </section>
  <section class="card wide">
    <h2>Round picker</h2>
    <p class="csub">choose a round to load its full record into the Current view</p>
    <div class="tablewrap"><table id="picker">
      <colgroup><col style="width:64px"><col style="width:110px"><col style="width:96px">
      <col style="width:26%"><col></colgroup>
      <thead><tr><th>#</th><th>Commit</th><th class="num">指标值</th><th>假设</th>
      <th>设备 / 预算</th></tr></thead><tbody id="pickerrows"></tbody></table></div>
    <p class="src">source: results.tsv + round-*.json</p>
  </section>
</div>
</div>

<footer>read-only view · operations live in the cli · local records are not independently attested</footer>
<script>
const state = {view:"current", round:-1, sig:""};
const NS = "http://www.w3.org/2000/svg";
const $ = id => document.getElementById(id);
function num(v){ return (typeof v === "number" && isFinite(v)) ? v : null; }
function fmt(v, d){ return num(v) === null ? "NOT_AVAILABLE" : Number(v).toFixed(d === undefined ? 4 : d); }
function short(h, n){ h = String(h || ""); return h.length > (n || 12) ? h.slice(0, n || 12) + "…" : h; }
function el(name, attrs, text){
  const n = document.createElementNS(NS, name);
  for (const k in (attrs || {})) n.setAttribute(k, attrs[k]);
  if (text !== undefined && text !== null) n.textContent = text;
  return n;
}
function css(name, fallback){
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
}

/* ---- charts ------------------------------------------------------------- */

function lineChart(host, rounds, metricName){
  host.replaceChildren();
  host.setAttribute("viewBox", "0 0 860 300");
  const scored = rounds.filter(r => num(r.value) !== null);
  if (!scored.length){
    host.appendChild(el("text", {x: 430, y: 150, "text-anchor": "middle",
      class: "axis", fill: css("--muted", "#8F8E88")}, "No scored round yet (NOT_AVAILABLE)"));
    return;
  }
  const W = 860, H = 300, L = 62, R = 24, T = 34, B = 46;
  const iw = W - L - R, ih = H - T - B;
  const vals = scored.map(r => r.value);
  const best = [];
  let champion = null;
  scored.forEach(r => { if (champion === null || r.value > champion) champion = r.value; best.push(champion); });
  const lo = Math.min(...vals), hi = Math.max(...vals, ...best);
  const pad = Math.max((hi - lo) * 0.3, Math.abs(hi) * 0.002, 1e-6);
  const y = v => T + ih - (v - (lo - pad)) / ((hi + pad) - (lo - pad)) * ih;
  const x = i => L + (scored.length === 1 ? iw / 2 : i * iw / (scored.length - 1));
  const ink = css("--ink", "#1C1C1A"), card = css("--card", "#F7F6F2");
  const grid = css("--grid", "#DEDDD6"), l3 = css("--l3", "#6A6963");
  const ok = css("--ok", "#2F6B4F"), l4 = css("--l4", "#8F8E88");

  for (let i = 0; i <= 4; i++){
    const v = (lo - pad) + ((hi + pad) - (lo - pad)) * i / 4;
    host.appendChild(el("line", {x1: L, x2: W - R, y1: y(v), y2: y(v), class: "gridline"}));
    host.appendChild(el("text", {x: L - 10, y: y(v) + 3.5, "text-anchor": "end",
      class: "axis", fill: l3}, v.toFixed(3)));
  }
  scored.forEach((r, i) => host.appendChild(el("text",
    {x: x(i), y: H - B + 20, "text-anchor": "middle", class: "axis", fill: l3}, "R" + r.round)));
  // area + line
  const pts = scored.map((r, i) => x(i) + "," + y(r.value)).join(" ");
  host.appendChild(el("polygon", {
    points: `${L},${T + ih} ${pts} ${x(scored.length - 1)},${T + ih}`,
    fill: ink, opacity: ".07"}));
  host.appendChild(el("polyline", {points: pts, class: "metricline", stroke: ink}));
  // best-so-far reference
  host.appendChild(el("polyline", {
    points: best.map((v, i) => x(i) + "," + y(v)).join(" "),
    fill: "none", stroke: l4, "stroke-width": "1.4", "stroke-dasharray": "5 4"}));
  // decision dots + hit areas
  const tip = $("tip"), box = $("chartbox");
  scored.forEach((r, i) => {
    const keep = r.decision === "KEEP";
    host.appendChild(el("circle", {cx: x(i), cy: y(r.value), r: keep ? 6 : 5.5,
      class: "dot " + (keep ? "keep" : "revert"),
      fill: keep ? ink : card, stroke: keep ? card : l3, "stroke-width": keep ? 2 : 2}));
    const hit = el("circle", {cx: x(i), cy: y(r.value), r: 18, class: "hit"});
    hit.setAttribute("tabindex", "0");
    hit.setAttribute("role", "button");
    hit.setAttribute("aria-label", "round " + r.round + " " + r.decision + " " + fmt(r.value));
    const show = () => {
      tip.innerHTML = "";
      const t = document.createElement("div"); t.className = "t";
      t.textContent = "R" + r.round + " · " + (r.decision || "—");
      const v = document.createElement("div");
      v.innerHTML = ""; v.append(document.createTextNode(metricName + " "));
      const b = document.createElement("b"); b.textContent = fmt(r.value); v.append(b);
      const s = document.createElement("div"); s.className = "r";
      s.textContent = (r.description || "").slice(0, 90);
      tip.append(t, v, s);
      tip.style.opacity = "1";
      const bx = box.getBoundingClientRect(), hx = hit.getBoundingClientRect();
      tip.style.left = Math.max(4, Math.min(bx.width - 200, hx.left - bx.left + hx.width / 2 - 100)) + "px";
      tip.style.top = Math.max(4, hx.top - bx.top - tip.offsetHeight - 12) + "px";
    };
    const hide = () => { tip.style.opacity = "0"; };
    hit.addEventListener("mouseenter", show);
    hit.addEventListener("focus", show);
    hit.addEventListener("mouseleave", hide);
    hit.addEventListener("blur", hide);
    hit.addEventListener("click", () => select(r.round));
    hit.addEventListener("keydown", e => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); select(r.round); }
    });
    host.appendChild(hit);
  });
}

function foldChart(host, detail){
  host.replaceChildren();
  host.setAttribute("viewBox", "0 0 560 240");
  const folds = (detail && detail.folds) || [];
  if (!folds.length){
    host.appendChild(el("text", {x: 280, y: 120, "text-anchor": "middle",
      class: "axis", fill: css("--muted", "#8F8E88")}, "No per-fold scores (NOT_AVAILABLE)"));
    return;
  }
  const W = 560, H = 240, L = 52, R = 16, T = 22, B = 40;
  const iw = W - L - R, ih = H - T - B;
  const vals = folds.map(f => num(f.macro_f1) || 0);
  const hi = Math.max(...vals) * 1.15 || 1, lo = 0;
  const y = v => T + ih - (v - lo) / (hi - lo) * ih;
  const bw = Math.min(46, iw / folds.length * 0.6);
  const ink = css("--ink", "#1C1C1A"), ok = css("--ok", "#2F6B4F");
  const grid = css("--grid", "#DEDDD6"), l3 = css("--l3", "#6A6963");
  const bestIdx = vals.indexOf(Math.max(...vals));
  for (let i = 0; i <= 3; i++){
    const v = lo + (hi - lo) * i / 3;
    host.appendChild(el("line", {x1: L, x2: W - R, y1: y(v), y2: y(v), class: "gridline"}));
    host.appendChild(el("text", {x: L - 8, y: y(v) + 3.5, "text-anchor": "end",
      class: "axis", fill: l3}, v.toFixed(3)));
  }
  folds.forEach((f, i) => {
    const cx = L + iw * (i + 0.5) / folds.length;
    host.appendChild(el("rect", {x: cx - bw / 2, y: y(num(f.macro_f1) || 0),
      width: bw, height: T + ih - y(num(f.macro_f1) || 0), rx: 4,
      class: "bar" + (i === bestIdx ? " best" : ""), fill: i === bestIdx ? ok : ink,
      opacity: i === bestIdx ? ".9" : ".55"}));
    host.appendChild(el("text", {x: cx, y: H - B + 18, "text-anchor": "middle",
      class: "axis", fill: l3}, "f" + f.fold + "/" + f.repeat));
    host.appendChild(el("text", {x: cx, y: y(num(f.macro_f1) || 0) - 7,
      "text-anchor": "middle", class: "axis", fill: l3}, (num(f.macro_f1) || 0).toFixed(3)));
  });
  $("foldsrc").textContent = folds.length + " folds · best highlighted";
}

function sparkline(host, values){
  host.replaceChildren();
  host.setAttribute("viewBox", "0 0 240 76");
  const vals = values.filter(v => num(v) !== null);
  if (vals.length < 2){
    host.appendChild(el("text", {x: 120, y: 40, "text-anchor": "middle",
      class: "axis", fill: css("--muted", "#8F8E88")}, "—"));
    return;
  }
  const W = 240, H = 76, P = 8;
  const lo = Math.min(...vals), hi = Math.max(...vals), span = (hi - lo) || Math.abs(hi) * .01 || 1;
  const x = i => P + i * (W - 2 * P) / (vals.length - 1);
  const y = v => P + (1 - (v - lo) / span) * (H - 2 * P);
  const ink = css("--ink", "#1C1C1A");
  host.appendChild(el("polygon", {
    points: `${P},${H - P} ${vals.map((v, i) => x(i) + "," + y(v)).join(" ")} ${x(vals.length - 1)},${H - P}`,
    fill: ink, opacity: ".08"}));
  host.appendChild(el("polyline", {points: vals.map((v, i) => x(i) + "," + y(v)).join(" "),
    fill: "none", stroke: ink, "stroke-width": "2", "stroke-linecap": "round"}));
  vals.forEach((v, i) => host.appendChild(el("circle", {cx: x(i), cy: y(v), r: 2.6,
    fill: ink, stroke: css("--card", "#F7F6F2"), "stroke-width": "1.4"})));
}

/* ---- views -------------------------------------------------------------- */

function renderChips(rounds){
  const box = $("chips");
  box.replaceChildren();
  rounds.forEach(r => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "chip" + (r.round === state.round ? " is-active" : "")
      + " " + String(r.decision || "").toLowerCase();
    b.textContent = "R" + r.round;
    b.title = (r.description || "").slice(0, 120);
    b.addEventListener("click", () => select(r.round));
    box.appendChild(b);
  });
}

function select(round){
  state.round = round;
  renderChips(lastRounds);
  const detail = lastRounds.find(r => r.round === round);
  const kv = $("roundkv");
  kv.replaceChildren();
  if (!detail){
    $("roundtitle").textContent = "Round detail";
    $("roundsub").textContent = "select a round from the chart, the chips or the table";
    foldChart($("folds"), null);
    return;
  }
  $("roundtitle").textContent = "R" + detail.round + " · " + (detail.decision || "—");
  $("roundsub").textContent = (detail.description || "").slice(0, 160);
  const rows = [
    ["指标值", fmt(detail.value)],
    ["fold std", fmt(detail.detail && detail.detail.fold_std)],
    ["commit", detail.commit],
    ["设备", [detail.detail && detail.detail.device, detail.detail && detail.detail.gpu]
      .filter(Boolean).join(" · ") || "NOT_AVAILABLE"],
    ["每折预算", detail.detail ? fmt(detail.detail.budget_seconds, 0) + " s" : "NOT_AVAILABLE"],
    ["seed", detail.detail ? String(detail.detail.seed) : "NOT_AVAILABLE"],
    ["折数", detail.detail ? detail.detail.fold_count + " × " + detail.detail.repeats : "NOT_AVAILABLE"],
    ["数据集", detail.detail ? short(detail.detail.dataset_sha256, 20) : "NOT_AVAILABLE"],
    ["protocol", short(detail.protocol, 20)],
  ];
  rows.forEach(([k, v]) => {
    const dt = document.createElement("dt"); dt.textContent = k;
    const dd = document.createElement("dd");
    if (k === "commit" || k === "protocol" || k === "数据集") dd.className = "mono";
    dd.textContent = v || "NOT_AVAILABLE";
    kv.append(dt, dd);
  });
  foldChart($("folds"), detail.detail);
  document.querySelectorAll("#rows tr").forEach(tr =>
    tr.classList.toggle("is-active", Number(tr.dataset.round) === round));
  document.querySelectorAll("#pickerrows tr").forEach(tr =>
    tr.classList.toggle("is-active", Number(tr.dataset.round) === round));
}

let lastRounds = [];

function renderTable(rounds){
  const tb = $("rows");
  tb.replaceChildren();
  rounds.forEach(r => {
    const tr = document.createElement("tr");
    tr.dataset.round = r.round;
    if (r.round === state.round) tr.classList.add("is-active");
    const cells = [
      [short(r.commit, 8), "mono"],
      [fmt(r.value), "num"],
      [null, null],
      [r.description || "—", null],
      [[r.evidence, r.detail && r.detail.device].filter(Boolean).join(" · "), null],
      [short(r.protocol, 8), "mono"],
    ];
    cells.forEach(([v, cls], i) => {
      const td = document.createElement("td");
      if (cls) td.className = cls;
      if (i === 2){
        const tag = document.createElement("span");
        tag.className = "tag " + (r.decision || "");
        tag.textContent = r.decision || "—";
        td.appendChild(tag);
      } else {
        td.textContent = v || "—";
      }
      tr.appendChild(td);
    });
    tr.addEventListener("click", () => select(r.round));
    tb.appendChild(tr);
  });
}

function renderPicker(rounds){
  const tb = $("pickerrows");
  tb.replaceChildren();
  rounds.forEach(r => {
    const tr = document.createElement("tr");
    tr.dataset.round = r.round;
    if (r.round === state.round) tr.classList.add("is-active");
    const cells = [
      ["R" + r.round, "num"],
      [short(r.commit, 10), "mono"],
      [fmt(r.value), "num"],
      [r.description || "—", null],
      [[r.detail && r.detail.device, r.detail && fmt(r.detail.budget_seconds, 0) + "s"]
        .filter(Boolean).join(" · "), null],
    ];
    cells.forEach(([v, cls]) => {
      const td = document.createElement("td");
      if (cls) td.className = cls;
      td.textContent = v || "—";
      tr.appendChild(td);
    });
    tr.addEventListener("click", () => { select(r.round); setView("current"); });
    tb.appendChild(tr);
  });
}

function trendCards(rounds){
  const host = $("trends");
  host.replaceChildren();
  const series = [
    ["Trusted metric", "primary metric · higher is better", r => num(r.value)],
    ["Fold dispersion", "std across folds", r => num(r.detail && r.detail.fold_std)],
    ["Mean fold time", "seconds per fold", r => {
      const f = (r.detail && r.detail.folds) || [];
      const t = f.map(x => num(x.train_seconds)).filter(v => v !== null);
      return t.length ? t.reduce((a, b) => a + b, 0) / t.length : null;
    }],
    ["Validation rows", "scored rows per fold", r => {
      const f = (r.detail && r.detail.folds) || [];
      const n = f.map(x => num(x.n_validation)).filter(v => v !== null);
      return n.length ? n.reduce((a, b) => a + b, 0) / n.length : null;
    }],
  ];
  series.forEach(([name, unit, get]) => {
    const vals = rounds.map(get);
    const card = document.createElement("section");
    card.className = "trend";
    const h = document.createElement("h3"); h.textContent = name;
    const u = document.createElement("p"); u.className = "u"; u.textContent = unit;
    const box = document.createElement("div"); box.className = "spark";
    const observed = vals.filter(v => v !== null);
    const v = document.createElement("div"); v.className = "tv";
    v.textContent = observed.length ? observed[observed.length - 1].toFixed(4) : "—";
    const s = document.createElement("p"); s.className = "ts";
    s.textContent = observed.length ? "last of " + observed.length : "NOT_AVAILABLE";
    card.append(h, u, box, v, s);
    host.appendChild(card);
    sparkline(box, vals);
  });
}

function setView(view){
  state.view = view;
  document.querySelectorAll(".tab").forEach(b => {
    const on = b.dataset.view === view;
    b.classList.toggle("is-active", on);
    b.setAttribute("aria-selected", on ? "true" : "false");
  });
  $("view-current").hidden = view !== "current";
  $("view-history").hidden = view !== "history";
}
// ?view=history lands straight on the history view — used by archived pages
// and by screenshots that must not depend on a click.
try{
  const wanted = new URLSearchParams(location.search).get("view");
  if (wanted === "history" || wanted === "current") setView(wanted);
}catch(e){}

function render(d){
  lastRounds = d.rounds || [];
  const metric = (d.rounds[0] && d.rounds[0].metric) || "metric";
  $("title").textContent = "AutoResearch · " + (d.task || "run");
  $("idsub").textContent = "";
  const b = document.createElement("b"); b.textContent = short(d.run, 28);
  $("idsub").append(b, document.createTextNode(" · " + metric
    + (d.direction === "minimize" ? " ↓" : " ↑")));
  const badge = document.createElement("span");
  badge.className = "badge" + (d.evidence === "LOCAL_EXPERIMENTAL" ? " local" : "");
  badge.textContent = d.evidence || "NOT_AVAILABLE";
  $("idsub").append(badge);

  const scored = lastRounds.filter(r => num(r.value) !== null);
  const best = scored.reduce((a, r) => (a === null || r.value > a) ? r.value : a, null);
  const keeps = lastRounds.filter(r => r.decision === "KEEP").length;
  const reverts = lastRounds.filter(r => r.decision === "REVERT").length;
  $("verdict").textContent = best === null ? "Waiting for the first scored round"
    : "Best " + metric + " so far: " + fmt(best) + ", decided in " + lastRounds.length + " round(s)";
  $("verdictsub").textContent = "decisions {\"KEEP\":" + keeps + ",\"REVERT\":" + reverts
    + "} · a KEEP must clear the fold dispersion, not just move the number";
  $("kpi").textContent = best === null ? "—" : fmt(best);
  const stats = $("stats");
  stats.replaceChildren();
  const detail = lastRounds[lastRounds.length - 1] && lastRounds[lastRounds.length - 1].detail;
  [["rounds", lastRounds.length], ["keeps", keeps], ["reverts", reverts],
   ["device", (detail && detail.device) || "NOT_AVAILABLE"],
   ["gpu", short(detail && detail.gpu, 22) || "NOT_AVAILABLE"],
   ["dataset", short(detail && detail.dataset_sha256, 12)]].forEach(([k, v]) => {
    const dt = document.createElement("dt"); dt.textContent = k;
    const dd = document.createElement("dd"); dd.textContent = v === undefined ? "NOT_AVAILABLE" : v;
    stats.append(dt, dd);
  });
  $("chartsub").textContent = metric + " per round · the line draws, the number counts · click a dot to inspect";

  if (state.round < 0 && scored.length) state.round = scored[scored.length - 1].round;
  if (!lastRounds.some(r => r.round === state.round)) state.round = scored.length ? scored[scored.length - 1].round : -1;
  renderChips(lastRounds);
  lineChart($("chart"), lastRounds, metric);
  renderTable(lastRounds);
  renderPicker(lastRounds);
  select(state.round);
  trendCards(lastRounds);
  $("slicetitle").textContent = "All " + lastRounds.length + " rounds";
  $("slicesub").textContent = metric + " and what the coordinator measured alongside it";
}

async function refresh(){
  try{
    const r = await fetch("/api", {cache: "no-store"});
    if (!r.ok) throw Error("HTTP " + r.status);
    const d = await r.json();
    const sig = JSON.stringify([d.rounds, d.status, d.evidence]);
    if (sig !== state.sig){ state.sig = sig; render(d); }
    else {
      // keep the live status line fresh even when nothing else changed
      $("verdictsrc").textContent = "source: results.tsv · status " + d.status;
    }
  }catch(e){
    $("idsub").textContent = "监控读取失败：" + e.message;
  }
}
document.querySelectorAll(".tab").forEach(b =>
  b.addEventListener("click", () => setView(b.dataset.view)));
addEventListener("resize", () => {
  const rounds = lastRounds;
  if (rounds.length){ lineChart($("chart"), rounds, (rounds[0] && rounds[0].metric) || "metric"); }
  const detail = rounds.find(r => r.round === state.round);
  foldChart($("folds"), detail && detail.detail);
});
refresh(); setInterval(refresh, 3000);
</script></body></html>'''


def _read_file(path: Path, limit: int) -> str:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0))
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError("monitor input is not a bounded regular file")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            data = stream.read(limit + 1)
        if len(data) > limit:
            raise ValueError("monitor input exceeds the limit")
        return data.decode("utf-8")
    finally:
        os.close(fd)


def _round_files(root: Path) -> list[Path]:
    """Newest round-*.json files first, bounded by MAX_ROUNDS."""
    try:
        found = sorted(root.glob("round-*.json"), reverse=True)
    except OSError:
        return []
    return found[:MAX_ROUNDS]


def read_run(root: Path) -> dict:
    """Everything the page shows, read from one run directory."""
    status = "暂无运行状态"
    evidence = "NOT_AVAILABLE"
    task = "NOT_AVAILABLE"
    reason = ""
    live: dict = {}
    try:
        live = json.loads(_read_file(root / "status.json", 16 * 1024))
        state = live.get("state", "NOT_AVAILABLE")
        evidence = live.get("evidence", "NOT_AVAILABLE")
        task = str(live.get("task", "NOT_AVAILABLE"))[:100]
        reason = str(live.get("reason", ""))[:100]
        if state in ("PREPARING", "TRAINING") and time.time() > float(live.get("deadline", 0)):
            state = "STALE"
        status = str(state)
        if live.get("fold") is not None:
            status += f" · 折 {live['fold']}/{live.get('folds', '?')}"
    except FileNotFoundError:
        pass
    except (OSError, ValueError, TypeError):
        status = "状态不可读"

    rows: list[dict] = []
    try:
        text = _read_file(root / "results.tsv", 2 * 1024 * 1024)
        rows = list(csv.DictReader(io.StringIO(text), delimiter="\t"))
    except FileNotFoundError:
        pass

    rounds = []
    for index, row in enumerate(rows):
        detail = None
        for path in _round_files(root):
            if path.name == f"round-{index:03d}.json":
                try:
                    detail = json.loads(_read_file(path, MAX_ROUND_BYTES))
                except (OSError, ValueError):
                    detail = None
                break
        # results.tsv stores the score as text; the page needs a number, and a
        # CRASH row carries none.
        raw = str(row.get("value") or "").strip()
        try:
            value = float(raw) if raw else None
        except ValueError:
            value = None
        rounds.append({
            "round": index,
            "commit": str(row.get("commit") or ""),
            "metric": str(row.get("metric") or ""),
            "value": value,
            "direction": str(row.get("direction") or ""),
            "decision": str(row.get("decision") or ""),
            "description": str(row.get("description") or ""),
            "evidence": str(row.get("evidence") or ""),
            "protocol": str(row.get("protocol_sha256") or ""),
            "detail": detail,
        })

    latest = next((r["detail"] for r in reversed(rounds) if r["detail"]), None)
    return {
        "run": root.name,
        "status": status,
        "task": task,
        "reason": reason,
        "evidence": evidence,
        "direction": (rounds[0]["direction"] if rounds else "maximize"),
        "state_detail": live,
        "rounds": rounds,
        "device": (latest or {}).get("device", "NOT_AVAILABLE"),
        "gpu": (latest or {}).get("gpu", "NOT_AVAILABLE"),
        "dataset_sha256": (latest or {}).get("dataset_sha256", "NOT_AVAILABLE"),
        "results": [
            {key: str(row.get(key) or "")[:400] for key in
             ("commit", "metric", "value", "direction", "decision", "description",
              "evidence", "peak_vram_mb")}
            for row in rows
        ],
    }


def make_server(root: Path, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            host = self.headers.get("Host", "").lower()
            port = self.server.server_port
            allowed = {"localhost", "127.0.0.1", f"localhost:{port}", f"127.0.0.1:{port}"}
            if host not in allowed:
                self.send_error(403)
                return
            if self.path == "/":
                payload, content_type = PAGE.encode(), "text/html; charset=utf-8"
            elif self.path == "/api":
                try:
                    payload = json.dumps(read_run(root), ensure_ascii=False).encode()
                except (OSError, ValueError):
                    self.send_error(500, "Cannot read run records")
                    return
                content_type = "application/json; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    make_server(args.run.resolve(strict=True), args.port).serve_forever()


if __name__ == "__main__":
    main()
