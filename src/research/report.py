"""Deterministic report for one run: what was tried, what was decided, what is unknown.

The report is written from persisted state only (the manifest, the events, the
last Doctor report and the run spec), so re-running it on the same run directory
produces the same file. Nothing is recomputed from a model and no score is
derived here: the trusted evaluator produced every number that appears.

Sections the issue asks for, and the honesty rule that comes with them:

* the round table is the research trail — hypothesis, controlled change,
  expected observation, observed evidence, decision, next action;
* the outcome column separates a model result (`KEEP`/`REVERT`) from an
  engineering refusal (`INVALID_PROTOCOL`, `UNSUPPORTED_ARCHITECTURE`, `CRASH`);
* anything the host did not measure is printed as `NOT_AVAILABLE`, never filled
  in with a plausible value.
"""

from __future__ import annotations

import json
from pathlib import Path

OUTCOMES = ("KEEP", "REVERT", "INVALID_PROTOCOL", "UNSUPPORTED_ARCHITECTURE",
            "CRASH", "BUDGET_EXHAUSTED")


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "NOT_AVAILABLE"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_report(session) -> dict:
    """The machine-readable report; `render_markdown` is its human view."""
    entries = session.entries()
    champion = session.champion()
    status = session.status()
    doctor = session.read_doctor() or {}
    outcomes = {name: 0 for name in OUTCOMES}
    for entry in entries:
        outcomes[entry.get("outcome") or entry.get("decision") or "UNKNOWN"] = (
            outcomes.get(entry.get("outcome") or entry.get("decision") or "UNKNOWN", 0) + 1
        )
    return {
        "schema_version": "autoresearch-report-v1",
        "run_id": session.run_id,
        "task": session.spec.get("task"),
        "created_at": session.spec.get("created_at"),
        "goal": session.spec.get("goal") or "NOT_AVAILABLE",
        "branch": session.spec.get("branch"),
        "editable": session.spec.get("editable"),
        "primary_metric": session.spec.get("primary_metric"),
        "selection_policy": session.spec.get("policy") or {},
        "budget": {
            "max_rounds": session.spec.get("max_rounds"),
            "rounds_completed": len(entries),
            "budget_seconds_per_fold": session.spec.get("budget_seconds"),
            "folds": session.spec.get("folds"),
            "repeats": session.spec.get("repeats"),
            "max_wall_clock_s": session.spec.get("max_wall_clock_s"),
        },
        "outcomes": outcomes,
        "champion": {
            "run_id": champion.get("run_id") if champion else None,
            "git_commit_sha": champion.get("git_commit_sha") if champion else None,
            "best_metric": champion.get("best_so_far") if champion else None,
        },
        "agent_kind": (entries[-1].get("agent_kind") if entries else None),
        "autonomous_research": bool(
            entries and entries[-1].get("agent_kind") not in (None, "deterministic-stub")
        ),
        "doctor": {
            "blocked": doctor.get("blocked"),
            "counts": doctor.get("counts"),
            "findings": [
                {"check": f.get("check"), "severity": f.get("severity"),
                 "message": f.get("message")}
                for f in (doctor.get("findings") or [])
            ],
        },
        "rounds": [
            {
                "run_id": e.get("run_id"),
                "round": e.get("round_index"),
                "hypothesis": e.get("hypothesis") or "NOT_AVAILABLE",
                "controlled_change": e.get("controlled_change") or "NOT_AVAILABLE",
                "expected_observation": e.get("expected_observation") or "NOT_AVAILABLE",
                "observed_evidence": e.get("observed_evidence") or "NOT_AVAILABLE",
                "decision": e.get("decision"),
                "outcome": e.get("outcome"),
                "status": e.get("status"),
                "reason": e.get("reason"),
                "next_action": e.get("next_action") or "NOT_AVAILABLE",
                "primary_value": (e.get("metrics") or {}).get("primary_value"),
                "primary_sem": (e.get("metrics") or {}).get("primary_sem"),
                "best_so_far": e.get("best_so_far"),
                "complexity": e.get("complexity"),
                "git_commit_sha": e.get("git_commit_sha"),
                "doctor_counts": (e.get("doctor") or {}).get("counts"),
                "per_class": e.get("per_class") or "NOT_AVAILABLE",
                "loss_history": (e.get("diagnostics") or {}).get("losses", "NOT_AVAILABLE"),
                "resource_telemetry": {
                    "elapsed_s": e.get("elapsed_s"),
                    "gpu_memory_mb": (e.get("diagnostics") or {}).get("gpu_memory_mb"),
                },
            }
            for e in entries
        ],
        "progress": status.get("progress"),
    }


def render_markdown(report: dict) -> str:
    lines = [
        f"# AutoResearch run {report['run_id']}",
        "",
        f"- task: `{report['task']}`  · primary metric: `{report['primary_metric']}`",
        f"- branch: `{report['branch']}`  · editable file: `{report['editable']}`",
        f"- frozen goal: {report['goal']}",
        f"- rounds: {report['budget']['rounds_completed']} / "
        f"{report['budget']['max_rounds']}  · budget per fold: "
        f"{report['budget']['budget_seconds_per_fold']}s  · "
        f"folds×repeats: {report['budget']['folds']}×{report['budget']['repeats']}",
        f"- autonomous research: {'yes' if report['autonomous_research'] else 'NO'} "
        f"(agent kind: {report['agent_kind'] or 'NOT_AVAILABLE'})",
        f"- champion: {report['champion']['run_id'] or 'NOT_AVAILABLE'} "
        f"@ `{(report['champion']['git_commit_sha'] or 'NOT_AVAILABLE')[:8]}` "
        f"· best: {_fmt(report['champion']['best_metric'])}",
        "",
        "## Outcomes",
        "",
        "| outcome | count |",
        "| --- | --- |",
    ]
    for name, count in report["outcomes"].items():
        lines.append(f"| {name} | {count} |")
    lines += [
        "",
        f"## Doctor ({'BLOCKED' if report['doctor']['blocked'] else 'not blocked'})",
        "",
    ]
    if report["doctor"]["findings"]:
        lines += ["| check | severity | message |", "| --- | --- | --- |"]
        for finding in report["doctor"]["findings"]:
            lines.append(f"| {finding['check']} | {finding['severity']} | "
                         f"{str(finding['message'])[:160]} |")
    else:
        lines.append("No diagnosis recorded for this run (`NOT_AVAILABLE`).")
    lines += [
        "",
        "## Rounds",
        "",
        "| # | hypothesis | change | outcome | value | best | next action | commit |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["rounds"]:
        lines.append(
            f"| {row['round']} | {str(row['hypothesis'])[:80]} | "
            f"{str(row['controlled_change'])[:60]} | {row['outcome']} | "
            f"{_fmt(row['primary_value'])} | {_fmt(row['best_so_far'])} | "
            f"{str(row['next_action'])[:80]} | "
            f"{(row['git_commit_sha'] or 'NOT_AVAILABLE')[:8]} |"
        )
    lines += [
        "",
        "## Saved artifacts (the chart outlives the browser)",
        "",
        f"- plotted series (data, re-drawable): `{report.get('chart_series')}`",
        f"- self-contained page (payload inlined, renders from a file): "
        f"`{report.get('page_archive')}`",
    ]
    if report.get("screenshot"):
        lines.append(f"- page screenshot: `{report['screenshot']}`")
    lines += [
        "",
        "## Evidence and limits",
        "",
        "- Every score came from the isolated trusted evaluator; no candidate "
        "reported its own metric.",
        "- Telemetry the host did not measure is printed as `NOT_AVAILABLE` "
        "(per-class breakdown, loss history, GPU memory) rather than estimated.",
        "- `INVALID_PROTOCOL`, `UNSUPPORTED_ARCHITECTURE` and `CRASH` are "
        "engineering outcomes; they are not evidence about model quality.",
    ]
    return "\n".join(lines) + "\n"


def write_report(session, out_dir=None, *, screenshot: bool = False,
                 browser: str | None = None) -> dict:
    out = Path(out_dir) if out_dir else session.root
    out.mkdir(parents=True, exist_ok=True)
    report = build_report(session)
    json_path = out / "report.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    # The chart must outlive the browser. A live page is a view, not a record, so
    # every report also writes (a) the plotted SERIES as data, which can be
    # re-drawn or diffed without this code, and (b) a self-contained page whose
    # payload is inlined, which renders identically from a file. The PNG is the
    # optional third form because it needs a browser on the host.
    charts_dir = out / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    series_path = charts_dir / f"{session.run_id}-metric.json"
    series = {
        "schema_version": "autoresearch-chart-v1",
        "run_id": session.run_id,
        "primary_metric": session.spec.get("primary_metric"),
        "points": [
            {"round": row.get("round"), "run_id": row.get("run_id"),
             "outcome": row.get("outcome"), "value": row.get("primary_value"),
             "best_so_far": row.get("best_so_far"),
             "hypothesis": row.get("hypothesis")}
            for row in report["rounds"]
        ],
    }
    series_path.write_text(
        json.dumps(series, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    from research.dashboard import render_static

    page_path = out / f"page-{session.run_id}.html"
    page_path.write_text(render_static(session, frozen=True), encoding="utf-8")
    md_path = out / f"report-{session.run_id}.md"
    paths = {"json": json_path, "markdown": md_path, "chart_series": series_path,
             "page_archive": page_path}
    report["chart_series"] = str(series_path)
    report["page_archive"] = str(page_path)
    if screenshot:
        # a delivery that claims a working page should carry its picture; the
        # failure mode is explicit (NOT_AVAILABLE when no browser exists), and it
        # never turns a report into a success it cannot back up
        from research.dashboard import capture_static

        png = out / f"report-{session.run_id}.png"
        try:
            paths["screenshot"] = capture_static(session, png, browser=browser)
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            paths["screenshot"] = f"NOT_AVAILABLE: {type(exc).__name__}: {exc}"
        report["screenshot"] = str(paths["screenshot"])
        json_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return paths