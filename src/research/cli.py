"""AutoResearch CLI — Karpathy shape, driven by the Skill.

    prepare   freeze one run (worktree + spec + inputs) and diagnose the protocol
    state     print the full state a proposer must react to
    step      apply ONE hypothesis, train, score, KEEP/REVERT, print new state
    status    coordinator summary (read-only)
    stop      request a stop
    doctor    diagnose a run's protocol
    dashboard serve the read-only live view

The Skill (skills/autoresearch/) is the proposer. There is no in-process agent,
no subprocess coding agent and no autonomous loop: each round is proposed after
reading the previous result, exactly like Karpathy's program.md loop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from research.agent import Hypothesis, ProposalRejected, digest
from research.loop import GitWorktree, SourceResearchLoop
from research.session import ResearchSession
from research.tasks import adapter_for

DEFAULT_PORT = 8765
DEFAULT_HOST = "127.0.0.1"

#: SHA-256 of the authoritative #56/#57 protocol-v2 short selection_records.json.
#: Source: MulitiModal-i56/runs/sd-v2/short/build-summary.json. Changing this
#: pin is a new data milestone (AGENTS.md rule 11).
SELECTION_DATASET_SHA256 = (
    "e63682908e6bed7488ada3cd68cf50b4a1817a4bfbfa28c0a00b74fbc0af4d55"
)


def _config_for(session_spec: dict, config: dict | None = None) -> dict:
    """The config the engine's subprocesses and the Doctor both read."""
    return {
        "task": session_spec["task"],
        "worktree_path": session_spec["worktree"],
        "editable": session_spec["editable"],
        "data": {
            "manifest_path": session_spec["manifest_path"],
            "dataset_path": session_spec["dataset_path"],
        },
        "time_budget_seconds": session_spec["budget_seconds"],
        "primary_metric": session_spec["primary_metric"],
        "folds": session_spec["folds"],
        "repeats": session_spec["repeats"],
        "trainer": {"seed": int(session_spec.get("seed", 0))},
        "train_provider": {"batch_size": 32},
        "frozen_hashes": session_spec.get("frozen_hashes") or {},
        **(config or {}),
    }


def _load_inputs(session: ResearchSession):
    adapter = adapter_for(session.spec["task"])
    manifest = adapter.load_manifest(session.spec["manifest_path"])
    records = adapter.load_records(session.spec["dataset_path"])
    return adapter, manifest, records


def _frozen_hashes(spec: dict, adapter) -> dict:
    def sha(path: str) -> str:
        try:
            return hashlib.sha256(Path(path).read_bytes()).hexdigest()
        except OSError:
            return "NOT_AVAILABLE"

    from research.frozen import TIME_BUDGET_SECONDS

    return {
        "dataset_hash": sha(spec["dataset_path"]),
        "data_manifest_hash": sha(spec["manifest_path"]),
        "metric": adapter.primary_metric,
        "evaluator": "research.evaluator.score_fold (isolated child)",
        "budget_seconds": spec["budget_seconds"],
        "frozen_time_budget": TIME_BUDGET_SECONDS,
        "editable": spec["editable"],
    }


def _build_loop(session: ResearchSession, *, task: str) -> SourceResearchLoop:
    spec = session.spec
    worktree = GitWorktree(Path(spec["worktree"]), spec["branch"])
    worktree.ensure_branch()
    return SourceResearchLoop(
        worktree=worktree, out_dir=session.root,
        manifest_path=spec["manifest_path"], dataset_path=spec["dataset_path"],
        editable=spec["editable"], folds=int(spec["folds"]),
        repeats=int(spec["repeats"]), task=task,
        budget_s=float(spec["budget_seconds"]),
        seed=int(spec.get("seed", 0)),
        max_wall_clock_s=float(spec.get("max_wall_clock_s") or 6 * 3600),
        frozen_goal=str(spec.get("goal") or ""),
    )


def _ensure_worktree(repo_root: Path, out_dir: Path, branch: str,
                     base_ref: str | None = None) -> Path:
    """Create a dedicated worktree so the main checkout is untouched."""
    path = out_dir / "worktree"
    if path.exists() and (path / ".git").exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", branch],
        cwd=repo_root, text=True, capture_output=True,
    ).returncode == 0
    command = (["git", "worktree", "add", str(path), branch] if exists
               else ["git", "worktree", "add", "-b", branch, str(path)])
    if base_ref and not exists:
        command.append(base_ref)
    result = subprocess.run(command, cwd=repo_root, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"could not create the run worktree at {path}: "
            f"{(result.stderr or result.stdout).strip()[:300]}"
        )
    return path


def cmd_prepare(args) -> int:
    """Freeze one run and diagnose the protocol. No training, no proposer."""
    out_dir = Path(args.out)
    if (out_dir / "run.json").exists():
        print(json.dumps({
            "ok": False,
            "error": f"run already prepared at {out_dir}; use state/step",
        }, ensure_ascii=False, indent=2))
        return 1
    repo_root = Path(args.repo or ".").resolve()
    branch = args.branch or f"autoresearch/{out_dir.name}"
    worktree = (Path(args.worktree) if args.worktree else
                _ensure_worktree(repo_root, out_dir, branch,
                                 base_ref=args.base_ref))
    branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=worktree,
        text=True, capture_output=True,
    ).stdout.strip() or branch
    task = args.task
    adapter = adapter_for(task)
    if args.manifest and args.dataset:
        # The runner runs in a git worktree: relative paths resolve elsewhere.
        # Pin the one authoritative protocol-v2 pair so a stale intermediate
        # builder output cannot produce numbers incomparable with #56/#57.
        manifest_path = Path(args.manifest).resolve(strict=True)
        dataset_path = Path(args.dataset).resolve(strict=True)
        dataset_sha = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
        if dataset_sha != SELECTION_DATASET_SHA256:
            print(json.dumps({
                "ok": False,
                "error": f"selection dataset sha256 {dataset_sha} does not "
                         f"match frozen protocol-v2 {SELECTION_DATASET_SHA256}",
            }, ensure_ascii=False, indent=2))
            return 2
        data_provenance = {"selection_dataset_sha256": dataset_sha}
    elif task == "toy":
        from research.task_toy import write_inputs

        data_provenance = None
        manifest_path, dataset_path = write_inputs(
            out_dir / "inputs", seed=args.seed
        )
    else:
        print(json.dumps({
            "ok": False,
            "error": "a real task needs --manifest and --dataset; only the "
                     "toy example task can materialize its own inputs",
        }))
        return 2
    spec = {
        "task": task, "editable": args.editable or adapter.default_editable,
        "worktree": str(worktree), "branch": branch,
        "folds": args.folds, "repeats": args.repeats,
        "max_rounds": args.max_rounds,
        "budget_seconds": args.budget,
        "primary_metric": adapter.primary_metric,
        "max_wall_clock_s": args.max_wall_clock_s,
        "manifest_path": str(manifest_path), "dataset_path": str(dataset_path),
        "seed": args.seed, "goal": args.goal or "",
    }
    session = ResearchSession.create(
        out_dir,
        task=task, editable=spec["editable"], worktree=spec["worktree"],
        branch=spec["branch"], folds=spec["folds"], repeats=spec["repeats"],
        max_rounds=spec["max_rounds"], budget_seconds=spec["budget_seconds"],
        primary_metric=adapter.primary_metric,
        max_wall_clock_s=spec["max_wall_clock_s"],
        notes="prepared by the Skill",
    )
    frozen = session.spec
    frozen.update({
        "manifest_path": str(manifest_path), "dataset_path": str(dataset_path),
        "seed": int(args.seed), "goal": args.goal or "",
        "frozen_hashes": _frozen_hashes(spec, adapter),
        "data_provenance": data_provenance,
    })
    (session.root / "run.json").write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    adapter, manifest, records = _load_inputs(session)
    from research import doctor as doctor_mod

    report = doctor_mod.preflight(
        _config_for(session.spec), manifest, records, adapter,
        run_id=session.run_id,
    )
    session.write_doctor(report.to_dict())
    print(json.dumps({
        "ok": True, "prepared": True, "blocked": report.blocked,
        "run_dir": str(session.root), "run_id": session.run_id,
        "doctor": report.to_dict(),
    }, ensure_ascii=False, indent=2))
    return 1 if report.blocked else 0


def cmd_state(args) -> int:
    """The FULL proposer state: what the next hypothesis must react to."""
    session = ResearchSession.load(args.run)
    loop = _build_loop(session, task=session.spec["task"])
    state = loop.research_state(session=session, agent_kind=args.agent_kind)
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0


def cmd_step(args) -> int:
    """Apply ONE externally-proposed round, then print the new state."""
    session = ResearchSession.load(args.run)
    if session.stop_requested():
        print(json.dumps({"ok": False, "error": "campaign stopped"}))
        return 1
    if session.rounds_remaining() <= 0:
        print(json.dumps({"ok": False, "error": "round budget exhausted"}))
        return 1
    if not session.within_wall_clock(time.monotonic()):
        print(json.dumps({"ok": False, "error": "wall clock budget exhausted"}))
        return 1
    adapter, manifest, records = _load_inputs(session)
    loop = _build_loop(session, task=session.spec["task"])
    payload = json.loads(Path(args.proposal_file).read_text(encoding="utf-8"))
    if str(payload.get("intent", "")).strip().upper() == "STOP":
        print(json.dumps({"ok": True, "stopped": True,
                          "reason": str(payload.get("hypothesis") or "proposer STOP")}))
        return 0
    state_before = loop.research_state(session=session,
                                       agent_kind="external-coding-agent")
    try:
        hypothesis = Hypothesis.from_dict(
            payload,
            raw_reply_sha256=digest(
                Path(args.proposal_file).read_text(encoding="utf-8")),
            context_sha256=digest(state_before),
        )
    except ProposalRejected as exc:
        print(json.dumps({"ok": False, "error": f"invalid proposal: {exc}"},
                         ensure_ascii=False, indent=2))
        return 1
    config = _config_for(session.spec)
    with session.writer() as writer:
        index = session.next_round_index()
        writer.emit("PROPOSING", "started", round_index=index,
                    message=f"external proposal from {args.proposer}")
        event_sink = {
            "spawn": (lambda pid, pgid: session.mark_active_child(pid, pgid)),
        }
        entry = loop.run_round(
            hypothesis.patch, hypothesis.intent,
            proposer=args.proposer, hypothesis=hypothesis.to_dict(),
            agent_kind="external-coding-agent", batch_supplied=False,
            doctor_context={"config": config, "manifest": manifest,
                            "records": records, "adapter": adapter},
            event_sink=event_sink,
        )
        writer.emit("DECIDING", "finished", round_index=index,
                    message=f"{entry.get('outcome')}: {str(entry.get('reason'))[:160]}")
    if entry.get("doctor"):
        session.write_doctor(entry["doctor"])
    state = loop.research_state(session=session, agent_kind="external-coding-agent")
    print(json.dumps({"ok": True, "entry": entry, "state": state},
                     ensure_ascii=False, indent=2))
    return 0


def cmd_doctor(args) -> int:
    from research import doctor as doctor_mod

    session = ResearchSession.load(args.run)
    adapter, manifest, records = _load_inputs(session)
    if args.postflight:
        entry = session.entries()[-1] if session.entries() else {}
        report = doctor_mod.postflight(
            {"decision": entry.get("decision"), "status": entry.get("status"),
             "reason": entry.get("reason"),
             "primary_metric": session.spec["primary_metric"],
             "primary_value": (entry.get("metrics") or {}).get("primary_value"),
             "primary_sem": (entry.get("metrics") or {}).get("primary_sem"),
             "previous_best": (entry.get("observed_evidence") or {}).get("previous_best"),
             "per_class": entry.get("per_class"),
             "diagnostics": entry.get("diagnostics")},
            _config_for(session.spec), adapter, run_id=session.run_id,
        )
    else:
        report = doctor_mod.preflight(
            _config_for(session.spec), manifest, records, adapter,
            run_id=session.run_id,
        )
    session.write_doctor(report.to_dict())
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 1 if report.blocked else 0


def cmd_status(args) -> int:
    session = ResearchSession.load(args.run)
    print(json.dumps(session.status(after_seq=args.after,
                                    events_limit=args.events),
                     ensure_ascii=False, indent=2))
    return 0


def cmd_stop(args) -> int:
    session = ResearchSession.load(args.run)
    print(json.dumps({"ok": True, "run_id": session.run_id,
                      **session.request_stop(args.reason)},
                     ensure_ascii=False, indent=2))
    return 0


def cmd_dashboard(args) -> int:
    from research.dashboard import render_static, serve

    if args.runs_root:
        from research.dashboard import serve_runs

        return serve_runs(args.runs_root, host=args.host, port=args.port)
    session = ResearchSession.load(args.run)
    if args.static:
        html = render_static(session)
        if args.out:
            Path(args.out).write_text(html, encoding="utf-8")
            print(json.dumps({"ok": True, "static_html": str(args.out),
                              "bytes": len(html)}))
        else:
            sys.stdout.write(html)
        return 0
    return serve(session, host=args.host, port=args.port, once=args.once)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autoresearch", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="freeze a run + Doctor preflight")
    p.add_argument("--out", required=True)
    p.add_argument("--task", default="toy")
    p.add_argument("--manifest", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--repo", default=".")
    p.add_argument("--worktree", default=None)
    p.add_argument("--branch", default=None)
    p.add_argument("--base-ref", default=None)
    p.add_argument("--editable", default=None)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--max-rounds", type=int, default=3)
    p.add_argument("--budget", type=float, default=2.0)
    p.add_argument("--max-wall-clock-s", type=float, default=6 * 3600)
    p.add_argument("--seed", type=int, default=20261105)
    p.add_argument("--goal", default=None)
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("state", help="full proposer state for the next round")
    p.add_argument("--run", required=True)
    p.add_argument("--agent-kind", default="external-coding-agent")
    p.set_defaults(func=cmd_state)

    p = sub.add_parser("step", help="apply one externally-proposed round")
    p.add_argument("--run", required=True)
    p.add_argument("--proposal-file", required=True)
    p.add_argument("--proposer", default="external-coding-agent")
    p.set_defaults(func=cmd_step)

    p = sub.add_parser("doctor", help="diagnose a run's protocol (read-only)")
    p.add_argument("--run", required=True)
    p.add_argument("--postflight", action="store_true")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("status", help="coordinator summary (read-only)")
    p.add_argument("--run", required=True)
    p.add_argument("--after", type=int, default=0)
    p.add_argument("--events", type=int, default=100)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("stop", help="ask a run to stop")
    p.add_argument("--run", required=True)
    p.add_argument("--reason", default="operator request")
    p.set_defaults(func=cmd_stop)

    p = sub.add_parser("dashboard", help="serve the read-only live view")
    p.add_argument("--run", default=None)
    p.add_argument("--runs-root", default=None)
    p.add_argument("--no-serve", action="store_true",
                   help="accepted for compatibility; the dashboard is loopback-only")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--once", action="store_true")
    p.add_argument("--static", action="store_true")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_dashboard)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not callable(getattr(args, "func", None)):
        parser.error(f"subcommand {args.command!r} has no handler wired")
    if args.command == "dashboard" and not (args.run or args.runs_root):
        parser.error("dashboard needs --run or --runs-root")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
