"""The ONE official AutoResearch entry point (issue #61, scope A).

    python -m research.cli <command> ...

    doctor     preflight/postflight diagnosis, read-only
    run        start an autonomous research campaign (needs a real Coding Agent)
    status     current round, stage, metrics, decisions (read-only)
    stop       ask a running campaign to stop (also ends the live training child)
    resume     continue from the manifest's completed rounds, never replaying them
    report     write a deterministic report of a run
    dashboard  serve the read-only live view on 127.0.0.1

The older `python -m runtime.cli research*` subcommands remain runnable for
regression but are deprecated: this module is the only place capabilities are
added. Nothing here re-implements training, isolation or scoring — it wires the
existing engine (`research.loop`, `research.runner`, `research.evaluator`,
`research.finalize`) to a stable interface and to a persisted run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from research.agent import (
    AgentUnavailable,
    ProposalRejected,
    build_proposer,
)
from research.doctor import BlockedProtocol
from research.loop import GitWorktree, SourceResearchLoop
from research.session import ResearchSession
from research.tasks import adapter_for

DEFAULT_PORT = 8765
DEFAULT_HOST = "127.0.0.1"


# -- shared helpers -------------------------------------------------------------


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
    """Manifest + records for the Doctor, loaded through the task adapter."""
    adapter = adapter_for(session.spec["task"])
    manifest = adapter.load_manifest(session.spec["manifest_path"])
    records = adapter.load_records(session.spec["dataset_path"])
    return adapter, manifest, records


def _frozen_hashes(session_spec: dict, adapter) -> dict:
    import hashlib

    def sha(path: str) -> str:
        try:
            return hashlib.sha256(Path(path).read_bytes()).hexdigest()
        except OSError:
            return "NOT_AVAILABLE"

    from research.frozen import TIME_BUDGET_SECONDS

    return {
        "dataset_hash": sha(session_spec["dataset_path"]),
        "data_manifest_hash": sha(session_spec["manifest_path"]),
        "metric": adapter.primary_metric,
        "evaluator": "research.evaluator.score_fold (isolated child)",
        "budget_seconds": session_spec["budget_seconds"],
        "frozen_time_budget": TIME_BUDGET_SECONDS,
        "editable": session_spec["editable"],
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
    """Create a dedicated worktree for a run, so the main checkout is untouched."""
    path = out_dir / "worktree"
    if path.exists() and (path / ".git").exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    # Ask git whether the branch exists instead of matching its error text: the
    # message is localised ("一个名为 ... 已经存在"), so a substring check both
    # misses real conflicts and would break on a different locale.
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


# -- commands --------------------------------------------------------------------


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


def _start_or_resume(args, *, resume: bool) -> int:
    out_dir = Path(args.out)
    if resume:
        session = ResearchSession.load(out_dir)
        spec = session.spec
    else:
        repo_root = Path(args.repo or ".").resolve()
        branch = args.branch or f"autoresearch/{Path(out_dir).name}"
        if args.worktree:
            worktree = Path(args.worktree)
        else:
            worktree = _ensure_worktree(repo_root, out_dir, branch,
                                        base_ref=args.base_ref)
            branch = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=worktree,
                text=True, capture_output=True,
            ).stdout.strip() or branch
        task = args.task
        adapter = adapter_for(task)
        if args.manifest and args.dataset:
            manifest_path, dataset_path = Path(args.manifest), Path(args.dataset)
        elif task == "toy":
            from research.task_toy import write_inputs

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
        spec_preview = {
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
            task=task,
            editable=spec_preview["editable"],
            worktree=spec_preview["worktree"],
            branch=spec_preview["branch"],
            folds=spec_preview["folds"],
            repeats=spec_preview["repeats"],
            max_rounds=spec_preview["max_rounds"],
            budget_seconds=spec_preview["budget_seconds"],
            primary_metric=adapter.primary_metric,
            max_wall_clock_s=spec_preview["max_wall_clock_s"],
            notes="created by research.cli run",
        )
        spec = session.spec
        # the frozen spec carries the data binding too, so doctor/status/report
        # can rebuild the same inputs without being told again
        spec.update({
            "manifest_path": str(manifest_path),
            "dataset_path": str(dataset_path),
            "seed": int(args.seed),
            "goal": args.goal or "",
            "agent_command": args.agent_command or "",
            "frozen_hashes": _frozen_hashes(spec_preview, adapter),
        })
        (session.root / "run.json").write_text(
            json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    adapter, manifest, records = _load_inputs(session)
    loop = _build_loop(session, task=session.spec["task"])
    proposer = build_proposer(args.agent_command, stub_steps=None,
                              timeout_s=args.agent_timeout)
    if args.allow_stub and not args.agent_command:
        from research.agent import DeterministicStubProposer

        worktree_path = Path(session.spec["worktree"])
        editable = session.spec["editable"]
        proposer = DeterministicStubProposer(steps=[
            _toy_width_step(worktree_path, editable, i)
            for i in range(int(args.block or args.max_rounds))
        ])
    block = args.block
    start = time.monotonic()
    try:
        summary = loop.run_autonomous(
            session, proposer, allow_stub=bool(args.allow_stub),
            doctor_context={"config": _config_for(session.spec), "manifest": manifest,
                            "records": records, "adapter": adapter},
            max_rounds=int(block) if block is not None else None,
        )
    except (AgentUnavailable, ProposalRejected) as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                         ensure_ascii=False, indent=2))
        return 1
    except BlockedProtocol as exc:
        print(json.dumps({"ok": False, "blocked": True, "error": str(exc)},
                         ensure_ascii=False, indent=2))
        return 1
    if args.dashboard:
        from research.dashboard import spawn_dashboard

        summary["dashboard"] = spawn_dashboard(session, port=args.port)
    summary.update({
        "run_dir": str(session.root),
        "wall_clock_s": round(time.monotonic() - start, 1),
    })
    print(json.dumps({"ok": True, **summary}, ensure_ascii=False, indent=2))
    return 0


def _toy_width_step(worktree: Path, editable: str, index: int):
    """A scripted engineering step for the toy task; explicitly NOT an agent.

    It reads the CURRENT editable file and builds a hunk from the lines that
    actually surround `WIDTH`, with context on both sides. A patch with no
    trailing context asserts "the file ends here", so an append-style hunk
    applies once and then fails forever — the first version of this stub made
    that mistake and every round was (correctly) refused as INVALID_PROTOCOL.

    It is still a stub, not an agent: the rule is fixed in advance, which is why
    runs driven by it are recorded as `autonomous_research=false`.
    """
    import re

    from research.agent import Hypothesis, ProposalRejected

    def run(_state: dict) -> Hypothesis:
        path = worktree / editable
        lines = path.read_text(encoding="utf-8").splitlines()
        anchor = next(
            (i for i, line in enumerate(lines) if re.match(r"^WIDTH = \d+$", line)),
            None,
        )
        if anchor is None:
            raise ProposalRejected("stub anchor 'WIDTH = <int>' not found")
        current = int(lines[anchor].split("=")[1])
        target = min(256, current + 32)
        before = lines[anchor - 1] if anchor > 0 else None
        after = lines[anchor + 1] if anchor + 1 < len(lines) else None
        if before is None or after is None:
            raise ProposalRejected("stub needs a context line on both sides of WIDTH")
        start_line = anchor  # 0-based context start == previous line
        old_side = [before, lines[anchor], after]
        new_side = [before, f"WIDTH = {target}", after]
        hunk = (
            f"@@ -{start_line},3 +{start_line},3 @@\n"
            f" {old_side[0]}\n"
            f"-{old_side[1]}\n"
            f"+{new_side[1]}\n"
            f" {old_side[2]}\n"
        )
        patch = (
            f"diff --git a/{editable} b/{editable}\n"
            f"--- a/{editable}\n"
            f"+++ b/{editable}\n"
            + hunk
        )
        return Hypothesis(
            hypothesis=f"width {current} -> {target} raises capacity",
            controlled_change="only the trunk width; data, budget and loss are fixed",
            expected_observation="macro-F1 rises if the toy task is capacity-limited",
            patch=patch,
            intent=f"stub: widen {current} -> {target}",
            next_action_if_keep=f"try width {min(256, target + 32)}",
            next_action_if_revert="stop widening and try regularisation",
        )
    return run


def cmd_step(args) -> int:
    """Apply ONE externally-proposed round, then print the new state.

    This is how a Coding Agent drives the loop one round at a time: the agent
    reads the state, writes a hypothesis file, calls `step`, and reads the state
    again. Every round is therefore proposed AFTER the previous result exists —
    which is the difference between autonomous research and replaying a plan, and
    why `batch_supplied` is False and both digests are recorded.
    """
    from research.agent import Hypothesis

    session = ResearchSession.load(args.run)
    adapter, manifest, records = _load_inputs(session)
    loop = _build_loop(session, task=session.spec["task"])
    payload = json.loads(Path(args.proposal_file).read_text(encoding="utf-8"))
    # provenance: which state the proposal claims to answer, and what was
    # submitted. `step` had been recording neither, so a reader could not tell
    # whether a proposal followed the previous round or replaced it.
    from research.agent import digest as agent_digest

    state_before = loop.research_state(session=session,
                                       agent_kind="external-coding-agent")
    try:
        hypothesis = Hypothesis.from_dict(
            payload,
            raw_reply_sha256=agent_digest(
                Path(args.proposal_file).read_text(encoding="utf-8")),
            context_sha256=agent_digest(state_before),
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
        entry = loop.run_round(
            hypothesis.patch, hypothesis.intent,
            proposer=args.proposer, hypothesis=hypothesis.to_dict(),
            agent_kind="external-coding-agent", batch_supplied=False,
            doctor_context={"config": config, "manifest": manifest,
                            "records": records, "adapter": adapter},
        )
        writer.emit("DECIDING", "finished", round_index=index,
                    message=f"{entry.get('outcome')}: {str(entry.get('reason'))[:160]}")
    if entry.get("doctor"):
        session.write_doctor(entry["doctor"])
    state = loop.research_state(session=session, agent_kind="external-coding-agent")
    print(json.dumps({"ok": True, "entry": entry, "state": state},
                     ensure_ascii=False, indent=2))
    return 0


def cmd_run(args) -> int:
    return _start_or_resume(args, resume=False)


def cmd_resume(args) -> int:
    return _start_or_resume(args, resume=True)


def cmd_status(args) -> int:
    session = ResearchSession.load(args.run)
    view = session.status(after_seq=args.after, events_limit=args.events)
    print(json.dumps(view, ensure_ascii=False, indent=2))
    return 0


def cmd_stop(args) -> int:
    session = ResearchSession.load(args.run)
    write = session.request_stop(args.reason)
    print(json.dumps({"ok": True, "run_id": session.run_id, **write},
                     ensure_ascii=False, indent=2))
    return 0


def cmd_report(args) -> int:
    from research.report import write_report

    session = ResearchSession.load(args.run)
    paths = write_report(session, out_dir=args.out)
    print(json.dumps({"ok": True, **{k: str(v) for k, v in paths.items()}},
                     ensure_ascii=False, indent=2))
    return 0


def cmd_dashboard(args) -> int:
    from research.dashboard import serve

    session = ResearchSession.load(args.run)
    return serve(session, host=args.host, port=args.port, once=args.once)


# -- argparse --------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autoresearch", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="diagnose a run's protocol (read-only)")
    p.add_argument("--run", required=True, help="run directory")
    p.add_argument("--postflight", action="store_true",
                   help="diagnose the last finished round instead of the preflight")
    p.set_defaults(func=cmd_doctor)

    def add_run_args(q, *, resume: bool) -> None:
        q.add_argument("--out", required=True, help="run directory")
        q.add_argument("--task", default="toy",
                       help="task adapter name (toy|constitution|syndrome)")
        q.add_argument("--manifest", default=None)
        q.add_argument("--dataset", default=None)
        q.add_argument("--repo", default=".", help="repository to cut a worktree from")
        q.add_argument("--worktree", default=None,
                       help="use this existing worktree instead of creating one")
        q.add_argument("--branch", default=None)
        q.add_argument("--base-ref", default=None)
        q.add_argument("--editable", default=None)
        q.add_argument("--folds", type=int, default=3)
        q.add_argument("--repeats", type=int, default=1)
        q.add_argument("--max-rounds", type=int, default=3)
        q.add_argument("--budget", type=float, default=2.0,
                       help="fixed wall clock per fold, in seconds")
        q.add_argument("--max-wall-clock-s", type=float, default=6 * 3600)
        q.add_argument("--seed", type=int, default=20261105)
        q.add_argument("--goal", default=None, help="frozen research objective")
        q.add_argument("--agent-command", default=None,
                       help="command for a real Coding Agent (reads the prompt on stdin, "
                            "prints the proposal JSON on stdout)")
        q.add_argument("--agent-timeout", type=float, default=600.0)
        q.add_argument("--allow-stub", action="store_true",
                       help="run the engineering path with a deterministic stub; the run "
                            "is recorded as autonomous_research=false")
        q.add_argument("--block", type=int, default=None,
                       help="rounds to run in this invocation (default: all remaining)")
        q.add_argument("--dashboard", action="store_true",
                       help="also start the read-only dashboard as its own process")
        q.add_argument("--port", type=int, default=DEFAULT_PORT)

    p = sub.add_parser("step", help="apply one externally-proposed round (agent-driven)")
    p.add_argument("--run", required=True)
    p.add_argument("--proposal-file", required=True)
    p.add_argument("--proposer", default="external-coding-agent")
    p.set_defaults(func=cmd_step)

    p = sub.add_parser("run", help="start an autonomous campaign")
    add_run_args(p, resume=False)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("resume", help="continue a campaign from its manifest")
    add_run_args(p, resume=True)
    p.set_defaults(func=cmd_resume)

    p = sub.add_parser("status", help="current state of a run (read-only)")
    p.add_argument("--run", required=True)
    p.add_argument("--after", type=int, default=0, help="events with seq > after")
    p.add_argument("--events", type=int, default=100)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("stop", help="ask a run to stop")
    p.add_argument("--run", required=True)
    p.add_argument("--reason", default="operator request")
    p.set_defaults(func=cmd_stop)

    p = sub.add_parser("report", help="write a report for a run")
    p.add_argument("--run", required=True)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("dashboard", help="serve the read-only live view")
    p.add_argument("--run", required=True)
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--once", action="store_true",
                   help="render the page to stdout and exit (used by tests)")
    p.set_defaults(func=cmd_dashboard)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not callable(getattr(args, "func", None)):
        parser.error(f"subcommand {args.command!r} has no handler wired")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())