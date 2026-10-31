"""Engineering CLI (#49): validate-architecture, smoke-tcm, smoke-physio,
checkpoint-reload, list-models. Synthetic fixtures only — never touches real
data, databases, or paid APIs (#49 out of scope). All outputs SMOKE_ONLY.

Usage: PYTHONPATH=src python -m runtime.cli <command> ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

from core.registry import MODEL_BUILDERS, batch_predict, resolve_device

from .batches import SyntheticBatchProvider
from .checkpoint import checkpoint_summary, load_training_checkpoint
from .protocols import TASK_BUILDERS
from .trainer import Trainer, TrainerConfig


def cmd_list_models(_args: argparse.Namespace) -> int:
    print(json.dumps({"registered_model_types": sorted(MODEL_BUILDERS)}))
    return 0


def cmd_validate_architecture(args: argparse.Namespace) -> int:
    task = args.task
    model, _ = TASK_BUILDERS[task]()
    provider = SyntheticBatchProvider(task, n_batches=1, batch_size=4, seed=0)
    batch = next(iter(provider)).batch.to(resolve_device(args.device))
    out = model(batch)
    loss, parts = model.compute_loss(batch)
    loss.backward()
    report = {
        "ok": True,
        "task": task,
        "model_type": model.model_type,
        "predictions": {k: list(v.shape) for k, v in out.predictions.items()},
        "auxiliary": {k: list(v.shape) for k, v in out.auxiliary.items()},
        "loss_parts": sorted(parts),
        "smoke_only": True,
    }
    print(json.dumps(report))
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    task = args.task
    model, config = TASK_BUILDERS[task]()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out) / f"{task}-{timestamp}"
    trainer_cfg = TrainerConfig(
        seed=args.seed,
        device=args.device,
        epochs=args.epochs,
        early_stop_patience=args.patience,
    )
    train_provider = SyntheticBatchProvider(
        task, n_batches=args.batches, batch_size=args.batch_size, seed=args.seed
    )
    eval_provider = SyntheticBatchProvider(
        task, n_batches=2, batch_size=args.batch_size, seed=args.seed + 1,
        subject_prefix=f"smoke-eval-{task}",
    )
    trainer = Trainer(model, trainer_cfg)
    report = trainer.fit(train_provider, eval_provider)
    trainer.save_checkpoint(out_dir)
    summary = {
        "command": f"smoke-{task}",
        "out_dir": str(out_dir),
        "model_config": config,
        "fit": report.to_jsonable(),
        "checkpoint": checkpoint_summary(out_dir),
        "smoke_only": True,
    }
    print(json.dumps(summary, indent=2))
    return 0


def cmd_checkpoint_reload(args: argparse.Namespace) -> int:
    model, state, metadata = load_training_checkpoint(args.dir, device=args.device)
    summary = checkpoint_summary(args.dir)
    provider = SyntheticBatchProvider(
        "tcm" if metadata["model_type"].startswith("tcm") else "physio",
        n_batches=1, batch_size=4, seed=7,
    )
    batch = next(iter(provider)).batch
    out = batch_predict(model, batch, device=args.device)
    finite = all(bool(torch.isfinite(v).all()) for v in out.predictions.values())
    print(json.dumps({
        "ok": finite,
        "epochs_completed": state.get("epoch"),
        "predictions": {k: list(v.shape) for k, v in out.predictions.items()},
        **summary,
    }))
    return 0


def cmd_research(args: argparse.Namespace) -> int:
    from runtime.harness import (
        REAL_TCM_PLAN,
        TCM_ONLY_MESSAGE,
        AutoResearchHarness,
        default_real_tcm_runspec,
        default_tcm_runspec,
    )

    if args.profile != "tcm":
        print(json.dumps({"ok": False, "error": TCM_ONLY_MESSAGE}))
        return 2
    if args.data_manifest:
        if not args.dataset:
            print(json.dumps({"ok": False, "error": "--dataset is required with --data-manifest"}))
            return 2
        runspec = default_real_tcm_runspec(
            args.data_manifest, args.dataset, rounds=args.rounds, timeout_s=args.timeout
        )
    else:
        runspec = default_tcm_runspec(rounds=args.rounds, timeout_s=args.timeout)
    runspec["seed"] = args.seed
    provider = None
    if args.data_manifest:
        from runtime.agent import DeterministicStubProvider

        provider = DeterministicStubProvider(plan=REAL_TCM_PLAN)
    harness = AutoResearchHarness(runspec, provider=provider, out_dir=args.out)
    print(json.dumps(harness.run(), indent=2))
    return 0


def cmd_research_step(args: argparse.Namespace) -> int:
    """Single-round AutoResearch driver for a REAL external agent: prints the
    research state (--print-context) or applies one proposal (--proposal-file)
    through the full harness (scope validation, frozen seed, subprocess
    isolation, append-only manifest)."""
    from runtime.agent import FixedProposalProvider, Proposal
    from runtime.harness import (
        TCM_ONLY_MESSAGE,
        AutoResearchHarness,
        context_digest,
        default_real_tcm_runspec,
        default_tcm_runspec,
    )

    if args.profile != "tcm":
        print(json.dumps({"ok": False, "error": TCM_ONLY_MESSAGE}))
        return 2
    if args.data_manifest:
        if not args.dataset:
            print(json.dumps({"ok": False, "error": "--dataset is required with --data-manifest"}))
            return 2
        runspec = default_real_tcm_runspec(
            args.data_manifest, args.dataset, rounds=args.rounds, timeout_s=args.timeout
        )
    else:
        runspec = default_tcm_runspec(rounds=args.rounds, timeout_s=args.timeout)
    runspec["seed"] = args.seed
    if not args.print_context and not args.proposal_file:
        print(json.dumps({"ok": False,
                          "error": "one of --print-context / --proposal-file is required"}))
        return 2
    if args.print_context and args.proposal_file:
        print(json.dumps({"ok": False,
                          "error": "--print-context and --proposal-file are exclusive"}))
        return 2
    harness = AutoResearchHarness(
        runspec, provider=FixedProposalProvider(None), out_dir=args.out
    )
    if args.print_context:
        context = harness.research_context()
        rendered = json.dumps(context, indent=2, ensure_ascii=False)
        if args.context_out:
            Path(args.context_out).write_text(rendered)
        print(rendered)
        return 0
    data = json.loads(Path(args.proposal_file).read_text())
    if not isinstance(data, dict) or "intent" not in data:
        print(json.dumps({"ok": False, "error": "proposal needs an 'intent'"}))
        return 2
    proposal = Proposal(
        intent=str(data["intent"]), config_patch=dict(data.get("config_patch", {}))
    )
    # agent provenance (#45): WHO proposed, WHEN-stated context it saw, and
    # the exact bytes it returned, so the round is auditable out-of-band
    provenance: dict = {}
    if args.proposer:
        provenance["proposer"] = args.proposer
        provenance["proposal_sha256"] = hashlib.sha256(
            Path(args.proposal_file).read_bytes()
        ).hexdigest()
        if args.context_file:
            seen = json.loads(Path(args.context_file).read_text())
            provenance["context_sha256"] = context_digest(seen)
        if args.proposal_raw:
            raw = Path(args.proposal_raw).read_bytes()
            provenance["proposal_raw_sha256"] = hashlib.sha256(raw).hexdigest()
    elif args.context_file or args.proposal_raw:
        print(json.dumps({"ok": False,
                          "error": "--context-file/--proposal-raw require --proposer"}))
        return 2
    try:
        entry = harness.step(proposal, provenance=provenance or None)
    except RuntimeError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1
    print(json.dumps(entry, indent=2, ensure_ascii=False))
    return 0


def cmd_research_status(args: argparse.Namespace) -> int:
    from runtime.manifest import MANIFEST_FILE, RunManifestStore

    store = RunManifestStore(Path(args.out) / MANIFEST_FILE)
    entries = store.load()
    champion = store.champion()
    print(json.dumps({
        "out_dir": str(args.out),
        "rounds_completed": len(entries),
        "runs": [
            {k: e.get(k) for k in ("run_id", "decision", "status", "reason", "best_so_far")}
            for e in entries
        ],
        "champion": {
            "run_id": champion.get("run_id"),
            "artifact_dir": champion.get("artifact_dir"),
            "best_so_far": champion.get("best_so_far"),
        } if champion else None,
    }, indent=2))
    return 0


def cmd_research_report(args: argparse.Namespace) -> int:
    from runtime.manifest import MANIFEST_FILE, RunManifestStore

    store = RunManifestStore(Path(args.out) / MANIFEST_FILE)
    entries = store.load()
    champion = store.champion()
    print(json.dumps({
        "rounds_completed": len(entries),
        "keep": [e["run_id"] for e in entries if e.get("decision") == "KEEP"],
        "revert": [e["run_id"] for e in entries if e.get("decision") == "REVERT"],
        "per_round": [
            {
                "run_id": e["run_id"],
                "intent": e.get("intent"),
                "decision": e.get("decision"),
                "primary_value": (e.get("metrics") or {}).get("primary_value"),
                "smoke_only": e.get("smoke_only"),
                "artifact_sha256": e.get("artifact_sha256"),
            }
            for e in entries
        ],
        "champion": {
            "run_id": champion.get("run_id"),
            "artifact_dir": champion.get("artifact_dir"),
            "best_so_far": champion.get("best_so_far"),
            "artifact_sha256": champion.get("artifact_sha256"),
        } if champion else None,
    }, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="runtime.cli", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-models").set_defaults(func=cmd_list_models)

    p = sub.add_parser("validate-architecture")
    p.add_argument("--task", choices=sorted(TASK_BUILDERS), required=True)
    p.add_argument("--device", default="cpu")
    p.set_defaults(func=cmd_validate_architecture)

    p = sub.add_parser("smoke")
    p.add_argument("--task", choices=sorted(TASK_BUILDERS), required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batches", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--out", default="runs/smoke")
    p.set_defaults(func=cmd_smoke)

    p = sub.add_parser("checkpoint-reload")
    p.add_argument("--dir", required=True)
    p.add_argument("--device", default="cpu")
    p.set_defaults(func=cmd_checkpoint_reload)

    p = sub.add_parser("research")
    p.add_argument("--profile", choices=["tcm", "physio"], required=True)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--out", default="runs/autoresearch")
    p.add_argument("--data-manifest", default=None,
                   help="frozen real-data manifest (#50); enables real training mode")
    p.add_argument("--dataset", default=None,
                   help="audited dataset file path (with --data-manifest)")
    p.set_defaults(func=cmd_research)

    p = sub.add_parser("research-step",
                       help="single-round driver for a real external agent")
    p.add_argument("--profile", choices=["tcm", "physio"], required=True)
    p.add_argument("--rounds", type=int, default=6,
                   help="total candidate budget for the experiment")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--out", default="runs/autoresearch")
    p.add_argument("--data-manifest", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--print-context", action="store_true",
                   help="print the research state JSON and exit")
    p.add_argument("--context-out", default=None,
                   help="also write the printed research state to this file "
                        "(hand it to the agent; pass it back with --context-file)")
    p.add_argument("--proposal-file", default=None,
                   help="JSON file with {'intent': ..., 'config_patch': {...}}")
    p.add_argument("--proposer", default=None,
                   help="agent identity for the round, e.g. "
                        "'antigravity:gemini-3.8-flash-high' (recorded in the "
                        "manifest; required for verifiable agent provenance)")
    p.add_argument("--context-file", default=None,
                   help="the context file the agent was shown; its digest is "
                        "recorded and must match the current research state")
    p.add_argument("--proposal-raw", default=None,
                   help="raw agent reply file; its digest is recorded")
    p.set_defaults(func=cmd_research_step)

    p = sub.add_parser("research-status")
    p.add_argument("--out", default="runs/autoresearch")
    p.set_defaults(func=cmd_research_status)

    p = sub.add_parser("research-report")
    p.add_argument("--out", default="runs/autoresearch")
    p.set_defaults(func=cmd_research_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
