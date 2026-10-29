"""Engineering CLI (#49): validate-architecture, smoke-tcm, smoke-physio,
checkpoint-reload, list-models. Synthetic fixtures only — never touches real
data, databases, or paid APIs (#49 out of scope). All outputs SMOKE_ONLY.

Usage: PYTHONPATH=src python -m runtime.cli <command> ...
"""

from __future__ import annotations

import argparse
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
