"""Reserved #45 AutoResearch protocol: train_candidate / trusted_score.

These two functions are the stable surface the AutoResearch harness (#45)
calls instead of touching Trainer internals. They are deterministic under a
fixed seed and operate on synthetic fixtures only in this phase (SMOKE_ONLY);
the trusted evaluator itself must later run in an agent-unwritable
environment (#45 hard rule 9).
"""

from __future__ import annotations

from pathlib import Path

import torch

from core.models import default_physio_model, default_tcm_model

from .batches import SyntheticBatchProvider
from .evaluation import Evaluator
from .trainer import FitReport, Trainer, TrainerConfig

TASK_BUILDERS = {
    "tcm": default_tcm_model,
    "physio": default_physio_model,
}


def train_candidate(config: dict) -> dict:
    """Train one candidate from a plain config dict; returns a JSON-able report.

    Recognized keys: task ('tcm'|'physio'), model (ctor kwargs), trainer
    (TrainerConfig kwargs), train/eval provider settings (n_batches,
    batch_size, seed), out_dir.
    """
    task = config.get("task", "tcm")
    if task not in TASK_BUILDERS:
        raise ValueError(f"unknown task {task!r}")
    model_cfg = dict(config.get("model", {}))
    trainer_cfg = TrainerConfig(**config.get("trainer", {}))
    # Seed before model construction: the worker runs in a fresh process, so
    # identical configs must produce identical inits (controlled comparison).
    torch.manual_seed(trainer_cfg.seed)
    train_cfg = dict(config.get("train_provider", {}))
    eval_cfg = dict(config.get("eval_provider", {}))
    out_dir = Path(config.get("out_dir", f"runs/autoresearch/{task}"))

    model, _ = TASK_BUILDERS[task](**model_cfg)
    trainer = Trainer(model, trainer_cfg)
    train_provider = SyntheticBatchProvider(task, **train_cfg)
    eval_provider = (
        SyntheticBatchProvider(task, **eval_cfg) if eval_cfg else None
    )
    report: FitReport = trainer.fit(train_provider, eval_provider)
    artifact_dir = trainer.save_checkpoint(out_dir)
    return {
        "artifact_dir": str(artifact_dir),
        "task": task,
        "model_type": model.model_type,
        "epochs_run": report.epochs_run,
        "best_epoch": report.best_epoch,
        "best_eval_loss": report.best_eval_loss,
        "smoke_only": True,
    }


def trusted_score(artifact_dir: str | Path, seed: int = 1234) -> dict:
    """Frozen-seed independent scoring of a saved candidate artifact.

    Uses its own synthetic eval provider with a distinct subject namespace so
    it can never share subjects with a training provider of the same seed.
    """
    from .checkpoint import load_training_checkpoint

    model, _, metadata = load_training_checkpoint(artifact_dir)
    task = "tcm" if metadata["model_type"].startswith("tcm") else "physio"
    provider = SyntheticBatchProvider(
        task, n_batches=3, batch_size=8, seed=seed,
        subject_prefix=f"trusted-eval-{task}",
    )
    metrics = Evaluator().evaluate(model, provider)
    return {"artifact_dir": str(artifact_dir), "model_type": metadata["model_type"],
            "smoke_only": True, **metrics}
