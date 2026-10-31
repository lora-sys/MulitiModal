"""#45 AutoResearch protocol: train_candidate / trusted_score.

These two functions are the stable surface the AutoResearch harness (#45)
calls instead of touching Trainer internals. Deterministic under a fixed
seed. Two data modes:

- synthetic (default): SMOKE_ONLY fixtures, engineering smoke only.
- real (#50): batches from a frozen real-data manifest (subject-disjoint
  split, train-fold scaler, audited labels); checkpoints are saved with
  smoke_only=False and the manifest hash in their provenance.
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


def _train_real(config: dict) -> dict:
    from .real_data import (
        RealBatchProvider,
        build_real_model,
        manifest_from_path,
    )

    data_cfg = dict(config["data"])
    manifest = manifest_from_path(data_cfg["manifest_path"])
    if data_cfg.get("task_type") and data_cfg["task_type"] != manifest["task_type"]:
        raise ValueError(
            "config task_type "
            f"{data_cfg['task_type']!r} != manifest {manifest['task_type']!r}"
        )
    model_cfg = dict(config.get("model", {}))
    # the dev partition is wired in as the eval provider below for reporting
    # only; it must never steer fitting (early stop / best-epoch restore
    # would double-use dev while the loop also selects on dev, review R-05)
    trainer_kwargs = dict(config.get("trainer", {}))
    for forbidden in ("early_stop_patience", "restore_best"):
        if trainer_kwargs.get(forbidden) is not None:
            raise ValueError(
                f"real-data training forbids trainer.{forbidden}: dev would "
                "influence fitting while selection also uses dev"
            )
    trainer_cfg = TrainerConfig(**trainer_kwargs)
    records = _load_records(data_cfg["dataset_path"])
    torch.manual_seed(trainer_cfg.seed)
    model = build_real_model(manifest, model_cfg)
    trainer = Trainer(model, trainer_cfg)
    train_bs = config.get("train_provider", {}).get("batch_size", 32)
    train_provider = RealBatchProvider(manifest, records, "train", batch_size=train_bs)
    eval_provider = RealBatchProvider(manifest, records, "dev", batch_size=64)
    report: FitReport = trainer.fit(train_provider, eval_provider)
    artifact_dir = trainer.save_checkpoint(
        Path(config.get("out_dir", "runs/autoresearch/real-tcm")),
        smoke_only=False,
        extra_state={
            "manifest_sha256": manifest["manifest_sha256"],
            "dataset_sha256": manifest["dataset_sha256"],
            "label_version": manifest["label_version"],
            "train_batch_size": train_bs,
        },
    )
    return {
        "artifact_dir": str(artifact_dir),
        "task": "tcm",
        "model_type": model.model_type,
        "epochs_run": report.epochs_run,
        "best_epoch": report.best_epoch,
        "best_eval_loss": report.best_eval_loss,
        "smoke_only": False,
        "data_mode": "real",
        "manifest_sha256": manifest["manifest_sha256"],
        "dataset_sha256": manifest["dataset_sha256"],
        "label_version": manifest["label_version"],
    }


def _load_records(dataset_path: str) -> list:
    from data.mendeley import load_mendeley_scores

    return load_mendeley_scores(dataset_path)


def train_candidate(config: dict) -> dict:
    """Train one candidate from a plain config dict; returns a JSON-able report.

    Synthetic mode keys: task ('tcm'|'physio'), model (ctor kwargs), trainer
    (TrainerConfig kwargs), train/eval provider settings, out_dir.
    Real mode (#50): data_mode='real' + data {manifest_path, dataset_path}.
    """
    task = config.get("task", "tcm")
    if task not in TASK_BUILDERS:
        raise ValueError(f"unknown task {task!r}")
    if config.get("data_mode") == "real":
        return _train_real(config)

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

    model, state, metadata = load_training_checkpoint(artifact_dir)
    if not state.get("smoke_only", False):
        raise ValueError(
            "trusted_score evaluates synthetic fixtures only; a real-data "
            "checkpoint must go through the worker's real-data scoring path"
        )
    task = "tcm" if metadata["model_type"].startswith("tcm") else "physio"
    provider = SyntheticBatchProvider(
        task, n_batches=3, batch_size=8, seed=seed,
        subject_prefix=f"trusted-eval-{task}",
    )
    metrics = Evaluator().evaluate(model, provider)
    return {"artifact_dir": str(artifact_dir), "model_type": metadata["model_type"],
            "smoke_only": True, **metrics}
