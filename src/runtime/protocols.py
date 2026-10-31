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


def cross_val_candidate(
    config: dict, folds: int = 4, split_seed: int = 0, repeats: int = 1
) -> dict:
    """Subject-level repeated k-fold score over the SELECTION POOL (train+dev).

    Why: a single 41-subject dev split cannot rank candidates — the spread
    between different splits of the same config (dev macro-F1 0.145 vs holdout
    0.345 in v7) dwarfs the spread between candidates. Each fold trains on the
    pool minus the fold and scores the held-out fold with the same trusted
    evaluator, and the harness compares the MEAN against a noise margin.

    `repeats` re-partitions the same pool with a different shuffle and averages
    over R x K folds: the per-fold spread stays what it is, but the spread of
    the mean falls as 1/sqrt(n), which is the quantity a KEEP decision actually
    rests on. One repeat (the default) is the single-partition behaviour.

    Two properties are structural, not conventions:
    - folds are drawn from train + dev only, so the holdout cannot be selected on;
    - each fold's manifest has its scaler refit on that fold's train rows and an
      EMPTY holdout partition (see data.real_split.fold_manifest).
    """
    import random

    import torch

    from data.real_split import fold_manifest

    from .real_data import (
        RealBatchProvider,
        build_real_model,
        manifest_from_path,
        real_eval_manifest,
    )
    from .tcm_eval import TCMTrustedEvaluator

    if repeats < 1 or folds < 2:
        raise ValueError(f"repeats={repeats} and folds={folds} must be >= 1 and >= 2")
    manifest = manifest_from_path(config["data"]["manifest_path"])
    records = _load_records(config["data"]["dataset_path"])
    pool = list(manifest["split"]["train"]) + list(manifest["split"]["dev"])
    if len(pool) < folds * 4:
        raise ValueError(f"selection pool of {len(pool)} subjects is too small for {folds} folds")

    trainer_kwargs = dict(config.get("trainer", {}))
    for forbidden in ("early_stop_patience", "restore_best"):
        if trainer_kwargs.get(forbidden) is not None:
            raise ValueError(
                f"real-data training forbids trainer.{forbidden}: a validation "
                "fold must never steer fitting"
            )
    batch_size = config.get("train_provider", {}).get("batch_size", 32)
    per_fold: list[float | None] = []
    fold_sizes: list[int] = []
    for repeat in range(repeats):
        # each repeat draws a fresh partition of the same pool; averaging over
        # R x K folds is what makes a small effect visible above fold noise
        shuffled = sorted(pool)
        random.Random(split_seed + repeat).shuffle(shuffled)
        fold_of = {s: i % folds for i, s in enumerate(shuffled)}
        for f in range(folds):
            train_ids = [s for s in shuffled if fold_of[s] != f]
            dev_ids = [s for s in shuffled if fold_of[s] == f]
            fold_sizes.append(len(dev_ids))
            # the scaler is refit on THIS fold's train rows: reusing the
            # parent manifest's statistics would put validation-fold feature
            # values into the training inputs of the model that scores them
            fold = fold_manifest(
                manifest, records, train_ids, dev_ids,
                notes=f"selection fold {f} of repeat {repeat}",
            ).data
            torch.manual_seed(trainer_kwargs.get("seed", 0))
            model = build_real_model(fold, config.get("model", {}))
            trainer = Trainer(model, TrainerConfig(**trainer_kwargs))
            provider = RealBatchProvider(fold, records, "train", batch_size=batch_size)
            trainer.fit(provider)  # no eval provider: nothing may steer fitting
            val = RealBatchProvider(fold, records, "dev", batch_size=64)
            evaluator = TCMTrustedEvaluator(real_eval_manifest(fold))
            per_fold.append(
                evaluator.evaluate(model, val, partition="dev").primary_value
            )

    scored = [v for v in per_fold if v is not None]
    if not scored:
        raise ValueError("no fold produced a scoreable primary metric")
    mean = sum(scored) / len(scored)
    variance = sum((v - mean) ** 2 for v in scored) / len(scored)
    std = variance ** 0.5
    return {
        "primary_value": mean,
        # spread of one fold's score: what the chart draws error bars from
        "primary_std": std,
        # spread of the MEAN: the only quantity a KEEP decision compares, so
        # the noise margin is derived from this, not from `primary_std`
        "primary_sem": std / (len(scored) ** 0.5),
        "folds": per_fold,
        "fold_sizes": fold_sizes,
        "repeats": repeats,
        "n_folds": len(scored),
        "selection": "repeated_group_kfold" if repeats > 1 else "group_kfold",
    }


def refit_champion(config: dict, selection: dict | None = None) -> dict:
    """Fit the selected configuration on the whole SELECTION POOL (train+dev).

    The deliverable model: every row that took part in the cross-validated
    selection is used for the final fit, and the frozen holdout is never
    read here (it is unlocked once, by the report). The checkpoint records
    the frozen manifest it descends from, so a report can verify the
    binding instead of trusting the directory name.
    """
    import torch

    from data.real_split import pool_manifest

    from .real_data import RealBatchProvider, build_real_model, manifest_from_path

    data_cfg = dict(config["data"])
    manifest = manifest_from_path(data_cfg["manifest_path"])
    records = _load_records(data_cfg["dataset_path"])
    pool = pool_manifest(
        manifest, records,
        notes="final refit on the AutoResearch selection pool (train+dev)",
    )
    trainer_kwargs = dict(config.get("trainer", {}))
    for forbidden in ("early_stop_patience", "restore_best"):
        if trainer_kwargs.get(forbidden) is not None:
            raise ValueError(
                f"refit forbids trainer.{forbidden}: there is no validation "
                "partition left to steer fitting"
            )
    trainer_cfg = TrainerConfig(**trainer_kwargs)
    torch.manual_seed(trainer_cfg.seed)
    model = build_real_model(pool.data, config.get("model", {}))
    trainer = Trainer(model, trainer_cfg)
    batch_size = config.get("train_provider", {}).get("batch_size", 32)
    provider = RealBatchProvider(pool.data, records, "train", batch_size=batch_size)
    report = trainer.fit(provider)
    artifact_dir = trainer.save_checkpoint(
        Path(config.get("out_dir", "runs/autoresearch/final")),
        smoke_only=False,
        extra_state={
            "manifest_sha256": pool.data["manifest_sha256"],
            "selection_parent_manifest_sha256": manifest["manifest_sha256"],
            "dataset_sha256": manifest["dataset_sha256"],
            "label_version": manifest["label_version"],
            "train_batch_size": batch_size,
            "refit_partition": "train+dev",
            "refit_rows": len(pool.data["split"]["train"]),
            "selection": selection or {},
        },
    )
    return {
        "artifact_dir": str(artifact_dir),
        "model_type": model.model_type,
        "epochs_run": report.epochs_run,
        "smoke_only": False,
        "data_mode": "real",
        "manifest_sha256": pool.data["manifest_sha256"],
        "selection_parent_manifest_sha256": manifest["manifest_sha256"],
        "dataset_sha256": manifest["dataset_sha256"],
        "label_version": manifest["label_version"],
        "refit_partition": "train+dev",
        "refit_rows": len(pool.data["split"]["train"]),
        "holdout_rows_untouched": len(pool.data["split"]["holdout"]),
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
