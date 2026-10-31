"""Isolated worker entry for the #45 harness.

The harness starts this module in a separate process so training and trusted
scoring run outside the harness/agent process boundary. The environment is
sanitized to a whitelist (no host secrets/credentials are inherited), a step
toward the OS/container isolation that remains AGENT_ISOLATION_UNVERIFIED
until verified per #45.

Usage:
  PYTHONPATH=src python -m runtime.worker train --config <config.json>
  PYTHONPATH=src python -m runtime.worker score --artifact <dir> \
      [--manifest <real-data-manifest.json> --dataset <data.xls>]

Real-data scoring (#50) evaluates against the frozen manifest's dev
partition and fails closed when nothing is observable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from runtime.protocols import cross_val_candidate, refit_champion, train_candidate

#: the only host environment variables a worker may inherit (#45 hard rule:
#: no credentials, no proxy/auth material from the harness host). Exact-match
#: on purpose — a PYTHON* prefix rule would admit PYTHONSTARTUP/PYTHONHOME,
#: which execute attacker-controlled code at interpreter startup.
ALLOWED_ENV = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "TMPDIR",
    "PYTHONPATH",
    "PYTHONUNBUFFERED",
    "PYTHONDONTWRITEBYTECODE",
)


def sanitize_env(env: dict) -> dict:
    return {k: v for k, v in env.items() if k in ALLOWED_ENV}


def _train(config_path: str) -> dict:
    config = json.loads(Path(config_path).read_text())
    return train_candidate(config)


def _cv(config_path: str, folds: int, seed: int) -> dict:
    config = json.loads(Path(config_path).read_text())
    return cross_val_candidate(config, folds=folds, split_seed=seed)


def _refit(config_path: str) -> dict:
    config = json.loads(Path(config_path).read_text())
    return refit_champion(config)


def _score_real(artifact_dir: str, manifest_path: str, dataset_path: str) -> dict:
    from data.mendeley import load_mendeley_scores

    from .checkpoint import load_training_checkpoint
    from .real_data import RealBatchProvider, manifest_from_path, real_eval_manifest
    from .tcm_eval import TCMTrustedEvaluator

    manifest = manifest_from_path(manifest_path)
    model, state, metadata = load_training_checkpoint(artifact_dir)
    if not model.model_type.startswith("tcm"):
        raise ValueError(
            f"trusted TCM scoring does not support model_type {model.model_type!r} (#45 TCM_ONLY)"
        )
    if state.get("smoke_only", True):
        raise ValueError("refusing real-data scoring of a SMOKE_ONLY checkpoint")
    if state.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError(
            "checkpoint was trained on a different real-data manifest "
            f"({state.get('manifest_sha256')}) than the scorer's ({manifest['manifest_sha256']})"
        )
    records = load_mendeley_scores(dataset_path)
    provider = RealBatchProvider(manifest, records, "dev")
    evaluator = TCMTrustedEvaluator(real_eval_manifest(manifest))
    report = evaluator.evaluate(model, provider, partition="dev")
    out = report.to_dict()
    out["manifest_sha256"] = manifest["manifest_sha256"]
    return out


def _score(artifact_dir: str, manifest: str | None, dataset: str | None) -> dict:
    if bool(manifest) != bool(dataset):
        raise ValueError("--manifest and --dataset must be given together")
    if manifest is not None:
        return _score_real(artifact_dir, manifest, dataset)
    from runtime.batches import SyntheticBatchProvider
    from runtime.checkpoint import load_training_checkpoint
    from runtime.tcm_eval import EvalManifest, TCMTrustedEvaluator

    model, _, metadata = load_training_checkpoint(artifact_dir)
    model_type = metadata["model_type"]
    if not model_type.startswith("tcm"):
        raise ValueError(
            f"trusted TCM scoring does not support model_type {model_type!r} (#45 TCM_ONLY)"
        )
    provider = SyntheticBatchProvider(
        "tcm", n_batches=2, batch_size=8, seed=1234,
        subject_prefix="trusted-eval-tcm",
    )
    manifest_obj = EvalManifest(
        version="trusted-eval-v0",
        input_spec_version="tcm-input-v0",
        standard_version="SYNTHETIC-CONTRACT-V0",
        label_tier="SMOKE_ONLY",
        train_subjects=(),
        dev_subjects=tuple(provider.subject_ids),
        holdout_subjects=(),
    )
    report = TCMTrustedEvaluator(manifest_obj).evaluate(model, provider, partition="dev")
    return report.to_dict()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="runtime.worker", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train")
    p.add_argument("--config", required=True)
    p.set_defaults(func=lambda a: _train(a.config))

    p = sub.add_parser("cv", help="subject-level k-fold score over the selection pool")
    p.add_argument("--config", required=True)
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=lambda a: _cv(a.config, a.folds, a.seed))

    p = sub.add_parser("refit", help="fit the selected config on train+dev (the deliverable)")
    p.add_argument("--config", required=True)
    p.set_defaults(func=lambda a: _refit(a.config))

    p = sub.add_parser("score")
    p.add_argument("--artifact", required=True)
    p.add_argument("--manifest", default=None)
    p.add_argument("--dataset", default=None)
    p.set_defaults(func=lambda a: _score(a.artifact, a.manifest, a.dataset))

    args = parser.parse_args(argv)
    print(json.dumps(args.func(args)))
    return 0


if __name__ == "__main__":
    os.environ = sanitize_env(dict(os.environ))  # noqa: B003 — deliberate whitelist
    sys.exit(main())
