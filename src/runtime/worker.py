"""Isolated worker entry for the #45 harness.

The harness starts this module in a separate process so training and trusted
scoring run outside the harness/agent process boundary (a first isolation
step; OS/container-level isolation is still AGENT_ISOLATION_UNVERIFIED until
the real-provider smoke in #45).

Usage:
  PYTHONPATH=src python -m runtime.worker train --config <config.json>
  PYTHONPATH=src python -m runtime.worker score --artifact <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from runtime.protocols import train_candidate


def _train(config_path: str) -> dict:
    config = json.loads(Path(config_path).read_text())
    return train_candidate(config)


def _score(artifact_dir: str, seed: int) -> dict:
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
        "tcm", n_batches=2, batch_size=8, seed=seed,
        subject_prefix="trusted-eval-tcm",
    )
    manifest = EvalManifest(
        version="trusted-eval-v0",
        input_spec_version="tcm-input-v0",
        standard_version="SYNTHETIC-CONTRACT-V0",
        label_tier="SMOKE_ONLY",
        train_subjects=(),
        dev_subjects=tuple(provider.subject_ids),
        holdout_subjects=(),
    )
    report = TCMTrustedEvaluator(manifest).evaluate(model, provider, partition="dev")
    return report.to_dict()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="runtime.worker", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train")
    p.add_argument("--config", required=True)
    p.set_defaults(func=lambda a: _train(a.config))

    p = sub.add_parser("score")
    p.add_argument("--artifact", required=True)
    p.add_argument("--seed", type=int, default=1234)
    p.set_defaults(func=lambda a: _score(a.artifact, a.seed))

    args = parser.parse_args(argv)
    print(json.dumps(args.func(args)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
