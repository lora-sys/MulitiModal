"""Finalise the source-level run: refit the champion on the selection pool, then
score the holdout once.

Two things this must not do:

* read the holdout while choosing anything. The champion is already fixed by the
  manifest before this module runs, and the holdout is touched exactly once, at
  the end, to produce a number;
* refit on train alone. The champion was selected by cross-validation over
  train+dev, so both partitions belong to the selection and the deliverable is
  fitted on all of it — with the pool manifest's scaler refit on those 232 rows,
  and its holdout partition intact and untouched.

The checkpoint records the candidate file's SHA-256 and the git commit it came
from, so "which code produced this model" is answerable without trusting a
directory name.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def refit_and_score(
    worktree: Path,
    out_dir: Path,
    manifest_path: str | Path,
    dataset_path: str | Path,
    editable: str,
    commit: str,
    seed: int = 0,
    budget_seconds: float | None = None,
) -> dict:
    import torch

    import research.candidate as candidate
    from data.mendeley import load_mendeley_scores
    from data.real_split import load_real_manifest, pool_manifest
    from research.frozen import TIME_BUDGET_SECONDS, FoldData
    from runtime.real_data import RealBatchProvider, real_eval_manifest, real_input_spec
    from runtime.tcm_eval import TCMTrustedEvaluator

    budget = TIME_BUDGET_SECONDS if budget_seconds is None else budget_seconds
    frozen = load_real_manifest(manifest_path)
    records = load_mendeley_scores(dataset_path)
    pool = pool_manifest(
        frozen, records,
        notes=f"source-level champion {commit[:8]} refit: selection pool train+dev",
    ).data

    # the candidate's own input contract: the pool's scaler, and a holdout that
    # this step is allowed to read exactly once, at the very end
    spec = real_input_spec(pool)
    dims = {"modalities": {n: p["dim"] for n, p in pool["modalities"].items()},
            "targets": 9}
    train_provider = RealBatchProvider(pool, records, "train", batch_size=32)
    fold = FoldData(
        train=train_provider, dev=None, input_spec=spec, dims=dims,
        n_train_subjects=len(pool["split"]["train"]), n_dev_subjects=0, seed=seed,
    )
    started = time.monotonic()
    torch.manual_seed(seed)
    model = candidate.build_model(fold)
    candidate.train(model, fold, budget, seed)
    fit_seconds = round(time.monotonic() - started, 2)

    candidate_path = worktree / editable
    champion_dir = out_dir / "champion"
    champion_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), champion_dir / "weights.pt")
    finalize = {
        "candidate_file": editable,
        "candidate_sha256": _sha256(candidate_path),
        "git_commit": commit,
        "pool_manifest_sha256": pool["manifest_sha256"],
        "selection_parent_manifest_sha256": frozen["manifest_sha256"],
        "refit_partition": "train+dev",
        "refit_rows": len(pool["split"]["train"]),
        "holdout_rows_untouched": len(pool["split"]["holdout"]),
        "seed": seed,
        "time_budget_seconds": budget,
        "fit_seconds": fit_seconds,
        "input_spec_version": spec.version,
    }
    (champion_dir / "finalize.json").write_text(json.dumps(finalize, indent=2))

    # ---- the single holdout read -----------------------------------------
    holdout = RealBatchProvider(pool, records, "holdout", batch_size=64)
    report = TCMTrustedEvaluator(real_eval_manifest(pool)).evaluate(
        model, holdout, partition="holdout"
    )
    from runtime.baselines import evaluate_baselines

    baselines = evaluate_baselines(pool, records)
    return {
        "finalize": finalize,
        "holdout": report.to_dict(),
        "baselines": {
            name: {
                "macro_ap": vals["holdout"]["macro_ap"],
                "macro_f1": vals["holdout"]["macro_f1"],
                "subject_mae": vals["holdout"]["subject_mae"],
            }
            for name, vals in baselines.items()
            if isinstance(vals, dict) and "holdout" in vals
        },
    }