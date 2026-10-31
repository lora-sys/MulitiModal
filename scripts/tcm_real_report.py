"""#50 final deliverable report generator (dev host only, real data).

Consumes the AutoResearch run directory + frozen manifest and produces:
- final dev/holdout evaluation of the champion (per-dim F1 + macro F1 +
  subject MAE)
- Dummy / BMI-only / logistic baselines on the same partitions with the
  same availability information the neural model receives
- row-QC sensitivity on the SAME frozen split minus fingerprint-anomalous
  subjects (never a reshuffled split)
- the separate 8-score PUBLISHED regression task with a train-mean
  predictor as the trivial reference
- charts (round metrics, per-dim F1, class support, regression MAE);
  undefined values render as absent bars, never as 0
- report.md and model-card.md

Everything lands under --out (git-ignored artifacts/); the committed report
references artifact hashes, never raw data.

Usage:
  PYTHONPATH=src python scripts/tcm_real_report.py \
    --runs-dir runs/real-tcm --manifest <manifest.json> --dataset <data.xls> \
    --out artifacts/tcm-50
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from data.mendeley import load_mendeley_scores  # noqa: E402
from data.real_split import (  # noqa: E402
    build_real_manifest,
    exclude_subjects_from_manifest,
    load_real_manifest,
)
from runtime.baselines import CANONICAL_DIMS, evaluate_baselines, evaluate_neural  # noqa: E402
from runtime.checkpoint import load_training_checkpoint  # noqa: E402
from runtime.manifest import MANIFEST_FILE, RunManifestStore  # noqa: E402
from runtime.real_data import (  # noqa: E402
    RealBatchProvider,
    build_real_model,
    real_eval_manifest,
    real_input_spec,
)
from runtime.tcm_eval import TCMTrustedEvaluator  # noqa: E402
from runtime.trainer import Trainer, TrainerConfig  # noqa: E402

KEEP_COLOR, REVERT_COLOR = "#2a9d8f", "#e76f51"
BASELINE_STYLE = {
    "dummy_majority": ("Dummy (majority)", "#adb5bd"),
    "bmi_only_logistic": ("BMI-only logistic", "#e9c46a"),
    "feature_logistic": ("Logistic (all features)", "#4361ee"),
    "neural_champion": ("Neural champion (AutoResearch)", "#2a9d8f"),
}


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def qc_rows_from_records(records) -> tuple[int, ...]:
    """QC exclusions derived from the adapter's audit trail, never hardcoded."""
    return tuple(
        sorted({
            row
            for record in records
            for rows in record.provenance["fingerprint_anomalies"].values()
            for row in rows
        })
    )


def assert_champion_matches_manifest(state: dict, manifest: dict) -> None:
    """Fail-closed binding on the REPORT path (not just the worker scoring
    path): the champion checkpoint must descend from exactly this frozen
    manifest and dataset, or the report would describe an artifact no
    contract can account for.

    Two shapes are accepted. (a) Trained on the frozen manifest's train
    partition. (b) The final refit on the selection pool (train+dev): its
    own manifest hash differs, so it must name the frozen manifest as its
    parent and state the refit partition. Anything else fails closed.
    """
    if state.get("smoke_only"):
        raise ValueError("champion checkpoint is SMOKE_ONLY; refusing to report on it")
    if state.get("manifest_sha256") != manifest["manifest_sha256"]:
        parent = state.get("selection_parent_manifest_sha256")
        if not (parent == manifest["manifest_sha256"]
                and state.get("refit_partition") == "train+dev"):
            raise ValueError(
                "champion checkpoint was trained on manifest "
                f"{state.get('manifest_sha256')!r} (selection parent {parent!r}, "
                f"refit {state.get('refit_partition')!r}), which does not descend "
                f"from the frozen manifest {manifest['manifest_sha256']!r}; "
                "refusing to build the report"
            )
    if state.get("dataset_sha256") != manifest["dataset_sha256"]:
        raise ValueError(
            "champion checkpoint dataset_sha256 "
            f"{state.get('dataset_sha256')!r} does not match the frozen manifest "
            f"{manifest['dataset_sha256']!r}; refusing to build the report"
        )


def availability_rates(manifest: dict, records) -> dict:
    """Per-partition, per-modality availability so readers can see how much
    of the 'full' modality set is actually observable (#eval E05)."""
    by_row = {r.subject_index: r for r in records}
    out: dict = {}
    for partition in ("train", "dev", "holdout"):
        rows = [manifest["subject_row_by_id"][s] for s in manifest["split"][partition]]
        for modality, params in manifest["modalities"].items():
            groups = params["feature_groups"]
            n_available = sum(
                1 for r in rows
                if not any(math.isnan(v) for g in groups for v in by_row[r].features[g])
            )
            out[f"{partition}/{modality}"] = {"n_rows": len(rows), "n_available": n_available}
    return out


def dev_primary(model, manifest, records) -> float:
    evaluator = TCMTrustedEvaluator(real_eval_manifest(manifest))
    provider = RealBatchProvider(manifest, records, "dev")
    report = evaluator.evaluate(model, provider, partition="dev")
    return report.subject_mae


def derived_manifest(
    manifest: dict, records, *, task_type: str | None = None,
    modality_set: str | None = None, notes: str = "",
) -> dict:
    """Rebuild a task-variant manifest that MIRRORS the frozen split: same
    seed, the frozen manifest's fractions and exclusions — then verify
    partition membership actually matches instead of relying on default-
    argument coincidence (#eval E29)."""
    derived = build_real_manifest(
        records,
        dataset_sha256=manifest["dataset_sha256"],
        task_type=task_type or manifest["task_type"],
        modality_set=modality_set or manifest["modality_set"],
        split_seed=manifest["split"]["seed"],
        fractions=tuple(manifest["split"]["fractions"]),
        exclude_subject_rows=tuple(manifest["excluded_subject_rows"]),
        notes=notes,
    ).data
    for partition in ("train", "dev", "holdout"):
        if set(derived["split"][partition]) != set(manifest["split"][partition]):
            raise ValueError(
                f"derived manifest {partition!r} membership drifted from the "
                "frozen split; refusing"
            )
    return derived


def _fixed_training(
    manifest: dict,
    records,
    model_cfg: dict,
    epochs: int,
    out_dir: Path,
    trainer_kwargs: dict | None = None,
    seed: int = 0,
    train_batch_size: int = 32,
):
    # seed BEFORE model construction: deterministic init regardless of what
    # this process ran before (#eval E06)
    torch.manual_seed(seed)
    model = build_real_model(manifest, model_cfg)
    config = TrainerConfig(seed=seed, epochs=epochs, **(trainer_kwargs or {}))
    trainer = Trainer(model, config)
    trainer.fit(
        RealBatchProvider(manifest, records, "train", batch_size=train_batch_size),
        # a pool manifest has no dev rows; passing an empty provider would be
        # both meaningless and (before the guard) an obscure crash
        RealBatchProvider(manifest, records, "dev", batch_size=64)
        if manifest["split"]["dev"] else None,
    )
    trainer.save_checkpoint(
        out_dir, smoke_only=False,
        extra_state={"manifest_sha256": manifest["manifest_sha256"]},
    )
    model, _, _ = load_training_checkpoint(out_dir)
    return model


def regression_task(manifest_cls, records, out_dir: Path) -> dict:
    """Separate 8-score PUBLISHED regression training (fixed config) with a
    per-dim train-mean predictor as the trivial reference (#eval E11)."""
    manifest = derived_manifest(
        manifest_cls, records, task_type="score_regression",
        notes="regression deliverable: mirrors the frozen classification split",
    )
    model = _fixed_training(
        manifest, records,
        {"include_category_head": False, "include_score_head": True}, 60,
        out_dir / "regression",
    )
    by_row = {r.subject_index: r for r in records}
    train_rows = [manifest["subject_row_by_id"][s] for s in manifest["split"]["train"]]
    train_mean = []
    for d in range(9):
        vals = [by_row[r].scores[d] for r in train_rows
                if by_row[r].scores[d] is not None]
        train_mean.append(sum(vals) / len(vals) if vals else None)

    per_dim: dict = {}
    model.eval()
    with torch.no_grad():
        for partition in ("dev", "holdout"):
            provider = RealBatchProvider(manifest, records, partition, batch_size=64)
            sq: dict[int, list] = {d: [] for d in range(9)}
            absd: dict[int, list] = {d: [] for d in range(9)}
            ref_absd: dict[int, list] = {d: [] for d in range(9)}
            for item in provider:
                out = model(item.batch)
                pred = out.predictions["constitution_scores"]
                target = item.batch.targets["constitution_scores"]
                mask = item.batch.target_mask["constitution_scores"]
                for d in range(9):
                    obs = mask[:, d]
                    if obs.any():
                        diffs = (pred[:, d] - target[:, d])[obs]
                        absd[d].extend(diffs.abs().tolist())
                        sq[d].extend((diffs**2).tolist())
                        if train_mean[d] is not None:
                            ref = (target[:, d] - train_mean[d])[obs]
                            ref_absd[d].extend(ref.abs().tolist())
            per_dim[partition] = {
                CANONICAL_DIMS[d]: {
                    "mae": sum(absd[d]) / len(absd[d]) if absd[d] else None,
                    "rmse": (sum(sq[d]) / len(sq[d])) ** 0.5 if sq[d] else None,
                    "mean_predictor_mae": (
                        sum(ref_absd[d]) / len(ref_absd[d]) if ref_absd[d] else None
                    ),
                }
                for d in range(9)
            }
    return {
        "manifest_sha256": manifest["manifest_sha256"],
        "aggregation": "pooled per-dim over observed entries",
        "includes_qc_rows": True,
        "per_dim": per_dim,
    }


def row_qc_sensitivity(
    manifest, records, out_dir: Path, champion_cfg: dict, champion_trainer: dict,
    champion_model=None, seed: int = 0, train_batch_size: int = 32,
) -> dict:
    """Retrain without fingerprint-anomalous subjects and compare on the SAME
    holdout (exclude_subjects_from_manifest, never a fresh shuffle).

    Both sides of the contrast are fitted on the selection pool (the manifest
    passed in) and scored on the untouched holdout: one denominator, one
    scoring partition, and each model under its OWN manifest — the champion
    keeps the scaler it was trained with (a checkpoint's input contract is
    bound to its training manifest), the QC model uses the QC manifest's
    refit scaler. The dev partition is empty in a pool manifest, so a dev
    contrast would have no subjects to average over.
    """
    from runtime.real_data import real_eval_manifest

    qc_rows = qc_rows_from_records(records)
    manifest_qc = exclude_subjects_from_manifest(
        manifest, records, qc_rows,
        notes="row-QC sensitivity: selection pool minus fingerprint-anomalous subjects",
    ).data
    trainer_kwargs = {
        k: champion_trainer[k] for k in ("learning_rate", "grad_clip")
        if champion_trainer.get(k) is not None
    }
    model_qc = _fixed_training(
        manifest_qc, records, champion_cfg, champion_trainer.get("epochs", 40),
        out_dir / "qc-exclude", trainer_kwargs, seed=seed,
        train_batch_size=train_batch_size,
    )
    holdout = tuple(manifest_qc["split"]["holdout"])
    champion_without = None
    if champion_model is not None:
        provider = RealBatchProvider(
            manifest, records, "holdout", subject_ids=holdout
        )
        champion_without = TCMTrustedEvaluator(real_eval_manifest(manifest)).evaluate(
            champion_model, provider, partition="holdout"
        ).subject_mae
    return {
        "manifest_sha256": manifest_qc["manifest_sha256"],
        "excluded_subject_rows": list(qc_rows),
        "fit_rows_without": len(manifest_qc["split"]["train"]),
        "holdout_subject_mae_without": holdout_primary(model_qc, manifest_qc, records),
        "holdout_subject_mae_champion_on_reduced": champion_without,
    }


def holdout_primary(model, manifest, records) -> float | None:
    """Subject MAE on the holdout partition (the only partition a model
    refit on the selection pool is allowed to be scored on)."""
    evaluator = TCMTrustedEvaluator(real_eval_manifest(manifest))
    provider = RealBatchProvider(manifest, records, "holdout")
    return evaluator.evaluate(model, provider, partition="holdout").subject_mae


def deployable_ablation(
    manifest, records, out_dir: Path, champion_cfg: dict, champion_trainer: dict,
    seed: int = 0, train_batch_size: int = 32,
) -> dict:
    """Body/Vitals-only training with the champion's configuration, isolating
    the feature-set effect.

    The ablation is fitted on the same SELECTION POOL (train+dev) as the
    deliverable: an ablation fitted on the frozen train fold alone would
    differ in training rows as well as in features, and could not be read as
    "only the feature set changed". Its dev partition is empty by
    construction, so it is scored on the holdout like the deliverable.
    """
    from data.real_split import pool_manifest

    manifest_dep = pool_manifest(
        derived_manifest(
            manifest, records, modality_set="deployable",
            notes="deployable Body/Vitals-only ablation (no exam hardware needed)",
        ),
        records,
        notes="deployable ablation fitted on the AutoResearch selection pool",
    ).data
    trainer_kwargs = {
        k: champion_trainer[k] for k in ("learning_rate", "grad_clip")
        if champion_trainer.get(k) is not None
    }
    model_dep = _fixed_training(
        manifest_dep, records, champion_cfg, champion_trainer.get("epochs", 40),
        out_dir / "deployable-only", trainer_kwargs, seed=seed,
        train_batch_size=train_batch_size,
    )
    return {
        "manifest_sha256": manifest_dep["manifest_sha256"],
        "fit_rows": len(manifest_dep["split"]["train"]),
        "holdout_subject_mae": holdout_primary(model_dep, manifest_dep, records),
        "f1": evaluate_neural(model_dep, manifest_dep, records),
    }


def chart_rounds(entries: list[dict], out: Path) -> None:
    rounds = [
        e for e in entries
        if e.get("metrics") and e["metrics"].get("primary_value") is not None
    ]
    if not rounds:
        return
    # x positions are the REAL round indices from the run ids (tcm-0003 -> 3),
    # not positional 1..n — reverted rounds must not compress the axis
    ys = [e["metrics"]["primary_value"] for e in rounds]
    xs = [int(e["run_id"].rsplit("-", 1)[-1]) for e in rounds]
    colors = [KEEP_COLOR if e["decision"] == "KEEP" else REVERT_COLOR for e in rounds]
    # fold-averaged rounds carry their spread; without the bars a 0.03
    # difference reads as a signal when it is inside the fold noise
    errs = [e.get("primary_std") for e in rounds]
    has_err = any(v is not None for v in errs)
    folded = any((e.get("selection") or {}).get("folds", 1) > 1 for e in rounds)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(xs, ys, color=colors,
           yerr=[v or 0.0 for v in errs] if has_err else None,
           ecolor="#264653", capsize=3)
    # Two different numbers matter and confusing them misreports the run:
    # the champion is the best ACCEPTED score (what the deliverable uses),
    # while the tallest bar is merely the best score ever observed — a
    # candidate that did not clear the noise margin and was rolled back.
    accepted = [e["best_so_far"] for e in rounds if e.get("best_so_far") is not None]
    champion = max(accepted) if accepted else None
    observed = max(ys)  # the primary metric is maximised (macro-F1)
    if champion is not None:
        ax.axhline(champion, color="#264653", linestyle="--", linewidth=1,
                   label=f"champion, accepted {champion:.4f}")
    if champion is None or observed > champion + 1e-9:
        ax.axhline(observed, color="#e76f51", linestyle=":", linewidth=1,
                   label=f"max observed, not accepted {observed:.4f}")
    # the two fill colours carry the decision; without a key the reader cannot
    # tell a kept round from a rolled-back one
    from matplotlib.patches import Patch

    handles, labels = ax.get_legend_handles_labels()
    handles += [Patch(color=KEEP_COLOR), Patch(color=REVERT_COLOR)]
    labels += ["KEEP (new champion)", "REVERT (rolled back)"]
    ax.set_xticks(xs)
    ax.set_xticklabels([e["run_id"] for e in rounds], fontsize=7, rotation=90)
    ax.set_xlabel("AutoResearch round")
    metric_label = (
        "selection macro-F1, k-fold over train+dev (mean ± fold std)"
        if folded else "dev macro-F1 (selection metric)"
    )
    ax.set_ylabel(metric_label, fontsize=9)
    ax.set_title("AutoResearch rounds — selection metric")
    ax.legend(handles, labels, fontsize=8)
    # label only the rounds that matter: 24 labels at 4 decimals collide into
    # an unreadable band, and the y-axis already carries the values
    for e, x, y in zip(rounds, xs, ys, strict=True):
        keep = e["decision"] == "KEEP"
        top = abs(y - observed) < 1e-12
        if not (keep or top):
            continue
        ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "chart-rounds.png", dpi=150)
    plt.close(fig)


def _series_values(per_dim: dict, dims: list[str]) -> tuple[list[float | None], int]:
    vals = [per_dim[d]["f1"] for d in dims]
    return vals, sum(1 for v in vals if v is None)


def chart_per_dim_f1(baselines: dict, neural: dict, out: Path) -> None:
    dims = [d for d in CANONICAL_DIMS if d != "balanced"]
    series = []
    for key, (label, color) in BASELINE_STYLE.items():
        source = neural if key == "neural_champion" else baselines.get(key)
        if not source:
            continue
        vals, n_undef = _series_values(source["holdout"]["per_dim"], dims)
        shown = label + (f" ({n_undef} undef)" if n_undef else "")
        series.append((shown, color, [np.nan if v is None else v for v in vals]))
    width = 0.8 / len(series)
    xs = range(len(dims))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, (label, color, vals) in enumerate(series):
        pos = [x + i * width for x in xs]
        ax.bar(pos, vals, width, label=label, color=color)
    ax.set_xticks([x + 0.4 - width / 2 for x in xs])
    ax.set_xticklabels(dims, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("holdout F1 (>=40 = YES)")
    ax.set_title(
        "Per-dimension holdout F1 — neural champion vs baselines "
        "(missing bar = F1 undefined, not zero)"
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "chart-per-dim-f1.png", dpi=150)
    plt.close(fig)


def chart_support(neural: dict, out: Path) -> None:
    # holdout partition — same partition as the per-dim F1 chart (#eval E13)
    dims = [d for d in CANONICAL_DIMS if d != "balanced"]
    support = [neural["holdout"]["per_dim"][d]["support_positive"] for d in dims]
    total = [neural["holdout"]["per_dim"][d]["n_observed"] for d in dims]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(dims, total, color="#dee2e6", label="observed")
    ax.bar(dims, support, color="#2a9d8f", label="positive (YES)")
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels(dims, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("holdout subjects")
    ax.set_title("Per-dimension label support (holdout partition)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "chart-support.png", dpi=150)
    plt.close(fig)


def chart_regression(regression: dict, out: Path) -> None:
    dims = [d for d in CANONICAL_DIMS if d != "balanced"]
    for partition, fname in (("dev", "chart-regression-dev.png"),
                             ("holdout", "chart-regression-holdout.png")):
        model_mae = [
            regression["per_dim"][partition][d]["mae"] for d in dims
        ]
        ref_mae = [
            regression["per_dim"][partition][d]["mean_predictor_mae"] for d in dims
        ]
        width = 0.4
        xs = range(len(dims))
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar([x - width / 2 for x in xs],
               [np.nan if v is None else v for v in model_mae],
               width, label="neural model", color="#4361ee")
        ax.bar([x + width / 2 for x in xs],
               [np.nan if v is None else v for v in ref_mae],
               width, label="train-mean predictor", color="#adb5bd")
        ax.set_xticks(list(xs))
        ax.set_xticklabels(dims, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(f"{partition} MAE (0-100 score points)")
        ax.set_title(
            f"8-score PUBLISHED regression — {partition} per-dimension MAE\n"
            "neural model vs train-mean predictor (balanced dim unobserved)"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / fname, dpi=150)
        plt.close(fig)


def seed_robustness(
    manifest, records, out_dir: Path, champion_cfg: dict, champion_trainer: dict,
    seeds: tuple[int, ...] = (1, 2), train_batch_size: int = 32,
) -> dict:
    """Refit the champion config on the pool with other seeds and score the
    SAME holdout.

    The headline number is one fit of one seed. If the margin over the
    logistic baseline is seed luck, it will not survive two more fits — and
    that is exactly the claim a reader is entitled to check before believing
    "the neural model is better". Same split, same holdout, same unlock
    event: this is not a second look at a fresh partition.
    """
    trainer_kwargs = {
        k: champion_trainer[k] for k in ("learning_rate", "grad_clip")
        if champion_trainer.get(k) is not None
    }
    epochs = champion_trainer.get("epochs", 40)
    per_seed: dict[str, dict] = {}
    for seed in seeds:
        model = _fixed_training(
            manifest, records, champion_cfg, epochs, out_dir / f"seed-{seed}",
            trainer_kwargs, seed=seed, train_batch_size=train_batch_size,
        )
        report = evaluate_neural(model, manifest, records)["holdout"]
        per_seed[str(seed)] = {
            "macro_f1": report["macro_f1"],
            "subject_mae": report["subject_mae"],
        }
    return {"seeds": list(seeds), "per_seed": per_seed,
            "fit_rows": len(manifest["split"]["train"])}


def _seed_verdict(seeds: dict, reference_f1: float | None, primary_f1: float | None) -> str:
    """Say whether the headline margin survives a change of seed."""
    values = [v["macro_f1"] for v in seeds.get("per_seed", {}).values()
              if v.get("macro_f1") is not None]
    if not values or reference_f1 is None:
        return "无法判定（缺少种子或基线指标）"
    above = sum(1 for v in values if v > reference_f1)
    all_above = above == len(values)
    if all_above:
        return (f"全部 {len(values)} 个额外种子都高于全特征逻辑回归 "
                f"{_fmt(reference_f1)}，差距不是单一种子的偶然")
    return (f"仅 {above}/{len(values)} 个额外种子高于全特征逻辑回归 "
            f"{_fmt(reference_f1)}——单种子领先不可复现，不得当作模型优势的证据")


def _regression_verdict(regression: dict) -> str:
    """State the outcome of the 8-score regression against its own reference,
    rather than leaving the reader to diff two bar series (#audit: the report
    said 'see summary.json' while the model lost to the mean predictor on
    every holdout dimension)."""
    dims = [d for d in CANONICAL_DIMS if d != "balanced"]
    parts: list[str] = []
    for partition in ("dev", "holdout"):
        per_dim = regression["per_dim"].get(partition, {})
        wins = 0
        compared = 0
        for dim in dims:
            entry = per_dim.get(dim) or {}
            model_mae, ref_mae = entry.get("mae"), entry.get("mean_predictor_mae")
            if model_mae is None or ref_mae is None:
                continue
            compared += 1
            wins += model_mae < ref_mae
        if not compared:
            continue
        verdict = (
            "未优于训练均值预测器" if wins == 0
            else f"在 {wins}/{compared} 维优于训练均值预测器"
        )
        parts.append(f"{partition} {verdict}")
    if not parts:
        return ""
    return (
        "逐维对比结果：" + "；".join(parts)
        + "。回归头与分类头共享训练、每维样本量更小，当前证据不支持把它当作"
        "评分预测已成立的证据。"
    )


def _zero_dims_line(neural: dict) -> str:
    """Name the dimensions whose holdout F1 is exactly 0 — a macro average
    hides a total miss on a supported dimension."""
    per_dim = neural["holdout"]["per_dim"]
    zeros = [
        dim for dim, v in per_dim.items()
        if v.get("f1") == 0.0 and (v.get("support_positive") or 0) > 0
    ]
    if not zeros:
        return "- 逐维失败维度：holdout 上没有被完全漏判的维度"
    detail = ", ".join(f"{d}（{per_dim[d]['support_positive']} 个正例）" for d in zeros)
    return f"- 逐维失败维度：holdout 上 F1 恰为 0 的维度——{detail}；macro 平均会掩盖它们"


def _fmt(value: float | None, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if value is not None else "N/A"


def check_proposal_ordering(runs_dir: Path) -> dict:
    """Did each proposal exist before the research state it claims to rest on?

    The manifest records `context_sha256` (the state at apply time) and
    `proposal_sha256`, which together cannot distinguish "the agent read the
    state, then proposed" from "a batch of proposals was written earlier and
    the context file was generated at apply time to pass the digest check".
    The drivers leave both files in the run directory, so their mtimes
    separate the two cases mechanically — including for runs recorded before
    the harness started flagging it.

    Returns counts plus the rounds concerned. Empty when the run directory
    keeps no proposal/context files (agent-driven runs through `step()` do,
    a scripted `research` run does not).
    """
    pairs: list[tuple[str, bool]] = []
    if not runs_dir.is_dir():
        return {"pairs": 0, "proposal_predates_context": 0, "rounds": []}
    for ctx in sorted(runs_dir.glob("ctx-[0-9][0-9][0-9][0-9].json")):
        n = ctx.stem.split("-", 1)[1]
        prop = runs_dir / f"p-{n}.json"
        if not prop.exists():
            continue
        pairs.append(
            (f"tcm-{n}", prop.stat().st_mtime < ctx.stat().st_mtime)
        )
    flagged = [run_id for run_id, predates in pairs if predates]
    return {
        "pairs": len(pairs),
        "proposal_predates_context": len(flagged),
        "rounds": flagged,
    }


def _ordering_line(ar: dict) -> str:
    """Report what the proposal/context mtime check found, or nothing."""
    ordering = ar.get("proposal_ordering") or {}
    pairs = ordering.get("pairs") or 0
    if not pairs:
        return ""
    predates = ordering.get("proposal_predates_context") or 0
    if not predates:
        return (f" 提案/上下文文件时间戳自检：{pairs} 对均正常，"
                "无提案早于其所依据的研究状态。")
    rounds = "、".join(ordering.get("rounds") or []) or "—"
    return (
        f" **提案/上下文文件时间戳自检：{predates}/{pairs} 轮的提案文件早于其所配的"
        f"上下文文件**（{rounds}）——这些轮的 digest 只证明「应用时刻的研究状态与所声称的"
        "一致」，不证明提案者事先读过该状态（批量预写提案的必然结果）。后续轮次须逐轮"
        "读取上下文后再提案。"
    )


def agent_status_line(ar: dict) -> str:
    """Provenance line derived from the run entries, never hardcoded."""
    rounds = ar.get("per_round", [])
    statuses = {e.get("agent_status") for e in rounds if e.get("agent_status")}
    proposers = sorted({e["proposer"] for e in rounds if e.get("proposer")})
    if proposers:
        provider = "外部 LLM agent " + "、".join(f"`{p}`" for p in proposers)
        agent_note = (
            f"每轮记录提案者与上下文摘要（{sum(1 for e in rounds if e.get('proposer'))}"
            f"/{len(rounds)} 轮带完整溯源）"
        )
    elif "AGENT_ISOLATION_UNVERIFIED" in statuses:
        provider = "外部 agent（提案者未记录，provenance 不完整）"
        agent_note = "manifest 未记录 proposer/context 摘要"
    else:
        provider = "确定性计划"
        agent_note = "无外部 agent"
    return (
        f"- 训练模式：REAL_DATA_TRAINING（agent provider={provider}；{agent_note}；"
        "agent 状态 AGENT_ISOLATION_UNVERIFIED：OS/container 级隔离仍未验证）。"
    )


def write_reports(out: Path, summary: dict, manifest: dict, records_count: int) -> None:
    ar = summary["autoresearch"]
    neural = summary["neural_evaluation"]
    base = summary["baselines"]
    regression = summary["regression_task"]
    qc = summary["row_qc_sensitivity"]
    dep = summary["deployable_ablation"]
    seeds = summary["seed_robustness"]
    avail = summary["availability"]

    keep = [e["run_id"] for e in ar["per_round"] if e["decision"] == "KEEP"]
    revert = [e["run_id"] for e in ar["per_round"] if e["decision"] == "REVERT"]
    split_sizes = summary["split_sizes"]
    split_total = sum(split_sizes.values())
    dummy_holdout = base["dummy_majority"]["holdout"]
    rates = [
        v["positive_rate"] for v in dummy_holdout.get("per_dim", {}).values()
        if v.get("positive_rate") is not None
    ]
    pos_rate = sum(rates) / len(rates) if rates else None
    excluded_note = (
        f"，剔除 {len(manifest.get('excluded_subject_rows', []))} 行"
        if manifest.get("excluded_subject_rows") else ""
    )
    deliverable = summary["deliverable"]
    in_sample_dev = deliverable["dev_metrics_are_in_sample"]
    folds = sorted({
        (e.get("selection") or {}).get("folds", 1) for e in ar["per_round"]
    })
    if folds == [1]:
        selection_line = (
            f"dev macro-F1（单划分，{split_sizes['dev']} 人）——该划分的划分间波动"
            "大于候选间差异，排序不可信"
        )
    else:
        k = folds[-1]
        selection_line = (
            f"subject 级 {k} 折交叉验证的 macro-F1 均值（池 = train+dev 共 "
            f"{split_sizes['train'] + split_sizes['dev']} 人，holdout 从不进入）；"
            "KEEP 需超过冠军一个由折间 std 派生的余量，避免把噪声当改进"
        )
    dev_label = (
        f"dev（{split_sizes['dev']} 人）**已并入选择池、不再单独评分**"
        if in_sample_dev else f"dev（{split_sizes['dev']} 人；样本外）"
    )
    # with a pool refit the trusted evaluator forbids a dev score (dev ⊆ train),
    # so the deliverable has exactly one honest partition: the holdout
    scope_line = (
        f"评分口径：交付模型在选择池（{_fmt(deliverable['refit_rows'], 0)} 行）上重训，"
        f"dev {split_sizes['dev']} 人已并入该池，受信评估器禁止 train/dev 重叠，"
        f"因此**只在 holdout {split_sizes['holdout']} 人上评分**；下方所有对照"
        "（Dummy / BMI / 全特征逻辑回归）都在同一池上拟合、同一 holdout 上评分，"
        "不靠少给基线 41 行来抬高神经模型的数字。"
        if in_sample_dev else
        f"评分口径：交付模型在冻结 train 折上训练，dev {split_sizes['dev']} 人"
        f"为样本外、holdout {split_sizes['holdout']} 人一次解锁。"
    )
    if in_sample_dev:
        neural_lines = [
            f"- 神经冠军 holdout macro-F1 {_fmt(neural['holdout']['macro_f1'])}"
            f"（macro 对全部可观测维度平均，未定义 F1 计 0，分母不随模型漂移；"
            f"未定义维度 {neural['holdout']['n_undefined_f1']} 个）",
            f"- 神经冠军 holdout subject MAE {_fmt(neural['holdout']['subject_mae'], 4)}"
            "（诊断口径；选择指标是 macro-F1，即上方口径）",
            f"- 神经冠军 holdout subject RMSE（Brier 诊断，非选择指标）"
            f"{_fmt(neural['holdout']['subject_rmse'], 4)}；Dummy（多数类）holdout "
            f"subject RMSE {_fmt(base['dummy_majority']['holdout']['subject_rmse'], 4)}",
        ]
    else:
        neural_lines = [
            f"- 神经冠军：{dev_label} macro-F1 {_fmt(neural['dev']['macro_f1'])}，"
            f"holdout macro-F1 {_fmt(neural['holdout']['macro_f1'])}"
            f"（macro 对全部可观测维度平均，未定义 F1 计 0，分母不随模型漂移；"
            f"dev 未定义维度 {neural['dev']['n_undefined_f1']} 个，holdout "
            f"{neural['holdout']['n_undefined_f1']} 个）",
            f"- 神经冠军 subject MAE：dev（同上口径）{_fmt(neural['dev']['subject_mae'], 4)}，"
            f"holdout {_fmt(neural['holdout']['subject_mae'], 4)}"
            "（诊断口径；选择指标是 macro-F1，即上方口径）",
            f"- 神经冠军 subject RMSE（Brier 诊断，非选择指标）：dev "
            f"{_fmt(neural['dev']['subject_rmse'], 4)}，holdout "
            f"{_fmt(neural['holdout']['subject_rmse'], 4)}；Dummy（多数类）holdout "
            f"subject RMSE {_fmt(base['dummy_majority']['holdout']['subject_rmse'], 4)}",
        ]
    selection_header = (
        "选择 macro-F1（dev 单划分）" if folds == [1]
        else f"选择 macro-F1（{folds[-1]} 折均值 ± 折间 std）"
    )

    def holdout_macro_f1(entry: dict) -> str:
        return _fmt(entry["holdout"]["macro_f1"])

    lines = [
        "# TCM 真实训练报告（#50，AutoResearch V1，"
        f"{summary.get('report_version', 'v2')}）",
        "",
        f"生成时间：{summary['generated_at']}  ",
        f"数据：`mendeley-cxtph3tjsg-v1`（SHA-256 `{summary['dataset_sha256'][:16]}…`，"
        f"{records_count} 名受试者，其中 {split_total} 名入冻结 manifest{excluded_note}；"
        f"QC 异常行 {qc['excluded_subject_rows']} 四维计分指纹不符）  ",
        f"标准：{summary['standard_version']}；标签契约：`{summary['label_version']}`  ",
        f"冻结 manifest：`{summary['manifest_sha256'][:16]}…`；划分："
        f"train {split_sizes['train']} / dev {split_sizes['dev']} / "
        f"holdout {split_sizes['holdout']}"
        "（行级互斥划分；受试者级不重叠未经验证——数据集无 ID 列）",
        "",
        "## AutoResearch 轮次",
        "",
        f"共 {ar['rounds_completed']} 轮：KEEP {keep}，REVERT {revert}。",
        f"选择口径：{selection_line}。",
        "冠军配置：`" + ar["champion"]["run_id"] + "`（"
        f"选择值 {_fmt(ar['champion']['best_so_far'], 4)}）。最终交付模型是按该配置"
        f"在**选择池（train+dev）全量重训**的结果（{_fmt(deliverable['refit_rows'], 0)} 行），"
        f"权重 SHA-256 `{(deliverable['weights_sha256'] or '')[:16]}…`。",
        "所有候选使用相同冻结种子与 epoch 级确定性 shuffle；batch_size 被改动时"
        "批次组成随之变化，其余超参独立比较。",
        "",
        f"| 轮 | 意图 | 提案者 | 决定 | {selection_header} |",
        "| --- | --- | --- | --- | --- |",
    ]
    for e in ar["per_round"]:
        value = e.get("primary_value")
        std = e.get("primary_std")
        cell = f"{value:.4f}" if value is not None else "—"
        if value is not None and std is not None:
            cell += f" ± {std:.4f}"
        lines.append(
            f"| {e['run_id']} | {e['intent']} | {e.get('proposer') or '—'} "
            f"| {e['decision']} | {cell} |"
        )
    lines += [
        "",
        "## 最终评估（holdout 在研究结束后解锁；历次解锁见「证据状态与限制」，"
        "均已如实披露）",
        "",
        scope_line,
        *neural_lines,
        _zero_dims_line(neural),
        f"- Dummy（多数类）：holdout macro-F1 {holdout_macro_f1(base['dummy_majority'])}，"
        f"holdout subject MAE {_fmt(base['dummy_majority']['holdout']['subject_mae'], 4)}",
        f"- BMI-only 逻辑回归：holdout macro-F1 {holdout_macro_f1(base['bmi_only_logistic'])}，"
        f"holdout subject MAE {_fmt(base['bmi_only_logistic']['holdout']['subject_mae'], 4)}",
        f"- 全特征逻辑回归（含可用性指示位）：holdout macro-F1 "
        f"{holdout_macro_f1(base['feature_logistic'])}，"
        f"holdout subject MAE {_fmt(base['feature_logistic']['holdout']['subject_mae'], 4)}",
        f"- **种子稳健性**（冠军同配置、同一池 {seeds['fit_rows']} 行、同一 holdout，"
        "属同一次解锁事件）：seed 0（交付模型）macro-F1 "
        f"{_fmt(neural['holdout']['macro_f1'])}；"
        + "，".join(
            f"seed {s}: {_fmt(v['macro_f1'])} / MAE {_fmt(v['subject_mae'], 4)}"
            for s, v in seeds["per_seed"].items()
        )
        + "。"
        + _seed_verdict(seeds, base["feature_logistic"]["holdout"]["macro_f1"],
                        neural["holdout"]["macro_f1"]),
        "- **口径说明（必读）**：AutoResearch 的选择指标就是这里报告的 "
        "macro-F1（0.5 阈值、固定观测维分母），保证爬山的落点和交付指标一致。"
        "两个概率损失代理都被实测否决：subject MAE 是 L1，二值标签下的最优常数是"
        f"中位数 0（本数据正类率 ≈{_fmt(pos_rate, 2)} < 0.5），按它爬山永远赢不过"
        "全零预测器；subject RMSE（Brier）虽是 proper scoring rule，但对没有可用信号的"
        "模型，最优解是预测基率常数——实测那一轮冠军 dev macro-F1 只有 0.021，"
        "而普通逻辑回归 holdout 是 "
        f"{_fmt(base['feature_logistic']['holdout']['macro_f1'])}。"
        "因此 MAE/RMSE 仅作诊断，选择看 F1：holdout macro-F1 "
        f"{_fmt(neural['holdout']['macro_f1'])} 对 Dummy "
        f"{_fmt(dummy_holdout['macro_f1'])}、全特征逻辑回归 "
        f"{_fmt(base['feature_logistic']['holdout']['macro_f1'])}。"
        "三个口径都给出，读者可自行权衡。"
        f"**统计强度**：holdout 仅 {split_sizes['holdout']} 人，macro-F1 的差异在"
        "数十个正例量级上属噪声范围，未做重采样置信区间；跨划分的重复性才是判据，"
        "单一划分上的领先不构成「该模型更好」的证据。",
        "",
        "图表：`chart-rounds.png`、`chart-per-dim-f1.png`、`chart-support.png`、"
        "`chart-regression-dev.png`、`chart-regression-holdout.png`"
        "（图中缺失柱 = F1/MAE 未定义，不是 0）。",
        "",
        "## 模态可用率（缺失≠被当作平均人：基线与神经模型拿到同一可用性信息）",
        "",
        "| 分区/模态 | 可用 / 行数 |",
        "| --- | --- |",
    ]
    for key in sorted(avail):
        lines.append(f"| {key} | {avail[key]['n_available']} / {avail[key]['n_rows']} |")
    lines += [
        "",
        "## 独立交付任务",
        "",
        "本节三个任务都在 holdout 上评分，训练口径各自不同：**8-score 回归**在"
        f"冻结 train 折（{split_sizes['train']} 行）上训练；**row-QC 敏感性**与"
        f"**Deployable 消融**在与交付模型相同的选择池"
        f"（{_fmt(deliverable['refit_rows'], 0)} 行）上训练。它们各自内部两臂同口径，"
        "用来回答「该组件/该特征集是否改变结果」；回归头训练的模型与交付模型不同，"
        "两个 holdout 数字不能直接相减。",
        "",
        f"- **8-score PUBLISHED 回归**（独立 manifest，固定配置）：逐维 MAE 与"
        f"训练均值预测器参照见 `summary.json`（regression_task）；"
        f"manifest `{regression['manifest_sha256'][:16]}…`。"
        "该任务包含 QC 异常行、未做敏感性对照，属已知限制。"
        + _regression_verdict(regression),
        f"- **row-QC 敏感性**（选择池剔除 {qc['excluded_subject_rows']} 后重训，"
        f"冠军同配置，{_fmt(qc.get('fit_rows_without'), 0)} 行 vs 交付模型 "
        f"{_fmt(deliverable['refit_rows'], 0)} 行）：剔除后模型 holdout subject MAE "
        f"{_fmt(qc.get('holdout_subject_mae_without'), 4)} vs 含 QC 行训练的冠军 "
        f"{_fmt(qc.get('holdout_subject_mae_champion_on_reduced'), 4)}。"
        "两侧都只在 holdout 上评分（同一分区、同一分母），差异只来自训练侧。",
        f"- **Deployable（Body/Vitals-only）消融**（冠军同超参、同一选择池 "
        f"{_fmt(dep.get('fit_rows'), 0)} 行，仅换特征集）："
        f"holdout subject MAE {_fmt(dep['holdout_subject_mae'], 4)}，"
        f"holdout macro-F1 {_fmt((dep.get('f1') or {}).get('holdout', {}).get('macro_f1'))}。"
        "单 seed、holdout 42 人、无区间估计——只作方向参考，不构成特征贡献的显著证据。",
        "",
        "## 证据状态与限制",
        "",
        agent_status_line(ar),
        "- 标签为 RULE 派生类别（规则阈值 ≥40=YES；TENDENCY/NO=0；平和维 ABSTAIN）——"
        "不是作者发布的分类标签，也不是临床诊断真值。注意 CCMQ 分数粒度使有效切点"
        "随各维题数漂移（6 题维 ≈41.7、7 题维 ≈42.9、8 题维 ≈40.6）。",
        "- 列映射为 FINGERPRINT_CROSS_CHECKED（论文定义表付费墙内）；"
        f"QC 异常行 {qc['excluded_subject_rows']} **包含在**主分析与全部基线中，"
        "只在 row-QC 敏感性分析里被剔除；QC 判定仅基于计分规则指纹，与标签、"
        "模型表现和 holdout 均无关，不构成 holdout 信息泄漏。",
        f"- 受试者身份 = 数据行序（{summary.get('subject_identity', 'UNVERIFIED')}）——"
        "数据集无受试者 ID 列，无法独立核验每行确为不同受试者。",
        f"- 冠军由 {ar['rounds_completed']} 轮候选按交叉验证选出（每轮都在上一轮冠军"
        f"基础上改动，属爬山；winner's curse 仍然存在，只是从 41 人单划分换成 k 折均值"
        f"加余量门槛）；消融为单 seed 固定配置；holdout 仅 {split_sizes['holdout']} 人，"
        "小数位差异属噪声，未做置信区间重采样。",
        "- holdout 解锁记录（逐项）：split seed 20261030 在 v1、v2 解锁；"
        "20261101 在 v3 前置报告解锁；20261102 在 v3 报告解锁（另有一次中止运行"
        "未产出报告）；20261103 在 v4 报告解锁；20261104 在 v5 报告解锁；"
        "20261105 在 v6 报告解锁（该轮的选择指标后来被否决，但其 holdout 已消耗）。"
        f"本报告使用全新冻结划分（manifest `{summary['manifest_sha256'][:12]}…`，"
        f"seed {manifest['split']['seed']}），其 holdout 仅本次解锁一次；"
        "同一报告的 8-score 回归、部署消融、row-QC 敏感性都在同一 holdout 上评估，"
        "属同一次解锁事件，不额外增加信息。如再次重评估须再换划分。",
        "- 溯源强度：`proposer` / `context_sha256` / `proposal_sha256` 由驱动方"
        "（操作者）写入，属于**自报归属**——它证明「提交的提案与它声称依据的研究状态」"
        "一致」，不证明「该 agent 真的读取了上下文」或「该 agent 真的生成了它」。"
        + _ordering_line(ar)
        + "要升级为可证明的 agent 驱动，需要 provider 端 request/response id 或签名。",
        "- 类别支持不均衡（见 chart-support.png）；结果仅代表研究口径，不可临床使用。",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    card = [
        f"# TCM Constitution Model Card（#50 交付 {summary.get('report_version', 'v2')}）",
        "",
        f"- model_type：`tcm_constitution_v1`；champion run：`{ar['champion']['run_id']}`",
        f"- 权重：`{ar['champion']['artifact_dir']}/weights.pt`（SHA-256 "
        f"`{(ar['champion']['artifact_sha256'] or '')}`）",
        "- 输入契约：`" + real_input_spec(manifest).version + "`（"
        + ", ".join(f"{k}({v['dim']})" for k, v in manifest["modalities"].items())
        + "；train-fold 标准化，退化列置零，缺失模态行 availability-masked）",
        "- 输出：9 维独立 sigmoid（`category_probs`）；平和维永远 ABSTAIN",
        f"- 训练数据：Mendeley cxtph3tjsg/1（CC BY 4.0）共 {records_count} 名受试者，"
        f"按 70/15/15 行级划分；最终模型在选择池（train {split_sizes['train']} + "
        f"dev {split_sizes['dev']}）上重训，holdout {split_sizes['holdout']} 人从不参与"
        "拟合或选择，"
        f"RULE 派生二值标签（`{summary['label_version']}`）",
        "- 适用人群：与研究样本同源的成年体检人群；不适用于临床诊断、无按摩场景外的任何干预决策",
        "- 已知偏差：类别支持不均衡；体型代理变量（BMI）基线较强；"
        "冠军在 dev 上选出（winner's curse）；单 seed",
        "- 禁止：临床诊断、GB/T 46939—2025 语义混用、LLM_SILVER 当真值",
    ]
    (out / "model-card.md").write_text("\n".join(card) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", default="artifacts/tcm-50")
    parser.add_argument("--label", default="v3",
                        help="report version label (goes into titles and the holdout log)")
    parser.add_argument("--champion-dir", default=None,
                        help="deliverable checkpoint directory (research-finalize "
                             "output); defaults to the champion round's artifact")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = load_real_manifest(args.manifest)
    records = load_mendeley_scores(args.dataset)

    store = RunManifestStore(Path(args.runs_dir) / MANIFEST_FILE)
    champion = store.champion()
    if not champion:
        raise SystemExit("no AutoResearch champion found; run the research loop first")
    # fold-scored rounds keep no per-round model; the deliverable comes from
    # `research-finalize` (one fit on the selection pool)
    champion_dir = args.champion_dir or champion.get("artifact_dir")
    if not champion_dir:
        raise SystemExit(
            "the champion round has no artifact directory (k-fold selection): "
            "run `runtime.cli research-finalize` and pass --champion-dir"
        )
    model, state, metadata = load_training_checkpoint(champion_dir)
    assert_champion_matches_manifest(state, manifest)
    entries = store.load()
    champion_cfg = dict(metadata.get("config", {}))
    champion_trainer = dict(state["trainer_config"])
    champion_seed = int(champion_trainer.get("seed", 0))
    champion_batch = int(state.get("train_batch_size", 32))
    refit_partition = state.get("refit_partition")
    # the deliverable was refit on train+dev, so the frozen dev partition is
    # inside its training data. Score it, and every model it is compared
    # against, on the selection pool: same 232 fit rows, same holdout, and the
    # trusted evaluator's no-overlap rule keeps dev out of the scoring path.
    from data.real_split import pool_manifest

    pool = pool_manifest(
        manifest, records,
        notes="report-time view: the selection pool the deliverable was fitted on",
    ).data
    in_sample_dev = refit_partition == "train+dev"
    deliverable = {
        "dir": str(champion_dir),
        "weights_sha256": _file_sha256(Path(champion_dir) / "weights.pt"),
        "refit_partition": refit_partition,
        "refit_rows": state.get("refit_rows"),
        "selection_parent_manifest_sha256": state.get(
            "selection_parent_manifest_sha256"
        ),
        "dev_metrics_are_in_sample": in_sample_dev,
    }

    neural_eval = evaluate_neural(model, pool, records)
    baselines = evaluate_baselines(pool, records)
    regression = regression_task(manifest, records, out)
    qc = row_qc_sensitivity(pool, records, out, champion_cfg, champion_trainer,
                            champion_model=model, seed=champion_seed,
                            train_batch_size=champion_batch)
    qc["holdout_subject_mae_with"] = neural_eval["holdout"]["subject_mae"]
    deployable = deployable_ablation(manifest, records, out, champion_cfg,
                                     champion_trainer, seed=champion_seed,
                                     train_batch_size=champion_batch)
    seeds = seed_robustness(pool, records, out, champion_cfg, champion_trainer,
                            train_batch_size=champion_batch)

    chart_rounds(entries, out)
    chart_per_dim_f1(baselines, neural_eval, out)
    chart_support(neural_eval, out)
    chart_regression(regression, out)

    per_round = [
        {**{k: e.get(k) for k in ("run_id", "intent", "decision", "status",
                                  "artifact_sha256", "agent_status",
                                  "proposer", "context_sha256",
                                  "proposal_sha256", "proposal_raw_sha256")},
         # the exact patch the agent asked for: without it a reader can see that
         # something was proposed but not what (audit A-01)
         "config_patch": e.get("config_patch"),
         "effective_config": e.get("effective_config"),
         "primary_value": (e.get("metrics") or {}).get("primary_value"),
         "primary_std": e.get("primary_std"),
         "selection": e.get("selection")}
        for e in entries
    ]
    split_sizes = {p: len(manifest["split"][p]) for p in ("train", "dev", "holdout")}
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "report_version": args.label,
        "manifest_sha256": manifest["manifest_sha256"],
        "dataset_sha256": manifest["dataset_sha256"],
        "label_version": manifest["label_version"],
        "standard_version": manifest["standard_version"],
        "split_sizes": split_sizes,
        "split_seed": manifest["split"]["seed"],
        "split_fractions": manifest["split"]["fractions"],
        "records_total": len(records),
        "subjects_in_manifest": sum(split_sizes.values()),
        "subject_identity": (
            "row_order (UNVERIFIED: dataset has no subject ID column; "
            "identity assumed from row position)"
        ),
        "primary_metric": {
            "name": "category_probs/macro_f1",
            "direction": "max",
            "why": "the deliverable metric (nine-dim classification at 0.5, "
                   "macro over the fixed observed-dimension denominator). "
                   "subject_mae is minimised by the median (=0 for a binary "
                   "label under 50% prevalence) and subject_rmse by the "
                   "base-rate constant, so neither can select a classifier",
        },
        "metric_params": {
            "classification_threshold": 0.5,
            "label_rule": manifest["label_version"],
            "macro_f1": "mean per-dim F1 over observed dims; undefined F1 counts as 0",
            "subject_mae": "mean over subjects of per-subject mean |p - label| "
                           "over observed dims (diagnostic, NOT the selection metric)",
            "subject_rmse": "sqrt(mean over subjects of per-subject MSE); Brier "
                            "score up to a monotone root — the selection metric",
            "partitions": split_sizes,
        },
        "deliverable": deliverable,
        "champion_provenance": {
            "run_id": champion["run_id"],
            "weights_sha256": deliverable["weights_sha256"],
            "round_artifact_sha256": champion.get("artifact_sha256"),
            "manifest_sha256": manifest["manifest_sha256"],
            "dataset_sha256": manifest["dataset_sha256"],
            "label_version": manifest["label_version"],
            "train_batch_size": champion_batch,
            "weight_epoch": state.get("weight_epoch"),
            "trainer_config": champion_trainer,
            "model_config": champion_cfg,
        },
        "availability": availability_rates(manifest, records),
        "autoresearch": {
            "proposal_ordering": check_proposal_ordering(Path(args.runs_dir)),
            "selection": {
                "folds": sorted({
                    (e.get("selection") or {}).get("folds", 1) for e in entries
                }),
                "margin_factor": next(
                    (e["selection"]["margin_factor"] for e in entries
                     if e.get("selection") and "margin_factor" in e["selection"]),
                    None,
                ),
                "pool_rows": split_sizes["train"] + split_sizes["dev"],
                "holdout_never_used_for_selection": True,
            },
            "rounds_completed": len(entries),
            "champion": {
                "run_id": champion["run_id"],
                "artifact_dir": deliverable["dir"],
                "artifact_sha256": deliverable["weights_sha256"],
                "round_artifact_sha256": champion.get("artifact_sha256"),
                "best_so_far": champion.get("best_so_far"),
                "refit_partition": refit_partition,
            },
            "per_round": per_round,
        },
        "neural_evaluation": neural_eval,
        "baselines": baselines,
        "regression_task": regression,
        "row_qc_sensitivity": qc,
        "deployable_ablation": deployable,
        "seed_robustness": seeds,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    write_reports(out, summary, manifest, len(records))
    print(json.dumps({
        "out": str(out),
        "champion_run": champion["run_id"],
        "champion_dev_primary_value": champion.get("best_so_far"),
        "dev_macro_f1": neural_eval["dev"]["macro_f1"],
        "holdout_macro_f1": neural_eval["holdout"]["macro_f1"],
        "holdout_subject_mae": neural_eval["holdout"]["subject_mae"],
        "baseline_holdout": {
            k: {"macro_f1": v["holdout"]["macro_f1"],
                "subject_mae": v["holdout"].get("subject_mae")}
            for k, v in baselines.items() if isinstance(v, dict) and "holdout" in v
        },
        "row_qc_without": qc["holdout_subject_mae_without"],
        "deployable_holdout_mae": deployable["holdout_subject_mae"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
