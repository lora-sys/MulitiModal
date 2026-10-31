"""TCM trusted evaluator (#40, architecture-first): frozen manifest, label-mask
metrics, subject-level MAE/RMSE, leakage rejection, SMOKE_ONLY reporting.

Independent of the Trainer: it consumes any model-like object (forward ->
ModelOutput) plus a BatchProvider, and refuses inconsistent standard/spec
versions, unknown prediction keys, and subject crossing between partitions.
Real-data science validation stays behind #37/#38/#47/#50; with synthetic
inputs every report is SMOKE_ONLY and real metrics are declared not
computable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from core.batch import mask_name
from core.registry import batch_predict

from .batches import BatchProvider

EVAL_METRIC_VERSION = "tcm-eval-v0"
# The selection metric is the deliverable metric: nine-dim classification at a
# 0.5 threshold, macro-F1 over the fixed observed-dimension denominator.
#
# Two probability-loss surrogates were tried first and both mis-rank models for
# THIS task:
#   * subject_mae (L1) is minimised by the median, which is 0 for a binary label
#     whose positive rate is below 0.5 — a hill climb on it cannot beat an
#     all-zero predictor (measured: dummy 0.2292 held the holdout lead);
#   * subject_rmse (Brier) is proper, but its optimum for a model with no usable
#     signal is the base-rate constant, so the loop converged on a model with
#     dev macro-F1 0.021 while a plain logistic regression scored 0.308.
# A calibrated constant is the honest answer to "how confident are you on this
# subject", but it is not a classifier. Optimising the reported metric keeps the
# search aligned with what the model is for; Brier/MAE stay as diagnostics.
PRIMARY_METRIC = "category_probs/macro_f1"
PRIMARY_DIRECTION = "max"
KNOWN_PREDICTION_KEYS = frozenset({"category_probs", "constitution_scores"})
PARTITIONS = ("train", "dev", "holdout")


@dataclass(frozen=True)
class EvalManifest:
    """Frozen partition lists + contract versions; immutable by design (#40)."""

    version: str
    input_spec_version: str
    standard_version: str
    label_tier: str
    train_subjects: tuple[str, ...]
    dev_subjects: tuple[str, ...]
    holdout_subjects: tuple[str, ...]

    def __post_init__(self) -> None:
        sets = {
            "train": set(self.train_subjects),
            "dev": set(self.dev_subjects),
            "holdout": set(self.holdout_subjects),
        }
        for a in PARTITIONS:
            for b in PARTITIONS:
                if a < b and sets[a] & sets[b]:
                    raise ValueError(
                        f"manifest leakage: subject present in both {a!r} and {b!r}"
                    )

    def subjects(self, partition: str) -> tuple[str, ...]:
        if partition not in PARTITIONS:
            raise ValueError(f"unknown partition {partition!r}; expected {PARTITIONS}")
        return getattr(self, f"{partition}_subjects")

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "input_spec_version": self.input_spec_version,
            "standard_version": self.standard_version,
            "label_tier": self.label_tier,
            "n_subjects": {p: len(self.subjects(p)) for p in PARTITIONS},
        }


@dataclass(frozen=True)
class TCMReport:
    """Immutable evaluation report; mutation attempts raise (read-only gate).

    the primary value is macro-F1 over the fixed observed-dimension
    denominator (see PRIMARY_METRIC); subject_mae/subject_rmse are reported as
    diagnostics. The probability losses are None
    when no label is observable (fail-closed: an empty observation must never
    look like a perfect score, #45 audit finding)."""

    metric_version: str
    primary_metric: str
    partition: str
    standard_version: str
    label_tier: str
    smoke_only: bool
    n_subjects: int
    n_labeled: int
    subject_mae: float | None
    subject_rmse: float | None
    macro_f1: float | None = None
    n_undefined_f1: int = 0
    per_dim: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    not_computable: tuple = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "metric_version": self.metric_version,
            "primary_metric": self.primary_metric,
            "primary_value": self.macro_f1,
            "partition": self.partition,
            "standard_version": self.standard_version,
            "label_tier": self.label_tier,
            "smoke_only": self.smoke_only,
            "n_subjects": self.n_subjects,
            "n_labeled": self.n_labeled,
            "subject_mae": self.subject_mae,
            "subject_rmse": self.subject_rmse,
            "macro_f1": self.macro_f1,
            "n_undefined_f1": self.n_undefined_f1,
            "per_dim": self.per_dim,
            "provenance": self.provenance,
            "not_computable": list(self.not_computable),
        }


def compute_subject_level_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    subject_ids: list[str],
) -> dict:
    """Hand-checkable core: subject-level MAE/RMSE over observed entries only.

    Per subject: pooled mean abs diff (and mean sq diff) over ALL of that
    subject's observed entries across rows; subjects with nothing observed
    are excluded (never zero-filled). Aggregation matches
    TCMTrustedEvaluator.evaluate — same subject, same number.
    """
    if pred.shape != target.shape or pred.shape != mask.shape:
        raise ValueError("pred/target/mask shape mismatch")
    if len(subject_ids) != pred.shape[0]:
        raise ValueError("subject_ids length must match batch rows")
    subject_sq: dict[str, float] = {}
    subject_abs: dict[str, float] = {}
    subject_obs: dict[str, int] = {}
    for i, subj in enumerate(subject_ids):
        obs = mask[i]
        if not bool(obs.any()):
            continue
        diffs = (pred[i] - target[i])[obs]
        subject_sq[subj] = subject_sq.get(subj, 0.0) + float((diffs**2).sum())
        subject_abs[subj] = subject_abs.get(subj, 0.0) + float(diffs.abs().sum())
        subject_obs[subj] = subject_obs.get(subj, 0) + int(obs.sum())
    n_labeled = len(subject_abs)
    if n_labeled == 0:
        # fail-closed: an empty observation must never look like a perfect
        # score (same semantics as TCMTrustedEvaluator, #45 audit finding)
        return {"n_subjects": len(set(subject_ids)), "n_labeled": 0,
                "subject_mae": None, "subject_rmse": None}
    per_subject_abs = {s: subject_abs[s] / subject_obs[s] for s in subject_abs}
    per_subject_sq = {s: subject_sq[s] / subject_obs[s] for s in subject_sq}
    mae = sum(per_subject_abs.values()) / n_labeled
    rmse = (sum(per_subject_sq.values()) / n_labeled) ** 0.5
    return {
        "n_subjects": len(set(subject_ids)),
        "n_labeled": n_labeled,
        "subject_mae": mae,
        "subject_rmse": rmse,
    }


def compute_per_dim_stats(
    target: torch.Tensor, mask: torch.Tensor
) -> dict[int, dict]:
    """Per-dimension observed counts, missing counts, MAE inputs and positive rate."""
    n_rows = target.shape[0]
    stats: dict[int, dict] = {}
    for d in range(target.shape[1]):
        obs = mask[:, d]
        n = int(obs.sum())
        stats[d] = {
            "n": n,
            "missing": n_rows - n,
            "mean_target": float(target[obs].mean()) if n else None,
            "positive_rate": (
                float((target[obs] > 0.5).float().mean()) if n else None
            ),
        }
    return stats


def _macro_f1_from_tensors(pred_bin, target, mask) -> tuple[float | None, int]:
    """Macro-F1 over the FIXED observed-dimension denominator: an undefined F1
    counts as 0 so the denominator cannot drift between candidates and reward a
    model for never predicting a rare dimension."""
    if pred_bin is None or target is None or mask is None:
        return None, 0
    mask = mask.bool()
    scores: list[float] = []
    undefined = 0
    for d in range(target.shape[1]):
        obs = mask[:, d]
        if not bool(obs.any()):
            continue
        t = target[obs, d] > 0.5
        p = pred_bin[obs, d] > 0.5
        tp = int((t & p).sum())
        fp = int((~t & p).sum())
        fn = int((t & ~p).sum())
        denom = 2 * tp + fp + fn
        if denom == 0:
            undefined += 1
            scores.append(0.0)
        else:
            scores.append(2 * tp / denom)
    if not scores:
        return None, 0
    return sum(scores) / len(scores), undefined


class TCMTrustedEvaluator:
    """Read-only scorer over a frozen manifest; the single primary metric is
    category_probs/macro_f1 (the deliverable metric — see PRIMARY_METRIC for
    why neither MAE nor Brier is used for selection). AutoResearch must
    optimize against this and never touch holdout labels (#40/#45)."""

    def __init__(
        self,
        manifest: EvalManifest,
        metric_version: str = EVAL_METRIC_VERSION,
    ) -> None:
        self.manifest = manifest
        self.metric_version = metric_version

    def evaluate(
        self,
        model,
        provider: BatchProvider,
        partition: str = "dev",
    ) -> TCMReport:
        allowed = set(self.manifest.subjects(partition))
        if not allowed:
            raise ValueError(f"manifest partition {partition!r} is empty")
        if not set(provider.subject_ids) <= allowed:
            crossing = sorted(set(provider.subject_ids) - allowed)[:3]
            raise ValueError(
                f"provider subjects outside manifest partition {partition!r} "
                f"(sample crossing): {crossing}"
            )
        spec = getattr(model, "input_spec", None)
        if spec is None or spec.version != self.manifest.input_spec_version:
            raise ValueError(
                "standard/spec version mismatch: model input spec "
                f"{getattr(spec, 'version', None)!r} != manifest "
                f"{self.manifest.input_spec_version!r}"
            )

        all_sq: dict[str, list[float]] = {}
        all_abs: dict[str, list[float]] = {}
        all_pred: list = []
        target_tensor = mask_tensor = None
        subject_ids: list[str] = []
        model_type = type(model).__name__
        for item in provider:
            batch = item.batch
            out = batch_predict(model, batch)
            unknown = set(out.predictions) - KNOWN_PREDICTION_KEYS
            if unknown:
                raise ValueError(f"unknown prediction keys from model: {sorted(unknown)}")
            if batch.targets is None or "category_labels" not in batch.targets:
                raise ValueError("eval batches must carry category_labels targets")
            pred = out.predictions["category_probs"]
            target = batch.targets["category_labels"]
            mask = mask_name(batch.target_mask, "category_labels")
            subj = item.subject_ids
            if len(subj) != pred.shape[0]:
                raise ValueError("subject_ids length must match batch rows")
            for i, s in enumerate(subj):
                obs = mask[i]
                if not bool(obs.any()):
                    continue
                diffs = (pred[i] - target[i])[obs].detach()
                all_sq.setdefault(s, []).extend((diffs**2).tolist())
                all_abs.setdefault(s, []).extend(diffs.abs().tolist())
            all_pred.append((pred.detach() >= 0.5).to(target.dtype))
            if target_tensor is None:
                target_tensor, mask_tensor = target.detach(), mask.detach()
                subject_ids = list(subj)
            else:
                target_tensor = torch.cat([target_tensor, target.detach()])
                mask_tensor = torch.cat([mask_tensor, mask.detach()])
                subject_ids.extend(subj)
            model_type = getattr(model, "model_type", model_type)

        n_labeled = len(all_abs)
        macro_f1, n_undefined_f1 = (
            _macro_f1_from_tensors(
                torch.cat(all_pred) if all_pred else None,
                target_tensor, mask_tensor,
            )
            if target_tensor is not None
            else (None, 0)
        )
        subject_mae = (
            sum(sum(v) / len(v) for v in all_abs.values()) / n_labeled
            if n_labeled
            else None
        )
        subject_rmse = (
            (sum(sum(v) / len(v) for v in all_sq.values()) / n_labeled) ** 0.5
            if n_labeled
            else None
        )
        smoke_only = self.manifest.label_tier == "SMOKE_ONLY"
        not_computable = []
        if smoke_only:
            not_computable.append(
                "real-data subject-level metrics require #37/#38 labels "
                "passing #47 QC; synthetic run reports SMOKE_ONLY numbers only"
            )
        if not n_labeled:
            not_computable.append(
                f"no observable labels in partition {partition!r}; "
                "primary metric is NOT_COMPUTABLE (fail-closed)"
            )
        per_dim = (
            compute_per_dim_stats(target_tensor, mask_tensor)
            if target_tensor is not None
            else {}
        )
        return TCMReport(
            metric_version=self.metric_version,
            primary_metric=PRIMARY_METRIC,
            partition=partition,
            standard_version=self.manifest.standard_version,
            label_tier=self.manifest.label_tier,
            smoke_only=smoke_only,
            n_subjects=len(set(subject_ids)),
            n_labeled=n_labeled,
            subject_mae=subject_mae,
            subject_rmse=subject_rmse,
            macro_f1=macro_f1,
            n_undefined_f1=n_undefined_f1,
            per_dim=per_dim,
            provenance={
                "model_type": model_type,
                "input_spec_version": self.manifest.input_spec_version,
                "manifest_version": self.manifest.version,
            },
            not_computable=tuple(not_computable),
        )


def make_synthetic_tcm_manifest(
    train_provider: BatchProvider,
    dev_provider: BatchProvider,
    holdout_provider: BatchProvider,
    *,
    version: str = "synthetic-manifest-v0",
) -> EvalManifest:
    """Build a frozen manifest from three disjoint synthetic providers."""
    return EvalManifest(
        version=version,
        input_spec_version="tcm-input-v0",
        standard_version="SYNTHETIC-CONTRACT-V0",
        label_tier="SMOKE_ONLY",
        train_subjects=tuple(train_provider.subject_ids),
        dev_subjects=tuple(dev_provider.subject_ids),
        holdout_subjects=tuple(holdout_provider.subject_ids),
    )
