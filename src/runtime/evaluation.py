"""Evaluator (#49): masked metrics over a BatchProvider, versioned.

Metric values are computed only on observed (target_mask=true) entries and
are SMOKE_ONLY in this phase. Category metrics are computed only when the
labels actually exist in the batch — never assumed available (#39/#49).
"""

from __future__ import annotations

import torch

from core.batch import mask_name
from core.registry import batch_predict

from .batches import BatchProvider

METRIC_VERSION = "masked-metrics-v0"


class Evaluator:
    """Aggregates per-target masked MSE/MAE plus the model's own eval_loss."""

    def __init__(self, metric_version: str = METRIC_VERSION) -> None:
        self.metric_version = metric_version

    def evaluate(
        self,
        model: torch.nn.Module,
        provider: BatchProvider,
        device: torch.device | str = "cpu",
    ) -> dict:
        model.eval()
        device = torch.device(device) if isinstance(device, str) else device
        prediction_targets = getattr(model, "prediction_targets", {})
        sums: dict[str, float] = {
            "eval_loss": 0.0,
            "n_batches": 0.0,
        }
        for item in provider:
            batch = item.batch.to(device)
            loss, _ = model.compute_loss(batch)
            sums["eval_loss"] += float(loss.detach())
            sums["n_batches"] += 1

            out = batch_predict(model, batch, device=device)
            for name, pred in out.predictions.items():
                target_name = prediction_targets.get(name, name)
                if batch.targets is None or target_name not in batch.targets:
                    continue
                target = batch.targets[target_name]
                mask = mask_name(batch.target_mask, target_name)
                if not bool(mask.any()):
                    sums[f"{name}/n"] = sums.get(f"{name}/n", 0.0)
                    continue
                diff = (pred - target)[mask].detach()
                sums[f"{name}/mse"] = sums.get(f"{name}/mse", 0.0) + float(
                    (diff**2).sum()
                )
                sums[f"{name}/mae"] = sums.get(f"{name}/mae", 0.0) + float(
                    diff.abs().sum()
                )
                sums[f"{name}/n"] = sums.get(f"{name}/n", 0.0) + int(mask.sum())
                if name == "category_probs":
                    hits = ((pred > 0.5) == (target > 0.5))[mask].float()
                    sums[f"{name}/acc_hits"] = sums.get(f"{name}/acc_hits", 0.0) + float(
                        hits.sum()
                    )

        metrics: dict[str, float | str] = {"metric_version": self.metric_version}
        metrics["n_batches"] = int(sums["n_batches"])
        metrics["eval_loss"] = sums["eval_loss"] / max(sums["n_batches"], 1.0)
        for name in sorted(set(k.rsplit("/", 1)[0] for k in sums if "/" in k)):
            n = sums.get(f"{name}/n", 0.0)
            if n:
                metrics[f"{name}/mse"] = sums[f"{name}/mse"] / n
                metrics[f"{name}/mae"] = sums[f"{name}/mae"] / n
                if f"{name}/acc_hits" in sums:
                    metrics[f"{name}/acc"] = sums[f"{name}/acc_hits"] / n
            metrics[f"{name}/n"] = n
        return metrics
