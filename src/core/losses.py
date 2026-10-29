"""Mask-aware loss contracts (#48/#39).

All losses compute over observed entries only (target_mask=true). An
all-unobserved batch returns a graph-connected zero so training can skip
without NaNs; it never fabricates labels.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _zero_like(pred: torch.Tensor) -> torch.Tensor:
    return pred.sum() * 0.0


def masked_mse_loss(
    pred: torch.Tensor, targets: torch.Tensor, target_mask: torch.Tensor
) -> torch.Tensor:
    observed = target_mask
    if not bool(observed.any()):
        return _zero_like(pred)
    return F.mse_loss(pred[observed], targets[observed])


def masked_mae_loss(
    pred: torch.Tensor, targets: torch.Tensor, target_mask: torch.Tensor
) -> torch.Tensor:
    observed = target_mask
    if not bool(observed.any()):
        return _zero_like(pred)
    return (pred[observed] - targets[observed]).abs().mean()


def masked_multilabel_bce_loss(
    logits: torch.Tensor, targets: torch.Tensor, target_mask: torch.Tensor
) -> torch.Tensor:
    """BCE-with-logits over observed entries only; unobserved dims are never
    counted and never default to negative (#38/#39/#50)."""
    observed = target_mask
    if not bool(observed.any()):
        return _zero_like(logits)
    return F.binary_cross_entropy_with_logits(logits[observed], targets[observed].float())
