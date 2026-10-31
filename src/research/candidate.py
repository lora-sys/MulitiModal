"""THE ONLY FILE THE AUTORESEARCH AGENT MAY EDIT.

Everything in this file is fair game: architecture, loss, optimizer, schedule,
batch handling, pooling, regularisation, calibration — the whole training loop.
Everything outside it is frozen (see `research/frozen.py`): data, split, scaler,
labels, evaluator, metric, and the training time budget.

Two rules the loop enforces mechanically:

1. `build_model(fold)` must return a module whose `input_spec` equals
   `fold.input_spec` and whose `forward(batch)` returns a ModelOutput with a
   `category_probs` entry of shape [B, 9]. Nothing else about the model is
   checked — that is the freedom.
2. `train(model, fold, budget_seconds, seed)` must return within
   `budget_seconds`. Overrunning is a crash, not a slow win.

The score is computed by the frozen evaluator on the fold's held-out subjects
AFTER this file returns, and this file never sees it.

Baseline (round 0) is the configuration the previous, config-only loop shipped
as its champion: a 32-wide per-modality MLP, concatenation fusion, masked
binary cross-entropy, plain Adam, 40 equivalent epochs. It exists so the first
KEEP has something to beat, and so a reader can see exactly what the search
started from.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn

from core.batch import ModelOutput
from core.losses import masked_multilabel_bce_loss
from research.frozen import TIME_BUDGET_SECONDS, FoldData


class FusionMLP(nn.Module):
    """Baseline: one MLP per modality, concatenated, one linear head."""

    def __init__(self, input_spec, width: int = 32, fusion: int = 32, n_targets: int = 9):
        super().__init__()
        self.model_type = "tcm_constitution_v1"
        self.input_spec = input_spec
        self.encoders = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(spec.dim + 1, width), nn.GELU(), nn.Linear(width, width)
            )
            for name, spec in ((m.name, m) for m in input_spec.modalities)
        })
        self.fusion = nn.Sequential(
            nn.Linear(width * len(input_spec.modalities), fusion), nn.GELU()
        )
        self.head = nn.Linear(fusion, n_targets)

    def forward(self, batch) -> ModelOutput:
        parts = []
        for name, encoder in self.encoders.items():
            x = batch.features[name]
            # the availability flag travels with the values as a feature, and
            # missing rows are already zeroed by the frozen provider
            flag = batch.availability[name].float().unsqueeze(-1)
            parts.append(encoder(torch.cat([x, flag], dim=-1)))
        hidden = self.fusion(torch.cat(parts, dim=-1))
        logits = self.head(hidden)
        return ModelOutput(
            predictions={"category_probs": torch.sigmoid(logits)},
            auxiliary={"category_logits": logits},
        )


WIDTH = 32
FUSION = 32
LEARNING_RATE = 3e-3
BATCH_PATIENCE = 0  # unused by the baseline; kept as a named knob to edit


def build_model(fold: FoldData):
    return FusionMLP(fold.input_spec, WIDTH, FUSION, fold.dims["targets"])


def train(model, fold: FoldData, budget_seconds: float = TIME_BUDGET_SECONDS,
          seed: int = 0) -> None:
    """Fit in place for a FIXED wall clock, not a fixed number of epochs.

    The loop is driven by elapsed time: it stops when the budget is gone,
    wherever the schedule happens to be. That is what makes "train longer"
    unavailable as an improvement.
    """
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    deadline = time.monotonic() + budget_seconds
    model.train()
    while time.monotonic() < deadline:
        epoch_done = False
        for item in fold.train:
            batch = item.batch
            out = model(batch)
            loss = masked_multilabel_bce_loss(
                out.auxiliary["category_logits"],
                batch.targets["category_labels"],
                batch.target_mask["category_labels"],
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if time.monotonic() >= deadline:
                epoch_done = True
                break
        if epoch_done:
            break
    model.eval()