"""THE TOY-TASK EDITABLE FILE (config task="toy", issue #61).

This is the example task's editable file, and it exists so the whole
AutoResearch loop — guard, fixed budget, isolation, trusted scoring, dashboard —
can be demonstrated with no TCM data, no GPU and no network. Same two mechanical
rules as `research/candidate.py`:

1. `build_model(fold)` must return a module whose `input_spec` equals
   `fold.input_spec` and whose `forward(batch)` returns a ModelOutput with a
   `toy_probs` entry of shape [B, n_classes] (softmax over the frozen classes).
2. `train(model, fold, budget_seconds, seed)` must return within the budget.

Everything in this file is fair game for the search: architecture, loss,
optimiser, schedule, regularisation. The data, the split, the labels, the
metric, the evaluator and the budget are frozen and live outside it.

Honesty: this task's labels come from a scripted generator, so this file's job is
to prove the plumbing, not to claim any medical result.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.batch import ModelOutput
from research.frozen import TIME_BUDGET_SECONDS, FoldData


class ToyMLP(nn.Module):
    """One encoder per modality, concatenated, one softmax head."""

    def __init__(self, input_spec, width: int = 32, fusion: int = 32,
                 n_targets: int = 3):
        super().__init__()
        self.model_type = "toy_classifier_v1"
        self.input_spec = input_spec
        self.n_targets = n_targets
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
            flag = batch.availability[name].float().unsqueeze(-1)
            parts.append(encoder(torch.cat([x, flag], dim=-1)))
        hidden = self.fusion(torch.cat(parts, dim=-1))
        logits = self.head(hidden)
        return ModelOutput(
            predictions={"toy_probs": torch.softmax(logits, dim=-1)},
            auxiliary={"toy_logits": logits},
        )


WIDTH = 32
FUSION = 32
LEARNING_RATE = 3e-3


def build_model(fold: FoldData):
    return ToyMLP(fold.input_spec, WIDTH, FUSION, fold.dims["targets"])


def train(model, fold: FoldData, budget_seconds: float = TIME_BUDGET_SECONDS,
          seed: int = 0) -> None:
    """Fit in place for a FIXED wall clock; unlabeled rows never contribute."""
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    deadline = time.monotonic() + budget_seconds
    model.train()
    while time.monotonic() < deadline:
        epoch_done = False
        for item in fold.train:
            batch = item.batch
            out = model(batch)
            mask = batch.target_mask["toy_labels"]
            labels = batch.targets["toy_labels"]
            if not bool(mask.any()):
                continue
            loss = F.cross_entropy(out.auxiliary["toy_logits"][mask], labels[mask])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if time.monotonic() >= deadline:
                epoch_done = True
                break
        if epoch_done:
            break
    model.eval()