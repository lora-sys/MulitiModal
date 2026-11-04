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

# ==============================================================================
# PIPELINE CONFIGURATION (Candidate Agent knobs)
# ==============================================================================
ENCODER_TYPE = "mlp"      # "mlp" (2-layer non-linear) or "linear" (1-layer projection)
HEAD_TYPE = "linear"       # "linear" (direct sigmoid) or "mlp" (2-layer MLP head)
WIDTH = 32                # per-modality encoder width
FUSION = 32               # multimodal fusion bottleneck width
DROPOUT = 0.0             # dropout probability (0.0 means disabled)
LEARNING_RATE = 3e-3      # optimizer learning rate
WEIGHT_DECAY = 0.0        # L2 regularization weight decay
BATCH_PATIENCE = 0        # unused by the baseline; kept as a named knob to edit


def _make_encoder(
    in_dim: int, width: int, encoder_type: str = "mlp", dropout: float = 0.0
) -> nn.Sequential:
    if encoder_type == "linear":
        layers: list[nn.Module] = [nn.Linear(in_dim, width)]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)
    layers = [
        nn.Linear(in_dim, width),
        nn.GELU(),
    ]
    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(width, width))
    return nn.Sequential(*layers)


def _make_head(
    in_dim: int, n_targets: int, head_type: str = "linear", dropout: float = 0.0
) -> nn.Module:
    if head_type == "mlp":
        layers: list[nn.Module] = [
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(in_dim, n_targets))
        return nn.Sequential(*layers)
    return nn.Linear(in_dim, n_targets)


class FusionMLP(nn.Module):
    """Modular feature + model Pipeline for TCM constitution classification."""

    def __init__(
        self,
        input_spec,
        width: int = WIDTH,
        fusion: int = FUSION,
        n_targets: int = 9,
        encoder_type: str = ENCODER_TYPE,
        head_type: str = HEAD_TYPE,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.model_type = "tcm_constitution_v1"
        self.input_spec = input_spec
        self.encoders = nn.ModuleDict({
            name: _make_encoder(spec.dim + 1, width, encoder_type=encoder_type, dropout=dropout)
            for name, spec in ((m.name, m) for m in input_spec.modalities)
        })
        self.fusion = nn.Sequential(
            nn.Linear(width * len(input_spec.modalities), fusion), nn.GELU()
        )
        self.head = _make_head(fusion, n_targets, head_type=head_type, dropout=dropout)

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


ConstitutionPipeline = FusionMLP


def build_model(fold: FoldData):
    return ConstitutionPipeline(
        fold.input_spec,
        width=WIDTH,
        fusion=FUSION,
        n_targets=fold.dims["targets"],
        encoder_type=ENCODER_TYPE,
        head_type=HEAD_TYPE,
        dropout=DROPOUT,
    )


def train(model, fold: FoldData, budget_seconds: float = TIME_BUDGET_SECONDS,
          seed: int = 0) -> None:
    """Fit in place for a FIXED wall clock, not a fixed number of epochs.

    The loop is driven by elapsed time: it stops when the budget is gone,
    wherever the schedule happens to be. That is what makes "train longer"
    unavailable as an improvement.
    """
    deadline = time.monotonic() + budget_seconds
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
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