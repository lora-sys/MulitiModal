"""THE SYNDROME-TASK EDITABLE FILE (config task="syndrome", issue #53).

Same contract as `research/candidate.py` (read its header for the two
mechanical rules), for TCM-SD syndrome classification:

1. `build_model(fold)` must return a module whose `input_spec` equals
   `fold.input_spec` and whose `forward(batch)` returns a ModelOutput with a
   `syndrome_probs` entry of shape [B, n_classes] (softmax over the frozen
   taxonomy). `fold.dims["targets"]` is n_classes.
2. `train(model, fold, budget_seconds, seed)` must return within
   `budget_seconds`.

Input contract: each case is ONE tabular modality whose dim is read from
`fold.input_spec` (see research/task_sd.py — the featurizer/scaler machinery
runs outside the loop; this file never sees raw text). Two manifest
modalities feed the same contract:

* "float_groups": frozen inline text features, train-fold z-scored (#29);
* "hashed_text": L2-normalized hashed char 1-3 gram features (2^16 buckets),
  rebuilt per batch by the provider from the manifest-frozen stateless
  HashingVectorizer — no scaler statistics exist or apply.

Either way the input is a plain float tensor of the declared dim, so the
model stays dim-agnostic and self-contained. Each record also carries a real
subject `user_id` (R1 F3): the frozen manifest split is subject-level — all
cases of one user land in one partition — delegated to
`data.tcm_sd.subject_partitions`; the split seed and ratios come from that
data-layer contract, not from this task. The score is computed by the frozen
`SyndromeTrustedEvaluator` after this file returns; this file never sees it.

This v0 skeleton is the smallest self-contained model that survives the full
isolation chain (train_worker -> state_dict + frozen-architecture export ->
evaluator rebuild). It exists so the engine plumbing is proven and the first
real candidate has something to beat; it makes no claim about TCM-SD
performance (SMOKE_ONLY until real data flows through #53 scope 1).
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.batch import ModelOutput
from research.frozen import TIME_BUDGET_SECONDS, FoldData


# ==============================================================================
# PIPELINE CONFIGURATION (Candidate Agent knobs)
# ==============================================================================
ENCODER_TYPE = "mlp"      # "mlp" (2-layer non-linear) or "linear" (1-layer projection)
HEAD_TYPE = "linear"       # "linear" (direct softmax) or "mlp" (2-layer MLP head)
WIDTH = 32                # per-modality encoder width
FUSION = 32               # multimodal fusion bottleneck width
DROPOUT = 0.0             # dropout probability (0.0 means disabled)
LEARNING_RATE = 3e-3      # optimizer learning rate
WEIGHT_DECAY = 0.0        # L2 regularization weight decay


def _make_encoder(in_dim: int, width: int, encoder_type: str = "mlp", dropout: float = 0.0) -> nn.Sequential:
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


def _make_head(in_dim: int, n_targets: int, head_type: str = "linear", dropout: float = 0.0) -> nn.Module:
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


class SyndromeMLP(nn.Module):
    """Modular feature + model Pipeline for TCM-SD syndrome classification."""

    def __init__(
        self,
        input_spec,
        width: int = WIDTH,
        fusion: int = FUSION,
        n_targets: int = 2,
        encoder_type: str = ENCODER_TYPE,
        head_type: str = HEAD_TYPE,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.model_type = "tcm_syndrome_v1"
        self.input_spec = input_spec
        self.n_targets = n_targets
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
            # the availability flag travels with the values as a feature;
            # unavailable rows are already zeroed by the frozen provider
            flag = batch.availability[name].float().unsqueeze(-1)
            parts.append(encoder(torch.cat([x, flag], dim=-1)))
        hidden = self.fusion(torch.cat(parts, dim=-1))
        logits = self.head(hidden)
        return ModelOutput(
            predictions={"syndrome_probs": torch.softmax(logits, dim=-1)},
            auxiliary={"syndrome_logits": logits},
        )


SyndromePipeline = SyndromeMLP


def build_model(fold: FoldData):
    return SyndromePipeline(
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
    """Fit in place for a FIXED wall clock, on labeled cases only.

    Unlabeled cases (mask False) carry no gradient — ABSTAIN is never
    trained into a guess. Time-driven loop, same as the constitution
    baseline: "train longer" is not an available win.
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
            logits = out.auxiliary["syndrome_logits"]
            mask = batch.target_mask["syndrome_labels"].squeeze(-1)
            labels = batch.targets["syndrome_labels"].squeeze(-1)[mask].long()
            if labels.numel() == 0:
                continue
            loss = F.cross_entropy(logits[mask], labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if time.monotonic() >= deadline:
                epoch_done = True
                break
        if epoch_done:
            break
    model.eval()
