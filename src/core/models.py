"""TCM and Physio dual-track models on the shared core (#48).

- TCMConstitutionModel: tabular/waveform encoders -> fusion -> [B, 9]
  continuous scores (sigmoid*100, mask-aware) + optional independent
  multi-label category head (BCE contract, disabled by default).
- PhysioBPSIEFRIModel: Pre-only body/physio branches -> fusion -> EFRI
  primary head + optional BPSI auxiliary head. During/Post data can never
  enter the input contract (enforced by ModelBatch.validate).
"""

from __future__ import annotations

import torch
from torch import nn

from .batch import ModelBatch, ModelOutput, mask_name
from .encoders import ConcatFusion, ScoreHead, build_encoder
from .losses import masked_mae_loss, masked_mse_loss, masked_multilabel_bce_loss
from .specs import (
    PHYSIO_TARGET_BPSI_AUX,
    PHYSIO_TARGET_EFRI,
    TCM_TARGET_CATEGORY,
    TCM_TARGET_SCORES,
    InputSpecSet,
    TargetSpec,
    make_physio_input_spec,
    make_tcm_input_spec,
)


class _MultiModalTrunk(nn.Module):
    """Shared encoder+fusion trunk driven entirely by the input spec.

    Modalities absent from a batch contribute exact zero embeddings for those
    rows (explicit mask contract — never imputed inputs, never NaN).
    """

    def __init__(self, spec: InputSpecSet, hidden_dim: int, fusion_dim: int) -> None:
        super().__init__()
        self.spec = spec
        self.hidden_dim = hidden_dim
        self.encoders = nn.ModuleDict(
            {m.name: build_encoder(m, hidden_dim) for m in spec.modalities}
        )
        self.fusion = ConcatFusion(spec.names(), hidden_dim, fusion_dim)

    def forward(self, batch: ModelBatch) -> torch.Tensor:
        embeddings = {
            name: self.encoders[name](
                batch.features[name], batch.availability.get(name)
            )
            for name in self.spec.names()
            if name in batch.features
        }
        if not embeddings:
            raise ValueError("batch has none of the modalities declared by the input spec")
        ref = next(iter(embeddings.values()))
        zeros = torch.zeros(
            batch.batch_size, self.hidden_dim, dtype=ref.dtype, device=ref.device
        )
        filled = {
            name: embeddings.get(name, zeros) for name in self.spec.names()
        }
        return self.fusion(filled)


class TCMConstitutionModel(nn.Module):
    """Nine-dimension constitution model (V0): continuous scores + optional
    multi-label category head. No default mutually-exclusive softmax (#39/#50)."""

    model_type = "tcm_constitution_v0"

    def __init__(
        self,
        input_spec: InputSpecSet,
        hidden_dim: int = 64,
        fusion_dim: int = 64,
        num_scores: int = 9,
        include_category_head: bool = False,
    ) -> None:
        super().__init__()
        self.input_spec = input_spec
        self.target_specs: tuple[TargetSpec, ...] = (
            TCM_TARGET_SCORES,
            *((TCM_TARGET_CATEGORY,) if include_category_head else ()),
        )
        self.num_scores = num_scores
        self.include_category_head = include_category_head
        self.trunk = _MultiModalTrunk(input_spec, hidden_dim, fusion_dim)
        self.score_head = ScoreHead(fusion_dim, num_scores, value_range=(0.0, 100.0))
        self.category_head = (
            ScoreHead(fusion_dim, num_scores, value_range=None)
            if include_category_head
            else None
        )

    def forward(self, batch: ModelBatch) -> ModelOutput:
        batch.validate(self.input_spec)
        h = self.trunk(batch)
        out = ModelOutput(predictions={"constitution_scores": self.score_head(h)})
        if self.category_head is not None:
            out.auxiliary["category_logits"] = self.category_head(h)
        return out

    def compute_loss(self, batch: ModelBatch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch.validate(self.input_spec)
        if batch.targets is None or batch.target_mask is None:
            raise ValueError("TCM training requires targets and target_mask")
        out = self.forward(batch)
        parts: dict[str, torch.Tensor] = {}
        scores = out.predictions["constitution_scores"]
        if "constitution_scores" in batch.targets:
            parts["score_mse"] = masked_mse_loss(
                scores, batch.targets["constitution_scores"],
                mask_name(batch.target_mask, "constitution_scores"),
            )
        if self.category_head is not None and "category_labels" in batch.targets:
            parts["category_bce"] = masked_multilabel_bce_loss(
                out.auxiliary["category_logits"], batch.targets["category_labels"],
                mask_name(batch.target_mask, "category_labels"),
            )
        if not parts:
            raise ValueError(
                f"no usable targets among {sorted(batch.targets)} for {self.model_type}"
            )
        loss = sum(parts.values())  # type: ignore[assignment]
        return loss, parts


class PhysioBPSIEFRIModel(nn.Module):
    """Pre-only physio model (V0): EFRI primary head + optional BPSI aux head
    (#41: single primary-output rule; BPSI is baseline/auxiliary, never a
    substitute performance claim)."""

    model_type = "physio_bpsi_efri_v0"

    def __init__(
        self,
        input_spec: InputSpecSet,
        hidden_dim: int = 64,
        fusion_dim: int = 64,
        include_bpsi_aux: bool = True,
        bpsi_aux_weight: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_spec = input_spec
        self.target_specs: tuple[TargetSpec, ...] = (
            PHYSIO_TARGET_EFRI,
            *((PHYSIO_TARGET_BPSI_AUX,) if include_bpsi_aux else ()),
        )
        self.include_bpsi_aux = include_bpsi_aux
        self.bpsi_aux_weight = bpsi_aux_weight
        self.trunk = _MultiModalTrunk(input_spec, hidden_dim, fusion_dim)
        self.efri_head = ScoreHead(fusion_dim, 1, value_range=None)
        self.bpsi_head = (
            ScoreHead(fusion_dim, 1, value_range=(0.0, 100.0))
            if include_bpsi_aux
            else None
        )

    def forward(self, batch: ModelBatch) -> ModelOutput:
        batch.validate(self.input_spec)
        h = self.trunk(batch)
        out = ModelOutput(predictions={"efri": self.efri_head(h)})
        if self.bpsi_head is not None:
            out.auxiliary["bpsi"] = self.bpsi_head(h)
        return out

    def compute_loss(self, batch: ModelBatch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch.validate(self.input_spec)
        if batch.targets is None or batch.target_mask is None:
            raise ValueError("Physio training requires targets and target_mask")
        out = self.forward(batch)
        parts: dict[str, torch.Tensor] = {}
        if "efri" in batch.targets:
            parts["efri_mse"] = masked_mse_loss(
                out.predictions["efri"], batch.targets["efri"],
                mask_name(batch.target_mask, "efri"),
            )
        if self.bpsi_head is not None and "bpsi" in batch.targets:
            parts["bpsi_aux_mae"] = self.bpsi_aux_weight * masked_mae_loss(
                out.auxiliary["bpsi"], batch.targets["bpsi"],
                mask_name(batch.target_mask, "bpsi"),
            )
        if not parts:
            raise ValueError(
                f"no usable targets among {sorted(batch.targets)} for {self.model_type}"
            )
        loss = sum(parts.values())  # type: ignore[assignment]
        return loss, parts


def default_tcm_model(
    body_dim: int = 12, physio_dim: int = 7, **kwargs: object
) -> tuple[TCMConstitutionModel, dict]:
    """Build the V0 TCM model from schema-provided dims; returns (model, config).

    The returned config contains exactly the model constructor kwargs, so the
    registry can rebuild the model from metadata + input_spec alone (the dims
    live inside the input spec).
    """
    spec = make_tcm_input_spec(body_dim, physio_dim)
    config = {
        "hidden_dim": kwargs.get("hidden_dim", 64),
        "fusion_dim": kwargs.get("fusion_dim", 64),
        "num_scores": kwargs.get("num_scores", 9),
        "include_category_head": kwargs.get("include_category_head", False),
    }
    return TCMConstitutionModel(spec, **config), config


def default_physio_model(
    body_dim: int = 12, physio_dim: int = 7, **kwargs: object
) -> tuple[PhysioBPSIEFRIModel, dict]:
    spec = make_physio_input_spec(body_dim, physio_dim)
    config = {
        "hidden_dim": kwargs.get("hidden_dim", 64),
        "fusion_dim": kwargs.get("fusion_dim", 64),
        "include_bpsi_aux": kwargs.get("include_bpsi_aux", True),
        "bpsi_aux_weight": kwargs.get("bpsi_aux_weight", 0.3),
    }
    return PhysioBPSIEFRIModel(spec, **config), config
