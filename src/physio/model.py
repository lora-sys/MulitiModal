"""Versioned realtime fatigue-state model contract (#97).

This is a small MLP over mask-aware window features. It is deliberately not a
validated clinical model: ``validation_status`` defaults to ``NOT_VALIDATED``,
and the inference layer refuses to emit a score until an independently
validated artifact is supplied. The module imports without PyTorch so the
contract and window plumbing stay usable on CPU-only research hosts.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .windows import WINDOW_FEATURE_VERSION

if TYPE_CHECKING:
    from core.specs import InputSpecSet

REALTIME_TARGET_VERSION = "fatigue_state_v1"
REALTIME_MODEL_VERSION = "physio-realtime-v1"
REALTIME_INPUT_VERSION = "physio-realtime-window-input-v0"

VALIDATION_NOT_VALIDATED = "NOT_VALIDATED"
VALIDATION_VALIDATED = "VALIDATED"
VALIDATION_STATUSES = (VALIDATION_NOT_VALIDATED, VALIDATION_VALIDATED)


@dataclass(frozen=True)
class RealtimeModelConfig:
    """Frozen model/target provenance and architecture settings."""

    input_version: str = REALTIME_INPUT_VERSION
    feature_version: str = WINDOW_FEATURE_VERSION
    target_version: str = REALTIME_TARGET_VERSION
    hidden_dim: int = 32
    fusion_dim: int = 32
    dropout: float = 0.1
    validation_status: str = VALIDATION_NOT_VALIDATED

    def __post_init__(self) -> None:
        for name in ("input_version", "feature_version", "target_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("hidden_dim", "fusion_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be within [0, 1)")
        if self.validation_status not in VALIDATION_STATUSES:
            raise ValueError(
                f"validation_status must be one of {VALIDATION_STATUSES}, "
                f"got {self.validation_status!r}"
            )


def make_realtime_input_spec(n_features: int) -> InputSpecSet:
    """Input contract for one normalized window row."""
    try:
        from core.specs import InputSpecSet as SpecSet
        from core.specs import ModalitySpec
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "building the realtime input spec requires the core package "
            "(and PyTorch, which core imports)"
        ) from exc

    if isinstance(n_features, bool) or not isinstance(n_features, int) or n_features <= 0:
        raise ValueError("n_features must be a positive integer")
    return SpecSet(
        version=REALTIME_INPUT_VERSION,
        modalities=(
            ModalitySpec(
                "window_features",
                "tabular",
                n_features,
                units="standardized",
                description=(
                    "causal window mean/trend/variance/baseline-delta/coverage "
                    f"features ({WINDOW_FEATURE_VERSION})"
                ),
            ),
        ),
    )


@dataclass(frozen=True)
class WindowModelBatch:
    """Torch-free single-row window input.

    ``values`` uses 0.0 placeholders at masked positions purely so a dense
    tensor can be built. The mask is the authority; models must gate every
    placeholder out. ``to_model_batch`` is the boundary where this becomes a
    ``core.batch.ModelBatch`` when PyTorch is installed.
    """

    feature_names: tuple[str, ...]
    values: tuple[float, ...]
    mask: tuple[bool, ...]

    def __post_init__(self) -> None:
        if len(self.feature_names) != len(self.values) or len(self.values) != len(self.mask):
            raise ValueError("feature_names, values and mask must have equal lengths")

    @property
    def batch_size(self) -> int:
        return 1

    def to_model_batch(self, input_spec: InputSpecSet | None = None) -> Any:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "converting a window batch requires PyTorch; install torch"
            ) from exc
        from core.batch import ModelBatch

        spec = input_spec or make_realtime_input_spec(len(self.values))
        batch = ModelBatch(
            features={
                "window_features": torch.tensor([self.values], dtype=torch.float32)
            },
            availability={
                "window_features": torch.tensor([any(self.mask)], dtype=torch.bool)
            },
        )
        batch.validate(spec)
        return batch


_TORCH_IMPORT_ERROR: ImportError | None = None
try:  # pragma: no cover - environment dependent
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc

if nn is not None:  # pragma: no cover - skipped when torch is absent
    from core.batch import ModelBatch, ModelOutput
    from core.encoders import build_encoder
    from core.specs import TargetSpec

    class RealtimeFatigueModel(nn.Module):
        """Small mask-aware MLP for the current fatigue-related state."""

        model_type = REALTIME_MODEL_VERSION

        def __init__(
            self,
            n_features: int,
            config: RealtimeModelConfig | None = None,
        ) -> None:
            super().__init__()
            self.config = config or RealtimeModelConfig()
            self.n_features = n_features
            self.input_spec = make_realtime_input_spec(n_features)
            self.target_spec = TargetSpec(
                name="fatigue_state",
                dim=1,
                semantic=(
                    "engineering proxy for current fatigue-related state; "
                    "not a clinical label and not end-of-massage efficacy"
                ),
                value_range=(0.0, 100.0),
                version=self.config.target_version,
            )
            self.encoder = build_encoder(
                self.input_spec.modalities[0], self.config.hidden_dim
            )
            self.mask_projection = nn.Linear(n_features, self.config.hidden_dim)
            self.head = nn.Sequential(
                nn.Linear(self.config.hidden_dim, self.config.fusion_dim),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.fusion_dim, 1),
                nn.Sigmoid(),
            )

        def forward(
            self,
            batch: ModelBatch | WindowModelBatch,
            feature_mask: Any = None,
        ) -> ModelOutput:
            if isinstance(batch, WindowModelBatch):
                if feature_mask is None:
                    feature_mask = batch.mask
                batch = batch.to_model_batch(self.input_spec)
            if not isinstance(batch, ModelBatch):
                raise TypeError("RealtimeFatigueModel expects a ModelBatch")
            batch.validate(self.input_spec)

            features = batch.features["window_features"]
            availability = batch.availability.get("window_features")
            if feature_mask is None:
                mask = torch.ones_like(features, dtype=torch.bool)
            elif isinstance(feature_mask, torch.Tensor):
                mask = feature_mask.to(device=features.device, dtype=torch.bool)
                if mask.ndim == 1:
                    mask = mask.unsqueeze(0)
            else:
                mask = torch.tensor(feature_mask, dtype=torch.bool, device=features.device)
                if mask.ndim == 1:
                    mask = mask.unsqueeze(0)

            if mask.shape != features.shape:
                raise ValueError(
                    "feature mask shape must match the window feature tensor"
                )
            gated = features * mask.to(features.dtype)
            hidden = self.encoder(gated, availability)
            mask_embedding = self.mask_projection(mask.to(features.dtype))
            if availability is not None:
                mask_embedding = mask_embedding * availability.to(features.dtype).unsqueeze(-1)
            hidden = hidden + mask_embedding
            return ModelOutput(
                predictions={"fatigue_state": 100.0 * self.head(hidden)}
            )

        def state_sha256(self) -> str:
            """Deterministic digest over parameter names, dtypes, shapes and bytes."""
            digest = hashlib.sha256()
            for name, tensor in sorted(self.state_dict().items()):
                array = tensor.detach().to("cpu").contiguous().numpy()
                digest.update(name.encode("utf-8"))
                digest.update(str(array.dtype).encode("utf-8"))
                digest.update(str(tuple(array.shape)).encode("utf-8"))
                digest.update(array.tobytes())
            return digest.hexdigest()

else:  # pragma: no cover - exercised only on hosts without torch

    class RealtimeFatigueModel:  # type: ignore[no-redef]
        """Placeholder that fails loudly when torch is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "RealtimeFatigueModel requires PyTorch; install torch to construct "
                f"the realtime model ({_TORCH_IMPORT_ERROR})"
            )


__all__ = [
    "REALTIME_INPUT_VERSION",
    "REALTIME_MODEL_VERSION",
    "REALTIME_TARGET_VERSION",
    "VALIDATION_NOT_VALIDATED",
    "VALIDATION_STATUSES",
    "VALIDATION_VALIDATED",
    "RealtimeFatigueModel",
    "RealtimeModelConfig",
    "WindowModelBatch",
    "make_realtime_input_spec",
]
