"""Model batch / output contracts with explicit modality masks (#48).

Missing modalities are expressed by *absence* from `features`; present-but-
unavailable rows are expressed by `availability` masks. Values are never
imputed, and validation rejects anything the contract does not allow.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .specs import InputSpecSet


@dataclass
class ModelBatch:
    """A validated training/inference batch.

    features:    modality name -> tensor [B, D] (tabular) or [B, C, T] (waveform)
    availability: modality name -> bool tensor [B]; rows where the modality was
                  measured but is unusable. Absent modality = fully missing.
    targets:     target name -> tensor [B, K_target]
    target_mask: target name -> bool tensor [B, K_target]; observed entries only
    """

    features: dict[str, torch.Tensor]
    availability: dict[str, torch.Tensor] = field(default_factory=dict)
    targets: dict[str, torch.Tensor] | None = None
    target_mask: dict[str, torch.Tensor] | None = None

    @property
    def batch_size(self) -> int:
        if not self.features:
            raise ValueError("ModelBatch.features is empty")
        return int(next(iter(self.features.values())).shape[0])

    def validate(self, spec: InputSpecSet) -> None:
        """Reject anything the input spec does not allow — loudly, never silently."""
        if not self.features:
            raise ValueError("ModelBatch.features must not be empty")

        batch_size: int | None = None
        for name, tensor in self.features.items():
            try:
                mod = spec.get(name)
            except KeyError as exc:
                # This is the boundary that keeps During/Post out of Pre-only models.
                raise ValueError(
                    f"feature {name!r} is not an approved input for spec "
                    f"{spec.version!r}: {exc}"
                ) from exc
            expected_ndim = 2 if mod.kind == "tabular" else 3
            if tensor.ndim != expected_ndim:
                raise ValueError(
                    f"modality {name!r} ({mod.kind}) expects ndim={expected_ndim}, "
                    f"got shape {tuple(tensor.shape)}"
                )
            if tensor.shape[1] != mod.dim:
                raise ValueError(
                    f"modality {name!r} expects dim={mod.dim}, got {tensor.shape[1]}"
                )
            if batch_size is None:
                batch_size = int(tensor.shape[0])
            elif int(tensor.shape[0]) != batch_size:
                raise ValueError(
                    f"modality {name!r} has batch size {tensor.shape[0]}, expected {batch_size}"
                )
            if not torch.isfinite(tensor).all():
                raise ValueError(
                    f"modality {name!r} contains NaN/Inf; imputation is not allowed here, "
                    "clean or mask the data upstream (#26/#29)"
                )

        assert batch_size is not None
        for name, mask in self.availability.items():
            spec.get(name)  # unknown availability key -> KeyError
            if mask.shape != (batch_size,):
                raise ValueError(
                    f"availability[{name!r}] must have shape ({batch_size},), "
                    f"got {tuple(mask.shape)}"
                )
            if mask.dtype != torch.bool:
                raise ValueError(f"availability[{name!r}] must be bool, got {mask.dtype}")

        if (self.targets is None) != (self.target_mask is None):
            raise ValueError("targets and target_mask must be provided together")
        if self.targets is not None:
            assert self.target_mask is not None
            for name, tensor in self.targets.items():
                if tensor.ndim != 2:
                    raise ValueError(f"targets[{name!r}] must be [B, K], got {tuple(tensor.shape)}")
                if tensor.shape[0] != batch_size:
                    raise ValueError(f"targets[{name!r}] batch size mismatch")
                if not torch.isfinite(tensor[mask_name(self.target_mask, name)]).all():
                    raise ValueError(
                        f"targets[{name!r}] has NaN/Inf on observed (mask=true) entries"
                    )
            for name, mask in self.target_mask.items():
                if name not in self.targets:
                    raise ValueError(f"target_mask[{name!r}] has no matching target")
                if mask.dtype != torch.bool:
                    raise ValueError(f"target_mask[{name!r}] must be bool")
                if mask.shape != self.targets[name].shape:
                    raise ValueError(f"target_mask[{name!r}] shape mismatch with target")


def mask_name(masks: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    mask = masks.get(name)
    if mask is None:
        raise KeyError(f"no mask for target {name!r}")
    return mask


@dataclass
class ModelOutput:
    """predictions: target name -> [B, K]; auxiliary: optional head outputs."""

    predictions: dict[str, torch.Tensor]
    auxiliary: dict[str, torch.Tensor] = field(default_factory=dict)

    def check_ranges(self, ranges: dict[str, tuple[float, float] | None]) -> None:
        for name, rng in ranges.items():
            if rng is None or name not in self.predictions:
                continue
            low, high = rng
            tensor = self.predictions[name]
            if not bool(((tensor >= low - 1e-4) & (tensor <= high + 1e-4)).all()):
                raise ValueError(
                    f"predictions[{name!r}] violate declared range {rng}"
                )
