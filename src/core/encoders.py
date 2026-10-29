"""Encoders, fusion, and prediction heads (#48).

Missing-modality contract: an encoder multiplies its output by the row
availability mask, so unavailable rows contribute exact zeros downstream —
no mean/zero imputation of input values anywhere.
"""

from __future__ import annotations

import torch
from torch import nn

from .specs import ModalitySpec


class TabularEncoder(nn.Module):
    """MLP encoder for [B, D] tabular features -> [B, hidden]."""

    def __init__(self, spec: ModalitySpec, hidden_dim: int) -> None:
        super().__init__()
        self.spec = spec
        self.net = nn.Sequential(
            nn.Linear(spec.dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor, availability: torch.Tensor | None) -> torch.Tensor:
        out = self.net(x)
        if availability is not None:
            out = out * availability.to(out.dtype).unsqueeze(-1)
        return out


class WaveformEncoder(nn.Module):
    """Minimal Conv1D reference encoder for [B, C, T] waveforms -> [B, hidden].

    Adapter-level reference implementation (#48); real signal encoders are a
    later, separately evaluated milestone.
    """

    def __init__(self, spec: ModalitySpec, hidden_dim: int) -> None:
        super().__init__()
        self.spec = spec
        self.conv = nn.Sequential(
            nn.Conv1d(spec.dim, hidden_dim, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
        )
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, availability: torch.Tensor | None) -> torch.Tensor:
        out = self.conv(x).mean(dim=-1)
        out = self.proj(out)
        if availability is not None:
            out = out * availability.to(out.dtype).unsqueeze(-1)
        return out


def build_encoder(spec: ModalitySpec, hidden_dim: int) -> nn.Module:
    if spec.kind == "tabular":
        return TabularEncoder(spec, hidden_dim)
    return WaveformEncoder(spec, hidden_dim)


class ConcatFusion(nn.Module):
    """Concatenate per-modality embeddings -> shared fusion representation.

    Module order follows the InputSpecSet order, so serialization is stable.
    """

    def __init__(self, modality_names: tuple[str, ...], hidden_dim: int, fusion_dim: int) -> None:
        super().__init__()
        self.modality_names = modality_names
        self.net = nn.Sequential(
            nn.Linear(len(modality_names) * hidden_dim, fusion_dim),
            nn.GELU(),
        )

    def forward(self, embeddings: dict[str, torch.Tensor]) -> torch.Tensor:
        missing = [n for n in self.modality_names if n not in embeddings]
        if missing:
            raise KeyError(f"fusion missing embeddings for: {missing}")
        return self.net(torch.cat([embeddings[n] for n in self.modality_names], dim=-1))


class ScoreHead(nn.Module):
    """Linear head with optional squashing to a declared value range."""

    def __init__(
        self, fusion_dim: int, out_dim: int, value_range: tuple[float, float] | None
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(fusion_dim, out_dim)
        self.value_range = value_range

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        out = self.linear(h)
        if self.value_range is not None:
            low, high = self.value_range
            out = low + (high - low) * torch.sigmoid(out)
        return out
