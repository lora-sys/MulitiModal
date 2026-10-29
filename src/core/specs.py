"""Versioned input/target specification contracts (#48).

Dims and semantics are provided by spec/config, never hardcoded in models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ModalityKind = Literal["tabular", "waveform"]


@dataclass(frozen=True)
class ModalitySpec:
    """One input modality: name, tensor kind, feature/channel dim, units."""

    name: str
    kind: ModalityKind
    dim: int
    units: str = "arbitrary"
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ModalitySpec.name must be non-empty")
        if self.kind not in ("tabular", "waveform"):
            raise ValueError(f"unknown modality kind: {self.kind!r}")
        if self.dim <= 0:
            raise ValueError(f"ModalitySpec.dim must be positive, got {self.dim}")


@dataclass(frozen=True)
class InputSpecSet:
    """Frozen, versioned set of input modalities a model accepts."""

    version: str
    modalities: tuple[ModalitySpec, ...]

    def __post_init__(self) -> None:
        if not self.version:
            raise ValueError("InputSpecSet.version must be non-empty")
        names = [m.name for m in self.modalities]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate modality names: {names}")
        if not names:
            raise ValueError("InputSpecSet needs at least one modality")

    def names(self) -> tuple[str, ...]:
        return tuple(m.name for m in self.modalities)

    def get(self, name: str) -> ModalitySpec:
        for m in self.modalities:
            if m.name == name:
                return m
        raise KeyError(
            f"modality {name!r} is not part of input spec {self.version!r} "
            f"(allowed: {self.names()})"
        )

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "modalities": [dict(name=m.name, kind=m.kind, dim=m.dim, units=m.units,
                                description=m.description) for m in self.modalities],
        }

    @classmethod
    def from_dict(cls, data: dict) -> InputSpecSet:
        return cls(
            version=data["version"],
            modalities=tuple(ModalitySpec(**m) for m in data["modalities"]),
        )


@dataclass(frozen=True)
class TargetSpec:
    """One prediction target: name, dim, semantic meaning, expected range."""

    name: str
    dim: int
    semantic: str
    value_range: tuple[float, float] | None = None
    version: str = "v0"

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError(f"TargetSpec.dim must be positive, got {self.dim}")
        if self.value_range is not None and self.value_range[0] >= self.value_range[1]:
            raise ValueError("TargetSpec.value_range must be (low, high) with low < high")

    def to_dict(self) -> dict:
        return dict(name=self.name, dim=self.dim, semantic=self.semantic,
                    value_range=self.value_range, version=self.version)

    @classmethod
    def from_dict(cls, data: dict) -> TargetSpec:
        vr = data.get("value_range")
        return cls(name=data["name"], dim=data["dim"], semantic=data["semantic"],
                   value_range=tuple(vr) if vr is not None else None,
                   version=data.get("version", "v0"))


def make_tcm_input_spec(body_dim: int, physio_dim: int) -> InputSpecSet:
    """TCM V0 pre-only tabular inputs. Dims come from the dataset schema (#37/#39)."""
    return InputSpecSet(
        version="tcm-input-v0",
        modalities=(
            ModalitySpec("body_features", "tabular", body_dim,
                         units="mixed_schema",
                         description="schema-provided body/vitals features"),
            ModalitySpec("physio_features", "tabular", physio_dim,
                         units="mixed_schema",
                         description="schema-provided physiological features"),
        ),
    )


def make_physio_input_spec(body_dim: int, physio_dim: int) -> InputSpecSet:
    """Physio V0 pre-only inputs (#23/#41): Pre body + Pre physiology only.

    During-EEG and Post measures are training-time label material (#42) and
    must never appear in the model input contract.
    """
    return InputSpecSet(
        version="physio-input-pre-only-v0",
        modalities=(
            ModalitySpec("body_pre", "tabular", body_dim,
                         units="mixed_schema",
                         description="pre-massage body information"),
            ModalitySpec("physio_pre", "tabular", physio_dim,
                         units="mixed_schema",
                         description="pre-massage physiological measurements"),
        ),
    )


TCM_TARGET_SCORES = TargetSpec(
    name="constitution_scores", dim=9,
    semantic="nine constitution scores 0-100; separate head, disabled unless "
             "real questionnaire scores exist (#39)",
    value_range=(0.0, 100.0),
)
TCM_TARGET_CATEGORY = TargetSpec(
    name="category_labels", dim=9,
    semantic="nine constitution category labels; non-exclusive multi-label by "
             "default, softmax-CE only for mutually-exclusive sources (#39)",
    value_range=None,
)
PHYSIO_TARGET_EFRI = TargetSpec(
    name="efri", dim=1,
    semantic="expected fatigue relief index, standardized composite response (experimental)",
    value_range=None,
)
PHYSIO_TARGET_BPSI_AUX = TargetSpec(
    name="bpsi_aux", dim=1,
    semantic="optional auxiliary head for the Pre-derived BPSI baseline index",
    value_range=(0.0, 100.0),
)
