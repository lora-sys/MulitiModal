"""Shared Phase-1 neural core: specs, batches, encoders, models, registry (#48).

Everything here is synthetic-testable and real-data-free by design; see
issue #48 for the acceptance contract.

`specs.py` is pure stdlib, but the model/batch/registry modules require torch.
Package attributes are resolved lazily (PEP 562) so a clean, torch-free CI can
still ``import core.specs`` without the optional neural stack. With torch
installed, every ``from core import X`` behaves exactly as before.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

# Pure-stdlib contract, safe to import on any host.
from .specs import (
    PHYSIO_TARGET_BPSI_AUX,
    PHYSIO_TARGET_EFRI,
    TCM_TARGET_CATEGORY,
    TCM_TARGET_SCORES,
    InputSpecSet,
    ModalitySpec,
    TargetSpec,
    make_physio_input_spec,
    make_tcm_input_spec,
)

_TORCH_BACKED: dict[str, tuple[str, str]] = {
    "ModelBatch": ("core.batch", "ModelBatch"),
    "ModelOutput": ("core.batch", "ModelOutput"),
    "ConcatFusion": ("core.encoders", "ConcatFusion"),
    "ScoreHead": ("core.encoders", "ScoreHead"),
    "TabularEncoder": ("core.encoders", "TabularEncoder"),
    "WaveformEncoder": ("core.encoders", "WaveformEncoder"),
    "build_encoder": ("core.encoders", "build_encoder"),
    "masked_mae_loss": ("core.losses", "masked_mae_loss"),
    "masked_mse_loss": ("core.losses", "masked_mse_loss"),
    "masked_multilabel_bce_loss": ("core.losses", "masked_multilabel_bce_loss"),
    "masked_softmax_ce_loss": ("core.losses", "masked_softmax_ce_loss"),
    "PhysioBPSIEFRIModel": ("core.models", "PhysioBPSIEFRIModel"),
    "TCMConstitutionModel": ("core.models", "TCMConstitutionModel"),
    "default_physio_model": ("core.models", "default_physio_model"),
    "default_tcm_model": ("core.models", "default_tcm_model"),
    "MODEL_BUILDERS": ("core.registry", "MODEL_BUILDERS"),
    "batch_predict": ("core.registry", "batch_predict"),
    "load_model": ("core.registry", "load_model"),
    "resolve_device": ("core.registry", "resolve_device"),
    "save_model": ("core.registry", "save_model"),
}

__all__ = [
    "ModelBatch",
    "ModelOutput",
    "ConcatFusion",
    "ScoreHead",
    "TabularEncoder",
    "WaveformEncoder",
    "build_encoder",
    "masked_mae_loss",
    "masked_mse_loss",
    "masked_multilabel_bce_loss",
    "masked_softmax_ce_loss",
    "PhysioBPSIEFRIModel",
    "TCMConstitutionModel",
    "default_physio_model",
    "default_tcm_model",
    "MODEL_BUILDERS",
    "batch_predict",
    "load_model",
    "resolve_device",
    "save_model",
    "PHYSIO_TARGET_BPSI_AUX",
    "PHYSIO_TARGET_EFRI",
    "TCM_TARGET_CATEGORY",
    "TCM_TARGET_SCORES",
    "InputSpecSet",
    "ModalitySpec",
    "TargetSpec",
    "make_physio_input_spec",
    "make_tcm_input_spec",
]


def __getattr__(name: str) -> Any:
    target = _TORCH_BACKED.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, attribute = target
    value = getattr(import_module(module_path), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
