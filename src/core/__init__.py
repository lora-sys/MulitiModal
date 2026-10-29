"""Shared Phase-1 neural core: specs, batches, encoders, models, registry (#48).

Everything here is synthetic-testable and real-data-free by design; see
issue #48 for the acceptance contract.
"""

from .batch import ModelBatch, ModelOutput
from .encoders import ConcatFusion, ScoreHead, TabularEncoder, WaveformEncoder, build_encoder
from .losses import masked_mae_loss, masked_mse_loss, masked_multilabel_bce_loss
from .models import (
    PhysioBPSIEFRIModel,
    TCMConstitutionModel,
    default_physio_model,
    default_tcm_model,
)
from .registry import MODEL_BUILDERS, batch_predict, load_model, resolve_device, save_model
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
