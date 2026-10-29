"""Shared synthetic batch factories for core tests (#48)."""

import torch

from core.batch import ModelBatch
from core.specs import make_physio_input_spec, make_tcm_input_spec

BODY_DIM = 12
PHYSIO_DIM = 7
HIDDEN = 32
NUM_SCORES = 9


def tcm_spec():
    return make_tcm_input_spec(BODY_DIM, PHYSIO_DIM)


def physio_spec():
    return make_physio_input_spec(BODY_DIM, PHYSIO_DIM)


def tcm_batch(
    batch_size: int = 8,
    seed: int = 0,
    *,
    with_targets: bool = True,
    drop_modality: str | None = None,
    zero_out_availability: str | None = None,
    inject_nan: bool = False,
) -> ModelBatch:
    g = torch.Generator().manual_seed(seed)
    features = {
        "body_features": torch.rand(batch_size, BODY_DIM, generator=g),
        "physio_features": torch.rand(batch_size, PHYSIO_DIM, generator=g),
    }
    if drop_modality is not None:
        del features[drop_modality]
    if inject_nan:
        features["body_features"][0, 0] = float("nan")
    availability: dict[str, torch.Tensor] = {}
    if zero_out_availability is not None:
        availability[zero_out_availability] = torch.zeros(batch_size, dtype=torch.bool)

    targets = target_mask = None
    if with_targets:
        targets = {
            "constitution_scores": torch.rand(batch_size, NUM_SCORES, generator=g) * 100.0
        }
        target_mask = {
            "constitution_scores": torch.rand(batch_size, NUM_SCORES, generator=g) > 0.3
        }
    return ModelBatch(
        features=features,
        availability=availability,
        targets=targets,
        target_mask=target_mask,
    )


def physio_batch(
    batch_size: int = 8,
    seed: int = 1,
    *,
    with_targets: bool = True,
    extra_during_post: bool = False,
) -> ModelBatch:
    g = torch.Generator().manual_seed(seed)
    features = {
        "body_pre": torch.rand(batch_size, BODY_DIM, generator=g),
        "physio_pre": torch.rand(batch_size, PHYSIO_DIM, generator=g),
    }
    if extra_during_post:
        # The exact leakage the Pre-only contract must reject (#23/#31).
        features["eeg_during"] = torch.rand(batch_size, 9, generator=g)
        features["physio_post"] = torch.rand(batch_size, 7, generator=g)

    targets = target_mask = None
    if with_targets:
        targets = {
            "efri": torch.randn(batch_size, 1, generator=g),
            "bpsi": torch.rand(batch_size, 1, generator=g) * 100.0,
        }
        target_mask = {
            "efri": torch.ones(batch_size, 1, dtype=torch.bool),
            "bpsi": torch.rand(batch_size, 1, generator=g) > 0.5,
        }
    return ModelBatch(
        features=features, targets=targets, target_mask=target_mask
    )
