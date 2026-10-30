"""Real-data batch pipeline for TCM training (#50): frozen manifest ->
subject-disjoint BatchProviders + model factory.

The manifest (src/data/real_split.py) carries the frozen split, the
train-fold-fitted scaler and the label contract. This module turns it
into ModelBatches per the core contract: a modality row with missing
features is availability-masked and zero-filled (never imputed); the
balanced dimension is unobserved in every target (never negative-filled).
All provenance travels in the manifest; batches stay in memory only.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

import torch

from core.batch import ModelBatch
from core.models import TCMConstitutionModel
from core.specs import InputSpecSet, ModalitySpec

from .batches import BatchItem
from .tcm_eval import EvalManifest

MANIFEST_FIELD = "real_data_manifest"


def real_input_spec(manifest: dict) -> InputSpecSet:
    modalities = tuple(
        ModalitySpec(
            name=name,
            kind="tabular",
            dim=params["dim"],
            units="standardized_schema",
            description=f"train-fold standardized: {params['feature_groups']}",
        )
        for name, params in manifest["modalities"].items()
    )
    return InputSpecSet(version="tcm-input-real-v1", modalities=modalities)


def build_real_model(manifest: dict, model_cfg: dict) -> TCMConstitutionModel:
    """Registry-compatible factory: model_config carries exactly the ctor
    kwargs so core.registry can rebuild from metadata + input_spec."""
    spec = real_input_spec(manifest)
    config = {
        "hidden_dim": model_cfg.get("hidden_dim", 32),
        "fusion_dim": model_cfg.get("fusion_dim", 32),
        "num_scores": model_cfg.get("num_scores", 9),
        "include_category_head": model_cfg.get("include_category_head", True),
        "category_exclusive": model_cfg.get("category_exclusive", False),
        "include_score_head": model_cfg.get("include_score_head", False),
    }
    model = TCMConstitutionModel(spec, **config)
    model.model_config = config
    return model


def _row_tensors(
    records_by_row: dict,
    manifest: dict,
    subject_ids: tuple[str, ...],
    task_type: str,
) -> tuple[dict[str, torch.Tensor], ...]:
    """Materialize features/availability/targets/masks for one partition."""
    modalities = manifest["modalities"]
    features = {name: [] for name in modalities}
    availability = {name: [] for name in modalities}
    targets: list[list[float]] = []
    target_mask: list[list[bool]] = []
    for sid in subject_ids:
        row = manifest["subject_row_by_id"][sid]
        record = records_by_row[row]
        for name, params in modalities.items():
            values = [v for g in params["feature_groups"] for v in record.features[g]]
            mean = manifest["scaler"][name]["mean"]
            std = manifest["scaler"][name]["std"]
            ok = not any(math.isnan(v) for v in values)
            scaled = [
                (v - m) / s if ok else 0.0 for v, m, s in zip(values, mean, std, strict=True)
            ]
            features[name].append(scaled)
            availability[name].append(ok)
        if task_type == "derived_multilabel_classification":
            from data.real_split import derive_classification_targets

            labels, mask = derive_classification_targets(record.scores)
            target_key = "category_labels"
        else:
            labels = [s if s is not None else 0.0 for s in record.scores]
            mask = [s is not None for s in record.scores]
            target_key = "constitution_scores"
        targets.append(labels)
        target_mask.append(mask)
    return (
        {name: torch.tensor(features[name], dtype=torch.float32) for name in modalities},
        {name: torch.tensor(availability[name], dtype=torch.bool) for name in modalities},
        {target_key: torch.tensor(targets, dtype=torch.float32)},
        {target_key: torch.tensor(target_mask, dtype=torch.bool)},
    )


class RealBatchProvider:
    """BatchProvider over one frozen manifest partition (#50).

    Deterministic: identical (manifest, partition, batch_size) -> identical
    batches. subject_ids are the manifest's partition list, so the #27
    disjointness assertion and the trusted evaluator's partition check hold
    by construction.
    """

    def __init__(
        self,
        manifest: dict,
        records: list,
        partition: str,
        batch_size: int = 32,
    ) -> None:
        if partition not in ("train", "dev", "holdout"):
            raise ValueError(f"unknown partition {partition!r}")
        self.manifest = manifest
        self.partition = partition
        self.batch_size = max(1, int(batch_size))
        self.subject_ids = tuple(manifest["split"][partition])
        self.records_by_row = {r.subject_index: r for r in records}
        missing = [s for s in self.subject_ids
                   if manifest["subject_row_by_id"][s] not in self.records_by_row]
        if missing:
            raise ValueError(f"manifest subjects missing from records: {missing[:3]}")
        self._features, self._availability, self._targets, self._target_mask = _row_tensors(
            self.records_by_row, manifest, self.subject_ids, manifest["task_type"]
        )

    def input_spec(self) -> InputSpecSet:
        return real_input_spec(self.manifest)

    def __len__(self) -> int:
        return (len(self.subject_ids) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[BatchItem]:
        n = len(self.subject_ids)
        for lo in range(0, n, self.batch_size):
            hi = min(lo + self.batch_size, n)
            sl = slice(lo, hi)
            yield BatchItem(
                batch=ModelBatch(
                    features={k: v[sl] for k, v in self._features.items()},
                    availability={k: v[sl] for k, v in self._availability.items()},
                    targets={k: v[sl] for k, v in self._targets.items()},
                    target_mask={k: v[sl] for k, v in self._target_mask.items()},
                ),
                subject_ids=list(self.subject_ids[lo:hi]),
            )


def real_eval_manifest(manifest: dict) -> EvalManifest:
    """Frozen EvalManifest mirroring the real split for the trusted evaluator."""
    return EvalManifest(
        version=f"real-{manifest['manifest_sha256'][:12]}",
        input_spec_version="tcm-input-real-v1",
        standard_version=manifest["standard_version"],
        label_tier=manifest["label_tier"],
        train_subjects=tuple(manifest["split"]["train"]),
        dev_subjects=tuple(manifest["split"]["dev"]),
        holdout_subjects=tuple(manifest["split"]["holdout"]),
    )


def manifest_from_path(path: str | Path) -> dict:
    from data.real_split import load_real_manifest

    return load_real_manifest(path)
