"""Pre-only TCM inference API (#40): single/batch entry with artifact version
checks, missing-field errors, and explicit ABSTAIN (拒判) — never guessing.

Evidence-state rule (#22/#47): an inference result carries an explicit
status; rows with no usable modality are ABSTAIN, their probs are null.
"""

from __future__ import annotations

from pathlib import Path

import torch

from core.batch import ModelBatch
from core.registry import MODEL_BUILDERS, load_model

TCM_OUTPUT_SCHEMA_VERSION = "tcm-inference-v0"
_FORMAT_VERSION = "1"


class TCMPredictor:
    """Loads a frozen TCM artifact and exposes Pre-only single/batch inference."""

    def __init__(self, model_dir: str | Path, device: str = "cpu") -> None:
        self.model, self.metadata = load_model(model_dir, device=device)
        spec_version = self.metadata["input_spec"]["version"]
        if not spec_version.startswith("tcm-input"):
            raise ValueError(
                f"artifact input spec {spec_version!r} is not a TCM contract"
            )
        if self.metadata["format_version"] != _FORMAT_VERSION:
            raise ValueError(
                f"unsupported artifact format_version {self.metadata['format_version']!r}"
            )
        if self.metadata["model_type"] not in MODEL_BUILDERS:
            raise ValueError(
                f"unknown model_type {self.metadata['model_type']!r}"
            )
        self.spec_version = spec_version

    def _row_usable(self, batch: ModelBatch, i: int) -> bool:
        for name in self.model.input_spec.names():
            if name not in batch.features:
                continue
            avail = batch.availability.get(name)
            if avail is None or bool(avail[i]):
                return True
        return False

    def predict_single(
        self,
        features: dict[str, list | torch.Tensor],
        availability: dict[str, list | torch.Tensor] | None = None,
    ) -> dict:
        """One sample in, schema-versioned result out. Unknown fields raise;
        no usable modality yields status ABSTAIN with null probabilities."""
        if not features:
            return self._abstain_result()
        spec_names = set(self.model.input_spec.names())
        unknown = set(features) - spec_names
        if unknown:
            raise KeyError(
                f"unknown feature keys {sorted(unknown)}; allowed: {sorted(spec_names)}"
            )
        batch = ModelBatch(
            features={
                k: torch.as_tensor(v, dtype=torch.float32).reshape(1, -1)
                for k, v in features.items()
            },
            availability={
                k: torch.as_tensor(v, dtype=torch.bool).reshape(-1)
                for k, v in (availability or {}).items()
            },
        )
        if not self._row_usable(batch, 0):
            return self._abstain_result()
        out = self.model(batch)
        return self._result([out.predictions["category_probs"][0].tolist()])

    def predict_batch(self, batch: ModelBatch) -> dict:
        """Batch inference with per-row ABSTAIN when no modality is usable."""
        out = self.model(batch)
        probs = out.predictions["category_probs"]
        usable = [self._row_usable(batch, i) for i in range(batch.batch_size)]
        result = self._result(
            [probs[i].tolist() if u else None for i, u in enumerate(usable)]
        )
        result["n_abstained"] = usable.count(False)
        return result

    def _result(self, probs_per_row: list) -> dict:
        return {
            "schema_version": TCM_OUTPUT_SCHEMA_VERSION,
            "status": "OK",
            "input_spec_version": self.spec_version,
            "model_type": self.metadata["model_type"],
            "smoke_only": True,
            "category_probs": probs_per_row,
        }

    def _abstain_result(self) -> dict:
        return {
            "schema_version": TCM_OUTPUT_SCHEMA_VERSION,
            "status": "ABSTAIN",
            "input_spec_version": self.spec_version,
            "model_type": self.metadata["model_type"],
            "smoke_only": True,
            "category_probs": None,
        }
