"""Model registry: build / save / load with full provenance metadata (#48).

Saved artifacts contain schema/model/version/checkpoint metadata and the
SHA-256 of the weights file, so a checkpoint can never be loaded into a
mismatched architecture or input schema silently.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
from torch import nn

from .batch import ModelBatch, ModelOutput
from .models import PhysioBPSIEFRIModel, TCMConstitutionModel
from .specs import InputSpecSet

MODEL_BUILDERS: dict[str, type[nn.Module]] = {
    TCMConstitutionModel.model_type: TCMConstitutionModel,
    PhysioBPSIEFRIModel.model_type: PhysioBPSIEFRIModel,
}

WEIGHTS_FILE = "weights.pt"
METADATA_FILE = "metadata.json"


def resolve_device(preference: str = "cpu") -> torch.device:
    """CPU is the default; cuda/mps/auto opt in explicitly (#48)."""
    if preference == "cpu":
        return torch.device("cpu")
    if preference in ("cuda", "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if preference == "auto" and torch.backends.mps.is_available():
            return torch.device("mps")
        if preference == "cuda":
            raise RuntimeError("cuda requested but not available")
        return torch.device("cpu")
    if preference == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("mps requested but not available")
    raise ValueError(f"unknown device preference: {preference!r}")


def save_model(
    model: nn.Module,
    config: dict,
    out_dir: str | Path,
    source_commit: str | None = None,
) -> Path:
    """Serialize weights + metadata. Model class must be registry-known."""
    model_type = getattr(model, "model_type", None)
    if model_type not in MODEL_BUILDERS:
        raise ValueError(f"model type {model_type!r} is not registered")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    weights_path = out / WEIGHTS_FILE
    torch.save(model.state_dict(), weights_path)
    sha256 = hashlib.sha256(weights_path.read_bytes()).hexdigest()

    metadata = {
        "model_type": model_type,
        "format_version": "1",
        "config": config,
        "input_spec": model.input_spec.to_dict(),  # type: ignore[attr-defined]
        "target_specs": [t.to_dict() for t in model.target_specs],  # type: ignore[attr-defined]
        "torch_version": torch.__version__,
        "source_commit": source_commit or os.environ.get("SOURCE_COMMIT", "unknown"),
        "weights_sha256": sha256,
    }
    (out / METADATA_FILE).write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    return out


def load_model(model_dir: str | Path, device: str = "cpu") -> tuple[nn.Module, dict]:
    """Rebuild from metadata and load weights strictly. Any mismatch raises."""
    model_dir = Path(model_dir)
    metadata = json.loads((model_dir / METADATA_FILE).read_text())
    model_type = metadata["model_type"]
    if model_type not in MODEL_BUILDERS:
        raise ValueError(f"unknown model_type in metadata: {model_type!r}")

    cls = MODEL_BUILDERS[model_type]
    spec = InputSpecSet.from_dict(metadata["input_spec"])
    model = cls(input_spec=spec, **metadata["config"])  # type: ignore[call-arg]
    state = torch.load(model_dir / WEIGHTS_FILE, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)

    actual_sha = hashlib.sha256((model_dir / WEIGHTS_FILE).read_bytes()).hexdigest()
    if actual_sha != metadata["weights_sha256"]:
        raise ValueError(
            f"weights checksum mismatch: metadata says {metadata['weights_sha256']}, "
            f"file is {actual_sha}"
        )
    model.to(resolve_device(device))
    model.eval()
    return model, metadata


def batch_predict(
    model: nn.Module, batch: ModelBatch, device: str = "cpu"
) -> ModelOutput:
    """CPU-default batch inference on the resolved device."""
    dev = resolve_device(device)
    was_training = model.training
    model.eval()
    features = {k: v.to(dev) for k, v in batch.features.items()}
    availability = {k: v.to(dev) for k, v in batch.availability.items()}
    with torch.no_grad():
        moved = ModelBatch(features=features, availability=availability)
        out = model(moved)
    if was_training:
        model.train()
    return out
