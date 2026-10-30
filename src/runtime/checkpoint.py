"""Training checkpoints (#49): registry weights/metadata + optimizer/epoch/RNG
state, with full provenance (seed, torch version, source commit, SMOKE_ONLY)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from core.registry import (
    METADATA_FILE,
    WEIGHTS_FILE,
    load_model,
    save_model,
)

TRAINING_STATE_FILE = "training_state.pt"


def _load_verified_state(model_dir: Path) -> dict:
    """Load training state with the weights file checksum verified first and
    safe (non-pickle) deserialization for tensors-only state dicts."""
    metadata = json.loads((model_dir / METADATA_FILE).read_text())
    digest = hashlib.sha256((model_dir / WEIGHTS_FILE).read_bytes()).hexdigest()
    if digest != metadata["weights_sha256"]:
        raise ValueError(
            f"weights file checksum {digest[:12]}… does not match metadata "
            f"{metadata['weights_sha256'][:12]}…; refusing to load the artifact"
        )
    return torch.load(model_dir / TRAINING_STATE_FILE, map_location="cpu",
                      weights_only=True)


def save_training_checkpoint(
    model: torch.nn.Module,
    training_state: dict,
    out_dir: str | Path,
    source_commit: str | None = None,
    smoke_only: bool = True,
    extra_state: dict | None = None,
) -> Path:
    model_config = getattr(model, "model_config", None)
    if model_config is None:
        raise ValueError(
            "model was not built by a default_*_model factory; cannot checkpoint"
        )
    out = save_model(model, model_config, out_dir, source_commit=source_commit)
    state = dict(training_state)
    if extra_state:
        state.update(extra_state)
    # explicit marker either way: the loader refuses a missing marker, so a
    # real-data run (smoke_only=False) is visible in provenance, never silent
    state["smoke_only"] = bool(smoke_only)
    torch.save(state, out / TRAINING_STATE_FILE)
    return out


def load_training_checkpoint(
    model_dir: str | Path, device: str = "cpu"
) -> tuple[torch.nn.Module, dict, dict]:
    """Returns (model, training_state, metadata); checksums verified by the registry."""
    model_dir = Path(model_dir)
    if not (model_dir / TRAINING_STATE_FILE).exists():
        raise FileNotFoundError(
            f"{model_dir} has no {TRAINING_STATE_FILE}; "
            "use core.registry.load_model for inference-only artifacts"
        )
    model, metadata = load_model(model_dir, device=device)
    state = _load_verified_state(model_dir)
    if "smoke_only" not in state:
        raise ValueError(
            "checkpoint is missing the explicit smoke_only marker; refusing to trust it"
        )
    if not state["smoke_only"] and not state.get("manifest_sha256"):
        raise ValueError(
            "SMOKE_ONLY marker flipped to False without real-data manifest "
            "provenance (manifest_sha256); refusing to trust the checkpoint"
        )
    return model, state, metadata


def checkpoint_summary(model_dir: str | Path) -> dict:
    """Human/CLI-facing provenance summary without loading weights."""
    model_dir = Path(model_dir)
    metadata = json.loads((model_dir / METADATA_FILE).read_text())
    state = _load_verified_state(model_dir)
    return {
        "model_type": metadata["model_type"],
        "weights_sha256": metadata["weights_sha256"],
        "weights_file": WEIGHTS_FILE,
        "epochs_completed": state.get("epoch"),
        "global_step": state.get("global_step"),
        "best_eval_loss": state.get("best_eval_loss"),
        "seed": state.get("trainer_config", {}).get("seed"),
        "smoke_only": state.get("smoke_only"),
    }
