"""Training checkpoints (#49): registry weights/metadata + optimizer/epoch/RNG
state, with full provenance (seed, torch version, source commit, SMOKE_ONLY)."""

from __future__ import annotations

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


def save_training_checkpoint(
    model: torch.nn.Module,
    training_state: dict,
    out_dir: str | Path,
    source_commit: str | None = None,
) -> Path:
    model_config = getattr(model, "model_config", None)
    if model_config is None:
        raise ValueError(
            "model was not built by a default_*_model factory; cannot checkpoint"
        )
    out = save_model(model, model_config, out_dir, source_commit=source_commit)
    state = dict(training_state)
    state["smoke_only"] = True  # no real-data training exists in this phase
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
    state = torch.load(model_dir / TRAINING_STATE_FILE, map_location="cpu", weights_only=False)
    if not state.get("smoke_only", False):
        raise ValueError("checkpoint is missing the SMOKE_ONLY marker; refusing to trust it")
    return model, state, metadata


def checkpoint_summary(model_dir: str | Path) -> dict:
    """Human/CLI-facing provenance summary without loading weights."""
    model_dir = Path(model_dir)
    metadata = json.loads((model_dir / METADATA_FILE).read_text())
    state = torch.load(model_dir / TRAINING_STATE_FILE, map_location="cpu", weights_only=False)
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
