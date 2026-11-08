"""Torch-optional smoke training for the realtime window model (#95/#97).

This module never turns synthetic data into a research claim. It exists to
prove the engineering path: masked loss, finite gradients, deterministic seed,
checkpoint save/reload and an experiment manifest. Real training requires
approved labels and is refused otherwise.

The module imports without PyTorch. On hosts without torch every training
entry raises ``TrainingUnavailable`` instead of pretending to train.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dataset import IDENTITY_TRUSTED, FeatureRow
from .model import REALTIME_MODEL_VERSION, RealtimeModelConfig, WindowModelBatch
from .preprocessing import WindowScaler

TRAINING_SCHEMA_VERSION = "physio-realtime-training-v1"
EVIDENCE_SMOKE = "SMOKE_ONLY"
EVIDENCE_REAL = "REAL_DATA"
EVIDENCE_LEVELS = (EVIDENCE_SMOKE, EVIDENCE_REAL)

TARGET_KEY = "fatigue_state"


class TrainingRefused(RuntimeError):
    """Raised when training lacks a trustworthy label set or host capability."""


class TrainingUnavailable(TrainingRefused):
    """Raised when PyTorch is not installed."""


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 3
    learning_rate: float = 1e-3
    seed: int = 7
    min_samples: int = 4
    hidden_dim: int = 16
    fusion_dim: int = 16
    dropout: float = 0.0
    evidence_level: str = EVIDENCE_SMOKE

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.min_samples < 1:
            raise ValueError("min_samples must be positive")
        if self.hidden_dim <= 0 or self.fusion_dim <= 0:
            raise ValueError("hidden_dim and fusion_dim must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be within [0, 1)")
        if self.evidence_level not in EVIDENCE_LEVELS:
            raise ValueError(f"unknown evidence_level: {self.evidence_level!r}")

    def payload(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "min_samples": self.min_samples,
            "hidden_dim": self.hidden_dim,
            "fusion_dim": self.fusion_dim,
            "dropout": self.dropout,
            "evidence_level": self.evidence_level,
        }


@dataclass(frozen=True)
class TrainingManifest:
    model_version: str
    feature_names: tuple[str, ...]
    scaler_sha256: str
    config: TrainingConfig
    n_samples: int
    n_labeled: int
    identity_trust: str
    status: str
    checkpoint_sha256: str
    state_sha256: str
    resume_from: str | None = None
    final_loss: float | None = None
    schema_version: str = TRAINING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TRAINING_SCHEMA_VERSION:
            raise ValueError(f"unsupported training schema: {self.schema_version!r}")
        if not self.feature_names:
            raise ValueError("feature_names must be non-empty")
        if self.n_samples < 0 or self.n_labeled < 0 or self.n_labeled > self.n_samples:
            raise ValueError("invalid sample counts")
        for name in ("scaler_sha256", "checkpoint_sha256", "state_sha256"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or set(value) - set("0123456789abcdef")
            ):
                raise ValueError(f"{name} must be a lowercase hex sha256 digest")
        if self.final_loss is not None and not math.isfinite(self.final_loss):
            raise ValueError("final_loss must be finite")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema_version,
            "model_version": self.model_version,
            "feature_names": list(self.feature_names),
            "scaler_sha256": self.scaler_sha256,
            "config": self.config.payload(),
            "n_samples": self.n_samples,
            "n_labeled": self.n_labeled,
            "identity_trust": self.identity_trust,
            "status": self.status,
            "checkpoint_sha256": self.checkpoint_sha256,
            "state_sha256": self.state_sha256,
            "resume_from": self.resume_from,
            "final_loss": self.final_loss,
        }

    def digest(self) -> str:
        encoded = json.dumps(
            self.payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "sha256": self.digest()}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> TrainingManifest:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("schema") != TRAINING_SCHEMA_VERSION:
            raise ValueError(f"unsupported manifest schema: {data.get('schema')!r}")
        config = TrainingConfig(**data["config"])
        manifest = cls(
            model_version=str(data["model_version"]),
            feature_names=tuple(str(name) for name in data["feature_names"]),
            scaler_sha256=str(data["scaler_sha256"]),
            config=config,
            n_samples=int(data["n_samples"]),
            n_labeled=int(data["n_labeled"]),
            identity_trust=str(data["identity_trust"]),
            status=str(data["status"]),
            checkpoint_sha256=str(data["checkpoint_sha256"]),
            state_sha256=str(data["state_sha256"]),
            resume_from=data.get("resume_from"),
            final_loss=(
                float(data["final_loss"]) if data.get("final_loss") is not None else None
            ),
        )
        if data.get("sha256") != manifest.digest():
            raise ValueError("manifest sha256 mismatch (corrupt or edited)")
        return manifest


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise TrainingUnavailable(
            "PyTorch is required for smoke training; install torch on the training host"
        ) from exc
    return torch


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_trainable(
    rows: Sequence[FeatureRow],
    targets: Sequence[float | None],
    *,
    config: TrainingConfig,
    identity_trust: str,
    allow_untrusted_pipeline: bool = False,
) -> tuple[int, int]:
    """Return ``(n_samples, n_labeled)`` or refuse with a concrete reason."""
    if len(rows) != len(targets):
        raise TrainingRefused("rows and targets must have equal length")
    if len(rows) < config.min_samples:
        raise TrainingRefused(
            f"at least {config.min_samples} samples are required for smoke training"
        )
    labeled = 0
    for index, (row, target) in enumerate(zip(rows, targets, strict=True)):
        if target is None:
            continue
        if not math.isfinite(float(target)):
            raise TrainingRefused(
                f"target at row {index} (session {row.session_id!r}) must be finite; "
                "NaN/Inf is not a label"
            )
        labeled += 1
        if not row.valid_mask:
            raise TrainingRefused(
                f"labeled row {index} (session {row.session_id!r}, subject "
                f"{row.subject_ref!r}) has no usable feature"
            )
    if labeled == 0:
        raise TrainingRefused("no available label; real training cannot start")
    if identity_trust != IDENTITY_TRUSTED and not allow_untrusted_pipeline:
        raise TrainingRefused(
            "shared/unknown identity may only run pipeline smoke with explicit opt-in "
            f"({len(rows)} rows, {labeled} labels, identity_trust={identity_trust!r})"
        )
    if config.evidence_level == EVIDENCE_REAL and identity_trust != IDENTITY_TRUSTED:
        raise TrainingRefused("real-data training requires verified subject identity")
    return len(rows), labeled


def train_realtime_smoke(
    rows: Sequence[FeatureRow],
    targets: Sequence[float | None],
    *,
    scaler: WindowScaler,
    config: TrainingConfig | None = None,
    identity_trust: str = IDENTITY_TRUSTED,
    allow_untrusted_pipeline: bool = False,
    out_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
) -> TrainingManifest:
    """Train one tiny masked-loss model on synthetic rows and save a checkpoint."""
    config = config or TrainingConfig()
    n_samples, n_labeled = assert_trainable(
        rows,
        targets,
        config=config,
        identity_trust=identity_trust,
        allow_untrusted_pipeline=allow_untrusted_pipeline,
    )
    torch = _require_torch()
    from .model import RealtimeFatigueModel

    if any(row.feature_names != scaler.feature_names for row in rows):
        raise TrainingRefused("rows do not share the scaler feature contract")

    torch.manual_seed(config.seed)
    model = RealtimeFatigueModel(
        len(scaler.feature_names),
        RealtimeModelConfig(
            hidden_dim=config.hidden_dim,
            fusion_dim=config.fusion_dim,
            dropout=config.dropout,
        ),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    target_tensor = torch.tensor(
        [[float(target) if target is not None else 0.0] for target in targets],
        dtype=torch.float32,
    )
    label_mask = torch.tensor(
        [[target is not None] for target in targets], dtype=torch.bool
    )
    if resume_from is not None:
        state = torch.load(Path(resume_from), map_location="cpu", weights_only=True)
        model.load_state_dict(state)

    feature_tensor = torch.tensor([row.values for row in rows], dtype=torch.float32)
    mask_tensor = torch.tensor([row.valid_mask for row in rows], dtype=torch.bool)

    model.train()
    last_loss: float | None = None
    from core.batch import ModelBatch

    batch = ModelBatch(
        features={"window_features": feature_tensor},
        availability={"window_features": mask_tensor.any(dim=1)},
    )
    for _epoch in range(config.epochs):
        optimizer.zero_grad()
        predictions = model(batch, feature_mask=mask_tensor).predictions[TARGET_KEY]
        squared = (predictions - target_tensor) ** 2
        masked = squared * label_mask.to(squared.dtype)
        loss = masked.sum() / label_mask.sum()
        if not torch.isfinite(loss):
            raise TrainingRefused("training produced a non-finite loss")
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise TrainingRefused("training produced a non-finite gradient")
        optimizer.step()
        last_loss = float(loss.detach())

    manifest_dir = Path(out_dir) if out_dir is not None else Path.cwd() / "runs" / "physio-smoke"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = manifest_dir / "model_state.pt"
    torch.save(model.state_dict(), checkpoint)
    manifest = TrainingManifest(
        model_version=REALTIME_MODEL_VERSION,
        feature_names=scaler.feature_names,
        scaler_sha256=scaler.digest(),
        config=config,
        n_samples=n_samples,
        n_labeled=n_labeled,
        identity_trust=identity_trust,
        status=config.evidence_level,
        checkpoint_sha256=_file_sha256(checkpoint),
        state_sha256=model.state_sha256(),
        resume_from=str(resume_from) if resume_from is not None else None,
        final_loss=last_loss,
    )
    manifest.save(manifest_dir / "manifest.json")
    return manifest


def load_realtime_model(
    *,
    checkpoint: str | Path,
    scaler: WindowScaler,
    hidden_dim: int = 16,
    fusion_dim: int = 16,
    dropout: float = 0.0,
):
    """Load a saved checkpoint into the model that matches ``scaler`` features."""
    torch = _require_torch()
    from .model import RealtimeFatigueModel

    model = RealtimeFatigueModel(
        len(scaler.feature_names),
        RealtimeModelConfig(hidden_dim=hidden_dim, fusion_dim=fusion_dim, dropout=dropout),
    )
    model.load_state_dict(torch.load(Path(checkpoint), map_location="cpu", weights_only=True))
    model.eval()
    return model


def predict_masked(
    rows: Sequence[FeatureRow],
    *,
    checkpoint: str | Path,
    scaler: WindowScaler,
    hidden_dim: int = 16,
    fusion_dim: int = 16,
    dropout: float = 0.0,
) -> tuple[float, ...]:
    """Reload a checkpoint and score rows with the same mask contract."""
    torch = _require_torch()
    model = load_realtime_model(
        checkpoint=checkpoint,
        scaler=scaler,
        hidden_dim=hidden_dim,
        fusion_dim=fusion_dim,
        dropout=dropout,
    )
    outputs: list[float] = []
    with torch.no_grad():
        for row in rows:
            batch = WindowModelBatch(
                feature_names=row.feature_names, values=row.values, mask=row.valid_mask
            )
            prediction = model(batch, feature_mask=row.valid_mask)
            outputs.append(float(prediction.predictions[TARGET_KEY].reshape(-1)[0]))
    return tuple(outputs)


__all__ = [
    "EVIDENCE_REAL",
    "EVIDENCE_SMOKE",
    "TARGET_KEY",
    "TRAINING_SCHEMA_VERSION",
    "TrainingConfig",
    "TrainingManifest",
    "TrainingRefused",
    "TrainingUnavailable",
    "assert_trainable",
    "load_realtime_model",
    "predict_masked",
    "train_realtime_smoke",
]
