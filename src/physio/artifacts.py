"""Validated model-artifact binding (#97).

An ``ModelArtifact`` binds the evidence gate to concrete bytes: the scaler
digest, the model state digest and the feature/target versions must all match
the objects handed to inference. A self-reported ``VALIDATED`` string alone is
not enough; a mismatched or missing artifact always yields ``NOT_AVAILABLE``.

The binding does not create validation evidence. ``validation_status`` remains
an external, human-reviewed decision recorded in the artifact; this module only
makes sure the artifact cannot be swapped for a different model or scaler.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .model import (
    REALTIME_TARGET_VERSION,
    VALIDATION_NOT_VALIDATED,
    VALIDATION_STATUSES,
    VALIDATION_VALIDATED,
)
from .preprocessing import WindowScaler
from .windows import WINDOW_FEATURE_VERSION

ARTIFACT_SCHEMA = "physio-model-artifact-v1"

_HEX_DIGITS = set("0123456789abcdef")


class ArtifactError(ValueError):
    """Raised when an artifact is malformed or does not match its objects."""


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= _HEX_DIGITS
    )


def _canonical_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ModelArtifact:
    """Version and digest binding for one validated model/scaler pair."""

    model_version: str
    scaler_sha256: str
    state_sha256: str
    feature_version: str = WINDOW_FEATURE_VERSION
    target_version: str = REALTIME_TARGET_VERSION
    validation_status: str = VALIDATION_NOT_VALIDATED
    manifest_sha256: str | None = None
    schema: str = ARTIFACT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != ARTIFACT_SCHEMA:
            raise ValueError(f"unsupported artifact schema: {self.schema!r}")
        for name in ("model_version", "feature_version", "target_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("scaler_sha256", "state_sha256"):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"{name} must be a lowercase hex sha256 digest")
        if self.manifest_sha256 is not None and not _is_sha256(self.manifest_sha256):
            raise ValueError("manifest_sha256 must be a lowercase hex sha256 digest")
        if self.validation_status not in VALIDATION_STATUSES:
            raise ValueError(
                f"validation_status must be one of {VALIDATION_STATUSES}, "
                f"got {self.validation_status!r}"
            )

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "model_version": self.model_version,
            "feature_version": self.feature_version,
            "target_version": self.target_version,
            "scaler_sha256": self.scaler_sha256,
            "state_sha256": self.state_sha256,
            "validation_status": self.validation_status,
            "manifest_sha256": self.manifest_sha256,
        }

    def digest(self) -> str:
        return _canonical_digest(self.payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "sha256": self.digest()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelArtifact:
        if not isinstance(data, dict):
            raise ArtifactError("artifact must be a JSON object")
        if data.get("schema") != ARTIFACT_SCHEMA:
            raise ArtifactError(f"unsupported artifact schema: {data.get('schema')!r}")
        payload = {key: value for key, value in data.items() if key != "sha256"}
        if data.get("sha256") != _canonical_digest(payload):
            raise ArtifactError("artifact sha256 mismatch (corrupt or edited)")
        try:
            return cls(
                model_version=str(payload["model_version"]),
                scaler_sha256=str(payload["scaler_sha256"]),
                state_sha256=str(payload["state_sha256"]),
                feature_version=str(payload["feature_version"]),
                target_version=str(payload["target_version"]),
                validation_status=str(payload["validation_status"]),
                manifest_sha256=(
                    str(payload["manifest_sha256"])
                    if payload.get("manifest_sha256") is not None
                    else None
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactError(f"malformed artifact: {exc}") from exc

    @classmethod
    def from_manifest(
        cls,
        manifest: Any,
        scaler: WindowScaler,
        *,
        model_version: str | None = None,
        validation_status: str = VALIDATION_NOT_VALIDATED,
    ) -> ModelArtifact:
        """Build an artifact from a completed training manifest.

        The manifest's scaler digest must already match ``scaler``; otherwise the
        manifest and the runtime scaler describe different preprocessing.
        """
        if manifest.scaler_sha256 != scaler.digest():
            raise ArtifactError(
                "manifest scaler_sha256 does not match the supplied scaler"
            )
        if not _is_sha256(manifest.state_sha256):
            raise ArtifactError("manifest state_sha256 is missing or malformed")
        if validation_status == VALIDATION_VALIDATED and manifest.status != "REAL_DATA":
            raise ArtifactError(
                "a SMOKE_ONLY manifest cannot be promoted to a VALIDATED artifact"
            )
        return cls(
            model_version=model_version or manifest.model_version,
            scaler_sha256=manifest.scaler_sha256,
            state_sha256=manifest.state_sha256,
            feature_version=WINDOW_FEATURE_VERSION,
            target_version=REALTIME_TARGET_VERSION,
            validation_status=validation_status,
            manifest_sha256=manifest.digest(),
        )

    def verify(self, scaler: WindowScaler, model: Any) -> None:
        """Raise ``ArtifactError`` unless this binding matches both objects."""
        if self.feature_version != WINDOW_FEATURE_VERSION:
            raise ArtifactError(
                f"artifact feature_version {self.feature_version!r} does not match "
                f"{WINDOW_FEATURE_VERSION!r}"
            )
        if self.target_version != REALTIME_TARGET_VERSION:
            raise ArtifactError(
                f"artifact target_version {self.target_version!r} does not match "
                f"{REALTIME_TARGET_VERSION!r}"
            )
        try:
            scaler_digest = scaler.digest()
        except (AttributeError, TypeError) as exc:
            raise ArtifactError("scaler does not expose a digest") from exc
        if self.scaler_sha256 != scaler_digest:
            raise ArtifactError("artifact scaler digest does not match the runtime scaler")
        digest = getattr(model, "state_sha256", None)
        if not callable(digest):
            raise ArtifactError("model does not expose a state_sha256 digest")
        try:
            state_digest = digest()
        except Exception as exc:  # noqa: BLE001 - a broken artifact must not score
            raise ArtifactError(
                f"cannot read the model state digest: {type(exc).__name__}"
            ) from exc
        if self.state_sha256 != state_digest:
            raise ArtifactError("artifact state digest does not match the runtime model")

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> ModelArtifact:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except OSError as exc:
            raise ArtifactError(f"cannot read artifact: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ArtifactError(f"artifact is not valid JSON: {exc}") from exc
        return cls.from_dict(data)


__all__ = [
    "ARTIFACT_SCHEMA",
    "ArtifactError",
    "ModelArtifact",
    "VALIDATION_NOT_VALIDATED",
    "VALIDATION_VALIDATED",
]
