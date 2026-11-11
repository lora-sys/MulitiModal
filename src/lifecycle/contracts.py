"""Versioned, immutable lifecycle contracts (#104-#112).

One vocabulary for the whole ML lifecycle: source -> dataset -> snapshot ->
features/targets -> train -> artifact -> gateway -> observe. Every manifest is
frozen, carries a sha256 digest, and names the evidence level and approval state
that produced it. Nothing in this module reads real data or starts training.

Deliberately dependency-free (stdlib only) so a clean CI install can import it on
any supported Python without torch, sklearn, or the private dataset.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CONTRACTS_VERSION = "ml-lifecycle-contracts-v1"

# Evidence and gate states. The three gates are independent: a green engineering
# run can never promote GO_REAL_DATA_TRAINING or GO_RESEARCH_CLAIMS.
EVIDENCE_SMOKE = "SMOKE_ONLY"
EVIDENCE_LOCAL = "LOCAL_EXPERIMENTAL"
EVIDENCE_SYNTHETIC = "SYNTHETIC"
EVIDENCE_LEVELS = (EVIDENCE_SMOKE, EVIDENCE_LOCAL, EVIDENCE_SYNTHETIC)

ISOLATION_UNVERIFIED = "AGENT_ISOLATION_UNVERIFIED"

GATE_ENGINEERING = "GO_ENGINEERING"
GATE_REAL_DATA_TRAINING = "GO_REAL_DATA_TRAINING"
GATE_RESEARCH_CLAIMS = "GO_RESEARCH_CLAIMS"
GATE_GRANTED = "GRANTED"
GATE_NOT_GRANTED = "NOT_GRANTED"
GATE_BLOCKED = "BLOCKED_BY_DATA_OR_DEVICE"

APPROVAL_PENDING = "PENDING"
APPROVAL_APPROVED = "APPROVED"
APPROVAL_REJECTED = "REJECTED"
APPROVAL_UNVERIFIED = "UNVERIFIED"
APPROVAL_STATES = (
    APPROVAL_PENDING,
    APPROVAL_APPROVED,
    APPROVAL_REJECTED,
    APPROVAL_UNVERIFIED,
)

TRUST_TRUSTED = "TRUSTED"
TRUST_SHARED = "SHARED_ACCOUNT_UNKNOWN"
TRUST_UNKNOWN = "UNKNOWN"
TRUST_STATES = (TRUST_TRUSTED, TRUST_SHARED, TRUST_UNKNOWN)

# Source capability states. NOT_CONNECTED is a hard fact about real devices and
# may never be overridden by a mock or simulated source.
SOURCE_DEVICE = "device"
SOURCE_SERVICE = "service"
SOURCE_OFFLINE = "offline"
SOURCE_SIMULATED = "simulated"
SOURCE_KINDS = (SOURCE_DEVICE, SOURCE_SERVICE, SOURCE_OFFLINE, SOURCE_SIMULATED)
CAP_CONNECTED = "CONNECTED"
CAP_NOT_CONNECTED = "NOT_CONNECTED"
CAP_DEGRADED = "DEGRADED"
CAP_UNAVAILABLE = "UNAVAILABLE"

STATUS_OK = "OK"
STATUS_ABSTAIN = "ABSTAIN"
STATUS_NOT_AVAILABLE = "NOT_AVAILABLE"
STATUS_BLOCKED = "BLOCKED"
STATUS_SAFE_STOP = "SAFE_STOP"
STATUS_STALE = "STALE"
STATUS_REJECTED = "REJECTED"
TERMINAL_STATUSES = (
    STATUS_OK,
    STATUS_ABSTAIN,
    STATUS_NOT_AVAILABLE,
    STATUS_BLOCKED,
    STATUS_SAFE_STOP,
    STATUS_STALE,
    STATUS_REJECTED,
)

# Lifecycle stages, in order. Each stage is independently gateable.
STAGES = (
    "ingest",
    "normalize",
    "validate",
    "reconcile",
    "snapshot",
    "ready_gate",
    "features",
    "prepare",
    "train",
    "validate_model",
    "compare",
    "package",
    "promote",
    "deploy",
    "serve",
    "observe",
)


class ContractError(ValueError):
    """Raised when a manifest would fabricate provenance or approval."""


def canonical_json(payload: Any) -> str:
    """Stable JSON used for every digest: sorted keys, compact, ASCII-safe."""
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def digest(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def file_sha256(path: str) -> str:
    """Streaming sha256 of a raw file. Raw data is read-only here."""
    handle = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(65536), b""):
            handle.update(chunk)
    return handle.hexdigest()


def utc_now() -> datetime:
    return datetime.now(UTC)


def atomic_write_text(path: str | Path, text: str) -> None:
    """Replace a file atomically so a reader never sees a truncated one.

    A process killed mid-write would otherwise leave invalid JSON behind, and
    every later run would fail to read its own state until someone deleted it.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _check_hex(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or set(value) - set("0123456789abcdef")
    ):
        raise ContractError(f"{name} must be a lowercase hex sha256 digest")


def _check_version(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{name} must be a non-empty version string")


@dataclass(frozen=True)
class VersionRef:
    """A named, versioned, hashed reference to an input or rule set."""

    name: str
    version: str
    sha256: str

    def __post_init__(self) -> None:
        _check_version(self.name, "name")
        _check_version(self.version, "version")
        _check_hex(self.sha256, "sha256")

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "version": self.version, "sha256": self.sha256}


@dataclass(frozen=True)
class SourceManifest:
    """One raw source. `sha256` is over the exact raw bytes, never a derived row."""

    source_id: str
    kind: str
    uri: str
    sha256: str
    license: str
    provenance: str
    identity_trust: str
    capability: str
    schema_version: str
    declared_at: str = ""

    def __post_init__(self) -> None:
        if self.kind not in SOURCE_KINDS:
            raise ContractError(f"unknown source kind {self.kind!r}")
        if self.capability not in (CAP_CONNECTED, CAP_NOT_CONNECTED, CAP_DEGRADED, CAP_UNAVAILABLE):
            raise ContractError(f"unknown capability {self.capability!r}")
        if self.identity_trust not in TRUST_STATES:
            raise ContractError(f"unknown identity_trust {self.identity_trust!r}")
        _check_hex(self.sha256, "source sha256")
        _check_version(self.schema_version, "schema_version")
        if self.kind == SOURCE_DEVICE and self.capability == CAP_CONNECTED:
            raise ContractError(
                "a real device may not declare CONNECTED without a vendor conformance record"
            )

    @property
    def id(self) -> str:
        return digest(self.to_dict())

    def to_dict(self) -> dict[str, str]:
        return {
            "source_id": self.source_id,
            "kind": self.kind,
            "uri": self.uri,
            "sha256": self.sha256,
            "license": self.license,
            "provenance": self.provenance,
            "identity_trust": self.identity_trust,
            "capability": self.capability,
            "schema_version": self.schema_version,
            "declared_at": self.declared_at,
        }


@dataclass(frozen=True)
class FieldSpec:
    """One feature in the data dictionary. Mask and fit scope are explicit."""

    name: str
    unit: str
    dtype: str
    timepoint: str
    window: str
    source: str
    direction: str
    fit_scope: str
    mask: str
    legal_range: tuple[float, float] | None = None
    unavailable_reason: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ContractError("FieldSpec.name is required")
        if self.timepoint not in ("pre", "during", "post", "static"):
            raise ContractError(f"unknown timepoint {self.timepoint!r}")
        if self.fit_scope not in ("train_fold", "none", "frozen"):
            raise ContractError(f"unknown fit_scope {self.fit_scope!r}")
        if self.legal_range is not None and self.legal_range[0] >= self.legal_range[1]:
            raise ContractError("legal_range must be (low, high) with low < high")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "unit": self.unit,
            "dtype": self.dtype,
            "timepoint": self.timepoint,
            "window": self.window,
            "source": self.source,
            "direction": self.direction,
            "fit_scope": self.fit_scope,
            "mask": self.mask,
            "legal_range": list(self.legal_range) if self.legal_range else None,
            "unavailable_reason": self.unavailable_reason,
        }


@dataclass(frozen=True)
class FeatureSpec:
    """Versioned feature contract shared by training and inference."""

    version: str
    fields: tuple[FieldSpec, ...]
    transform_id: str
    evidence_level: str = EVIDENCE_SYNTHETIC

    def __post_init__(self) -> None:
        _check_version(self.version, "feature version")
        _check_version(self.transform_id, "transform_id")
        names = [f.name for f in self.fields]
        if not names or len(names) != len(set(names)):
            raise ContractError("FeatureSpec fields must be non-empty and unique")

    @property
    def sha256(self) -> str:
        return digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "fields": [f.to_dict() for f in self.fields],
            "transform_id": self.transform_id,
            "evidence_level": self.evidence_level,
        }


@dataclass(frozen=True)
class TargetSpec:
    """Versioned target contract. The output head must match `legal_range`."""

    name: str
    version: str
    head: str
    legal_range: tuple[float, float] | None
    unit: str
    timepoint: str
    provenance: str
    approval: str
    unavailable_reason: str = ""

    def __post_init__(self) -> None:
        _check_version(self.version, "target version")
        if self.approval not in APPROVAL_STATES:
            raise ContractError(f"unknown approval {self.approval!r}")
        if self.timepoint not in ("pre", "during", "post", "derived"):
            raise ContractError(f"unknown timepoint {self.timepoint!r}")
        if self.legal_range is not None and self.legal_range[0] >= self.legal_range[1]:
            raise ContractError("legal_range must be (low, high) with low < high")
        if self.approval != APPROVAL_APPROVED and not self.unavailable_reason:
            raise ContractError(
                "an unapproved target must state why it is unavailable; "
                "no fabricated fatigue ground truth is permitted"
            )

    @property
    def sha256(self) -> str:
        return digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "head": self.head,
            "legal_range": list(self.legal_range) if self.legal_range else None,
            "unit": self.unit,
            "timepoint": self.timepoint,
            "provenance": self.provenance,
            "approval": self.approval,
            "unavailable_reason": self.unavailable_reason,
        }


@dataclass(frozen=True)
class QCReport:
    """Per-batch quality counts. Every rejected row carries a reason."""

    n_input: int
    n_accepted: int
    n_rejected: int
    n_duplicate: int
    n_imputed: int
    rejections: tuple[dict[str, str], ...] = ()
    drift: tuple[str, ...] = ()
    blocked: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.n_accepted + self.n_rejected + self.n_duplicate != self.n_input:
            raise ContractError(
                "accepted + rejected + duplicate must equal input rows"
            )
        if any(not r.get("reason") for r in self.rejections):
            raise ContractError("every rejected row needs a reason (no silent skip)")

    @property
    def sha256(self) -> str:
        return digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_input": self.n_input,
            "n_accepted": self.n_accepted,
            "n_rejected": self.n_rejected,
            "n_duplicate": self.n_duplicate,
            "n_imputed": self.n_imputed,
            "rejections": [dict(r) for r in self.rejections],
            "drift": list(self.drift),
            "blocked": list(self.blocked),
        }


@dataclass(frozen=True)
class PipelineState:
    """Durable, resumable pipeline cursor. Written as JSON, never a database."""

    run_id: str
    stage: str
    cursor: str
    processed_keys: tuple[str, ...]
    dataset_sha256: str
    schema_version: str
    code_version: str
    config_sha256: str
    rules_sha256: str
    status: str
    started_at: str
    updated_at: str
    error: str = ""

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ContractError(f"unknown stage {self.stage!r}")
        if self.status not in ("RUNNING", "PAUSED", "DONE", "FAILED", "BLOCKED"):
            raise ContractError(f"unknown status {self.status!r}")
        _check_hex(self.dataset_sha256, "dataset_sha256")
        _check_hex(self.config_sha256, "config_sha256")
        _check_hex(self.rules_sha256, "rules_sha256")

    @property
    def cache_key(self) -> str:
        """Recompute only when source hash, code, config or rules change."""
        return digest(
            {
                "dataset": self.dataset_sha256,
                "schema": self.schema_version,
                "code": self.code_version,
                "config": self.config_sha256,
                "rules": self.rules_sha256,
                "stage": self.stage,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "stage": self.stage,
            "cursor": self.cursor,
            "processed_keys": list(self.processed_keys),
            "dataset_sha256": self.dataset_sha256,
            "schema_version": self.schema_version,
            "code_version": self.code_version,
            "config_sha256": self.config_sha256,
            "rules_sha256": self.rules_sha256,
            "status": self.status,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "error": self.error,
            "cache_key": self.cache_key,
        }


@dataclass(frozen=True)
class DatasetManifest:
    """A frozen, approved dataset a training run may consume."""

    dataset_id: str
    source_sha256: tuple[str, ...]
    schema_version: str
    label_version: str
    feature_version: str
    qc_sha256: str
    n_rows: int
    n_subjects: int
    identity_trust: str
    evidence_level: str
    approval: str
    frozen: bool
    created_at: str

    def __post_init__(self) -> None:
        if self.identity_trust not in TRUST_STATES:
            raise ContractError(f"unknown identity_trust {self.identity_trust!r}")
        if self.approval not in APPROVAL_STATES:
            raise ContractError(f"unknown approval {self.approval!r}")
        if self.frozen and self.approval != APPROVAL_APPROVED:
            raise ContractError("a frozen dataset must be approved")
        for value in self.source_sha256:
            _check_hex(value, "source_sha256 entry")
        _check_hex(self.qc_sha256, "qc_sha256")

    @property
    def sha256(self) -> str:
        return digest(self.to_dict())

    def trainable(self) -> tuple[bool, str]:
        """Hard subject-identity and approval gate for real training."""
        if self.approval != APPROVAL_APPROVED:
            return False, f"dataset approval is {self.approval}"
        if self.identity_trust != TRUST_TRUSTED:
            return False, f"subject identity is {self.identity_trust}"
        if not self.frozen:
            return False, "dataset is not frozen"
        if self.evidence_level == EVIDENCE_SMOKE:
            return False, "SMOKE_ONLY evidence cannot enter real training"
        return True, "admitted"

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "source_sha256": list(self.source_sha256),
            "schema_version": self.schema_version,
            "label_version": self.label_version,
            "feature_version": self.feature_version,
            "qc_sha256": self.qc_sha256,
            "n_rows": self.n_rows,
            "n_subjects": self.n_subjects,
            "identity_trust": self.identity_trust,
            "evidence_level": self.evidence_level,
            "approval": self.approval,
            "frozen": self.frozen,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class SplitManifest:
    """Subject-level folds. Same subject never crosses folds."""

    version: str
    seed: int
    n_folds: int
    assignment_sha256: str
    trainable_subjects: tuple[str, ...]
    held_out: tuple[str, ...]

    def __post_init__(self) -> None:
        _check_hex(self.assignment_sha256, "assignment_sha256")
        overlap = set(self.trainable_subjects) & set(self.held_out)
        if overlap:
            raise ContractError(f"subject leakage across folds: {sorted(overlap)}")

    @property
    def sha256(self) -> str:
        return digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "seed": self.seed,
            "n_folds": self.n_folds,
            "assignment_sha256": self.assignment_sha256,
            "trainable_subjects": list(self.trainable_subjects),
            "held_out": list(self.held_out),
        }


@dataclass(frozen=True)
class EvalSpec:
    """One frozen evaluator. Status is the authority, never a raw metric."""

    version: str
    metric: str
    min_samples: int
    min_subjects: int
    status: str = "INSUFFICIENT_EVIDENCE"

    def __post_init__(self) -> None:
        _check_version(self.version, "eval version")
        if self.min_samples < 1 or self.min_subjects < 1:
            raise ContractError("minimums must be positive")

    @property
    def sha256(self) -> str:
        return digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "metric": self.metric,
            "min_samples": self.min_samples,
            "min_subjects": self.min_subjects,
            "status": self.status,
        }


@dataclass(frozen=True)
class TrainConfig:
    """Immutable, replayable training configuration."""

    seed: int
    model_kind: str
    epochs: int
    evidence_level: str
    commit: str
    environment: str
    license: str
    approval: str
    budget_s: float = 60.0

    def __post_init__(self) -> None:
        if self.approval not in APPROVAL_STATES:
            raise ContractError(f"unknown approval {self.approval!r}")
        if self.evidence_level not in EVIDENCE_LEVELS:
            raise ContractError(f"unknown evidence_level {self.evidence_level!r}")
        if self.budget_s <= 0:
            raise ContractError("budget_s must be positive")

    @property
    def sha256(self) -> str:
        return digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "model_kind": self.model_kind,
            "epochs": self.epochs,
            "evidence_level": self.evidence_level,
            "commit": self.commit,
            "environment": self.environment,
            "license": self.license,
            "approval": self.approval,
            "budget_s": self.budget_s,
        }


@dataclass(frozen=True)
class ArtifactManifest:
    """Everything a deployed artifact is bound to. Never self-promotes."""

    artifact_id: str
    model_sha256: str
    feature_sha256: str
    target_sha256: str
    scaler_sha256: str
    dataset_sha256: str
    split_sha256: str
    eval_sha256: str
    commit: str
    dependencies: tuple[str, ...]
    license: str
    approval: str
    validation_status: str
    evidence_level: str

    def __post_init__(self) -> None:
        for name in (
            "model_sha256", "feature_sha256", "target_sha256", "scaler_sha256",
            "dataset_sha256", "split_sha256", "eval_sha256",
        ):
            _check_hex(getattr(self, name), name)
        if self.approval not in APPROVAL_STATES:
            raise ContractError(f"unknown approval {self.approval!r}")
        if self.validation_status == "VALIDATED" and self.approval != APPROVAL_APPROVED:
            raise ContractError("VALIDATED requires an APPROVED record")
        if (
            self.validation_status == "VALIDATED"
            and self.evidence_level in (EVIDENCE_SMOKE, EVIDENCE_SYNTHETIC)
        ):
            raise ContractError("a SMOKE/SYNTHETIC artifact cannot be VALIDATED")

    @property
    def sha256(self) -> str:
        return digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "model_sha256": self.model_sha256,
            "feature_sha256": self.feature_sha256,
            "target_sha256": self.target_sha256,
            "scaler_sha256": self.scaler_sha256,
            "dataset_sha256": self.dataset_sha256,
            "split_sha256": self.split_sha256,
            "eval_sha256": self.eval_sha256,
            "commit": self.commit,
            "dependencies": list(self.dependencies),
            "license": self.license,
            "approval": self.approval,
            "validation_status": self.validation_status,
            "evidence_level": self.evidence_level,
        }


@dataclass(frozen=True)
class Envelope:
    """Versioned gateway request/response envelope. No raw physiology in logs."""

    schema_version: str
    request_id: str
    session_id: str
    trace_id: str
    mode: str
    event_time: str
    feature_version: str
    model_version: str
    target_version: str
    quality: str
    status: str
    provenance: str
    license: str
    evidence_level: str
    display_gate: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in TERMINAL_STATUSES:
            raise ContractError(f"unknown status {self.status!r}")
        if self.mode not in ("A_pre_only", "B_realtime", "C_tcm"):
            raise ContractError(f"unknown mode {self.mode!r}")
        if (
            self.status == STATUS_OK
            and self.evidence_level == EVIDENCE_SMOKE
            and self.mode != "A_pre_only"
        ):
            raise ContractError("a SMOKE model may only answer research mode A")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "trace_id": self.trace_id,
            "mode": self.mode,
            "event_time": self.event_time,
            "feature_version": self.feature_version,
            "model_version": self.model_version,
            "target_version": self.target_version,
            "quality": self.quality,
            "status": self.status,
            "provenance": self.provenance,
            "license": self.license,
            "evidence_level": self.evidence_level,
            "display_gate": self.display_gate,
            "payload": dict(self.payload),
        }


@dataclass(frozen=True)
class StageEvidence:
    """Per-stage GO/NO-GO record, kept separate from raw test counts."""

    stage: str
    go: bool
    evidence_level: str
    gate: str
    reason: str
    artifacts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ContractError(f"unknown stage {self.stage!r}")
        if self.gate not in (GATE_GRANTED, GATE_NOT_GRANTED, GATE_BLOCKED):
            raise ContractError(f"unknown gate {self.gate!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "go": self.go,
            "evidence_level": self.evidence_level,
            "gate": self.gate,
            "reason": self.reason,
            "artifacts": list(self.artifacts),
        }


def gate_bundle(engineering_go: bool) -> dict[str, str]:
    """Engineering success must never promote the two research gates."""
    return {
        GATE_ENGINEERING: GATE_GRANTED if engineering_go else GATE_NOT_GRANTED,
        GATE_REAL_DATA_TRAINING: GATE_NOT_GRANTED,
        GATE_RESEARCH_CLAIMS: GATE_NOT_GRANTED,
    }
