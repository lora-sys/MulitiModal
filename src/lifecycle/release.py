"""Immutable release bundles, explicit champion transitions and measured reports (#109).

Registration never changes the champion. Promotion and rollback are separate,
logged transitions that replace one registry file atomically. Benchmark and
deployment reports describe supplied measurements only; they grant no gates.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .contracts import (
    APPROVAL_APPROVED,
    CAP_NOT_CONNECTED,
    CONTRACTS_VERSION,
    EVIDENCE_LEVELS,
    EVIDENCE_LOCAL,
    EVIDENCE_SMOKE,
    EVIDENCE_SYNTHETIC,
    STATUS_BLOCKED,
    STATUS_NOT_AVAILABLE,
    STATUS_OK,
    ArtifactManifest,
    canonical_json,
    digest,
    file_sha256,
    gate_bundle,
    utc_now,
)

RELEASE_VERSION = "ml-lifecycle-release-v1"
VALIDATION_VALIDATED = "VALIDATED"
VALIDATION_STATUSES = ("NOT_VALIDATED", "PENDING", VALIDATION_VALIDATED, "REJECTED")
PROMOTION_ALLOWED_EVIDENCE = (EVIDENCE_LOCAL,)
CPU_TARGET = "cpu-first"
SUPPORTED_HARDWARE = ("cpu",)
SUPPORTED_FORMATS = ("python-stdlib",)
COMPATIBILITY_KEYS = frozenset({"hardware", "format"})

_MEASURED = "MEASURED"
_HEX_DIGITS = frozenset("0123456789abcdef")
_RESERVED_EVIDENCE = frozenset({"evidence_level", "validation_status", "approval"})


class ReleaseError(ValueError):
    """Raised when a release transition or artifact binding is not admissible."""


def _check_hex(value: Any, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or set(value) - _HEX_DIGITS:
        raise ReleaseError(f"{name} must be a lowercase hex sha256 digest")


def _relative_path(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ReleaseError("file keys must be non-empty relative paths")
    path = Path(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != value
        or value.startswith("./")
        or "\\" in value
    ):
        raise ReleaseError(f"unsafe file path {value!r}")
    return path


def _non_negative(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ReleaseError(f"{name} must be a non-negative number")
    return float(value)


def _percentile(samples: Sequence[float], fraction: float) -> float:
    """Linear interpolation between order statistics (inclusive percentile)."""
    ordered = sorted(samples)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def _artifact_id(entry: Any) -> str | None:
    return entry.get("artifact_id") if isinstance(entry, dict) else None


@dataclass(frozen=True)
class ArtifactBundle:
    """One immutable artifact directory bound to an ArtifactManifest."""

    root: Path
    manifest: ArtifactManifest
    files: Mapping[str, str]
    schema_version: str = RELEASE_VERSION
    environment: str = "local-cpu"
    compatibility: Mapping[str, str] = field(default_factory=dict)
    extra_evidence: Mapping[str, Any] = field(default_factory=dict)
    bundle_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != RELEASE_VERSION:
            raise ReleaseError(f"unsupported release schema {self.schema_version!r}")
        object.__setattr__(self, "root", Path(self.root))
        if not isinstance(self.environment, str) or not self.environment.strip():
            raise ReleaseError("environment must be a non-empty string")
        items = []
        for key, value in dict(self.files).items():
            path = _relative_path(key)
            _check_hex(value, f"declared sha256 for {key}")
            items.append((path.as_posix(), value))
        normalized = dict(items)
        if not normalized:
            raise ReleaseError("a bundle must declare at least one file")
        if len(normalized) != len(items):
            raise ReleaseError("bundle file paths must be unique")
        object.__setattr__(self, "files", normalized)
        compatibility = dict(self.compatibility)
        if not all(isinstance(key, str) for key in compatibility):
            raise ReleaseError("compatibility keys must be strings")
        object.__setattr__(self, "compatibility", compatibility)
        extra = dict(self.extra_evidence)
        if set(extra) & _RESERVED_EVIDENCE:
            raise ReleaseError("extra_evidence cannot override manifest evidence fields")
        object.__setattr__(self, "extra_evidence", extra)
        if self.bundle_sha256 is not None:
            _check_hex(self.bundle_sha256, "bundle_sha256")
            if self.bundle_sha256 != digest(self.to_dict()):
                raise ReleaseError("bundle_sha256 mismatch (corrupt or edited)")

    @property
    def artifact_id(self) -> str:
        return self.manifest.artifact_id

    @property
    def sha256(self) -> str:
        return digest(self.to_dict())

    @classmethod
    def from_directory(
        cls,
        root: str | Path,
        manifest: ArtifactManifest,
        *,
        files: Sequence[str],
        environment: str = "local-cpu",
        compatibility: Mapping[str, str] | None = None,
        extra_evidence: Mapping[str, Any] | None = None,
    ) -> ArtifactBundle:
        directory = Path(root)
        hashes = {str(name): file_sha256(directory / name) for name in files}
        bundle = cls(
            root=directory,
            manifest=manifest,
            files=hashes,
            environment=environment,
            compatibility=dict(compatibility or {}),
            extra_evidence=dict(extra_evidence or {}),
        )
        return replace(bundle, bundle_sha256=bundle.sha256)

    def to_dict(self, *, include_hash: bool = False) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "contracts_version": CONTRACTS_VERSION,
            "artifact_id": self.artifact_id,
            "manifest": self.manifest.to_dict(),
            "manifest_sha256": self.manifest.sha256,
            "files": dict(self.files),
            "root": self.root.as_posix(),
            "dependencies": list(self.manifest.dependencies),
            "environment": self.environment,
            "license": self.manifest.license,
            "commit": self.manifest.commit,
            "evidence": {
                "evidence_level": self.manifest.evidence_level,
                "validation_status": self.manifest.validation_status,
                "approval": self.manifest.approval,
                **self.extra_evidence,
            },
            "compatibility": dict(self.compatibility),
        }
        if include_hash:
            payload["bundle_sha256"] = digest(payload)
        return payload

    def verify(self) -> dict[str, Any]:
        """Recompute every file and reject missing, changed or undeclared files."""
        if self.root.is_symlink() or not self.root.is_dir():
            raise ReleaseError(f"bundle root is not a plain directory: {self.root}")
        actual: dict[str, str] = {}
        for path in sorted(self.root.rglob("*"), key=lambda item: item.as_posix()):
            if path.is_symlink():
                raise ReleaseError(f"symlink in bundle is not allowed: {path.name}")
            if path.is_file():
                relative = path.relative_to(self.root).as_posix()
                try:
                    actual[relative] = file_sha256(str(path))
                except OSError as exc:
                    raise ReleaseError(f"cannot read bundle file {relative}: {exc}") from exc
            elif not path.is_dir():
                raise ReleaseError(f"unsupported bundle entry: {path.name}")
        declared = dict(self.files)
        missing = sorted(set(declared) - set(actual))
        extra = sorted(set(actual) - set(declared))
        changed = sorted(
            name for name in set(declared) & set(actual) if actual[name] != declared[name]
        )
        if missing or extra or changed:
            details = {"missing": missing, "extra": extra, "changed": changed}
            raise ReleaseError(f"bundle file verification failed: {canonical_json(details)}")
        return {
            "status": "VERIFIED",
            "artifact_id": self.artifact_id,
            "bundle_sha256": self.sha256,
            "files": declared,
        }


@dataclass(frozen=True)
class TransitionResult:
    """The outcome of a promotion or rollback, including its refusal reason."""

    action: str
    accepted: bool
    reason: str
    state: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "accepted": self.accepted,
            "reason": self.reason,
            "state": dict(self.state),
        }


class ReleaseRegistry:
    """A JSON-file registry whose champion changes only through logged transitions."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.exists():
            self._read()

    def _empty_state(self) -> dict[str, Any]:
        return {
            "schema_version": RELEASE_VERSION,
            "contracts_version": CONTRACTS_VERSION,
            "candidates": {},
            "champion": None,
            "previous": None,
            "log": [],
        }

    def _read(self) -> dict[str, Any]:
        try:
            state = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReleaseError(f"cannot read registry: {exc}") from exc
        if not isinstance(state, dict):
            raise ReleaseError("registry must be a JSON object")
        if state.get("schema_version") != RELEASE_VERSION:
            raise ReleaseError(f"unknown registry version: {state.get('schema_version')!r}")
        if state.get("contracts_version") != CONTRACTS_VERSION:
            raise ReleaseError("registry contracts version is incompatible")
        if not isinstance(state.get("candidates"), dict) or not isinstance(
            state.get("log"), list
        ):
            raise ReleaseError("registry candidates or log has an incompatible shape")
        for key in ("champion", "previous"):
            if key not in state:
                raise ReleaseError(f"registry is missing {key}")
        return state

    def state(self) -> dict[str, Any]:
        """Return the durable state as plain JSON values."""
        current = self._read() if self.path.exists() else self._empty_state()
        return json.loads(canonical_json(current))

    def _write(self, state: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8") as stream:
                stream.write(canonical_json(state))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def _refuse(self, action: str, reason: str) -> TransitionResult:
        return TransitionResult(action, False, reason, self.state())

    def _commit(
        self,
        action: str,
        *,
        success_reason: str,
        state: dict[str, Any],
        approver: str,
        reason: str,
    ) -> TransitionResult:
        next_state = {
            key: value for key, value in state.items() if key not in ("champion", "previous", "log")
        }
        next_state["previous"] = state["previous"]
        next_state["champion"] = state["champion"]
        entry = self._log_entry(action, approver=approver, reason=reason, state=next_state)
        entry["state_sha256"] = digest(next_state)
        next_state["log"] = [*state["log"], entry]
        try:
            self._write(next_state)
        except OSError as exc:
            return self._refuse(action, f"atomic write failed: {exc}")
        return TransitionResult(action, True, success_reason, next_state)

    @staticmethod
    def _compatibility_reason(compatibility: Any) -> str | None:
        if not isinstance(compatibility, dict):
            return "version-incompatible compatibility declaration"
        unknown = set(compatibility) - COMPATIBILITY_KEYS
        if unknown:
            return f"version-incompatible compatibility keys {sorted(unknown)!r}"
        hardware = compatibility.get("hardware", SUPPORTED_HARDWARE[0])
        if hardware not in SUPPORTED_HARDWARE:
            return f"unsupported hardware target {hardware!r}"
        artifact_format = compatibility.get("format", SUPPORTED_FORMATS[0])
        if artifact_format not in SUPPORTED_FORMATS:
            return f"unsupported format {artifact_format!r}"
        return None

    def register(self, bundle: ArtifactBundle, validation_status: str) -> dict[str, Any]:
        """Add a verified candidate; never change champion or previous."""
        if validation_status not in VALIDATION_STATUSES:
            raise ReleaseError(f"unknown validation_status {validation_status!r}")
        if bundle.schema_version != RELEASE_VERSION:
            raise ReleaseError(f"unknown release version {bundle.schema_version!r}")
        if bundle.manifest.validation_status == VALIDATION_VALIDATED and (
            validation_status != VALIDATION_VALIDATED
        ):
            raise ReleaseError("validated manifest conflicts with candidate validation status")
        if validation_status == VALIDATION_VALIDATED and (
            bundle.manifest.validation_status != VALIDATION_VALIDATED
        ):
            raise ReleaseError("VALIDATED candidate requires a validated manifest")
        if bundle.manifest.evidence_level not in EVIDENCE_LEVELS:
            raise ReleaseError(f"unknown evidence level {bundle.manifest.evidence_level!r}")
        bundle.verify()
        entry = bundle.to_dict(include_hash=True)
        entry["validation_status"] = validation_status
        state = self._read() if self.path.exists() else self._empty_state()
        state["candidates"][bundle.artifact_id] = entry
        self._write(state)
        return entry

    @staticmethod
    def _log_entry(
        action: str, *, approver: str, reason: str, state: dict[str, Any]
    ) -> dict[str, Any]:
        champion = state["champion"]
        return {
            "action": action,
            "approver": approver,
            "reason": reason,
            "timestamp": utc_now().isoformat(),
            "artifact_sha256": (
                champion.get("bundle_sha256") if isinstance(champion, dict) else None
            ),
            "champion": _artifact_id(champion),
            "previous": _artifact_id(state["previous"]),
            "gates": gate_bundle(True),
        }

    def promote(self, artifact_id: str, *, approver: str, reason: str) -> TransitionResult:
        """Atomically promote one explicitly approved candidate to champion."""
        action = "promote"
        if not isinstance(approver, str) or not approver.strip():
            return self._refuse(action, "approver is required")
        if not isinstance(reason, str) or not reason.strip():
            return self._refuse(action, "reason is required")
        state = self._read() if self.path.exists() else self._empty_state()
        entry = state["candidates"].get(artifact_id)
        if not isinstance(entry, dict):
            return self._refuse(action, f"unknown artifact id {artifact_id!r}")
        if entry.get("schema_version") != RELEASE_VERSION:
            return self._refuse(action, f"version-incompatible bundle {artifact_id!r}")
        if entry.get("contracts_version") != CONTRACTS_VERSION:
            return self._refuse(action, f"contracts-version-incompatible bundle {artifact_id!r}")
        compatibility_reason = self._compatibility_reason(entry.get("compatibility"))
        if compatibility_reason:
            return self._refuse(action, f"{compatibility_reason} for {artifact_id!r}")
        manifest = entry.get("manifest")
        evidence = entry.get("evidence")
        if not isinstance(manifest, dict) or not isinstance(evidence, dict):
            return self._refuse(action, f"version-incompatible bundle {artifact_id!r}")
        if entry.get("validation_status") != VALIDATION_VALIDATED:
            return self._refuse(action, "validation_status is not VALIDATED")
        if manifest.get("validation_status") != VALIDATION_VALIDATED or (
            evidence.get("validation_status") != VALIDATION_VALIDATED
        ):
            return self._refuse(action, "manifest validation_status is not VALIDATED")
        if manifest.get("approval") != APPROVAL_APPROVED or (
            evidence.get("approval") != APPROVAL_APPROVED
        ):
            return self._refuse(action, "approval is not APPROVED")
        evidence_level = manifest.get("evidence_level")
        if evidence.get("evidence_level") != evidence_level:
            return self._refuse(action, "evidence_level is inconsistent")
        if evidence_level in (EVIDENCE_SMOKE, EVIDENCE_SYNTHETIC):
            return self._refuse(action, f"evidence level {evidence_level!r} cannot be promoted")
        if evidence_level not in PROMOTION_ALLOWED_EVIDENCE:
            return self._refuse(action, f"unrecognized evidence level {evidence_level!r}")
        try:
            bundle = ArtifactBundle(
                root=Path(entry["root"]),
                manifest=ArtifactManifest(
                    **{key: manifest[key] for key in ArtifactManifest.__dataclass_fields__}
                ),
                files=entry["files"],
                schema_version=entry["schema_version"],
                environment=entry["environment"],
                compatibility=entry["compatibility"],
                extra_evidence={
                    key: value
                    for key, value in evidence.items()
                    if key not in _RESERVED_EVIDENCE
                },
                bundle_sha256=entry.get("bundle_sha256"),
            )
            bundle.verify()
        except (KeyError, TypeError, ValueError) as exc:
            return self._refuse(action, f"version-incompatible bundle: {exc}")
        if entry.get("manifest_sha256") != bundle.manifest.sha256:
            return self._refuse(action, f"version-incompatible bundle {artifact_id!r}: bad hash")
        state["previous"] = state["champion"]
        state["champion"] = entry
        return self._commit(
            action,
            success_reason="promoted",
            state=state,
            approver=approver,
            reason=reason,
        )

    def rollback(self, approver: str, reason: str) -> TransitionResult:
        """Atomically restore the recorded previous champion."""
        action = "rollback"
        if not isinstance(approver, str) or not approver.strip():
            return self._refuse(action, "approver is required")
        if not isinstance(reason, str) or not reason.strip():
            return self._refuse(action, "reason is required")
        state = self._read() if self.path.exists() else self._empty_state()
        previous = state["previous"]
        if previous is None:
            return self._refuse(action, "no previous champion is recorded")
        state["champion"] = previous
        state["previous"] = None
        return self._commit(
            action,
            success_reason="rolled back",
            state=state,
            approver=approver,
            reason=reason,
        )


@dataclass(frozen=True)
class BenchmarkReport:
    """Latency, throughput and memory values supplied from an actual measurement."""

    latency_ms: tuple[float, ...]
    throughput_per_s: float
    peak_memory_bytes: int
    environment: str
    evidence_level: str = EVIDENCE_SYNTHETIC
    measurements: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = tuple(_non_negative(value, "latency sample") for value in self.latency_ms)
        object.__setattr__(self, "latency_ms", values)
        if not values:
            raise ReleaseError("latency samples must not be empty")
        if (
            isinstance(self.throughput_per_s, bool)
            or not isinstance(self.throughput_per_s, (int, float))
            or self.throughput_per_s < 0
        ):
            raise ReleaseError("throughput_per_s must be a non-negative number")
        if (
            isinstance(self.peak_memory_bytes, bool)
            or not isinstance(self.peak_memory_bytes, int)
            or self.peak_memory_bytes < 0
        ):
            raise ReleaseError("peak_memory_bytes must be a non-negative integer")
        if not isinstance(self.environment, str) or not self.environment.strip():
            raise ReleaseError("environment must be a non-empty string")
        if self.evidence_level not in EVIDENCE_LEVELS:
            raise ReleaseError(f"unknown evidence level {self.evidence_level!r}")
        object.__setattr__(self, "measurements", dict(self.measurements))

    @property
    def p50_ms(self) -> float:
        return _percentile(self.latency_ms, 0.50)

    @property
    def p95_ms(self) -> float:
        return _percentile(self.latency_ms, 0.95)

    @property
    def p99_ms(self) -> float:
        return _percentile(self.latency_ms, 0.99)

    def to_dict(self) -> dict[str, Any]:
        return {
            "measurement": _MEASURED,
            "evidence_level": self.evidence_level,
            "environment": self.environment,
            "sample_count": len(self.latency_ms),
            "latency_ms": {
                "samples": list(self.latency_ms),
                "p50": self.p50_ms,
                "p95": self.p95_ms,
                "p99": self.p99_ms,
            },
            "throughput_per_s": self.throughput_per_s,
            "peak_memory_bytes": self.peak_memory_bytes,
            "measurements": dict(self.measurements),
        }

    @classmethod
    def compare(
        cls, baseline: BenchmarkReport, candidate: BenchmarkReport
    ) -> dict[str, Any]:
        """Report measured candidate-minus-baseline deltas without a pass/fail verdict."""

        def delta(before: float, after: float) -> dict[str, float]:
            return {"baseline": before, "candidate": after, "delta": after - before}

        return {
            "measurement": "MEASURED_DELTA",
            "baseline_environment": baseline.environment,
            "candidate_environment": candidate.environment,
            "baseline_evidence_level": baseline.evidence_level,
            "candidate_evidence_level": candidate.evidence_level,
            "p50_ms": delta(baseline.p50_ms, candidate.p50_ms),
            "p95_ms": delta(baseline.p95_ms, candidate.p95_ms),
            "p99_ms": delta(baseline.p99_ms, candidate.p99_ms),
            "throughput_per_s": delta(
                baseline.throughput_per_s, candidate.throughput_per_s
            ),
            "peak_memory_bytes": delta(
                baseline.peak_memory_bytes, candidate.peak_memory_bytes
            ),
            "verdict": "NOT_EVALUATED",
            "note": "measured delta only; no validated performance limit or guarantee",
            "gates": gate_bundle(True),
        }


def cpu_first_check(hardware: str = CPU_TARGET) -> dict[str, Any]:
    """Observe local CPU-first capability without connecting or enabling hardware."""
    if not isinstance(hardware, str) or not hardware.strip():
        raise ReleaseError("hardware must be a non-empty string")
    normalized = hardware.strip().lower()
    supported = normalized == CPU_TARGET or normalized in SUPPORTED_HARDWARE
    return {
        "measurement": _MEASURED,
        "target": CPU_TARGET,
        "hardware": hardware,
        "capability": STATUS_OK if supported else CAP_NOT_CONNECTED,
        "status": STATUS_OK if supported else STATUS_BLOCKED,
        "connected": supported,
        "gates": gate_bundle(supported),
    }


def deployment_plan(
    *, benchmark: BenchmarkReport | None = None, hardware: str = CPU_TARGET
) -> dict[str, Any]:
    """Describe an offline CPU-first plan; unsupported hardware stays blocked."""
    check = cpu_first_check(hardware)
    return {
        "schema_version": RELEASE_VERSION,
        "target": CPU_TARGET,
        "environment": benchmark.environment if benchmark else "unmeasured",
        "cpu_first": check,
        "unsupported_hardware": {
            "capability": CAP_NOT_CONNECTED,
            "status": STATUS_BLOCKED,
        },
        "inference_status": STATUS_NOT_AVAILABLE,
        "benchmark": benchmark.to_dict() if benchmark else None,
        "gates": gate_bundle(check["connected"]),
        "notes": (
            "offline engineering plan only; no real deployment, network or device control",
            "no clinical or performance guarantee is claimed",
        ),
    }


__all__ = [
    "CPU_TARGET",
    "PROMOTION_ALLOWED_EVIDENCE",
    "RELEASE_VERSION",
    "VALIDATION_STATUSES",
    "VALIDATION_VALIDATED",
    "ArtifactBundle",
    "BenchmarkReport",
    "ReleaseError",
    "ReleaseRegistry",
    "TransitionResult",
    "cpu_first_check",
    "deployment_plan",
]
