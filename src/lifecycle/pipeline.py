"""Incremental, idempotent, resumable data pipeline (#105).

Stages: ingest -> normalize -> validate -> reconcile -> snapshot -> ready_gate.
Raw data is read-only; every derived artifact is reconstructible from the raw
sha256 plus the declared code/config/rule versions. No database, no external
orchestrator: state is a single JSON file under runs/.

Key invariants:
- Re-running the same input yields byte-identical output and QC (idempotent).
- Late, out-of-order and duplicate batches are reconciled by an idempotency key,
  never silently skipped.
- A checkpoint/cursor lets a failed or interrupted run resume without re-reading
  or re-emitting accepted rows.
- Recompute happens only when the source hash, code version, config or rules
  change; otherwise the cached snapshot is reused.
- New data can refresh ingest/QC/features automatically, but the ready gate
  never auto-admits formal training (that requires an APPROVED frozen snapshot).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contracts import (
    APPROVAL_APPROVED,
    APPROVAL_PENDING,
    APPROVAL_UNVERIFIED,
    CONTRACTS_VERSION,
    EVIDENCE_SYNTHETIC,
    SOURCE_OFFLINE,
    STATUS_ABSTAIN,
    STATUS_BLOCKED,
    STATUS_OK,
    TRUST_SHARED,
    TRUST_TRUSTED,
    DatasetManifest,
    PipelineState,
    QCReport,
    SourceManifest,
    atomic_write_text,
    canonical_json,
    digest,
    file_sha256,
    utc_now,
)

PIPELINE_VERSION = "ml-lifecycle-pipeline-v1"
STATE_FILE = "pipeline_state.json"
QUARANTINE_FILE = "quarantine.jsonl"
QC_FILE = "qc_report.json"
SNAPSHOT_FILE = "snapshot.json"
READY_FILE = "readiness.json"

# Declared field contract for the synthetic/offline batch shape. A field that is
# absent or has a different name is drift, not a silently dropped column.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "batch_id",
    "record_id",
    "session_id",
    "subject_ref",
    "phase",
    "metric",
    "value",
    "unit",
    "quality",
    "source_record",
)
OPTIONAL_COLUMNS: tuple[str, ...] = ("event_time", "received_at", "origin")
ALL_COLUMNS: frozenset[str] = frozenset(REQUIRED_COLUMNS + OPTIONAL_COLUMNS)

VALID_PHASES = frozenset({"pre", "during", "post"})
VALID_QUALITY = frozenset({"ok", "degraded", "rejected", "stale"})
VALID_ORIGIN = frozenset({"measured", "imputed", "derived", "unknown"})
ALLOWED_UNITS: dict[str, str] = {
    "heart_rate": "bpm",
    "spo2": "%",
    "respiration_rate": "breaths/min",
    "systolic_bp": "mmHg",
    "diastolic_bp": "mmHg",
    "fatigue_index": "index",
    "vendor_health_index": "index",
    "microcirculation_index": "index",
}
PLAUSIBLE_RANGE: dict[str, tuple[float, float]] = {
    "heart_rate": (20.0, 250.0),
    "spo2": (50.0, 100.0),
    "respiration_rate": (0.0, 60.0),
    "systolic_bp": (50.0, 260.0),
    "diastolic_bp": (30.0, 200.0),
    "fatigue_index": (0.0, 100.0),
    "vendor_health_index": (0.0, 100.0),
    "microcirculation_index": (1.0, 100.0),
}


class PipelineError(ValueError):
    """Raised when the pipeline cannot proceed without fabricating data."""


@dataclass(frozen=True)
class Batch:
    """One declared input batch. `path` is raw and read-only."""

    batch_id: str
    path: Path
    sha256: str

    def __post_init__(self) -> None:
        if not self.batch_id:
            raise PipelineError("batch_id is required")


@dataclass(frozen=True)
class NormalizedRow:
    """One row after normalize, carrying observed/valid/imputed provenance."""

    key: str
    batch_id: str
    record_id: str
    session_id: str
    subject_ref: str
    phase: str
    metric: str
    value: float | None
    unit: str
    quality: str
    origin: str
    source_record: str
    event_time: str
    received_at: str
    observed: bool
    valid: bool
    imputed: bool
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "batch_id": self.batch_id,
            "record_id": self.record_id,
            "session_id": self.session_id,
            "subject_ref": self.subject_ref,
            "phase": self.phase,
            "metric": self.metric,
            "value": self.value,
            "unit": self.unit,
            "quality": self.quality,
            "origin": self.origin,
            "source_record": self.source_record,
            "event_time": self.event_time,
            "received_at": self.received_at,
            "observed": self.observed,
            "valid": self.valid,
            "imputed": self.imputed,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class Snapshot:
    """A frozen, reconstructible view of accepted rows."""

    snapshot_id: str
    source_sha256: tuple[str, ...]
    dataset_sha256: str
    qc_sha256: str
    rows: tuple[NormalizedRow, ...]
    schema_version: str
    code_version: str
    config_sha256: str
    rules_sha256: str
    created_at: str
    rows_sha256: str = ""

    def __post_init__(self) -> None:
        expected = digest([row.to_dict() for row in self.rows])
        if not self.rows_sha256:
            object.__setattr__(self, "rows_sha256", expected)
        elif self.rows_sha256 != expected:
            raise ValueError("cached snapshot rows do not match rows_sha256")

    @property
    def sha256(self) -> str:
        return digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "source_sha256": list(self.source_sha256),
            "dataset_sha256": self.dataset_sha256,
            "qc_sha256": self.qc_sha256,
            "rows": [row.to_dict() for row in self.rows],
            "schema_version": self.schema_version,
            "code_version": self.code_version,
            "config_sha256": self.config_sha256,
            "rules_sha256": self.rules_sha256,
            "created_at": self.created_at,
            "rows_sha256": self.rows_sha256,
        }


@dataclass(frozen=True)
class RunResult:
    """Complete, auditable result of one pipeline invocation."""

    run_id: str
    status: str
    state: PipelineState
    snapshot: Snapshot | None
    qc: QCReport
    readiness: Mapping[str, Any]
    batches_seen: tuple[str, ...]
    batches_new: tuple[str, ...]
    quarantine: tuple[dict[str, str], ...]
    reused_snapshot: bool
    error: str = ""

    @property
    def passed(self) -> bool:
        return self.status in (STATUS_OK, STATUS_ABSTAIN)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PIPELINE_VERSION,
            "run_id": self.run_id,
            "status": self.status,
            "state": self.state.to_dict(),
            "snapshot_sha256": self.snapshot.sha256 if self.snapshot else None,
            "qc": self.qc.to_dict(),
            "readiness": dict(self.readiness),
            "batches_seen": list(self.batches_seen),
            "batches_new": list(self.batches_new),
            "quarantine": [dict(q) for q in self.quarantine],
            "reused_snapshot": self.reused_snapshot,
            "error": self.error,
        }


@dataclass(frozen=True)
class PipelineConfig:
    """Declared, hashable configuration. A change forces recompute."""

    schema_version: str = "lifecycle-batch-csv-v1"
    min_subjects: int = 2
    max_bad_fraction: float = 0.5
    allow_untrusted_identity: bool = True
    require_trusted_identity_for_ready: bool = True

    def __post_init__(self) -> None:
        if self.min_subjects < 1:
            raise PipelineError("min_subjects must be positive")
        if not 0.0 <= self.max_bad_fraction <= 1.0:
            raise PipelineError("max_bad_fraction must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "min_subjects": self.min_subjects,
            "max_bad_fraction": self.max_bad_fraction,
            "allow_untrusted_identity": self.allow_untrusted_identity,
            "require_trusted_identity_for_ready": self.require_trusted_identity_for_ready,
        }

    @property
    def sha256(self) -> str:
        return digest(self.to_dict())


@dataclass
class _Workspace:
    """Accumulator used across one run; not persisted as-is."""

    out_dir: Path
    rows: dict[str, NormalizedRow] = field(default_factory=dict)
    rejections: list[dict[str, str]] = field(default_factory=list)
    quarantine: list[dict[str, str]] = field(default_factory=list)
    drift: set[str] = field(default_factory=set)
    blocked: set[str] = field(default_factory=set)
    duplicates: int = 0
    imputed: int = 0
    input_rows: int = 0
    batch_input_counts: dict[str, int] = field(default_factory=dict)
    batches_new: list[str] = field(default_factory=list)


def declare_batch(path: str | Path, *, batch_id: str | None = None) -> Batch:
    """Hash the raw bytes once; the raw file is never modified."""
    file_path = Path(path)
    if not file_path.is_file():
        raise PipelineError(f"batch not found: {file_path}")
    return Batch(
        batch_id=batch_id or file_path.stem,
        path=file_path,
        sha256=file_sha256(file_path),
    )


def declare_source(
    batch: Batch,
    *,
    license: str = "PRIVATE_NOT_FOR_COMMIT",
    identity_trust: str = TRUST_SHARED,
    capability: str = "UNAVAILABLE",
    provenance: str = "offline research import",
) -> SourceManifest:
    return SourceManifest(
        source_id=batch.batch_id,
        kind=SOURCE_OFFLINE,
        uri=batch.path.name,
        sha256=batch.sha256,
        license=license,
        provenance=provenance,
        identity_trust=identity_trust,
        capability=capability,
        schema_version=CONTRACTS_VERSION,
    )


def _read_rows(batch: Batch, ws: _Workspace) -> list[dict[str, str]]:
    raw = batch.path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != batch.sha256:
        ws.blocked.add(
            f"{batch.batch_id}: raw bytes changed after declaration; re-declare the batch"
        )
        raise PipelineError(
            f"batch {batch.batch_id}: raw bytes do not match the declared sha256"
        )
    with io.StringIO(raw.decode("utf-8-sig"), newline="") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        missing = [name for name in REQUIRED_COLUMNS if name not in columns]
        if missing:
            ws.blocked.add(f"{batch.batch_id}: missing required columns {missing}")
            raise PipelineError(f"batch {batch.batch_id}: missing required columns: {missing}")
        unknown = [name for name in columns if name not in ALL_COLUMNS]
        if unknown:
            ws.drift.add(f"{batch.batch_id}: unknown columns {unknown}")
        rows: list[dict[str, str]] = []
        for raw in reader:
            ws.input_rows += 1
            rows.append(
                {
                    key: (str(value).strip() if value is not None else "")
                    for key, value in raw.items()
                }
            )
        return rows


def _row_key(row: Mapping[str, str]) -> str:
    """Idempotency key: stable across replay, resends and out-of-order arrival."""
    return digest(
        {
            "session_id": row.get("session_id", ""),
            "subject_ref": row.get("subject_ref", ""),
            "phase": row.get("phase", ""),
            "metric": row.get("metric", ""),
            "source_record": row.get("source_record", ""),
            "event_time": row.get("event_time", ""),
        }
    )


def _normalize_row(raw: Mapping[str, str], batch: Batch) -> NormalizedRow:
    reasons: list[str] = []
    key = _row_key(raw)
    raw_value = raw.get("value", "")
    observed = raw_value not in ("", "None", "null", "NA", "NaN")
    origin = raw.get("origin") or ("measured" if observed else "unknown")
    if origin not in VALID_ORIGIN:
        reasons.append(f"unknown origin {origin!r}")
        origin = "unknown"

    value: float | None = None
    if observed:
        try:
            value = float(raw_value)
        except ValueError:
            reasons.append(f"non-numeric value {raw_value!r}")
            observed = False
        else:
            if value != value or value in (float("inf"), float("-inf")):
                reasons.append("non-finite value")
                value = None
                observed = False

    unit = raw.get("unit", "")
    metric = raw.get("metric", "")
    expected_unit = ALLOWED_UNITS.get(metric)
    if expected_unit and unit != expected_unit:
        reasons.append(f"unit mismatch: {metric} expects {expected_unit}, got {unit!r}")
    if metric not in ALLOWED_UNITS:
        reasons.append(f"unknown metric {metric!r}")
    if observed and value is not None and metric in PLAUSIBLE_RANGE:
        low, high = PLAUSIBLE_RANGE[metric]
        if not low <= value <= high:
            reasons.append(f"value {value:g} outside plausible range [{low:g}, {high:g}]")

    phase = raw.get("phase", "")
    if phase not in VALID_PHASES:
        reasons.append(f"unknown phase {phase!r}")
    quality = raw.get("quality", "")
    if quality not in VALID_QUALITY:
        reasons.append(f"unknown quality {quality!r}")
    if quality in ("rejected", "stale"):
        reasons.append(f"quality={quality} is not usable")

    valid = observed and value is not None and not reasons
    imputed = origin == "imputed" and observed
    return NormalizedRow(
        key=key,
        batch_id=batch.batch_id,
        record_id=raw.get("record_id", ""),
        session_id=raw.get("session_id", ""),
        subject_ref=raw.get("subject_ref", ""),
        phase=phase,
        metric=metric,
        value=value if valid else None,
        unit=unit,
        quality=quality,
        origin=origin,
        source_record=raw.get("source_record", ""),
        event_time=raw.get("event_time", ""),
        received_at=raw.get("received_at", ""),
        observed=observed,
        valid=valid,
        imputed=imputed,
        reasons=tuple(reasons),
    )


def _batch_key(batch: Batch) -> str:
    """Idempotency key includes the raw hash: a re-issued batch is not a resend."""
    return f"{batch.batch_id}:{batch.sha256}"


def _ingest(
    batches: Sequence[Batch], ws: _Workspace, processed: set[str]
) -> None:
    """Read each batch at most once per run; a resend is a duplicate, not new data.

    A batch re-uploaded under the same id with different bytes is a correction,
    not a duplicate: the superseded bytes are quarantined and every row they
    contributed is withdrawn before the new bytes are read.

    Duplicate resolution is a pure function of the declared batch set. For each
    idempotency key the row with the lowest ``(batch_id, record_id)`` wins and
    the rest count as duplicates, so the declared order of the batch list cannot
    change the snapshot.

    The withdrawal path only fires when a single run lists the same batch id
    twice with different bytes. A correction across runs changes the dataset
    hash, so the resume dependency check forces a full recompute instead.
    """
    candidates: dict[str, list[NormalizedRow]] = {}
    try:
        for batch in batches:
            key = _batch_key(batch)
            if key in processed:
                ws.drift.add(
                    f"{batch.batch_id}: identical sha256 already ingested; "
                    "replay skipped"
                )
                continue
            superseded = [
                old for old in processed
                if old.rsplit(":", 1)[0] == batch.batch_id
            ]
            if superseded:
                ws.drift.add(
                    f"{batch.batch_id}: re-issued with a different sha256; "
                    "previous bytes replaced after explicit quarantine"
                )
                # Withdraw exactly the raw rows this batch contributed, rejected
                # and duplicate ones included, so QC conservation and
                # bad_fraction stay correct.
                removed_input = ws.batch_input_counts.pop(batch.batch_id, 0)
                ws.input_rows = max(ws.input_rows - removed_input, 0)
                for row_key in tuple(candidates):
                    kept = [
                        row for row in candidates[row_key]
                        if row.batch_id != batch.batch_id
                    ]
                    if kept:
                        candidates[row_key] = kept
                    else:
                        del candidates[row_key]
                for old_key in superseded:
                    processed.discard(old_key)
                ws.quarantine.append(
                    {
                        "key": key,
                        "batch_id": batch.batch_id,
                        "reason": (
                            "batch id collision with different content; "
                            "forced re-ingest"
                        ),
                    }
                )
            before = ws.input_rows
            raw_rows = _read_rows(batch, ws)
            ws.batch_input_counts[batch.batch_id] = ws.input_rows - before
            for raw in raw_rows:
                row = _normalize_row(raw, batch)
                candidates.setdefault(row.key, []).append(row)
            processed.add(key)
            ws.batches_new.append(batch.batch_id)
    finally:
        # Also runs when a later batch blocks: whatever was read must still be
        # accounted for, or QC conservation would fail on the error path.
        _resolve_duplicates(ws, candidates)


def _payload(row: NormalizedRow) -> dict[str, Any]:
    """Row content without the identity of the copy it came from."""
    data = row.to_dict()
    data.pop("batch_id", None)
    data.pop("record_id", None)
    return data


def _resolve_duplicates(
    ws: _Workspace, candidates: Mapping[str, list[NormalizedRow]]
) -> None:
    """Pick one row per idempotency key; count the rest instead of dropping them."""
    resolved: dict[str, NormalizedRow] = {}
    duplicates = 0
    for row_key, rows in candidates.items():
        ordered = sorted(
            rows, key=lambda r: (r.batch_id, r.record_id, r.source_record)
        )
        winner = ordered[0]
        resolved[row_key] = winner
        for other in ordered[1:]:
            duplicates += 1
            if _payload(other) != _payload(winner):
                ws.quarantine.append(
                    {
                        "key": row_key,
                        "batch_id": other.batch_id,
                        "reason": (
                            "conflicting duplicate with the same idempotency key"
                        ),
                        "resolution": "lowest (batch_id, record_id) wins",
                        "kept_record_id": winner.record_id,
                        "kept_batch_id": winner.batch_id,
                    }
                )
    ws.rows = resolved
    ws.duplicates = duplicates


def _validate(ws: _Workspace, config: PipelineConfig) -> None:
    """Reject unusable rows with reasons; never silently drop one."""
    accepted: dict[str, NormalizedRow] = {}
    for key, row in ws.rows.items():
        if not row.valid:
            reason = "; ".join(row.reasons) or "unusable"
            ws.rejections.append(
                {"key": key, "batch_id": row.batch_id, "reason": reason}
            )
            ws.quarantine.append(
                {"key": key, "batch_id": row.batch_id, "reason": reason}
            )
            continue
        if not row.session_id or not row.subject_ref:
            reason = "missing session or subject identity"
            ws.rejections.append(
                {"key": key, "batch_id": row.batch_id, "reason": reason}
            )
            ws.quarantine.append(
                {"key": key, "batch_id": row.batch_id, "reason": reason}
            )
            continue
        accepted[key] = row
        if row.imputed:
            ws.imputed += 1
    ws.rows = accepted

    bad_fraction = len(ws.rejections) / ws.input_rows if ws.input_rows else 0.0
    if bad_fraction > config.max_bad_fraction:
        ws.blocked.add(
            f"bad row fraction {bad_fraction:.3f} exceeds {config.max_bad_fraction:.3f}"
        )

    # Direct contamination: one session mixing two subject labels.
    by_session: dict[str, set[str]] = {}
    for row in ws.rows.values():
        by_session.setdefault(row.session_id, set()).add(row.subject_ref)
    for session_id, subjects in by_session.items():
        if len(subjects) > 1:
            ws.blocked.add(f"session {session_id!r} mixes subjects {sorted(subjects)}")

    # The inverse is NOT treated as contamination: one physical subject may
    # legitimately have several sessions. But the export gives no identity
    # dictionary, so physical-subject isolation stays an explicit deferral.
    by_subject: dict[str, set[str]] = {}
    for row in ws.rows.values():
        by_subject.setdefault(row.subject_ref, set()).add(row.session_id)
    repeated = {
        subject: sessions
        for subject, sessions in by_subject.items()
        if len(sessions) > 1
    }
    if repeated:
        ws.drift.add(
            f"{len(repeated)} subject_ref values span multiple sessions; "
            "physical-subject identity is unverified without a vendor dictionary"
        )


def _reconcile(ws: _Workspace) -> tuple[NormalizedRow, ...]:
    """Deterministic order so replay and out-of-order arrival converge."""
    return tuple(
        sorted(
            ws.rows.values(),
            key=lambda r: (
                r.session_id,
                r.subject_ref,
                r.event_time,
                r.received_at,
                r.phase,
                r.metric,
                r.key,
            ),
        )
    )


def _qc(ws: _Workspace) -> QCReport:
    n_rejected = len(ws.rejections)
    n_accepted = len(ws.rows)
    return QCReport(
        n_input=ws.input_rows,
        n_accepted=n_accepted,
        n_rejected=n_rejected,
        n_duplicate=ws.duplicates,
        n_imputed=ws.imputed,
        rejections=tuple(ws.rejections),
        drift=tuple(sorted(ws.drift)),
        blocked=tuple(sorted(ws.blocked)),
    )


def _readiness(
    ws: _Workspace,
    config: PipelineConfig,
    qc: QCReport,
    *,
    approval: str,
    identity_trust: str,
    frozen: bool,
    evidence_level: str,
) -> dict[str, Any]:
    subjects = {row.subject_ref for row in ws.rows.values()}
    reasons: list[str] = []
    if qc.blocked:
        reasons.extend(sorted(qc.blocked))
    if len(subjects) < config.min_subjects:
        reasons.append(f"subjects {len(subjects)} < required {config.min_subjects}")
    if not ws.rows:
        reasons.append("no accepted rows")
    if config.require_trusted_identity_for_ready and identity_trust != TRUST_TRUSTED:
        reasons.append(f"identity trust is {identity_trust}, not TRUSTED")
    if approval != APPROVAL_APPROVED:
        reasons.append(f"dataset approval is {approval}")
    if not frozen:
        reasons.append("snapshot is not frozen")
    if evidence_level == "SMOKE_ONLY":
        reasons.append("SMOKE_ONLY evidence cannot be admitted to formal training")

    train_ready = not reasons
    return {
        "status": STATUS_OK if train_ready else STATUS_ABSTAIN,
        "train_ready": train_ready,
        "reasons": reasons,
        "n_subjects": len(subjects),
        "n_rows": qc.n_accepted,
        "approval": approval,
        "identity_trust": identity_trust,
        "frozen": frozen,
        "evidence_level": evidence_level,
        "auto_train": False,
        "note": (
            "New data may refresh ingest/QC/features automatically; formal training "
            "requires a separate human-approved, frozen snapshot."
        ),
    }


def _load_state(path: Path) -> PipelineState | None:
    """Read the prior cursor; an unreadable one means "no prior", never a crash.

    A run killed mid-write can leave a truncated file. Treating it as absent
    lets the next run recompute instead of failing forever on resume.
    """
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return PipelineState(
            run_id=data["run_id"],
            stage=data["stage"],
            cursor=data["cursor"],
            processed_keys=tuple(data["processed_keys"]),
            dataset_sha256=data["dataset_sha256"],
            schema_version=data["schema_version"],
            code_version=data["code_version"],
            config_sha256=data["config_sha256"],
            rules_sha256=data["rules_sha256"],
            status=data["status"],
            started_at=data["started_at"],
            updated_at=data["updated_at"],
            error=data.get("error", ""),
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _read_snapshot(path: Path) -> Snapshot | None:
    """Read a cached snapshot; a truncated or malformed one counts as absent."""
    if not path.is_file():
        return None
    try:
        return _parse_snapshot(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _parse_snapshot(data: Mapping[str, Any]) -> Snapshot:
    rows = tuple(
        NormalizedRow(
            key=r["key"], batch_id=r["batch_id"], record_id=r["record_id"],
            session_id=r["session_id"], subject_ref=r["subject_ref"], phase=r["phase"],
            metric=r["metric"], value=r["value"], unit=r["unit"], quality=r["quality"],
            origin=r["origin"], source_record=r["source_record"],
            event_time=r["event_time"], received_at=r["received_at"],
            observed=r["observed"], valid=r["valid"], imputed=r["imputed"],
            reasons=tuple(r["reasons"]),
        )
        for r in data["rows"]
    )
    return Snapshot(
        snapshot_id=data["snapshot_id"],
        source_sha256=tuple(data["source_sha256"]),
        dataset_sha256=data["dataset_sha256"],
        qc_sha256=data["qc_sha256"],
        rows=rows,
        schema_version=data["schema_version"],
        code_version=data["code_version"],
        config_sha256=data["config_sha256"],
        rules_sha256=data["rules_sha256"],
        created_at=data["created_at"],
        rows_sha256=data["rows_sha256"],
    )


def run_pipeline(
    batches: Sequence[Batch],
    out_dir: str | Path,
    *,
    config: PipelineConfig | None = None,
    run_id: str | None = None,
    approval: str = APPROVAL_UNVERIFIED,
    identity_trust: str = TRUST_SHARED,
    frozen: bool = False,
    evidence_level: str = EVIDENCE_SYNTHETIC,
    code_version: str = PIPELINE_VERSION,
    rules_version: str = CONTRACTS_VERSION,
    resume: bool = True,
    fail_stage: str | None = None,
) -> RunResult:
    """Run the full incremental pipeline. Deterministic, resumable, fail-closed.

    `fail_stage` injects a failure before a named stage for recovery tests; it is
    never used in production and is recorded in the state error field.
    """
    if not batches:
        raise PipelineError("run_pipeline requires at least one batch")
    cfg = config or PipelineConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = utc_now().isoformat()
    rules_sha256 = digest({"schema": cfg.schema_version, "rules": rules_version})
    dataset_sha256 = digest(sorted((b.batch_id, b.sha256) for b in batches))
    # G2: a cold run's identity must be reproducible from declared inputs, not
    # from wall-clock time. Two cold runs of the same data/config/code/rules in
    # different directories now produce the same run_id.
    run = run_id or digest(
        {
            "dataset": dataset_sha256,
            "schema": cfg.schema_version,
            "code": code_version,
            "config": cfg.sha256,
            "rules": rules_sha256,
        }
    )[:16]
    state_path = out / STATE_FILE
    snapshot_path = out / SNAPSHOT_FILE

    prior = _load_state(state_path) if resume else None
    processed: set[str] = set()
    # The resume cursor is only valid for the exact dependency set that produced
    # it. Inheriting it after a config/schema/code/rule change would skip every
    # batch and overwrite a good snapshot with an empty one.
    deps_match = (
        prior is not None
        and prior.dataset_sha256 == dataset_sha256
        and prior.schema_version == cfg.schema_version
        and prior.code_version == code_version
        and prior.config_sha256 == cfg.sha256
        and prior.rules_sha256 == rules_sha256
    )
    if deps_match and prior is not None:
        processed = set(prior.processed_keys)
    cached = _read_snapshot(snapshot_path)
    recompute_key = digest(
        {
            "dataset": dataset_sha256,
            "schema": cfg.schema_version,
            "code": code_version,
            "config": cfg.sha256,
            "rules": rules_sha256,
        }
    )
    cache_hit = (
        cached is not None
        and cached.dataset_sha256 == dataset_sha256
        and cached.config_sha256 == cfg.sha256
        and digest(
            {
                "dataset": cached.dataset_sha256,
                "schema": cached.schema_version,
                "code": cached.code_version,
                "config": cached.config_sha256,
                "rules": cached.rules_sha256,
            }
        )
        == recompute_key
        and set(cached.source_sha256) == {b.sha256 for b in batches}
    )

    ws = _Workspace(out_dir=out)
    error = ""
    snapshot: Snapshot | None = None

    if cached is None and snapshot_path.is_file():
        ws.drift.add("cached snapshot unreadable; full recompute")

    cache_valid = cache_hit
    if cache_hit and cached is not None:
        try:
            qc_data = json.loads((out / QC_FILE).read_text(encoding="utf-8"))
            qc = QCReport(
                n_input=qc_data["n_input"], n_accepted=qc_data["n_accepted"],
                n_rejected=qc_data["n_rejected"], n_duplicate=qc_data["n_duplicate"],
                n_imputed=qc_data["n_imputed"],
                rejections=tuple(qc_data["rejections"]),
                drift=tuple(qc_data["drift"]), blocked=tuple(qc_data["blocked"]),
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            cache_valid = False
            ws.drift.add(f"cached QC unreadable ({type(exc).__name__}); full recompute")
        else:
            if qc.sha256 != cached.qc_sha256:
                # Structurally valid but not the report this snapshot was built
                # from: reusing it would publish a self-contradicting manifest.
                cache_valid = False
                ws.drift.add(
                    "cached QC does not match the cached snapshot; full recompute"
                )
    if cache_valid and cached is not None:
        snapshot = cached
        ws.rows = {row.key: row for row in cached.rows}
    else:
        # A cursor without a reusable snapshot would skip every batch and
        # overwrite a good run with an empty one.
        processed.clear()
        try:
            if fail_stage == "ingest":
                raise PipelineError("injected failure before ingest")
            _ingest(batches, ws, processed)
            if fail_stage == "normalize":
                raise PipelineError("injected failure after ingest")
            _validate(ws, cfg)
            if fail_stage == "validate":
                raise PipelineError("injected failure after validate")
            rows = _reconcile(ws)
            qc = _qc(ws)
            source_hashes = tuple(sorted(b.sha256 for b in batches))
            # Deterministic as-of marker: the earliest event time in the accepted
            # data, never wall-clock. Otherwise two identical runs produce two
            # different snapshot hashes.
            as_of = min(
                (r.event_time for r in rows if r.event_time), default=""
            ) or "1970-01-01T00:00:00+00:00"
            snapshot = Snapshot(
                snapshot_id=digest({"dataset": dataset_sha256, "code": code_version}),
                source_sha256=source_hashes,
                dataset_sha256=dataset_sha256,
                qc_sha256=qc.sha256,
                rows=rows,
                schema_version=cfg.schema_version,
                code_version=code_version,
                config_sha256=cfg.sha256,
                rules_sha256=rules_sha256,
                created_at=as_of,
            )
            atomic_write_text(
                out / SNAPSHOT_FILE, canonical_json(snapshot.to_dict()) + "\n"
            )
            atomic_write_text(out / QC_FILE, canonical_json(qc.to_dict()) + "\n")
            atomic_write_text(
                out / QUARANTINE_FILE,
                "".join(canonical_json(q) + "\n" for q in ws.quarantine),
            )
        except PipelineError as exc:
            error = str(exc)
            ws.blocked.add(f"stage failure: {error}")
            qc = _qc(ws)

    readiness = _readiness(
        ws, cfg, qc, approval=approval, identity_trust=identity_trust,
        frozen=frozen, evidence_level=evidence_level,
    )
    status = STATUS_BLOCKED if error else (
        STATUS_OK if readiness["train_ready"] else STATUS_ABSTAIN
    )
    inherit = (
        cache_valid
        and prior is not None
        and prior.status != "FAILED"
        and prior.dataset_sha256 == dataset_sha256
        and run_id is None
    )
    state = PipelineState(
        run_id=prior.run_id if inherit else run,
        stage="ready_gate",
        cursor=digest(sorted(processed)),
        processed_keys=tuple(sorted(processed)),
        dataset_sha256=dataset_sha256,
        schema_version=cfg.schema_version,
        code_version=code_version,
        config_sha256=cfg.sha256,
        rules_sha256=rules_sha256,
        status="FAILED" if error else ("DONE" if status == STATUS_OK else "PAUSED"),
        started_at=prior.started_at if inherit else (
            prior.started_at if prior else (snapshot.created_at if snapshot else started)
        ),
        updated_at=prior.updated_at if inherit else (
            snapshot.created_at if snapshot else started
        ),
        error=error,
    )
    atomic_write_text(out / STATE_FILE, canonical_json(state.to_dict()) + "\n")
    atomic_write_text(out / READY_FILE, canonical_json(readiness) + "\n")

    return RunResult(
        run_id=run,
        status=status,
        state=state,
        snapshot=snapshot,
        qc=qc,
        readiness=readiness,
        batches_seen=tuple(b.batch_id for b in batches),
        batches_new=tuple(ws.batches_new),
        quarantine=tuple(ws.quarantine),
        reused_snapshot=cache_valid,
        error=error,
    )


def dataset_manifest_from_run(
    result: RunResult,
    *,
    dataset_id: str,
    label_version: str,
    feature_version: str,
    approval: str = APPROVAL_PENDING,
    identity_trust: str = TRUST_SHARED,
    frozen: bool = False,
    evidence_level: str = EVIDENCE_SYNTHETIC,
) -> DatasetManifest:
    if result.snapshot is None:
        raise PipelineError("cannot declare a dataset without a snapshot")
    subjects = {row.subject_ref for row in result.snapshot.rows}
    return DatasetManifest(
        dataset_id=dataset_id,
        source_sha256=result.snapshot.source_sha256,
        schema_version=result.snapshot.schema_version,
        label_version=label_version,
        feature_version=feature_version,
        qc_sha256=result.qc.sha256,
        n_rows=result.qc.n_accepted,
        n_subjects=len(subjects),
        identity_trust=identity_trust,
        evidence_level=evidence_level,
        approval=approval,
        frozen=frozen,
        created_at=result.snapshot.created_at,
    )
