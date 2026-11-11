"""Append-only stage observability and data-update plans (#111).

Every event is an aggregated anonymous counter. The hash chain makes edits to
past JSONL lines detectable. Reports describe declared measurements only; no
training, promotion, publishing or research claim is performed here.
"""

from __future__ import annotations

import ast
import json
import re
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .contracts import (
    APPROVAL_APPROVED,
    APPROVAL_STATES,
    APPROVAL_UNVERIFIED,
    EVIDENCE_LEVELS,
    GATE_ENGINEERING,
    GATE_GRANTED,
    GATE_NOT_GRANTED,
    GATE_REAL_DATA_TRAINING,
    GATE_RESEARCH_CLAIMS,
    ISOLATION_UNVERIFIED,
    STAGES,
    STATUS_ABSTAIN,
    STATUS_BLOCKED,
    STATUS_NOT_AVAILABLE,
    STATUS_OK,
    STATUS_REJECTED,
    STATUS_SAFE_STOP,
    STATUS_STALE,
    TERMINAL_STATUSES,
    QCReport,
    StageEvidence,
    canonical_json,
    digest,
    gate_bundle,
)
from .training import DECISIONS

try:
    import fcntl
except ImportError:  # pragma: no cover - POSIX-only advisory lock
    fcntl = None

_CHAIN_LOCK = threading.Lock()

OBSERVE_VERSION = "ml-lifecycle-observe-v1"
GENESIS_SHA256 = "0" * 64
MEASURED = "MEASURED"
DECLARED = "DECLARED"
HUMAN_APPROVAL_REQUIRED = (
    "separate human approval is required before training, AutoResearch, promotion or publishing"
)
DATA_REFRESH_STAGES = ("ingest", "validate", "features")
DECISION_KEEP, DECISION_REVERT, DECISION_CRASH = DECISIONS
EXPERIMENT_DECISIONS = DECISIONS
AGGREGATE_COUNT_KEYS = frozenset(
    {
        "n_available",
        "n_unavailable",
        "n_refused",
        "n_abstain",
        "n_version_drift",
        "n_rollback",
        "n_rows",
        "n_rejected",
        "n_duplicate",
        "n_unknown_columns",
    }
)
_UNAVAILABLE_STATUSES = frozenset(
    {
        STATUS_ABSTAIN,
        STATUS_NOT_AVAILABLE,
        STATUS_STALE,
        STATUS_BLOCKED,
        STATUS_REJECTED,
        STATUS_SAFE_STOP,
    }
)
_REFUSAL_STATUSES = frozenset({STATUS_REJECTED, STATUS_SAFE_STOP, STATUS_BLOCKED})
_IDENTIFIER_WORDS = frozenset(
    {"subject", "patient", "person", "identity", "record", "session", "trace", "request"}
)
_RAW_WORDS = frozenset({"raw", "value", "physiology", "signal", "waveform", "sample"})
_RESERVED_WORDS = frozenset({"id", "ref", "key"})
_VALUE_WORDS = frozenset({"age", "sex", "gender", "height", "weight", "bmi", "heart", "spo2", "bp"})
_WORD_PATTERN = re.compile(r"[A-Za-z0-9]+")


class ObserveError(ValueError):
    """Raised when an observation would record private data or forged evidence."""


def _check_counts(counts: Any) -> None:
    if not isinstance(counts, Mapping):
        raise ObserveError("counts must be a mapping of aggregated anonymous counters")
    for key, value in counts.items():
        if not isinstance(key, str):
            raise ObserveError("count keys must be strings")
        words = frozenset(_WORD_PATTERN.findall(key.lower()))
        if words & _IDENTIFIER_WORDS or words & _RESERVED_WORDS:
            raise ObserveError(f"counts key {key!r} names a raw subject identifier")
        if words & _RAW_WORDS or words & _VALUE_WORDS:
            raise ObserveError(f"counts key {key!r} names a per-row raw value")
        parts = words - {
            "n", "count", "total", "rate", "ratio", "avg", "mean", "p50", "p95", "p99",
        }
        if not parts:
            raise ObserveError(f"counts key {key!r} is not an aggregate counter")
        if isinstance(value, Mapping) or isinstance(value, (list, tuple, set)):
            raise ObserveError(f"count {key!r} must be a scalar, not a per-row collection")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if value < 0:
                raise ObserveError(f"count {key!r} must be non-negative")
            if isinstance(value, float) and (
                value != value or value in (float("inf"), float("-inf"))
            ):
                raise ObserveError(f"count {key!r} must be finite")
        else:
            raise ObserveError(f"count {key!r} must be a numeric aggregate")


@dataclass(frozen=True)
class StageEvent:
    """One anonymous stage observation; event_time is declared or measured."""

    stage: str
    run_id: str
    event_time: str
    status: str
    counts: Mapping[str, float]
    reason: str
    evidence_level: str
    event_time_source: str = DECLARED
    counts_source: str = MEASURED
    version: str = OBSERVE_VERSION

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ObserveError(f"unknown stage {self.stage!r}")
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ObserveError("run_id is required")
        if not isinstance(self.event_time, str) or not self.event_time.strip():
            raise ObserveError("event_time is required")
        if self.event_time_source not in (DECLARED, MEASURED):
            raise ObserveError(f"unknown event_time_source {self.event_time_source!r}")
        if self.counts_source != MEASURED:
            raise ObserveError("counts_source must be MEASURED")
        if self.status not in TERMINAL_STATUSES:
            raise ObserveError(f"unknown status {self.status!r}")
        if self.evidence_level not in EVIDENCE_LEVELS:
            raise ObserveError(f"unknown evidence_level {self.evidence_level!r}")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ObserveError("reason is required")
        if self.version != OBSERVE_VERSION:
            raise ObserveError(f"unsupported observe schema {self.version!r}")
        object.__setattr__(self, "counts", dict(self.counts))
        self.validate()

    def validate(self) -> None:
        """Refuse raw subject identifiers and per-row values in the event."""
        _parse_timestamp(self.event_time)
        _check_counts(self.counts)
        words = frozenset(_WORD_PATTERN.findall(self.run_id.lower()))
        if words & _IDENTIFIER_WORDS or words & _RAW_WORDS:
            raise ObserveError("run_id must not contain a raw subject identifier")
        reason_words = frozenset(_WORD_PATTERN.findall(self.reason.lower()))
        if reason_words & _IDENTIFIER_WORDS or reason_words & _RAW_WORDS:
            raise ObserveError("reason must not describe raw subject data")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.version,
            "stage": self.stage,
            "run_id": self.run_id,
            "event_time": self.event_time,
            "event_time_source": self.event_time_source,
            "status": self.status,
            "counts": dict(self.counts),
            "counts_source": self.counts_source,
            "reason": self.reason,
            "evidence_level": self.evidence_level,
            "isolation": ISOLATION_UNVERIFIED,
        }

    def __str__(self) -> str:
        return canonical_json(self.to_dict())


def _parse_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ObserveError(f"invalid event_time {value!r}") from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


class ObservabilityStore:
    """Append-only JSONL log whose entries bind to the previous line's hash."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._tail = GENESIS_SHA256
        if self.path.exists():
            self.verify_chain()

    def append(self, event: StageEvent) -> str:
        """Append under a shared lock after re-reading the durable chain tail.

        A second store instance (or process) must not extend a stale tail; the
        verification and the write are one critical section.
        """
        event.validate()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _CHAIN_LOCK:
            with self.path.open("a+", encoding="utf-8") as stream:
                if fcntl is not None:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                try:
                    self.verify_chain()
                    previous = self._tail
                    record = {
                        "event": event.to_dict(),
                        "previous_sha256": previous,
                    }
                    record["sha256"] = digest(record)
                    stream.write(canonical_json(record) + "\n")
                    stream.flush()
                finally:
                    if fcntl is not None:
                        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        self._tail = record["sha256"]
        return self._tail

    def read_lines(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            lines = [
                json.loads(line)
                for line in self.path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, json.JSONDecodeError) as exc:
            raise ObserveError(f"cannot read event log: {exc}") from exc
        if not all(isinstance(line, dict) for line in lines):
            raise ObserveError("event log lines must be JSON objects")
        return lines

    def verify_chain(self) -> dict[str, Any]:
        """Recompute each line's digest and its link to the preceding line."""
        lines = self.read_lines()
        previous = GENESIS_SHA256
        for index, line in enumerate(lines):
            if line.get("previous_sha256") != previous:
                raise ObserveError(f"hash chain broken at line {index + 1}: previous_sha256")
            expected = digest(
                {"event": line.get("event"), "previous_sha256": line.get("previous_sha256")}
            )
            if line.get("sha256") != expected:
                raise ObserveError(f"hash chain broken at line {index + 1}: sha256")
            try:
                StageEvent(
                    stage=line["event"]["stage"],
                    run_id=line["event"]["run_id"],
                    event_time=line["event"]["event_time"],
                    status=line["event"]["status"],
                    counts=line["event"]["counts"],
                    reason=line["event"]["reason"],
                    evidence_level=line["event"]["evidence_level"],
                    event_time_source=line["event"].get("event_time_source", DECLARED),
                    counts_source=line["event"].get("counts_source", MEASURED),
                    version=line["event"].get("schema_version", OBSERVE_VERSION),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ObserveError(f"invalid event at line {index + 1}: {exc}") from exc
            previous = line["sha256"]
        self._tail = previous
        return {"verified": True, "n_events": len(lines), "head_sha256": previous}

    def summary(self, *, measured_at: str | None = None) -> dict[str, Any]:
        """Count declared event statuses and counters per stage; no thresholds."""
        self.verify_chain()
        stages: dict[str, dict[str, Any]] = {}
        for line in self.read_lines():
            event = line["event"]
            stage = event["stage"]
            report = stages.setdefault(
                stage,
                {
                    "n_events": 0,
                    "n_available": 0,
                    "n_unavailable": 0,
                    "n_refused": 0,
                    "n_abstain": 0,
                    "n_version_drift": 0,
                    "n_rollback": 0,
                    "status_counts": {},
                    "declared_reasons": [],
                    "declared_event_times": [],
                },
            )
            report["n_events"] += 1
            status = event["status"]
            report["status_counts"][status] = report["status_counts"].get(status, 0) + 1
            report["n_available"] += int(status not in _UNAVAILABLE_STATUSES)
            report["n_unavailable"] += int(status in _UNAVAILABLE_STATUSES)
            report["n_refused"] += int(status in _REFUSAL_STATUSES)
            report["n_abstain"] += int(status == STATUS_ABSTAIN)
            counts = event["counts"]
            for key in ("n_refused", "n_abstain", "n_version_drift", "n_rollback"):
                if key in counts:
                    report[key] += counts[key]
            if event["reason"] not in report["declared_reasons"]:
                report["declared_reasons"].append(event["reason"])
            if event["event_time"] not in report["declared_event_times"]:
                report["declared_event_times"].append(event["event_time"])

        for report in stages.values():
            total = report["n_events"]
            report["availability"] = {
                "n_available": report["n_available"],
                "n_unavailable": report["n_unavailable"],
            }
            for name in ("refusal_rate", "abstain_rate", "version_drift_rate"):
                numerator = report[
                    {
                        "refusal_rate": "n_refused",
                        "abstain_rate": "n_abstain",
                        "version_drift_rate": "n_version_drift",
                    }[name]
                ]
                report[name] = numerator / total if total else None
            report["rollback_count"] = report["n_rollback"]
            report["staleness"] = _staleness(report, measured_at)
        return {
            "schema_version": OBSERVE_VERSION,
            "measurement": MEASURED,
            "counts_source": MEASURED,
            "isolation": ISOLATION_UNVERIFIED,
            "n_events": sum(report["n_events"] for report in stages.values()),
            "stages": stages,
            "note": "measured counts and rates only; no validated availability threshold",
        }


def _staleness(report: Mapping[str, Any], measured_at: str | None) -> dict[str, Any]:
    """Report the age of the newest declared event, without a threshold."""
    times = report["declared_event_times"]
    if not times or measured_at is None:
        return {
            "measurement": MEASURED,
            "latest_event_time": max(times) if times else None,
            "measured_at": measured_at,
            "lag_s": None,
            "marker": "NO_MEASUREMENT_TIME_DECLARED" if times else "NO_EVENTS",
        }
    latest = max(times, key=_parse_timestamp)
    lag = (_parse_timestamp(measured_at) - _parse_timestamp(latest)).total_seconds()
    # A client clock slightly ahead of the measurement clock yields a negative
    # age. Report it as clock skew rather than a negative duration.
    skew = lag < 0.0
    return {
        "measurement": MEASURED,
        "latest_event_time": latest,
        "measured_at": measured_at,
        "lag_s": max(0.0, lag),
        "clock_skew": skew,
        "marker": "CLOCK_SKEW_DETECTED" if skew else "MEASURED_LAG_FROM_DECLARED_EVENT_TIME",
    }


def readiness_report(evidence: Sequence[StageEvidence]) -> dict[str, Any]:
    """Combine per-stage evidence; engineering GO cannot grant the other gates."""
    reasons: list[str] = []
    stages: dict[str, Any] = {}
    engineering_go = bool(evidence)
    for item in evidence:
        if not isinstance(item, StageEvidence):
            raise ObserveError("readiness_report requires StageEvidence records")
        stages[item.stage] = {
            "decision": "GO" if item.go else "NO-GO",
            "gate": item.gate,
            "evidence_level": item.evidence_level,
            "reason": item.reason,
            "artifacts": list(item.artifacts),
        }
        if not item.go:
            engineering_go = False
            reasons.append(f"{item.stage}: NO-GO ({item.reason})")
        elif item.gate != GATE_GRANTED:
            engineering_go = False
            reasons.append(f"{item.stage}: gate is {item.gate}, not {GATE_GRANTED}")
    if not evidence:
        reasons.append("no stage evidence declared")
    reasons.append(
        f"{GATE_REAL_DATA_TRAINING} requires an approved frozen dataset and independent evidence"
    )
    reasons.append(f"{GATE_RESEARCH_CLAIMS} requires independent validation")
    gates = gate_bundle(engineering_go)
    return {
        "schema_version": OBSERVE_VERSION,
        "measurement": DECLARED,
        "status": STATUS_OK if gates[GATE_ENGINEERING] == GATE_GRANTED else STATUS_ABSTAIN,
        "decision": "GO" if gates[GATE_ENGINEERING] == GATE_GRANTED else "NO-GO",
        "gates": gates,
        "reasons": tuple(reasons),
        "stages": stages,
        "isolation": ISOLATION_UNVERIFIED,
    }


def on_new_data(
    *,
    batch_ids: Sequence[str],
    qc: QCReport | None = None,
    evidence: Sequence[StageEvidence] | None = None,
    previous_qc: QCReport | None = None,
    dataset_approval: str = APPROVAL_UNVERIFIED,
    frozen: bool = False,
) -> dict[str, Any]:
    """Return a data-only refresh plan; never execute a lifecycle action."""
    if not batch_ids or any(not isinstance(value, str) or not value.strip() for value in batch_ids):
        raise ObserveError("batch_ids must be non-empty strings")
    if dataset_approval not in APPROVAL_STATES:
        raise ObserveError(f"unknown dataset_approval {dataset_approval!r}")
    normalized = tuple(batch_ids)
    plan = {
        "schema_version": OBSERVE_VERSION,
        "trigger": "DATA_UPDATE",
        "batch_ids": list(normalized),
        "batch_sha256": digest({"batch_ids": list(normalized)}),
        "allowed_stages": list(DATA_REFRESH_STAGES),
        "refresh_plan": ("ingest", "audit QC", "feature refresh", "readiness report"),
        "executed": False,
        "auto_train": False,
        "auto_promote": False,
        "auto_publish": False,
        "reason": HUMAN_APPROVAL_REQUIRED,
        "gates": gate_bundle(True),
        "measurement": DECLARED,
    }
    plan["gates"][GATE_REAL_DATA_TRAINING] = GATE_NOT_GRANTED
    plan["gates"][GATE_RESEARCH_CLAIMS] = GATE_NOT_GRANTED
    plan["qc_sha256"] = qc.sha256 if qc else None
    plan["drift"] = drift_check(previous_qc, qc) if qc and previous_qc else None
    plan["readiness"] = readiness_report(tuple(evidence or ()))
    plan["dataset_approval"] = dataset_approval
    plan["frozen"] = frozen
    plan["candidate_preparation"] = {
        "allowed": dataset_approval == APPROVAL_APPROVED and frozen,
        "auto_train": False,
        "reason": HUMAN_APPROVAL_REQUIRED,
    }
    return plan


def _reject_rate(qc: QCReport) -> float:
    return qc.n_rejected / qc.n_input if qc.n_input else 0.0


def _duplicate_rate(qc: QCReport) -> float:
    return qc.n_duplicate / qc.n_input if qc.n_input else 0.0


def _unknown_columns(qc: QCReport) -> tuple[str, ...]:
    """Return exact column identifiers from structured drift messages.

    ``unknown columns ['sensor_x']`` is emitted by the pipeline with a Python
    list literal. Tokenizing words loses underscores, punctuation and non-ASCII
    names, and filtering x/y/z hides real columns. Parse the list instead. Any
    legacy free-text drift note is kept verbatim rather than reconstructed.
    """
    marker = "unknown columns"
    names: list[str] = []
    for item in qc.drift:
        if not isinstance(item, str) or marker not in item:
            continue
        raw = item.split(marker, 1)[1].strip()
        parsed = None
        if raw.startswith("[") and raw.endswith("]"):
            try:
                parsed = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                parsed = None
        if isinstance(parsed, (list, tuple)):
            names.extend(str(name) for name in parsed)
        else:
            names.append(raw)
    return tuple(dict.fromkeys(names))


def drift_check(previous_qc: QCReport, current_qc: QCReport) -> dict[str, Any]:
    """Report measured QC deltas; no threshold or pass/fail verdict is invented."""
    for name, report in (("previous_qc", previous_qc), ("current_qc", current_qc)):
        if not isinstance(report, QCReport):
            raise ObserveError(f"{name} must be a QCReport")
    before_unknown = _unknown_columns(previous_qc)
    after_unknown = _unknown_columns(current_qc)

    def measured(before: float, after: float) -> dict[str, float]:
        return {"baseline": before, "candidate": after, "delta": after - before}

    return {
        "schema_version": OBSERVE_VERSION,
        "observation": "DRIFT_OBSERVED",
        "measurement": MEASURED,
        "input_volume": measured(previous_qc.n_input, current_qc.n_input),
        "reject_rate": measured(_reject_rate(previous_qc), _reject_rate(current_qc)),
        "duplicate_rate": measured(_duplicate_rate(previous_qc), _duplicate_rate(current_qc)),
        "unknown_columns": {
            "baseline": list(before_unknown),
            "candidate": list(after_unknown),
            "delta": len(after_unknown) - len(before_unknown),
            "new": [item for item in after_unknown if item not in before_unknown],
        },
        "verdict": "NOT_EVALUATED",
        "note": "measured deltas only; no validated drift threshold or pass/fail verdict",
        "gates": gate_bundle(True),
    }


def experiment_record(
    decision: str,
    *,
    run_id: str,
    reason: str,
    path: str | Path,
) -> dict[str, Any]:
    """Append one durable, reasoned KEEP / REVERT / CRASH decision; no execution."""
    if decision not in EXPERIMENT_DECISIONS:
        raise ObserveError(f"unknown experiment decision {decision!r}")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ObserveError("run_id is required")
    if not isinstance(reason, str) or not reason.strip():
        raise ObserveError("reason is required")
    record = {
        "schema_version": OBSERVE_VERSION,
        "decision": decision,
        "run_id": run_id,
        "reason": reason,
        "executed": False,
        "retried": False,
        "measurement": DECLARED,
    }
    record["sha256"] = digest(record)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        stream.write(canonical_json(record) + "\n")
    return record


__all__ = [
    "AGGREGATE_COUNT_KEYS",
    "DATA_REFRESH_STAGES",
    "DECLARED",
    "DECISION_CRASH",
    "DECISION_KEEP",
    "DECISION_REVERT",
    "EXPERIMENT_DECISIONS",
    "HUMAN_APPROVAL_REQUIRED",
    "MEASURED",
    "OBSERVE_VERSION",
    "ObservabilityStore",
    "ObserveError",
    "StageEvent",
    "drift_check",
    "experiment_record",
    "on_new_data",
    "readiness_report",
]
