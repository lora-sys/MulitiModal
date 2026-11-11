"""Thin unified inference gateway and mode-isolation adapter (#110).

Translates incoming requests across three isolated research and product modes
into versioned envelopes, enforcing fail-closed gates, idempotency, latency
budgets, and transport failure handling without device control.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from lifecycle.contracts import (
    CONTRACTS_VERSION,
    EVIDENCE_LEVELS,
    EVIDENCE_LOCAL,
    EVIDENCE_SMOKE,
    EVIDENCE_SYNTHETIC,
    GATE_BLOCKED,
    GATE_ENGINEERING,
    GATE_GRANTED,
    GATE_NOT_GRANTED,
    GATE_REAL_DATA_TRAINING,
    GATE_RESEARCH_CLAIMS,
    SOURCE_DEVICE,
    SOURCE_OFFLINE,
    SOURCE_SERVICE,
    SOURCE_SIMULATED,
    STATUS_ABSTAIN,
    STATUS_BLOCKED,
    STATUS_NOT_AVAILABLE,
    STATUS_OK,
    STATUS_REJECTED,
    STATUS_SAFE_STOP,
    STATUS_STALE,
    TERMINAL_STATUSES,
    ContractError,
    Envelope,
    canonical_json,
    digest,
    gate_bundle,
)

GATEWAY_SCHEMA_VERSION = CONTRACTS_VERSION

MODE_A_PRE_ONLY = "A_pre_only"
MODE_B_REALTIME = "B_realtime"
MODE_C_TCM = "C_tcm"
ALLOWED_MODES = (MODE_A_PRE_ONLY, MODE_B_REALTIME, MODE_C_TCM)

PHYSIO_METRIC_NAMES = frozenset(
    {
        "blood_pressure",
        "dbp",
        "diastolic",
        "eeg",
        "heart_rate",
        "hr",
        "hrv",
        "pulse",
        "respiration_rate",
        "rr_interval",
        "sbp",
        "skin_temp",
        "spo2",
        "systolic",
        "temperature",
    }
)


class GatewayError(Exception):
    """Base exception for gateway failures."""


class ModeIsolationError(GatewayError):
    """Raised on illegal cross-mode feature or data leakage."""


class ThrottleError(GatewayError):
    """Raised when in-flight request capacity is exhausted."""


@dataclass(frozen=True)
class SessionState:
    """Immutable reconstructed session state without invented measurements."""

    session_id: str
    event_count: int
    events: tuple[dict[str, Any], ...]
    last_event_time: str | None
    metrics_present: tuple[str, ...]
    state_hash: str
    invented_measurements: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "event_count": self.event_count,
            "events": [dict(e) for e in self.events],
            "last_event_time": self.last_event_time,
            "metrics_present": list(self.metrics_present),
            "state_hash": self.state_hash,
            "invented_measurements": self.invented_measurements,
        }


def make_envelope(
    *,
    schema_version: str = GATEWAY_SCHEMA_VERSION,
    request_id: str = "",
    session_id: str = "",
    trace_id: str = "",
    mode: str = MODE_A_PRE_ONLY,
    event_time: str = "",
    feature_version: str = "v1",
    model_version: str = "v1",
    target_version: str = "v1",
    quality: str = "ok",
    status: str = STATUS_OK,
    provenance: str = "gateway",
    license: str = "internal",
    evidence_level: str = EVIDENCE_SYNTHETIC,
    display_gate: str = "DEMO_ONLY",
    payload: Mapping[str, Any] | None = None,
) -> Envelope:
    now_iso = datetime.now(UTC).isoformat()
    clean_payload = dict(payload or {})
    clean_payload["control_command"] = None
    return Envelope(
        schema_version=schema_version or GATEWAY_SCHEMA_VERSION,
        request_id=request_id or f"req-{digest(now_iso)[:12]}",
        session_id=session_id or f"sess-{digest(now_iso)[:12]}",
        trace_id=trace_id or f"tr-{request_id or digest(now_iso)[:12]}",
        mode=mode,
        event_time=event_time or now_iso,
        feature_version=feature_version or "v1",
        model_version=model_version or "v1",
        target_version=target_version or "v1",
        quality=quality or "ok",
        status=status,
        provenance=provenance or "gateway",
        license=license or "internal",
        evidence_level=evidence_level,
        display_gate=display_gate,
        payload=clean_payload,
    )


def envelope_from_dict(data: Mapping[str, Any]) -> Envelope:
    payload = dict(data.get("payload", {}))
    payload["control_command"] = None
    return Envelope(
        schema_version=str(data.get("schema_version", GATEWAY_SCHEMA_VERSION)),
        request_id=str(data.get("request_id", "")),
        session_id=str(data.get("session_id", "")),
        trace_id=str(data.get("trace_id", "")),
        mode=str(data.get("mode", MODE_A_PRE_ONLY)),
        event_time=str(data.get("event_time", "")),
        feature_version=str(data.get("feature_version", "")),
        model_version=str(data.get("model_version", "")),
        target_version=str(data.get("target_version", "")),
        quality=str(data.get("quality", "ok")),
        status=str(data.get("status", STATUS_OK)),
        provenance=str(data.get("provenance", "")),
        license=str(data.get("license", "")),
        evidence_level=str(data.get("evidence_level", EVIDENCE_SYNTHETIC)),
        display_gate=str(data.get("display_gate", "")),
        payload=payload,
    )


def _is_during_or_post_feature(name: str) -> bool:
    norm = name.lower().strip()
    return (
        norm.startswith("during_")
        or norm.startswith("during.")
        or norm.startswith("post_")
        or norm.startswith("post.")
        or norm.endswith("_during")
        or norm.endswith("_post")
        or "_during_" in norm
        or "_post_" in norm
        or norm in ("during", "post")
    )


_FORBIDDEN_PHASES = ("during", "post")


def _declared_phase(item: Any) -> str:
    if isinstance(item, Mapping):
        return str(item.get("timepoint") or item.get("phase") or "").lower()
    return ""


def _structured_names(container: Any) -> set[str]:
    """Keep declared during/post phase metadata instead of trusting field names."""
    names: set[str] = set()
    if isinstance(container, Mapping):
        for key, value in container.items():
            phase = _declared_phase(value)
            name = str(key)
            names.add(f"{phase}_{name}" if phase in _FORBIDDEN_PHASES else name)
    elif isinstance(container, Sequence) and not isinstance(container, (str, bytes)):
        for item in container:
            if isinstance(item, str):
                names.add(item)
            elif isinstance(item, Mapping) and "name" in item:
                phase = _declared_phase(item)
                name = str(item["name"])
                names.add(f"{phase}_{name}" if phase in _FORBIDDEN_PHASES else name)
    return names


def _request_identity(req: Envelope) -> str | None:
    """Bind an idempotency key to the request that first used it.

    Only canonical-JSON payloads are admissible: ``None`` means the request
    cannot be honestly bound to an identity and must be refused, never hashed.
    """
    try:
        payload_json = canonical_json(dict(req.payload))
    except (TypeError, ValueError, RecursionError, OverflowError):
        return None
    return digest(
        {
            "mode": req.mode,
            "session_id": req.session_id,
            "trace_id": req.trace_id,
            "quality": req.quality,
            "status": req.status,
            "feature_version": req.feature_version,
            "model_version": req.model_version,
            "target_version": req.target_version,
            "provenance": req.provenance,
            "license": req.license,
            "evidence_level": req.evidence_level,
            "display_gate": req.display_gate,
            "payload": payload_json,
        }
    )


def _is_safe_stop(req: Envelope) -> bool:
    """Evaluate stop signals without calling arbitrary object methods."""
    payload = req.payload
    signal = payload.get("signal") if isinstance(payload, Mapping) else None
    action = payload.get("action") if isinstance(payload, Mapping) else None
    return (
        req.status == STATUS_SAFE_STOP
        or (isinstance(req.quality, str) and req.quality.lower() == "safe_stop")
        or (isinstance(signal, str) and signal.upper() == "SAFE_STOP")
        or (isinstance(payload, Mapping) and payload.get("safe_stop") is True)
        or (isinstance(payload, Mapping) and payload.get("user_stop") is True)
        or (isinstance(action, str) and action == "safe_stop")
    )


def _extract_feature_names(payload: Mapping[str, Any]) -> set[str]:
    names: set[str] = set()
    for key in ("features", "feature_names", "fields"):
        names.update(_structured_names(payload.get(key)))
    payload_phase = _declared_phase(payload)
    if payload_phase in _FORBIDDEN_PHASES:
        names.add(f"{payload_phase}_declared")
    for k in payload:
        if _is_during_or_post_feature(k):
            names.add(k)
    return names


def _detect_physio_leak_in_mode_c(payload: Mapping[str, Any]) -> str | None:
    physio_keys: list[str] = []
    for k, v in payload.items():
        if k.lower() in PHYSIO_METRIC_NAMES and v is not None:
            physio_keys.append(k)
    measurements = payload.get("measurements")
    if isinstance(measurements, Sequence) and not isinstance(measurements, (str, bytes)):
        for m in measurements:
            if isinstance(m, Mapping):
                metric = str(m.get("metric", "")).lower()
                if metric in PHYSIO_METRIC_NAMES:
                    physio_keys.append(metric)
    if not physio_keys:
        return None

    task = str(payload.get("task", "")).lower()
    infer_syndrome = payload.get("infer_syndrome")
    infer_constitution = payload.get("infer_constitution")
    target = str(payload.get("target", "")).lower()
    has_syndrome_model = bool(payload.get("model_id"))

    if (
        infer_syndrome is True
        or infer_constitution is True
        or task in ("syndrome", "constitution", "syndrome_inference", "tcm_sd", "classify")
        or target in ("syndrome", "constitution")
        or has_syndrome_model
        or payload.get("infer_from_physio") is True
    ):
        return (
            "mode C forbids inferring syndrome or constitution from "
            f"physiological numbers: {sorted(set(physio_keys))}"
        )
    return None


class Gateway:
    """Thin unified inference gateway enforcing mode isolation and contracts."""

    def __init__(
        self,
        *,
        timeout_s: float = 2.0,
        max_concurrency_per_mode: int = 10,
        concurrency_limits: Mapping[str, int] | None = None,
        stale_after_s: float = 120.0,
        known_feature_versions: Sequence[str] | None = None,
        known_model_versions: Sequence[str] | None = None,
        known_target_versions: Sequence[str] | None = None,
        known_artifacts: Mapping[str, str] | None = None,
        strict_mode: bool = False,
    ) -> None:
        self.timeout_s = timeout_s
        self.max_concurrency_per_mode = max_concurrency_per_mode
        self._concurrency_limits = dict(concurrency_limits or {})
        self.stale_after_s = stale_after_s
        self.known_feature_versions = (
            set(known_feature_versions) if known_feature_versions is not None else None
        )
        self.known_model_versions = (
            set(known_model_versions) if known_model_versions is not None else None
        )
        self.known_target_versions = (
            set(known_target_versions) if known_target_versions is not None else None
        )
        self.known_artifacts = dict(known_artifacts or {})
        self.strict_mode = strict_mode

        self._cache: dict[str, Envelope] = {}
        self._cache_identities: dict[str, str] = {}
        self.idempotency_counter: dict[str, int] = {}
        self.idempotency_hits: int = 0
        self._in_flight: dict[str, int] = {m: 0 for m in ALLOWED_MODES}
        self._session_events: dict[str, list[dict[str, Any]]] = {}

    def handle(self, request: Envelope | Mapping[str, Any]) -> Envelope:
        return self.handle_batch([request])[0]

    def handle_batch(self, requests: Sequence[Envelope | Mapping[str, Any]]) -> list[Envelope]:
        return [self._process_single(req) for req in requests]

    def _process_single(self, raw_request: Envelope | Mapping[str, Any]) -> Envelope:
        mode_str = str(
            raw_request.mode if isinstance(raw_request, Envelope) else raw_request.get("mode", "")
        )
        if mode_str not in ALLOWED_MODES:
            req_id = (
                raw_request.request_id
                if isinstance(raw_request, Envelope)
                else str(raw_request.get("request_id", ""))
            )
            return make_envelope(
                request_id=req_id,
                mode=MODE_A_PRE_ONLY,
                status=STATUS_REJECTED,
                quality="rejected",
                display_gate="REFUSED",
                payload={
                    "control_command": None,
                    "reason": f"unknown mode: {mode_str!r}",
                    "requested_mode": mode_str,
                },
            )

        if isinstance(raw_request, Envelope):
            req = raw_request
        else:
            req = envelope_from_dict(raw_request)

        if _is_safe_stop(req):
            res = make_envelope(
                schema_version=req.schema_version,
                request_id=req.request_id,
                session_id=req.session_id,
                trace_id=req.trace_id,
                mode=req.mode,
                event_time=req.event_time,
                feature_version=req.feature_version,
                model_version=req.model_version,
                target_version=req.target_version,
                quality="safe_stop",
                status=STATUS_SAFE_STOP,
                provenance=req.provenance,
                license=req.license,
                evidence_level=req.evidence_level,
                display_gate="SAFE_STOP_OVERRIDE",
                payload={
                    "control_command": None,
                    "reason": "SAFE_STOP signal received: all operations stopped",
                    "safe_stop": True,
                },
            )
            return res

        identity = _request_identity(req)
        if identity is None:
            return self._refuse_envelope(
                req,
                status=STATUS_REJECTED,
                reason=(
                    "payload is not canonical JSON (NaN/infinity, bytes, cyclic or "
                    "non-serializable values); refusing instead of fabricating an identity"
                ),
            )

        if req.request_id in self._cache:
            if self._cache_identities.get(req.request_id) != identity:
                return self._refuse_envelope(
                    req,
                    status=STATUS_REJECTED,
                    reason=(
                        "idempotency key reused with a different request identity "
                        "(mode/session/versions/payload must match)"
                    ),
                )
            self.idempotency_hits += 1
            self.idempotency_counter[req.request_id] = (
                self.idempotency_counter.get(req.request_id, 1) + 1
            )
            return self._cache[req.request_id]

        limit = self._concurrency_limits.get(req.mode, self.max_concurrency_per_mode)
        if self._in_flight[req.mode] >= limit:
            return self._refuse_envelope(
                req,
                status=STATUS_REJECTED,
                reason=(
                    f"throttle overflow: concurrency limit ({limit}) exceeded for mode {req.mode}"
                ),
            )

        if req.payload.get("transport_failure") is True:
            res = self._refuse_envelope(
                req,
                status=STATUS_NOT_AVAILABLE,
                reason="transport failure: network communication failure, no score emitted",
                extra_payload={"transport_error": True},
            )
            self._store_cache(req, res, identity)
            return res

        is_stale = (
            req.status == STATUS_STALE
            or req.quality.lower() == "stale"
            or req.payload.get("stale") is True
        )
        if not is_stale and req.event_time:
            try:
                dt = datetime.fromisoformat(req.event_time)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
                age = (datetime.now(UTC) - dt).total_seconds()
                if age > self.stale_after_s:
                    is_stale = True
            except (ValueError, TypeError):
                pass
        if is_stale:
            res = self._refuse_envelope(
                req,
                status=STATUS_STALE,
                reason=f"stale signal: measurement timestamp is older than {self.stale_after_s}s",
            )
            self._store_cache(req, res, identity)
            return res

        artifact_hash = req.payload.get("artifact_sha256") or req.payload.get("artifact_hash")
        if artifact_hash is not None:
            if (
                not isinstance(artifact_hash, str)
                or len(artifact_hash) != 64
                or bool(set(artifact_hash) - set("0123456789abcdef"))
            ):
                res = self._refuse_envelope(
                    req,
                    status=STATUS_BLOCKED,
                    reason=f"bad artifact hash format: {artifact_hash!r}",
                )
                self._store_cache(req, res, identity)
                return res
            if self.known_artifacts:
                expected = self.known_artifacts.get(req.model_version)
                if expected is None and artifact_hash not in self.known_artifacts.values():
                    res = self._refuse_envelope(
                        req,
                        status=STATUS_BLOCKED,
                        reason=f"unknown artifact hash: {artifact_hash}",
                    )
                    self._store_cache(req, res, identity)
                    return res
                if expected is not None and artifact_hash != expected:
                    res = self._refuse_envelope(
                        req,
                        status=STATUS_BLOCKED,
                        reason=(
                            f"tampered artifact hash: expected {expected}, got {artifact_hash}"
                        ),
                    )
                    self._store_cache(req, res, identity)
                    return res

        if (
            self.known_feature_versions is not None
            and req.feature_version not in self.known_feature_versions
        ):
            res = self._refuse_envelope(
                req,
                status=STATUS_NOT_AVAILABLE,
                reason=f"unknown feature version: {req.feature_version!r}",
            )
            self._store_cache(req, res, identity)
            return res

        if (
            self.known_model_versions is not None
            and req.model_version not in self.known_model_versions
        ):
            res = self._refuse_envelope(
                req,
                status=STATUS_NOT_AVAILABLE,
                reason=f"unknown model version: {req.model_version!r}",
            )
            self._store_cache(req, res, identity)
            return res

        if (
            self.known_target_versions is not None
            and req.target_version not in self.known_target_versions
        ):
            res = self._refuse_envelope(
                req,
                status=STATUS_NOT_AVAILABLE,
                reason=f"unknown target version: {req.target_version!r}",
            )
            self._store_cache(req, res, identity)
            return res

        is_synthetic_model = (
            req.model_version.startswith("synthetic")
            or req.payload.get("synthetic_model") is True
            or req.payload.get("model_kind") == "synthetic"
            or req.evidence_level == EVIDENCE_SYNTHETIC
        )
        is_product_request = (
            req.payload.get("is_product") is True
            or req.payload.get("environment") == "production"
            or req.display_gate in ("PRODUCT_APPROVED", "PRODUCTION")
            or (
                req.evidence_level in ("PRODUCT", "PRODUCTION", EVIDENCE_LOCAL)
                and not req.payload.get("synthetic_allowed", False)
            )
        )
        if is_synthetic_model and is_product_request:
            res = self._refuse_envelope(
                req,
                status=STATUS_REJECTED,
                reason="synthetic model cannot answer a real product request",
            )
            self._store_cache(req, res, identity)
            return res

        self._in_flight[req.mode] += 1
        try:
            budget_s = float(req.payload.get("budget_s", self.timeout_s))
            if req.payload.get("simulate_timeout") is True or budget_s <= 0:
                res = self._refuse_envelope(
                    req,
                    status=STATUS_STALE,
                    reason=f"timeout: latency budget of {budget_s}s exceeded",
                )
                self._store_cache(req, res, identity)
                return res

            start_t = time.monotonic()
            if req.mode == MODE_A_PRE_ONLY:
                res = self._handle_mode_a(req)
            elif req.mode == MODE_B_REALTIME:
                res = self._handle_mode_b(req)
            elif req.mode == MODE_C_TCM:
                res = self._handle_mode_c(req)
            else:
                res = self._refuse_envelope(
                    req, status=STATUS_REJECTED, reason=f"unknown mode {req.mode}"
                )

            elapsed = time.monotonic() - start_t
            if elapsed > budget_s:
                res = self._refuse_envelope(
                    req,
                    status=STATUS_STALE,
                    reason=f"timeout: elapsed {elapsed:.3f}s exceeded budget {budget_s}s",
                )

            self._store_cache(req, res, identity)
            return res
        finally:
            self._in_flight[req.mode] -= 1

    def _handle_mode_a(self, req: Envelope) -> Envelope:
        extracted = _extract_feature_names(req.payload)
        forbidden = sorted({name for name in extracted if _is_during_or_post_feature(name)})
        if forbidden:
            if self.strict_mode:
                raise ModeIsolationError(
                    f"mode A (pre_only) rejects during/post features: {forbidden}"
                )
            return self._refuse_envelope(
                req,
                status=STATUS_REJECTED,
                reason=f"mode A (pre_only) rejects during/post features: {forbidden}",
                extra_payload={"rejected_features": forbidden},
            )

        return make_envelope(
            schema_version=req.schema_version,
            request_id=req.request_id,
            session_id=req.session_id,
            trace_id=req.trace_id,
            mode=MODE_A_PRE_ONLY,
            event_time=req.event_time,
            feature_version=req.feature_version,
            model_version=req.model_version,
            target_version=req.target_version,
            quality=req.quality,
            status=STATUS_OK,
            provenance=req.provenance,
            license=req.license,
            evidence_level=req.evidence_level,
            display_gate="RESEARCH_PRE_ONLY",
            payload={
                "bpsi_index": req.payload.get("bpsi_index", 0.42),
                "efri_index": req.payload.get("efri_index", 0.55),
                "status": STATUS_OK,
                "control_command": None,
            },
        )

    def _handle_mode_b(self, req: Envelope) -> Envelope:
        """Realtime fatigue scoring stays NOT_AVAILABLE until trusted validation exists.

        Request-supplied flags cannot authorize a prediction: no validated model
        or server-side safety clearance is wired into this gateway.
        """
        return self._refuse_envelope(
            req,
            status=STATUS_NOT_AVAILABLE,
            reason=(
                "fatigue score not available: unvalidated realtime model; "
                "no trusted safety clearance is wired in; claims forbidden"
            ),
            extra_payload={"prediction": None, "control_command": None},
        )

    def _handle_mode_c(self, req: Envelope) -> Envelope:
        physio_leak = _detect_physio_leak_in_mode_c(req.payload)
        if physio_leak is not None:
            if self.strict_mode:
                raise ModeIsolationError(physio_leak)
            return self._refuse_envelope(
                req,
                status=STATUS_REJECTED,
                reason=physio_leak,
            )

        evidence = (
            req.evidence_level if req.evidence_level != EVIDENCE_SMOKE else EVIDENCE_SYNTHETIC
        )
        return make_envelope(
            schema_version=req.schema_version,
            request_id=req.request_id,
            session_id=req.session_id,
            trace_id=req.trace_id,
            mode=MODE_C_TCM,
            event_time=req.event_time,
            feature_version=req.feature_version,
            model_version=req.model_version,
            target_version=req.target_version,
            quality=req.quality,
            status=STATUS_OK,
            provenance=req.provenance,
            license=req.license,
            evidence_level=evidence,
            display_gate="TCM_KNOWLEDGE_GATE",
            payload={
                "syndrome_judgment": None,
                "classifier_invoked": False,
                "knowledge": req.payload.get("knowledge", []),
                "control_command": None,
                "status": STATUS_OK,
            },
        )

    def _refuse_envelope(
        self,
        req: Envelope,
        *,
        status: str,
        reason: str,
        extra_payload: Mapping[str, Any] | None = None,
    ) -> Envelope:
        payload = {
            "control_command": None,
            "prediction": None,
            "reason": reason,
            **(dict(extra_payload or {})),
        }
        return make_envelope(
            schema_version=req.schema_version,
            request_id=req.request_id,
            session_id=req.session_id,
            trace_id=req.trace_id,
            mode=req.mode,
            event_time=req.event_time,
            feature_version=req.feature_version,
            model_version=req.model_version,
            target_version=req.target_version,
            quality="rejected" if status in (STATUS_REJECTED, STATUS_BLOCKED) else "abstain",
            status=status,
            provenance=req.provenance,
            license=req.license,
            evidence_level=req.evidence_level,
            display_gate="REFUSED",
            payload=payload,
        )

    def _store_cache(self, request: Envelope, response: Envelope, identity: str) -> None:
        if request.request_id:
            self._cache_identities[request.request_id] = identity
            self._cache[request.request_id] = response
            self.idempotency_counter[request.request_id] = 1

    def record_event(self, session_id: str, event: Mapping[str, Any]) -> None:
        if not session_id:
            raise ValueError("session_id must not be empty")
        self._session_events.setdefault(session_id, []).append(dict(event))

    def rebuild_session(
        self, session_id: str, events: Sequence[Mapping[str, Any]] | None = None
    ) -> SessionState:
        if events is not None:
            raw_events = [dict(e) for e in events]
        else:
            raw_events = list(self._session_events.get(session_id, []))

        def _sort_key(ev: dict[str, Any]) -> str:
            return str(ev.get("event_time") or ev.get("measured_at") or "")

        sorted_events = sorted(raw_events, key=_sort_key)
        metrics_present: set[str] = set()
        for ev in sorted_events:
            metric = ev.get("metric")
            if metric:
                metrics_present.add(str(metric))
            if isinstance(ev.get("metrics"), Mapping):
                metrics_present.update(str(k) for k in ev["metrics"])

        last_time = None
        if sorted_events:
            last_time = sorted_events[-1].get("event_time") or sorted_events[-1].get("measured_at")

        state_hash = digest(canonical_json(sorted_events))
        return SessionState(
            session_id=session_id,
            event_count=len(sorted_events),
            events=tuple(sorted_events),
            last_event_time=str(last_time) if last_time else None,
            metrics_present=tuple(sorted(metrics_present)),
            state_hash=state_hash,
            invented_measurements=0,
        )

    def handle_transport_failure(
        self,
        session_id: str,
        reason: str = "network disconnect",
        request: Envelope | None = None,
    ) -> Envelope:
        req_id = request.request_id if request else f"transport-fail-{session_id}"
        trace_id = request.trace_id if request else f"tr-{req_id}"
        return make_envelope(
            request_id=req_id,
            session_id=session_id,
            trace_id=trace_id,
            mode=request.mode if request else MODE_B_REALTIME,
            status=STATUS_NOT_AVAILABLE,
            quality="unavailable",
            display_gate="TRANSPORT_FAILED",
            payload={
                "control_command": None,
                "prediction": None,
                "transport_failure": True,
                "reason": f"transport failure: {reason}",
            },
        )


__all__ = [
    "ALLOWED_MODES",
    "CONTRACTS_VERSION",
    "ContractError",
    "EVIDENCE_LEVELS",
    "EVIDENCE_LOCAL",
    "EVIDENCE_SMOKE",
    "EVIDENCE_SYNTHETIC",
    "GATEWAY_SCHEMA_VERSION",
    "GATE_BLOCKED",
    "GATE_ENGINEERING",
    "GATE_GRANTED",
    "GATE_NOT_GRANTED",
    "GATE_REAL_DATA_TRAINING",
    "GATE_RESEARCH_CLAIMS",
    "Gateway",
    "GatewayError",
    "ModeIsolationError",
    "MODE_A_PRE_ONLY",
    "MODE_B_REALTIME",
    "MODE_C_TCM",
    "PHYSIO_METRIC_NAMES",
    "SOURCE_DEVICE",
    "SOURCE_OFFLINE",
    "SOURCE_SERVICE",
    "SOURCE_SIMULATED",
    "STATUS_ABSTAIN",
    "STATUS_BLOCKED",
    "STATUS_NOT_AVAILABLE",
    "STATUS_OK",
    "STATUS_REJECTED",
    "STATUS_SAFE_STOP",
    "STATUS_STALE",
    "SessionState",
    "TERMINAL_STATUSES",
    "ThrottleError",
    "canonical_json",
    "digest",
    "envelope_from_dict",
    "gate_bundle",
    "make_envelope",
]
