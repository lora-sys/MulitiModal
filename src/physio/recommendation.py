"""Safe massage-program recommendation and state machine (#99).

Phase 1 only offers reviewed static candidates. The service never ranks by a
model-predicted fatigue reduction, never claims a program is optimal, and never
emits a device-control command while the control port is NOT_CONNECTED.

Separation of concerns:
- availability of a program (catalog)
- condition matching (reviewed static rules + user preference)
- model effect ranking (disabled; requires independent safety validation)
- device control (separate port; NOT_CONNECTED)
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .programs import (
    ProgramCapability,
    ProgramCapabilityCatalog,
    ProgramValidationError,
    parse_program_context,
)
from .schema import PHASE_DURING, ProtocolContextV1, SchemaError

POLICY_VERSION = "massage-rec-v1"
CONTROL_STATUS_NOT_CONNECTED = "NOT_CONNECTED"

STATE_IDLE = "IDLE"
STATE_BASELINE = "BASELINE"
STATE_MONITORING = "MONITORING"
STATE_RECOMMENDING = "RECOMMENDING"
STATE_USER_CONFIRM = "USER_CONFIRM"
STATE_SAFE_STOP = "SAFE_STOP"
STATES = (
    STATE_IDLE,
    STATE_BASELINE,
    STATE_MONITORING,
    STATE_RECOMMENDING,
    STATE_USER_CONFIRM,
    STATE_SAFE_STOP,
)

EVENT_BASELINE_READY = "baseline_ready"
EVENT_MONITORING_STARTED = "monitoring_started"
EVENT_RECOMMEND = "recommend"
EVENT_REQUEST_CONFIRMATION = "request_confirmation"
EVENT_USER_CONFIRMED = "user_confirmed"
EVENT_USER_CANCELLED = "user_cancelled"
EVENT_DATA_STALE = "data_stale"
EVENT_SAFE_STOP = "safe_stop"
EVENT_RESET = "reset"

_TRANSITIONS: dict[str, dict[str, str]] = {
    STATE_IDLE: {EVENT_BASELINE_READY: STATE_BASELINE, EVENT_SAFE_STOP: STATE_SAFE_STOP},
    STATE_BASELINE: {
        EVENT_MONITORING_STARTED: STATE_MONITORING,
        EVENT_DATA_STALE: STATE_BASELINE,
        EVENT_SAFE_STOP: STATE_SAFE_STOP,
    },
    STATE_MONITORING: {
        EVENT_RECOMMEND: STATE_RECOMMENDING,
        EVENT_DATA_STALE: STATE_MONITORING,
        EVENT_SAFE_STOP: STATE_SAFE_STOP,
    },
    STATE_RECOMMENDING: {
        EVENT_REQUEST_CONFIRMATION: STATE_USER_CONFIRM,
        EVENT_DATA_STALE: STATE_MONITORING,
        EVENT_SAFE_STOP: STATE_SAFE_STOP,
    },
    STATE_USER_CONFIRM: {
        EVENT_USER_CONFIRMED: STATE_MONITORING,
        EVENT_USER_CANCELLED: STATE_MONITORING,
        EVENT_DATA_STALE: STATE_MONITORING,
        EVENT_SAFE_STOP: STATE_SAFE_STOP,
    },
    STATE_SAFE_STOP: {EVENT_RESET: STATE_IDLE},
}

STATUS_OK = "OK"
STATUS_ABSTAIN = "ABSTAIN"
STATUS_SAFE_STOP = "SAFE_STOP"
STATUS_NOT_AVAILABLE = "NOT_AVAILABLE"


class StateTransitionError(RuntimeError):
    """An event that is not allowed from the current state."""


class ControlNotAuthorized(RuntimeError):
    """Raised if anything tries to send a device command without a real gate."""


class RecommendationStateMachine:
    def __init__(self, state: str = STATE_IDLE) -> None:
        if state not in STATES:
            raise ValueError(f"unknown recommendation state: {state!r}")
        self.state = state
        self.history: list[tuple[str, str]] = []

    def can(self, event: str) -> bool:
        return event in _TRANSITIONS[self.state]

    def trigger(self, event: str) -> str:
        target = _TRANSITIONS[self.state].get(event)
        if target is None:
            raise StateTransitionError(
                f"event {event!r} is not allowed in state {self.state!r}"
            )
        self.history.append((self.state, event))
        self.state = target
        return self.state


@dataclass(frozen=True)
class RecommendationCandidate:
    """A reviewed static suggestion, never an efficacy-ranked prescription."""

    program_id: str
    duration_s: int | None
    intensity: int | None
    reason: str
    evidence: tuple[str, ...]
    source: str = "REVIEWED_STATIC_RULE"
    expected_effect: None = None
    rank: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "program_id": self.program_id,
            "duration_s": self.duration_s,
            "intensity": self.intensity,
            "reason": self.reason,
            "evidence": list(self.evidence),
            "source": self.source,
            "expected_effect": None,
            "rank": self.rank,
        }


@dataclass
class RecommendationResult:
    state: str
    status: str
    reason: str
    candidates: list[RecommendationCandidate] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    policy_version: str = POLICY_VERSION
    model_version: str = "NOT_AVAILABLE"
    data_quality: str = "UNKNOWN"
    requires_user_confirmation: bool = True
    control_command: None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "status": self.status,
            "reason": self.reason,
            "candidates": [c.to_dict() for c in self.candidates],
            "rejected": [dict(r) for r in self.rejected],
            "evidence": dict(self.evidence),
            "policy_version": self.policy_version,
            "model_version": self.model_version,
            "data_quality": self.data_quality,
            "requires_user_confirmation": self.requires_user_confirmation,
            "control_command": None,
        }


@dataclass
class UserPreference:
    """Explicit user choices; absent means no preference, never a default dose."""

    preferred_program_ids: tuple[str, ...] = ()
    avoid_program_ids: tuple[str, ...] = ()
    max_intensity: int | None = None
    max_duration_s: int | None = None
    preferred_actions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "preferred_program_ids": list(self.preferred_program_ids),
            "avoid_program_ids": list(self.avoid_program_ids),
            "max_intensity": self.max_intensity,
            "max_duration_s": self.max_duration_s,
            "preferred_actions": list(self.preferred_actions),
        }


class DeviceControlPort:
    """Separate control boundary. There is no vendor API or consent yet."""

    status = CONTROL_STATUS_NOT_CONNECTED

    def send(self, command: Mapping[str, Any]) -> None:
        raise ControlNotAuthorized(
            "device control is not connected; recommendations must not become commands"
        )


class MassageRecommendationService:
    """Produce reviewed candidates or an explicit refusal; never control hardware."""

    def __init__(
        self,
        catalog: ProgramCapabilityCatalog,
        *,
        state_machine: RecommendationStateMachine | None = None,
        control_port: DeviceControlPort | None = None,
    ) -> None:
        self.catalog = catalog
        self.machine = state_machine or RecommendationStateMachine()
        self.control_port = control_port or DeviceControlPort()
        self.current_program: ProtocolContextV1 | None = None
        self.user_preference = UserPreference()
        self.last_result: RecommendationResult | None = None
        self.audit_log: list[dict[str, Any]] = []

    def user_stop(self) -> RecommendationResult:
        return self._safe_stop("user requested stop")

    def safety_alert(self, reason: str) -> RecommendationResult:
        return self._safe_stop(reason)

    def update(
        self,
        inference: Any | None,
        *,
        now: datetime | None = None,
        current_program: ProtocolContextV1 | None = None,
        user_preference: UserPreference | None = None,
        safety_alert: str | None = None,
        user_stop: bool = False,
    ) -> RecommendationResult:
        if user_stop:
            return self._safe_stop("user requested stop")
        if safety_alert:
            return self._safe_stop(safety_alert)
        if now is not None and now.tzinfo is None:
            raise SchemaError("recommendation time must be timezone-aware")

        self.current_program = current_program
        if user_preference is not None:
            self.user_preference = user_preference

        quality, model_version, inference_status = _inference_state(inference)
        if self.machine.state == STATE_SAFE_STOP:
            return self.last_result or self._safe_stop("service is in SAFE_STOP")

        rejected: list[dict[str, Any]] = []
        if self.current_program is not None:
            program_rejection = self._check_current_program(self.current_program)
            if program_rejection is not None:
                rejected.append(program_rejection)

        if self.machine.state == STATE_USER_CONFIRM:
            if inference_status == STATUS_ABSTAIN or quality in ("rejected", "stale"):
                self.machine.trigger(EVENT_DATA_STALE)
            elif self.last_result is not None:
                return self.last_result

        self.machine.state = self._advance_monitoring()
        if self.machine.state == STATE_SAFE_STOP:
            return self.last_result or self._safe_stop("service is in SAFE_STOP")

        if inference is None or inference_status == STATUS_ABSTAIN:
            result = self._refuse(
                STATUS_ABSTAIN,
                "no usable live telemetry; automatic change suggestions are stopped",
                quality=quality,
                model_version=model_version,
                rejected=rejected,
            )
            return result
        if quality in ("rejected", "stale"):
            self.machine.trigger(EVENT_DATA_STALE)
            return self._refuse(
                STATUS_ABSTAIN,
                f"telemetry quality {quality!r} blocks automatic change suggestions",
                quality=quality,
                model_version=model_version,
                rejected=rejected,
            )
        if not self.catalog.reviewed:
            return self._refuse(
                STATUS_ABSTAIN,
                "program capability catalog is missing or unreviewed",
                quality=quality,
                model_version=model_version,
                rejected=rejected,
            )

        candidates = self._candidates(rejected)
        if not candidates:
            return self._refuse(
                STATUS_ABSTAIN,
                "no reviewed program matches the user constraints",
                quality=quality,
                model_version=model_version,
                rejected=rejected,
            )

        self.machine.trigger(EVENT_RECOMMEND)
        self.machine.trigger(EVENT_REQUEST_CONFIRMATION)
        result = RecommendationResult(
            state=self.machine.state,
            status=STATUS_OK,
            reason="reviewed static candidates; no model effect ranking",
            candidates=candidates,
            rejected=rejected,
            evidence={
                "rules": "reviewed static capability matching",
                "model_effect_ranking": False,
                "expected_effect_available": False,
                "fatigue_prediction_used": False,
                "control_connected": self.control_port.status,
            },
            model_version=model_version,
            data_quality=quality,
            requires_user_confirmation=True,
        )
        self._record(result)
        return result

    def confirm(self, program_id: str, *, confirmed: bool = True) -> dict[str, Any]:
        if self.machine.state != STATE_USER_CONFIRM:
            raise StateTransitionError("no recommendation is waiting for confirmation")
        if self.last_result is None or program_id not in {
            c.program_id for c in self.last_result.candidates
        }:
            raise ValueError("program_id is not one of the offered candidates")
        self.machine.trigger(
            EVENT_USER_CONFIRMED if confirmed else EVENT_USER_CANCELLED
        )
        entry = {
            "event": "user_confirmed" if confirmed else "user_cancelled",
            "program_id": program_id,
            "policy_version": POLICY_VERSION,
            "control_command": None,
        }
        self.audit_log.append(entry)
        return entry

    def reset(self) -> RecommendationStateMachine:
        if self.machine.state == STATE_SAFE_STOP:
            self.machine.trigger(EVENT_RESET)
        return self.machine

    def _advance_monitoring(self) -> str:
        if self.machine.state == STATE_IDLE:
            self.machine.trigger(EVENT_BASELINE_READY)
        if self.machine.state == STATE_BASELINE:
            self.machine.trigger(EVENT_MONITORING_STARTED)
        if self.machine.state == STATE_RECOMMENDING:
            self.machine.trigger(EVENT_REQUEST_CONFIRMATION)
        return self.machine.state

    def _check_current_program(self, protocol: ProtocolContextV1) -> dict[str, Any] | None:
        try:
            parse_program_context(protocol, self.catalog)
        except ProgramValidationError as exc:
            return {
                "program_id": protocol.program_id,
                "reason": str(exc),
                "action": "current program not recognized; no change suggested",
            }
        return None

    def _candidates(self, rejected: list[dict[str, Any]]) -> list[RecommendationCandidate]:
        pref = self.user_preference
        available = [
            capability
            for capability in self.catalog.capabilities()
            if capability.program_id not in pref.avoid_program_ids
        ]
        if pref.preferred_program_ids:
            preferred = [
                c for c in available if c.program_id in pref.preferred_program_ids
            ]
            available = preferred or available
        candidates: list[RecommendationCandidate] = []
        for capability in available:
            rejection = self._reject_capability(capability, pref)
            if rejection is not None:
                rejected.append(rejection)
                continue
            intensity = _preferred_intensity(capability, pref)
            duration = _preferred_duration(capability, pref)
            candidates.append(
                RecommendationCandidate(
                    program_id=capability.program_id,
                    duration_s=duration,
                    intensity=intensity,
                    reason="matches reviewed capability and user constraints",
                    evidence=(
                        f"catalog_version={self.catalog.catalog_version}",
                        f"reviewed={capability.reviewed}",
                        "no expected fatigue effect claimed",
                    ),
                    rank=len(candidates) + 1,
                )
            )
        return candidates

    @staticmethod
    def _reject_capability(
        capability: ProgramCapability, pref: UserPreference
    ) -> dict[str, Any] | None:
        if not capability.selectable:
            return {"program_id": capability.program_id, "reason": "not user selectable"}
        if not capability.reviewed:
            return {"program_id": capability.program_id, "reason": "unreviewed capability"}
        if pref.max_intensity is not None and capability.intensity_min > pref.max_intensity:
            return {
                "program_id": capability.program_id,
                "reason": "minimum intensity exceeds the user's explicit limit",
            }
        if pref.max_duration_s is not None and capability.duration_min_s > pref.max_duration_s:
            return {
                "program_id": capability.program_id,
                "reason": "minimum duration exceeds the user's explicit limit",
            }
        if pref.preferred_actions:
            if not set(pref.preferred_actions) & set(capability.actions):
                return {
                    "program_id": capability.program_id,
                    "reason": "no requested action is supported",
                }
        return None

    def _refuse(
        self,
        status: str,
        reason: str,
        *,
        quality: str,
        model_version: str,
        rejected: list[dict[str, Any]],
    ) -> RecommendationResult:
        result = RecommendationResult(
            state=self.machine.state,
            status=status,
            reason=reason,
            rejected=rejected,
            evidence={
                "model_effect_ranking": False,
                "control_connected": self.control_port.status,
            },
            model_version=model_version,
            data_quality=quality,
            requires_user_confirmation=False,
        )
        self._record(result)
        return result

    def _safe_stop(self, reason: str) -> RecommendationResult:
        if self.machine.state != STATE_SAFE_STOP:
            self.machine.trigger(EVENT_SAFE_STOP)
        result = RecommendationResult(
            state=self.machine.state,
            status=STATUS_SAFE_STOP,
            reason=reason,
            evidence={
                "priority": "user_stop/alerts highest",
                "control_connected": self.control_port.status,
            },
            requires_user_confirmation=False,
        )
        self._record(result)
        return result

    def _record(self, result: RecommendationResult) -> None:
        self.last_result = result
        self.audit_log.append(
            {
                "state": result.state,
                "status": result.status,
                "reason": result.reason,
                "candidates": [c.program_id for c in result.candidates],
                "policy_version": result.policy_version,
                "model_version": result.model_version,
                "data_quality": result.data_quality,
                "control_command": None,
            }
        )


def _inference_state(inference: Any | None) -> tuple[str, str, str]:
    if inference is None:
        return "UNKNOWN", "NOT_AVAILABLE", STATUS_ABSTAIN
    status = str(getattr(inference, "status", STATUS_ABSTAIN))
    quality = str(getattr(inference, "quality", "UNKNOWN"))
    model_version = str(getattr(inference, "model_version", "NOT_AVAILABLE"))
    if status not in (STATUS_OK, STATUS_ABSTAIN, STATUS_NOT_AVAILABLE):
        status = STATUS_ABSTAIN
    return quality, model_version, status


def _preferred_intensity(
    capability: ProgramCapability, pref: UserPreference
) -> int | None:
    if pref.max_intensity is None:
        return None
    return min(pref.max_intensity, capability.intensity_max)


def _preferred_duration(
    capability: ProgramCapability, pref: UserPreference
) -> int | None:
    if pref.max_duration_s is None:
        return None
    return min(pref.max_duration_s, capability.duration_max_s)


def recommendation_for_phase(phase: str) -> bool:
    return phase == PHASE_DURING


__all__ = [
    "CONTROL_STATUS_NOT_CONNECTED",
    "ControlNotAuthorized",
    "DeviceControlPort",
    "MassageRecommendationService",
    "POLICY_VERSION",
    "RecommendationCandidate",
    "RecommendationResult",
    "RecommendationStateMachine",
    "STATES",
    "STATE_BASELINE",
    "STATE_IDLE",
    "STATE_MONITORING",
    "STATE_RECOMMENDING",
    "STATE_SAFE_STOP",
    "STATE_USER_CONFIRM",
    "STATUS_ABSTAIN",
    "STATUS_NOT_AVAILABLE",
    "STATUS_OK",
    "STATUS_SAFE_STOP",
    "StateTransitionError",
    "UserPreference",
]
