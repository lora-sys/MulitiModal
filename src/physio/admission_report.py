"""Auditable, fail-closed admission report for real physiology Sessions.

Reports distinguish engineering regression labels from independent fatigue
reference labels. This report never promotes a research gate or modifies raw
participant files. Existing dataset/real_training masks remain authoritative.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .dataset import IDENTITY_TRUSTED
from .real_export import (
    REFERENCE_LABEL_KEYS,
    RealSession,
    session_feature_row,
    session_labels,
)

ADMISSION_SCHEMA_VERSION = "physio-session-admission-v1"
TIER_VENDOR_PROXY = "VENDOR_PROXY_DELTA"
TIER_PHYSIO_RESPONSE = "OBSERVED_PHYSIO_DELTA"
TIER_UNAVAILABLE = "UNAVAILABLE"
GATE_NOT_GRANTED = "NOT_GRANTED"
ENGINEERING_ONLY = "ENGINEERING_ONLY"

PROXY_LABELS = frozenset({
    "y_delta_vendor_fatigue",
    "y_delta_vendor_health",
})


@dataclass(frozen=True)
class SessionAdmissionReport:
    session_id: str
    subject_ref: str
    identity_trust: str
    collection_status: str
    input_observed: tuple[str, ...]
    input_missing: tuple[str, ...]
    labels: dict[str, dict[str, Any]]
    engineering_status: str
    real_training_gate: str
    research_claims_gate: str
    blockers: tuple[str, ...]
    schema_version: str = ADMISSION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "subject_ref": self.subject_ref,
            "identity_trust": self.identity_trust,
            "collection_status": self.collection_status,
            "input_observed": list(self.input_observed),
            "input_missing": list(self.input_missing),
            "labels": self.labels,
            "engineering_status": self.engineering_status,
            "real_training_gate": self.real_training_gate,
            "research_claims_gate": self.research_claims_gate,
            "blockers": list(self.blockers),
        }


def inspect_session_admission(session: RealSession) -> SessionAdmissionReport:
    """Build a per-field admission manifest without inventing observed values.

    A numeric vendor delta is a proxy response target, not a self-report or
    clinically validated fatigue target. A zero placeholder under a false
    feature mask is never counted as an observed input. No local report may
    grant GO_REAL_DATA_TRAINING or GO_RESEARCH_CLAIMS.
    """
    row = session_feature_row(session)
    labels = session_labels(session)
    observed = tuple(name for name, seen in zip(
        row.feature_names, row.observed_mask, strict=True
    ) if seen)
    missing = tuple(name for name, seen in zip(
        row.feature_names, row.observed_mask, strict=True
    ) if not seen)
    manifest: dict[str, dict[str, Any]] = {}
    for name in sorted(REFERENCE_LABEL_KEYS):
        accepted = bool(labels.mask.get(name, False))
        proxy = name in PROXY_LABELS
        manifest[name] = {
            "available_for_engineering_smoke": accepted,
            "value": labels.values.get(name) if accepted else None,
            "tier": (TIER_VENDOR_PROXY if proxy else TIER_PHYSIO_RESPONSE)
            if accepted else TIER_UNAVAILABLE,
            "independent_fatigue_reference": False,
            "reason": labels.reasons.get(name, "missing_or_not_derivable"),
            "derived_from": "observed_post_minus_pre" if accepted else None,
        }
    blockers = []
    if session.identity_trust != IDENTITY_TRUSTED:
        blockers.append("physical_subject_identity_not_verified")
    if not any(item["available_for_engineering_smoke"] for item in manifest.values()):
        blockers.append("no_admissible_paired_response_target")
    blockers.extend([
        "independent_fatigue_reference_not_collected",
        "protocol_and_treatment_context_not_verified",
        "separate_research_gate_review_required",
    ])
    return SessionAdmissionReport(
        session_id=session.session_id,
        subject_ref=session.subject_ref,
        identity_trust=session.identity_trust,
        collection_status=(
            "ENDPOINTS_PRESENT"
            if session.endpoints.get("bp") and session.endpoints.get("health")
            else "PARTIAL"
        ),
        input_observed=observed,
        input_missing=missing,
        labels=manifest,
        engineering_status=ENGINEERING_ONLY,
        real_training_gate=GATE_NOT_GRANTED,
        research_claims_gate=GATE_NOT_GRANTED,
        blockers=tuple(blockers),
    )
