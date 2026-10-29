"""Evidence governance package (#47): registry + GO gates."""

from .registry import (
    ENGINEERING,
    NOT_APPLICABLE,
    NOT_AVAILABLE,
    NOT_VERIFIED,
    REAL_DATA_TRAINING,
    RESEARCH_CLAIMS,
    UNVERIFIED,
    VERIFIED,
    EvidenceClaim,
    EvidenceRegistry,
    GateDecision,
    evaluate_gates,
)

__all__ = [
    "ENGINEERING",
    "NOT_APPLICABLE",
    "NOT_AVAILABLE",
    "NOT_VERIFIED",
    "REAL_DATA_TRAINING",
    "RESEARCH_CLAIMS",
    "UNVERIFIED",
    "VERIFIED",
    "EvidenceClaim",
    "EvidenceRegistry",
    "GateDecision",
    "evaluate_gates",
]
