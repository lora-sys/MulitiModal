"""LLM_SILVER weak-label interface (#38): abstain-default, never gold.

LLM_SILVER labels may only be produced from truthful, quoted source text
with full provenance (model id, prompt id, quoted evidence). Confidence,
votes, or model agreement never upgrade silver to RULE/PUBLISHED; silver
never enters reference validation sets. The default provider is a
deterministic ABSTAIN stub so CI never calls a paid model and no silver
label can exist without an explicitly validated provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from labels.ccmq60 import LabelRecord


@dataclass(frozen=True)
class SilverRequest:
    source_id: str
    source_row: int | None
    prompt_id: str
    model_id: str
    text_evidence: str  # verbatim quote from a truthful source; may be empty


class SilverLabelProvider(Protocol):
    def label(self, request: SilverRequest) -> LabelRecord: ...


def ensure_reference_tier(record: LabelRecord) -> LabelRecord:
    """Guard for evaluators/reference exports: silver must never be treated
    as ground truth (#22 rule 5, #38)."""
    if record.label_tier == "LLM_SILVER":
        raise ValueError("LLM_SILVER labels must never be used as reference truth")
    return record


class DeterministicAbstainProvider:
    """Default provider: no model call, always a fully-masked ABSTAIN
    record carrying the refusal reason in provenance."""

    model_id = "deterministic-abstain-v0"

    def label(self, request: SilverRequest) -> LabelRecord:
        if not request.text_evidence.strip():
            reason = "no quoted text evidence"
        else:
            reason = (
                "no validated LLM labeling method available; silver labels "
                "require an explicitly approved provider (AGENT_NOT_VERIFIED)"
            )
        return LabelRecord(
            scores=[None] * 9,
            mask=[False] * 9,
            label_tier="LLM_SILVER",
            standard_version="none-abstain",
            instrument_id="LLM_SILVER_ABSTAIN",
            missing_policy="abstain",
            source_id=request.source_id,
            source_row=request.source_row,
            extra_provenance={
                "abstain_reason": reason,
                "prompt_id": request.prompt_id,
                "model_id": request.model_id,
                "quoted_evidence": request.text_evidence[:500],
            },
        )
