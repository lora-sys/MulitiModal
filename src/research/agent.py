"""Proposal schema and provenance helpers for the Skill-driven loop.

The Skill in ``skills/autoresearch/`` IS the proposer: it reads ``state``,
writes one hypothesis to ``p.json``, and calls ``step``. There is deliberately
no in-process proposer, no subprocess coding agent and no deterministic stub.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

#: fields every hypothesis must carry. A proposal missing one is refused
#: rather than defaulted: an experiment whose expected observation was never
#: stated cannot be falsified afterwards.
REQUIRED_HYPOTHESIS_FIELDS = (
    "hypothesis",
    "controlled_change",
    "expected_observation",
    "patch",
)


class ProposalRejected(ValueError):
    """The proposal is not a usable, falsifiable experiment."""


@dataclass(frozen=True)
class Hypothesis:
    """One controlled experiment."""

    hypothesis: str
    controlled_change: str
    expected_observation: str
    patch: str
    intent: str = ""
    next_action_if_keep: str = ""
    next_action_if_revert: str = ""
    raw_reply_sha256: str = ""
    context_sha256: str = ""

    def __post_init__(self) -> None:
        for name in REQUIRED_HYPOTHESIS_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ProposalRejected(f"hypothesis field {name!r} must be a non-empty string")
        if not _looks_like_unified_diff(self.patch):
            raise ProposalRejected(
                "patch must be a unified diff (a '--- a/<file>' and '+++ b/<file>' "
                "header plus at least one @@ hunk); free-form text cannot be applied "
                "and would be recorded as a protocol error"
            )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict, *, raw_reply_sha256: str = "",
                  context_sha256: str = "") -> Hypothesis:
        missing = [k for k in REQUIRED_HYPOTHESIS_FIELDS if not payload.get(k)]
        if missing:
            raise ProposalRejected(
                f"proposal is missing required field(s) {missing}; a proposal "
                "must state what it changes, what it holds fixed and what it "
                "expects to observe"
            )
        return cls(
            hypothesis=str(payload["hypothesis"]),
            controlled_change=str(payload["controlled_change"]),
            expected_observation=str(payload["expected_observation"]),
            patch=str(payload["patch"]),
            intent=str(payload.get("intent") or payload["hypothesis"])[:200],
            next_action_if_keep=str(payload.get("next_action_if_keep") or ""),
            next_action_if_revert=str(payload.get("next_action_if_revert") or ""),
            raw_reply_sha256=raw_reply_sha256,
            context_sha256=context_sha256,
        )


def _looks_like_unified_diff(patch: str) -> bool:
    has_old = "\n--- a/" in "\n" + patch
    has_new = "\n+++ b/" in "\n" + patch
    has_hunk = "\n@@ " in "\n" + patch
    return bool(has_old and has_new and has_hunk)


def digest(payload) -> str:
    """SHA-256 of a JSON-serialisable payload or raw text, for provenance."""
    text = payload if isinstance(payload, str) else json.dumps(
        payload, ensure_ascii=False, sort_keys=True
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
