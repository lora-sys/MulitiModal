"""Evidence Registry and three-tier GO gate (#47).

Every claimable scientific statement gets a machine-checkable record:
primary source, edition/version, access date, retrieved files with hashes,
the exact claim it supports, limitations, and an explicit verification
state (VERIFIED / UNVERIFIED / NOT_VERIFIED / NOT_AVAILABLE /
NOT_APPLICABLE). Nothing may be marked VERIFIED without file hashes and an
access date, and GO_REAL_DATA_TRAINING / GO_RESEARCH_CLAIMS can never be
derived from engineering smoke results — only from the required evidence
claims being VERIFIED.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

VERIFIED = "VERIFIED"
UNVERIFIED = "UNVERIFIED"
NOT_VERIFIED = "NOT_VERIFIED"
NOT_AVAILABLE = "NOT_AVAILABLE"
NOT_APPLICABLE = "NOT_APPLICABLE"
STATES = (VERIFIED, UNVERIFIED, NOT_VERIFIED, NOT_AVAILABLE, NOT_APPLICABLE)

ENGINEERING = "GO_ENGINEERING"
REAL_DATA_TRAINING = "GO_REAL_DATA_TRAINING"
RESEARCH_CLAIMS = "GO_RESEARCH_CLAIMS"


@dataclass(frozen=True)
class EvidenceClaim:
    """One auditable evidence record (#47 Scope)."""

    claim_id: str
    primary_url: str
    exact_supported_claim: str
    verification_state: str
    edition_version: str = "unspecified"
    access_date: str = ""  # ISO date of last retrieval attempt
    retrieved_files: tuple[dict, ...] = field(default_factory=tuple)
    limitations: str = ""
    owner: str = "lora-sys"
    reviewer: str = "AGENT_NOT_VERIFIED"
    blocks: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.claim_id or not self.primary_url or not self.exact_supported_claim:
            raise ValueError("claim_id, primary_url and exact_supported_claim are required")
        if self.verification_state not in STATES:
            raise ValueError(
                f"unknown verification_state {self.verification_state!r}; expected {STATES}"
            )
        if self.verification_state == VERIFIED and (
            not self.access_date or not self.retrieved_files
        ):
            raise ValueError(
                f"claim {self.claim_id!r}: VERIFIED requires access_date and "
                "retrieved_files with hashes (no 'license checked' shortcuts, #47)"
            )
        if self.verification_state in (NOT_VERIFIED, NOT_AVAILABLE) and not (
            self.limitations or self.blocks
        ):
            raise ValueError(
                f"claim {self.claim_id!r}: {self.verification_state} requires "
                "limitations or blocked tasks"
            )
        for f in self.retrieved_files:
            if "sha256" not in f or "path" not in f:
                raise ValueError(
                    f"claim {self.claim_id!r}: retrieved_files entries need path+sha256"
                )

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "primary_url": self.primary_url,
            "edition_version": self.edition_version,
            "access_date": self.access_date,
            "retrieved_files": list(self.retrieved_files),
            "exact_supported_claim": self.exact_supported_claim,
            "limitations": self.limitations,
            "verification_state": self.verification_state,
            "owner": self.owner,
            "reviewer": self.reviewer,
            "blocks": list(self.blocks),
        }

    @classmethod
    def from_dict(cls, data: dict) -> EvidenceClaim:
        return cls(
            claim_id=data["claim_id"],
            primary_url=data["primary_url"],
            exact_supported_claim=data["exact_supported_claim"],
            verification_state=data["verification_state"],
            edition_version=data.get("edition_version", "unspecified"),
            access_date=data.get("access_date", ""),
            retrieved_files=tuple(data.get("retrieved_files", ())),
            limitations=data.get("limitations", ""),
            owner=data.get("owner", "lora-sys"),
            reviewer=data.get("reviewer", "AGENT_NOT_VERIFIED"),
            blocks=tuple(data.get("blocks", ())),
        )


class EvidenceRegistry:
    """Loads a tracked JSON registry; read-only gate input (#47)."""

    def __init__(self, claims: list[EvidenceClaim]) -> None:
        self.claims: dict[str, EvidenceClaim] = {}
        for claim in claims:
            if claim.claim_id in self.claims:
                raise ValueError(f"duplicate claim_id {claim.claim_id!r}")
            self.claims[claim.claim_id] = claim

    @classmethod
    def load(cls, path: str | Path) -> EvidenceRegistry:
        data = json.loads(Path(path).read_text())
        return cls([EvidenceClaim.from_dict(c) for c in data["claims"]])

    def state(self, claim_id: str) -> str:
        if claim_id not in self.claims:
            return NOT_AVAILABLE
        return self.claims[claim_id].verification_state

    def audit(self) -> list[str]:
        """Human/CI-readable gaps: what is not yet VERIFIED and what it blocks."""
        lines = []
        for claim in self.claims.values():
            if claim.verification_state != VERIFIED:
                line = f"{claim.claim_id}: {claim.verification_state}"
                if claim.blocks:
                    blocks = ", ".join(claim.blocks)
                    line += f" (blocks: {blocks})"
                lines.append(line)
        return lines


# Gate requirements: a gate is GRANTED only when every listed claim is
# VERIFIED in the registry. GO_ENGINEERING is derived from engineering
# smoke claims only; the higher gates require real-data evidence, so a
# smoke score can never promote them (#47 hard rule 8).
GATE_REQUIREMENTS: dict[str, dict[str, tuple[str, ...]]] = {
    "tcm_constitution": {
        ENGINEERING: ("tcm-engineering-smoke",),
        REAL_DATA_TRAINING: ("mendeley-dataset-files", "tcm-label-rules-verified"),
        RESEARCH_CLAIMS: (
            "mendeley-dataset-files",
            "tcm-label-rules-verified",
            "independent-reference-data",
        ),
    },
    "physio_bpsi_efri": {
        ENGINEERING: ("physio-engineering-smoke",),
        REAL_DATA_TRAINING: ("physio-real-data", "physio-label-rules-verified"),
        RESEARCH_CLAIMS: (
            "physio-real-data",
            "physio-label-rules-verified",
            "independent-reference-data",
        ),
    },
}


@dataclass(frozen=True)
class GateDecision:
    model: str
    gate: str
    granted: bool
    state: str  # GRANTED / NOT_GRANTED
    missing_claims: tuple[str, ...]
    reason: str


def evaluate_gates(registry: EvidenceRegistry, model: str) -> dict[str, GateDecision]:
    if model not in GATE_REQUIREMENTS:
        raise ValueError(f"unknown model {model!r}; expected {sorted(GATE_REQUIREMENTS)}")
    decisions: dict[str, GateDecision] = {}
    for gate, required in GATE_REQUIREMENTS[model].items():
        missing = tuple(c for c in required if registry.state(c) != VERIFIED)
        granted = not missing
        decisions[gate] = GateDecision(
            model=model,
            gate=gate,
            granted=granted,
            state="GRANTED" if granted else "NOT_GRANTED",
            missing_claims=missing,
            reason=(
                "all required evidence claims VERIFIED"
                if granted
                else f"missing/invalid evidence: {', '.join(missing)}"
            ),
        )
    return decisions
