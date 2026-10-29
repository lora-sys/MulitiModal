"""Evidence registry and GO-gate tests (#47)."""

from pathlib import Path

import pytest

from evidence.registry import (
    ENGINEERING,
    REAL_DATA_TRAINING,
    RESEARCH_CLAIMS,
    EvidenceClaim,
    EvidenceRegistry,
    evaluate_gates,
)

pytestmark = pytest.mark.synthetic

REGISTRY_PATH = Path(__file__).resolve().parents[2] / "docs" / "evidence" / "registry.json"


@pytest.fixture(scope="module")
def registry():
    return EvidenceRegistry.load(REGISTRY_PATH)


def test_registry_loads_with_valid_claims(registry):
    assert {"pdf-spec", "mendeley-dataset-files"} <= set(registry.claims)
    assert registry.state("pdf-spec") == "VERIFIED"


def test_verified_claims_carry_hashes_and_dates(registry):
    for claim in registry.claims.values():
        if claim.verification_state == "VERIFIED":
            assert claim.access_date, claim.claim_id
            assert claim.retrieved_files, claim.claim_id


def test_engineering_granted_but_higher_gates_not(registry):
    for model in ("tcm_constitution", "physio_bpsi_efri"):
        gates = evaluate_gates(registry, model)
        assert gates[ENGINEERING].granted is True
        assert gates[REAL_DATA_TRAINING].granted is False
        assert gates[RESEARCH_CLAIMS].granted is False

    tcm = evaluate_gates(registry, "tcm_constitution")
    # mendeley-dataset-files was audited VERIFIED on 2026-10-30; the only
    # remaining GO_REAL_DATA_TRAINING(tcm) blocker is label-rule certification
    assert tcm[REAL_DATA_TRAINING].missing_claims == ("tcm-label-rules-verified",)
    assert tcm[RESEARCH_CLAIMS].missing_claims == (
        "tcm-label-rules-verified",
        "independent-reference-data",
    )
    physio = evaluate_gates(registry, "physio_bpsi_efri")
    assert "physio-real-data" in physio[REAL_DATA_TRAINING].missing_claims


def test_verified_claim_requires_files_and_date():
    with pytest.raises(ValueError, match="retrieved_files"):
        EvidenceClaim(
            claim_id="bad",
            primary_url="https://example.com",
            exact_supported_claim="x",
            verification_state="VERIFIED",
        )


def test_not_verified_requires_gap_note():
    with pytest.raises(ValueError, match="limitations or blocked tasks"):
        EvidenceClaim(
            claim_id="bad2",
            primary_url="https://example.com",
            exact_supported_claim="x",
            verification_state="NOT_VERIFIED",
        )


def test_retrieved_files_need_hashes():
    with pytest.raises(ValueError, match="path plus sha256 or md5"):
        EvidenceClaim(
            claim_id="bad3",
            primary_url="https://example.com",
            exact_supported_claim="x",
            verification_state="VERIFIED",
            access_date="2026-10-30",
            retrieved_files=({"path": "f"},),
        )


def test_unknown_state_rejected():
    with pytest.raises(ValueError, match="unknown verification_state"):
        EvidenceClaim(
            claim_id="bad4",
            primary_url="https://example.com",
            exact_supported_claim="x",
            verification_state="TRUST_ME_BRO",
        )


def test_audit_lists_open_gaps(registry):
    lines = registry.audit()
    joined = "\n".join(lines)
    assert "tcm-label-rules-verified: NOT_VERIFIED" in joined
    assert "autoresearch-agent-isolation: NOT_VERIFIED" in joined
    assert "physio-real-data: NOT_AVAILABLE" in joined
    assert "pdf-spec" not in joined  # verified claims are not gaps
    assert "mendeley-dataset-files" not in joined  # audited VERIFIED 2026-10-30


def test_gate_cannot_be_promoted_without_evidence(registry):
    """Negative: even a fully forged engineering score cannot promote the
    real-data gate — only VERIFIED data/label claims can, and those cannot
    be constructed without file hashes (#47 hard rule)."""
    forged = EvidenceRegistry(list(registry.claims.values()) + [
        EvidenceClaim(
            claim_id="tcm-engineering-smoke-perfect-score",
            primary_url="https://example.com",
            exact_supported_claim="smoke score 0.0",
            verification_state="UNVERIFIED",
            limitations="score alone never promotes gates",
        )
    ])
    gates = evaluate_gates(forged, "tcm_constitution")
    assert gates[REAL_DATA_TRAINING].granted is False


def test_unknown_model_gate_rejected(registry):
    with pytest.raises(ValueError, match="unknown model"):
        evaluate_gates(registry, "gpt-9")
