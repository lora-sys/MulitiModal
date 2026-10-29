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


def test_gate_levels_per_model(registry):
    for model in ("tcm_constitution", "physio_bpsi_efri"):
        gates = evaluate_gates(registry, model)
        assert gates[ENGINEERING].granted is True
        assert gates[RESEARCH_CLAIMS].granted is False  # no independent refs yet
    # tcm: both real-data claims VERIFIED on 2026-10-30 -> GRANTED
    assert evaluate_gates(registry, "tcm_constitution")[REAL_DATA_TRAINING].granted
    # physio still has no real data at all
    assert not evaluate_gates(registry, "physio_bpsi_efri")[REAL_DATA_TRAINING].granted

    tcm = evaluate_gates(registry, "tcm_constitution")
    # mendeley-dataset-files + tcm-label-rules-historical-verified became
    # VERIFIED on 2026-10-30: GO_REAL_DATA_TRAINING(tcm) is now GRANTED
    assert tcm[REAL_DATA_TRAINING].granted is True
    assert tcm[RESEARCH_CLAIMS].missing_claims == ("independent-reference-data",)
    physio = evaluate_gates(registry, "physio_bpsi_efri")
    assert "physio-real-data" in physio[REAL_DATA_TRAINING].missing_claims

    # the GB/T 46939-2025 claim stays NOT_VERIFIED (official full text not
    # obtained) but must never gate the historical-edition data path (#38)
    assert registry.state("tcm-label-rules-verified") == "NOT_VERIFIED"
    assert registry.state("tcm-label-rules-historical-verified") == "VERIFIED"


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
    assert "tcm-label-rules-verified: NOT_VERIFIED" in joined  # GB/T 2025 gap
    assert "tcm-label-rules-historical-verified" not in joined  # VERIFIED 2026-10-30
    assert "autoresearch-agent-isolation: NOT_VERIFIED" in joined
    assert "physio-real-data: NOT_AVAILABLE" in joined
    assert "pdf-spec" not in joined  # verified claims are not gaps
    assert "mendeley-dataset-files" not in joined  # audited VERIFIED 2026-10-30


def test_gate_cannot_be_promoted_without_evidence(registry):
    """Negative: the real-data gate is granted only while its required
    claims are VERIFIED — downgrading either one (or adding a forged
    engineering score) demotes it; smoke scores can never promote gates
    (#47 hard rule)."""
    demoted = []
    for claim in registry.claims.values():
        if claim.claim_id in ("mendeley-dataset-files", "tcm-label-rules-historical-verified"):
            demoted.append(
                EvidenceClaim(
                    claim_id=claim.claim_id,
                    primary_url=claim.primary_url,
                    exact_supported_claim=claim.exact_supported_claim,
                    verification_state="UNVERIFIED",
                    limitations="downgraded for the demotion negative test",
                )
            )
        else:
            demoted.append(claim)
    gates = evaluate_gates(EvidenceRegistry(demoted), "tcm_constitution")
    assert gates[REAL_DATA_TRAINING].granted is False

    forged = EvidenceRegistry(list(registry.claims.values()) + [
        EvidenceClaim(
            claim_id="tcm-engineering-smoke-perfect-score",
            primary_url="https://example.com",
            exact_supported_claim="smoke score 0.0",
            verification_state="UNVERIFIED",
            limitations="score alone never promotes or demotes gates",
        )
    ])
    assert evaluate_gates(forged, "tcm_constitution")[REAL_DATA_TRAINING].granted


def test_unknown_model_gate_rejected(registry):
    with pytest.raises(ValueError, match="unknown model"):
        evaluate_gates(registry, "gpt-9")
