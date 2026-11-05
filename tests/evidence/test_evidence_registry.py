"""Evidence registry and GO-gate tests (#47)."""

import shutil
import subprocess
from pathlib import Path

import pytest

from evidence.registry import (
    ENGINEERING,
    REAL_DATA_TRAINING,
    RESEARCH_CLAIMS,
    TRACKED_IN_GIT,
    VERIFIED,
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

    forged = EvidenceRegistry(
        list(registry.claims.values())
        + [
            EvidenceClaim(
                claim_id="tcm-engineering-smoke-perfect-score",
                primary_url="https://example.com",
                exact_supported_claim="smoke score 0.0",
                verification_state="UNVERIFIED",
                limitations="score alone never promotes or demotes gates",
            )
        ]
    )
    assert evaluate_gates(forged, "tcm_constitution")[REAL_DATA_TRAINING].granted


def test_unknown_model_gate_rejected(registry):
    with pytest.raises(ValueError, match="unknown model"):
        evaluate_gates(registry, "gpt-9")


def _claim(path: str, digest: str) -> EvidenceClaim:
    return EvidenceClaim(
        claim_id="c",
        primary_url="https://example.com",
        exact_supported_claim="x",
        verification_state=VERIFIED,
        access_date="2026-11-05",
        retrieved_files=({"path": path, "sha256": digest},),
    )


def test_repo_tracked_references_resolve(registry):
    """A TRACKED-IN-GIT entry means the git blob is the evidence itself, so
    deleting or renaming that file silently voids a VERIFIED claim. This
    already happened once: 37f4d11 removed
    tests/runtime/test_autoresearch.py while tcm-engineering-smoke stayed
    VERIFIED. No registry entry may cite a repository path that is gone."""
    defects = [
        line for line in registry.audit_references() if line.startswith("MISSING-REPO-REFERENCE")
    ]
    assert defects == []


def test_engineering_smoke_claims_cite_existing_evidence(registry):
    """Both smoke claims gate GO_ENGINEERING, so their evidence body must be
    readable on any host that has the checkout."""
    for claim_id in ("tcm-engineering-smoke", "physio-engineering-smoke"):
        claim = registry.claims[claim_id]
        assert claim.retrieved_files, claim_id
        for entry in claim.retrieved_files:
            assert (REGISTRY_PATH.parents[2] / entry["path"]).is_file(), (
                claim_id,
                entry["path"],
            )


def test_missing_repo_reference_is_reported_as_defect(tmp_path):
    lines = EvidenceRegistry([_claim("tests/removed_suite.py", TRACKED_IN_GIT)]).audit_references(
        root=tmp_path
    )
    assert lines == [
        "MISSING-REPO-REFERENCE c: tests/removed_suite.py is cited as the "
        "evidence body but is not in the checkout",
        "UNRESOLVABLE-ON-CHECKOUT c: all 1 reference(s) absent; re-fetch per "
        "its limitations before treating this claim as checkable",
    ]


def test_host_local_gap_is_not_reported_as_repo_defect(tmp_path):
    """Real data and run artifacts are git-ignored by design (#47), so a
    missing one is a host gap, never a broken evidence reference."""
    lines = EvidenceRegistry([_claim("data/local/share.xls", "6206f432b683f8a7")]).audit_references(
        root=tmp_path
    )
    assert not any(line.startswith("MISSING-REPO-REFERENCE") for line in lines)
    assert lines[0] == (
        "host-local-absent c: data/local/share.xls (git-ignored; not present on this host)"
    )


def test_partially_resolvable_claim_is_not_called_unresolvable(tmp_path):
    """One missing git-ignored file must not escalate a claim that still has
    a readable reference in the checkout."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "report.md").write_text("x")
    registry = EvidenceRegistry(
        [
            EvidenceClaim(
                claim_id="c",
                primary_url="https://example.com",
                exact_supported_claim="x",
                verification_state=VERIFIED,
                access_date="2026-11-05",
                retrieved_files=(
                    {"path": "docs/report.md", "sha256": TRACKED_IN_GIT},
                    {"path": "data/local/gone.xls", "md5": "363d3e388df6"},
                ),
            )
        ]
    )
    assert [line.split(":")[0] for line in registry.audit_references(root=tmp_path)] == [
        "host-local-absent c"
    ]


def test_tracked_reference_with_a_real_digest_is_a_defect(tmp_path):
    """Scope comes from Git, not from the digest field.

    `pdf-spec` cites the tracked framework PDF with a real SHA-256. Treating
    "has a digest" as "lives outside Git" would downgrade its deletion to a host
    gap and let the guard pass while a VERIFIED repository claim lost its body.
    """
    git = shutil.which("git")
    if git is None:
        pytest.skip("需要 git 才能构造 tracked-but-deleted 场景")
    ref = "docs/source/reference.pdf"
    (tmp_path / "docs" / "source").mkdir(parents=True)
    (tmp_path / ref).write_bytes(b"pdf")
    for cmd in (
        ("init", "-q"),
        ("config", "user.email", "t@example.com"),
        ("config", "user.name", "t"),
        ("add", ref),
        ("commit", "-qm", "add reference"),
    ):
        subprocess.run(
            (git, "-C", str(tmp_path), *cmd), check=True, capture_output=True
        )
    (tmp_path / ref).unlink()  # tracked in the index, gone from the checkout

    lines = EvidenceRegistry(
        [_claim(ref, "29727daced33cc95506f580ea542f4bd")]
    ).audit_references(root=tmp_path)
    assert lines[0].startswith("MISSING-REPO-REFERENCE"), lines


def test_untracked_reference_with_a_real_digest_stays_a_host_gap(tmp_path):
    """Same digest, nothing in Git: calling that a repository defect would make
    every non-dev host look broken."""
    lines = EvidenceRegistry(
        [_claim("data/local/share.xls", "6206f432b683f8a7")]
    ).audit_references(root=tmp_path)
    assert lines[0] == (
        "host-local-absent c: data/local/share.xls (git-ignored; not present "
        "on this host)"
    )


def test_v8_real_training_claim_keeps_a_git_reference(registry):
    """Its three run artifacts are gone on both hosts and the reproduction
    probe was deleted by 37f4d11, so the in-git report is the only reference
    that lets anyone re-check the claim."""
    claim = registry.claims["tcm-real-training-v8"]
    tracked = [
        entry["path"]
        for entry in claim.retrieved_files
        if (entry.get("sha256") or entry.get("md5")) == TRACKED_IN_GIT
    ]
    assert "docs/reports/tcm-50-real-training.md" in tracked
    for path in tracked:
        assert (REGISTRY_PATH.parents[2] / path).is_file(), path


def test_annotated_reference_path_still_resolves(tmp_path):
    """Regression: registry entries append "(dev host, git-ignored)" to the
    path, which made an existing file read as absent."""
    (tmp_path / "data" / "local").mkdir(parents=True)
    (tmp_path / "data" / "local" / "share.xls").write_bytes(b"x")
    claim = _claim("data/local/share.xls (dev host, git-ignored)", "6206f432b683f8a7")
    assert EvidenceRegistry([claim]).audit_references(root=tmp_path) == []


def test_parenthesis_in_real_filename_is_not_read_as_annotation(tmp_path):
    """A path containing " (" but not ending in that paren is a real file;
    truncating it would raise a false MISSING-REPO-REFERENCE defect."""
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "foo (old).py").write_text("")
    lines = EvidenceRegistry([_claim("tests/foo (old).py", TRACKED_IN_GIT)]).audit_references(
        root=tmp_path
    )
    assert lines == []
