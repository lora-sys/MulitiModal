"""Evidence registry and GO-gate tests (#47)."""

import shutil
import subprocess
from pathlib import Path

import pytest

from evidence.registry import (
    ENGINEERING,
    REAL_DATA_TRAINING,
    RESEARCH_CLAIMS,
    SCOPES,
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


def _claim(path: str, digest: str, scope: str) -> EvidenceClaim:
    return EvidenceClaim(
        claim_id="c",
        primary_url="https://example.com",
        exact_supported_claim="x",
        verification_state=VERIFIED,
        access_date="2026-11-05",
        retrieved_files=({"path": path, "sha256": digest, "scope": scope},),
    )


def test_repo_tracked_references_resolve(registry):
    """A repo-scoped reference is the evidence body itself, so a renamed or
    deleted file silently voids a VERIFIED claim. This already happened once:
    37f4d11 removed tests/runtime/test_autoresearch.py while
    tcm-engineering-smoke stayed VERIFIED, and nothing could see it."""
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
    lines = EvidenceRegistry(
        [_claim("tests/removed_suite.py", TRACKED_IN_GIT, "repo")]
    ).audit_references(root=tmp_path)
    assert lines == [
        "MISSING-REPO-REFERENCE c: tests/removed_suite.py is cited as the "
        "evidence body but is not in the checkout",
        "UNRESOLVABLE-ON-CHECKOUT c: all 1 reference(s) absent; re-fetch per "
        "its limitations before treating this claim as checkable",
    ]


def test_host_scoped_gap_is_not_reported_as_repo_defect(tmp_path):
    """data/, runs/ and artifacts/ are outside Git by design (#47), so a missing
    file there is a host gap, never a broken evidence reference."""
    lines = EvidenceRegistry(
        [_claim("data/local/share.xls", "6206f432b683f8a7", "host")]
    ).audit_references(root=tmp_path)
    assert lines == [
        "host-local-absent c: data/local/share.xls (declared host scope; not present on this host)",
        "UNRESOLVABLE-ON-CHECKOUT c: all 1 reference(s) absent; re-fetch per "
        "its limitations before treating this claim as checkable",
    ]


def test_partially_resolvable_claim_is_not_called_unresolvable(tmp_path):
    """One missing host file must not escalate a claim that still has a readable
    repository reference."""
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
                    {"path": "docs/report.md", "sha256": TRACKED_IN_GIT, "scope": "repo"},
                    {"path": "data/local/gone.xls", "md5": "363d3e388df6", "scope": "host"},
                ),
            )
        ]
    )
    assert [line.split(":")[0] for line in registry.audit_references(root=tmp_path)] == [
        "host-local-absent c"
    ]


def test_external_references_are_not_checked_here(tmp_path):
    """`upstream:` paths live in another repository; inventing a verdict for them
    would report failures that mean nothing."""
    lines = EvidenceRegistry(
        [_claim("upstream:README.md@228791fb", "3958fd4195ac2f98", "external")]
    ).audit_references(root=tmp_path)
    assert lines == []


@pytest.mark.parametrize("stage", ["unstaged", "staged", "committed"])
def test_repo_scope_survives_every_deletion_stage(tmp_path, stage):
    """Scope is declared, so no deletion bookkeeping can downgrade it.

    `git ls-files` lists cached entries only: once the deletion is committed the
    path is gone from the index, and an index-derived verdict reported
    `host-local-absent` for exactly the case this guard exists for.
    """
    git = shutil.which("git")
    if git is None:
        pytest.skip("需要 git 才能构造各阶段的删除")

    def run(*args):
        return subprocess.run((git, "-C", str(tmp_path), *args), check=True, capture_output=True)

    ref = "docs/source/reference.pdf"
    run("init", "-q")
    run("config", "user.email", "t@example.com")
    run("config", "user.name", "t")
    (tmp_path / "docs" / "source").mkdir(parents=True)
    (tmp_path / ref).write_bytes(b"pdf")
    run("add", ref)
    run("commit", "-qm", "add reference")
    (tmp_path / ref).unlink()
    if stage != "unstaged":
        run("add", "-A")
        if stage == "committed":
            run("commit", "-qm", "delete reference")
        # the index no longer knows about it - what the old code keyed on
        assert ref not in run("ls-files").stdout.decode()

    lines = EvidenceRegistry(
        [_claim(ref, "29727daced33cc95506f580ea542f4bd", "repo")]
    ).audit_references(root=tmp_path)
    assert lines[0].startswith("MISSING-REPO-REFERENCE"), (stage, lines)


def test_a_private_gitignore_cannot_reclassify_a_repo_reference(tmp_path):
    """Host-local Git configuration must not change the verdict.

    `git check-ignore` also honours `.git/info/exclude` and the user's global
    excludes, so one private `*.pdf` rule turned this repository reference into
    an informational gap on the reviewing machine alone.
    """
    git = shutil.which("git")
    if git is None:
        pytest.skip("需要 git 才能构造本机私有排除配置")

    def run(*args, env=None):
        return subprocess.run(
            (git, "-C", str(tmp_path), *args),
            check=True,
            capture_output=True,
            env=env,
        )

    ref = "docs/source/reference.pdf"
    run("init", "-q")
    run("config", "user.email", "t@example.com")
    run("config", "user.name", "t")
    (tmp_path / "docs" / "source").mkdir(parents=True)
    (tmp_path / ref).write_bytes(b"pdf")
    run("add", ref)
    run("commit", "-qm", "add reference")
    (tmp_path / ref).unlink()
    run("add", "-A")
    run("commit", "-qm", "delete reference")

    claim = _claim(ref, "29727daced33cc95506f580ea542f4bd", "repo")
    before = EvidenceRegistry([claim]).audit_references(root=tmp_path)

    (tmp_path / ".git" / "info" / "exclude").write_text("*.pdf\n")
    home = tmp_path / "home"
    (home / ".config" / "git").mkdir(parents=True)
    (home / ".config" / "git" / "ignore").write_text("*.pdf\n")
    after = EvidenceRegistry([claim]).audit_references(root=tmp_path)

    # 证明 Git 自己已经把它当成本机忽略路径，而结论不受影响
    ignored = subprocess.run(
        (git, "-C", str(tmp_path), "check-ignore", "-v", "--", ref),
        capture_output=True,
        text=True,
    )
    assert ignored.returncode == 0 and ".git/info/exclude" in ignored.stdout
    assert after == before
    assert after[0].startswith("MISSING-REPO-REFERENCE")


def test_v8_real_training_claim_keeps_a_repo_reference(registry):
    """Its three run artifacts are gone on both hosts and the reproduction probe
    was deleted by 37f4d11, so the in-git report is the only reference that lets
    anyone re-check the claim."""
    claim = registry.claims["tcm-real-training-v8"]
    repo = [entry["path"] for entry in claim.retrieved_files if entry.get("scope") == "repo"]
    assert "docs/reports/tcm-50-real-training.md" in repo
    for path in repo:
        assert (REGISTRY_PATH.parents[2] / path).is_file(), path


def test_registry_declares_scope_for_every_reference(registry):
    """Schema contract: scope is data, not a per-host guess."""
    for claim in registry.claims.values():
        for entry in claim.retrieved_files:
            assert entry.get("scope") in SCOPES, (claim.claim_id, entry)


def test_reference_without_scope_is_rejected():
    with pytest.raises(ValueError, match="needs scope"):
        EvidenceClaim(
            claim_id="bad",
            primary_url="https://example.com",
            exact_supported_claim="x",
            verification_state=VERIFIED,
            access_date="2026-11-05",
            retrieved_files=({"path": "docs/x.md", "sha256": "abc"},),
        )


def test_git_blob_digest_must_be_repo_scoped():
    """TRACKED-IN-GIT means "the git blob is the evidence"; pairing it with a
    host/external scope would be a contradiction that hides a repo reference."""
    with pytest.raises(ValueError, match="git blob"):
        EvidenceClaim(
            claim_id="bad",
            primary_url="https://example.com",
            exact_supported_claim="x",
            verification_state=VERIFIED,
            access_date="2026-11-05",
            retrieved_files=({"path": "docs/x.md", "sha256": TRACKED_IN_GIT, "scope": "host"},),
        )


def test_upstream_path_must_be_declared_external():
    with pytest.raises(ValueError, match="external"):
        EvidenceClaim(
            claim_id="bad",
            primary_url="https://example.com",
            exact_supported_claim="x",
            verification_state=VERIFIED,
            access_date="2026-11-05",
            retrieved_files=({"path": "upstream:x@y", "sha256": "abc", "scope": "repo"},),
        )


def test_annotated_reference_path_still_resolves(tmp_path):
    """Regression: registry entries append "(dev host, git-ignored)" to the path,
    which made an existing file read as absent."""
    (tmp_path / "data" / "local").mkdir(parents=True)
    (tmp_path / "data" / "local" / "share.xls").write_bytes(b"x")
    claim = _claim("data/local/share.xls (dev host, git-ignored)", "6206f432b683f8a7", "host")
    assert EvidenceRegistry([claim]).audit_references(root=tmp_path) == []


def test_parenthesis_in_real_filename_is_not_read_as_annotation(tmp_path):
    """A path containing " (" but not ending in that paren is a real file;
    truncating it would raise a false MISSING-REPO-REFERENCE defect."""
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "foo (old).py").write_text("")
    lines = EvidenceRegistry(
        [_claim("tests/foo (old).py", TRACKED_IN_GIT, "repo")]
    ).audit_references(root=tmp_path)
    assert lines == []
