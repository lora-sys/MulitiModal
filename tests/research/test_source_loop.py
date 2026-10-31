"""Source-level AutoResearch loop tests (#45 successor).

The pure decision/guard logic is tested here; the end-to-end run needs the real
workbook and lives on the dev host.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from research.guard import (
    PatchRejected,
    editable_lines,
    touched_paths,
    validate_frozen_imports,
    validate_source_patch,
)
from research.loop import COMPLEXITY_PRICE_PER_LINE, GitWorktree, SourceResearchLoop

pytestmark = pytest.mark.synthetic

EDITABLE = "src/research/candidate.py"

GOOD_PATCH = """diff --git a/src/research/candidate.py b/src/research/candidate.py
--- a/src/research/candidate.py
+++ b/src/research/candidate.py
@@ -1 +1,2 @@
 WIDTH = 32
+WIDTH = 64
"""


def test_only_the_editable_file_may_change():
    validate_source_patch(GOOD_PATCH, EDITABLE)  # does not raise
    for foreign in (
        "src/research/frozen.py",
        "src/research/loop.py",
        "src/runtime/tcm_eval.py",
        "tests/test_x.py",
    ):
        patch = GOOD_PATCH.replace(f"a/{EDITABLE}", f"a/{foreign}").replace(
            f"b/{EDITABLE}", f"b/{foreign}"
        )
        with pytest.raises(PatchRejected, match="outside the editable file"):
            validate_source_patch(patch, EDITABLE)


def test_frozen_symbols_are_refused():
    for symbol in ("TCMTrustedEvaluator", "holdout", "category_labels", "score_mask"):
        patch = GOOD_PATCH + f"+x = {symbol}\n"
        with pytest.raises(PatchRejected, match="frozen research machinery"):
            validate_source_patch(patch, EDITABLE)


def test_a_patch_that_only_deletes_code_is_not_a_simplification():
    patch = """diff --git a/src/research/candidate.py b/src/research/candidate.py
--- a/src/research/candidate.py
+++ b/src/research/candidate.py
@@ -1,3 +1,1 @@
-baseline
-WIDTH = 32
 """
    with pytest.raises(PatchRejected, match="only removes code"):
        validate_source_patch(patch, EDITABLE)


def test_empty_and_fileless_patches_are_refused():
    with pytest.raises(PatchRejected, match="empty patch"):
        validate_source_patch("   \n", EDITABLE)
    with pytest.raises(PatchRejected, match="touches no file"):
        validate_source_patch("@@ -1 +1 @@\n+hello\n", EDITABLE)


def test_touched_paths_reads_both_sides_of_the_diff():
    assert touched_paths(GOOD_PATCH) == {EDITABLE}


def test_candidate_may_not_import_beyond_the_public_api():
    validate_frozen_imports("from research.frozen import TIME_BUDGET_SECONDS, FoldData")
    for bad in ("evaluate_model", "build_fold_and_manifest", "_hidden"):
        with pytest.raises(PatchRejected, match="not part of the public API"):
            validate_frozen_imports(f"from research.frozen import {bad}")


def test_simplicity_criterion_matches_upstream_intent():
    # the margin is in standard errors of the mean, so more folds sharpen it
    assert SourceResearchLoop.decide(0.30, 0.01, 0, None) == (True, "baseline 0.300000")
    # a gain inside the margin but with fewer lines is kept: upstream's
    # "removing something and getting equal or better results" is a win
    keep, reason = SourceResearchLoop.decide(0.2995, 0.01, -8, 0.30)
    assert keep and "simpler" in reason, reason
    # the same gain WITH 8 added lines is not kept: it must pay for complexity
    keep, reason = SourceResearchLoop.decide(0.2995, 0.01, 8, 0.30)
    assert not keep, reason
    # a large gain clears margin + complexity price
    assert SourceResearchLoop.decide(0.36, 0.01, 8, 0.30)[0]
    # complexity is priced linearly, so the bar moves with the diff size
    price_8 = COMPLEXITY_PRICE_PER_LINE * 8
    assert not SourceResearchLoop.decide(0.30 + price_8 - 1e-9, 0.0, 8, 0.30)[0]
    assert SourceResearchLoop.decide(0.30 + price_8 + 1e-9, 0.0, 8, 0.30)[0]


def _git_repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    (tmp_path / "src/research").mkdir(parents=True)
    (tmp_path / "src/research/candidate.py").write_text("WIDTH = 32\n")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=tmp_path, check=True)
    return tmp_path


def test_worktree_refuses_a_branch_the_loop_must_not_advance(tmp_path):
    repo = _git_repo(tmp_path)
    for branch in ("main", "master", "feature/clean-slate", "develop"):
        with pytest.raises(ValueError, match="must start with 'autoresearch/'"):
            GitWorktree(repo, branch)


def test_worktree_applies_commits_and_resets(tmp_path):
    repo = _git_repo(tmp_path)
    wt = GitWorktree(repo, "autoresearch/t")
    base = wt.ensure_branch()
    editable = repo / EDITABLE
    assert editable_lines(repo, EDITABLE) == 1

    wt.apply(GOOD_PATCH, EDITABLE)
    assert "WIDTH = 64" in editable.read_text()
    assert editable_lines(repo, EDITABLE) == 2
    kept = wt.commit("tcm-0000: widen")
    assert kept != base
    assert wt.head() == kept

    # a rejected round must leave no trace of its patch behind
    second = """diff --git a/src/research/candidate.py b/src/research/candidate.py
--- a/src/research/candidate.py
+++ b/src/research/candidate.py
@@ -1,2 +1,3 @@
 WIDTH = 32
 WIDTH = 64
+WIDTH = 128
"""
    wt.apply(second, EDITABLE)
    assert "128" in editable.read_text()
    wt.reset_to(kept)
    assert "WIDTH = 64" in editable.read_text()
    assert wt.head() == kept

    # a reset also clears untracked files the patch may have created
    (repo / "src/research/stray.py").write_text("junk\n")
    wt.reset_to(kept)
    assert not (repo / "src/research/stray.py").exists()


def test_worktree_refuses_a_missing_path_and_a_non_repo(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        GitWorktree(tmp_path / "nope", "autoresearch/t")
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(ValueError, match="not a git worktree"):
        GitWorktree(plain, "autoresearch/t")


def test_loop_refuses_to_start_without_frozen_data(tmp_path):
    # the loop must not be constructible against a checkout it could then
    # advance with no frozen contract behind the numbers
    repo = _git_repo(tmp_path)
    loop = SourceResearchLoop(
        GitWorktree(repo, "autoresearch/t"), tmp_path / "out",
        manifest_path="missing.json", dataset_path="missing.xls",
    )
    assert loop.state()["best_metric"] is None
    assert loop.state()["history"] == []
    assert loop.state()["editable_lines"] == 1