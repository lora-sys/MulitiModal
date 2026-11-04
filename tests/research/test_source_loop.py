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
    editable_complexity,
    touched_paths,
    validate_frozen_imports,
    validate_source_patch,
)
from research.loop import MIN_GAIN_PER_ADDED_NODE, GitWorktree, SourceResearchLoop

pytestmark = pytest.mark.synthetic

EDITABLE = "src/research/candidate.py"

GOOD_PATCH = """diff --git a/src/research/candidate.py b/src/research/candidate.py
--- a/src/research/candidate.py
+++ b/src/research/candidate.py
@@ -1 +1,2 @@
 WIDTH = 32
+WIDTH = 64
"""


def test_frozen_paths_may_not_change():
    """Editable surface v2: a denylist, not a one-file allowlist."""
    validate_source_patch(GOOD_PATCH, EDITABLE)  # anchor candidate is writable
    for frozen in (
        "src/research/frozen.py",
        "src/research/loop.py",
        "src/research/evaluator.py",
        "src/runtime/tcm_eval.py",
        "tests/test_x.py",
        "AGENTS.md",
        "docs/reports/x.md",
    ):
        patch = GOOD_PATCH.replace(f"a/{EDITABLE}", f"a/{frozen}").replace(
            f"b/{EDITABLE}", f"b/{frozen}"
        )
        with pytest.raises(PatchRejected, match="frozen path|outside the writable"):
            validate_source_patch(patch, EDITABLE)


def test_any_writable_source_file_may_change_and_new_modules_may_be_added():
    """The Agent is no longer limited to one file or to hyperparameter edits."""
    other = """--- a/src/core/models.py
+++ b/src/core/models.py
@@ -1 +1,2 @@
 WIDTH = 32
+WIDTH = 64
"""
    validate_source_patch(other, EDITABLE)  # a second source file is fine
    new_module = """--- /dev/null
+++ b/src/research/new_encoder.py
@@ -0,0 +1 @@
+class Encoder: pass
"""
    validate_source_patch(new_module, EDITABLE)  # a NEW module is fine
    multi = GOOD_PATCH + other
    assert touched_paths(multi) == {EDITABLE, "src/core/models.py"}
    validate_source_patch(multi, EDITABLE)
    for outside in ("README.md", "scripts/x.py", "src/notes.txt",
                    "../outside.py", "/etc/passwd"):
        patch = GOOD_PATCH.replace(f"a/{EDITABLE}", f"a/{outside}").replace(
            f"b/{EDITABLE}", f"b/{outside}"
        )
        with pytest.raises(PatchRejected):
            validate_source_patch(patch, EDITABLE)


def test_frozen_symbols_are_refused_when_added():
    for symbol in ("TCMTrustedEvaluator", "holdout", "fold_manifest", "_fit_scaler"):
        patch = GOOD_PATCH + f"+x = {symbol}\n"
        with pytest.raises(PatchRejected, match="frozen research machinery"):
            validate_source_patch(patch, EDITABLE)


def test_only_added_lines_are_judged():
    # a patch that merely QUOTES frozen-machinery text as context, or deletes a
    # reference to it, is not reaching for the evaluator — the ensemble patch of
    # the first real run was rejected for exactly this
    context = """diff --git a/src/research/candidate.py b/src/research/candidate.py
--- a/src/research/candidate.py
+++ b/src/research/candidate.py
@@ -1,3 +1,4 @@
 WIDTH = 32
 batch.targets["category_labels"]  # quoting the baseline's own line
+TARGETS = "category_labels"
"""
    validate_source_patch(context, EDITABLE)  # does not raise
    deletion = GOOD_PATCH + "-old = TCMTrustedEvaluator\n"
    validate_source_patch(deletion, EDITABLE)  # does not raise


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


def test_complexity_metric_prices_semantics_not_lines(tmp_path):
    # the first implementation counted non-blank text lines, so a patch could
    # compress twenty statements onto one dense line and score as a
    # "simplification" while adding constructs. The metric counts AST
    # statement+expression nodes: densifying cannot reduce it.
    (tmp_path / "spread.py").write_text("WIDTH = 32\nFUSION = 32\n")
    (tmp_path / "dense.py").write_text("WIDTH = 32; FUSION = 32\n")
    spread = editable_complexity(tmp_path, "spread.py")
    dense = editable_complexity(tmp_path, "dense.py")
    assert dense == spread > 0


def test_complexity_metric_counts_removed_constructs_as_negative_growth(tmp_path):
    # deleting statements and re-adding an equivalent rewrite must not reset
    # the complexity floor: the metric is a before/after node delta, so only
    # genuinely removing constructs reads as negative growth
    (tmp_path / "a.py").write_text("x = 1\ny = 2\nz = x + y\n")
    (tmp_path / "b.py").write_text("x = 1\nz = x + 2\n")
    before = editable_complexity(tmp_path, "a.py")
    after = editable_complexity(tmp_path, "b.py")
    assert after < before


def test_complexity_counts_string_wrapped_code_so_simplification_cannot_be_faked(tmp_path):
    # F-07: `exec("...")` hides a construct tree inside one string node — a
    # patch could delete real code, re-add it wrapped in a string, and buy the
    # negative growth the simplification criterion rewards. String constants
    # are parsed recursively and their nodes counted: hiding code in a string
    # costs what the code costs.
    (tmp_path / "honest.py").write_text("x = 1\ny = x + 1\nz = y * 2\n")
    (tmp_path / "wrapped.py").write_text('exec("x = 1\\ny = x + 1\\nz = y * 2\\n")\n')
    assert editable_complexity(tmp_path, "wrapped.py") >= (
        editable_complexity(tmp_path, "honest.py")
    )
    # a plain short string is still just a couple of nodes
    (tmp_path / "small.py").write_text('NAME = "category_labels"\n')
    assert editable_complexity(tmp_path, "small.py") > 0


def test_dynamic_execution_calls_are_refused_zero_tolerance():
    # F-07: any bare-name call to exec/eval/compile/__import__ is refused
    # outright, whatever the payload — `eval(` as a text symbol would also
    # match the legitimate `model.eval()`, so this is an AST check
    from research.guard import validate_source_ast

    validate_source_ast("model.eval()\n")  # the training idiom is fine
    validate_source_ast("self.compile_all()\n")  # attribute names are fine
    for call in (
        'exec("payload = 1")',
        "eval(input)",
        "compile(src, 'x', 'exec')",
        "__import__('os')",
    ):
        with pytest.raises(PatchRejected, match="dynamic code execution"):
            validate_source_ast(call)


def test_dynamic_execution_aliases_are_refused_two_generations():
    # H-01: binding the builtin to a name (`f = exec`) and calling the name
    # dodged the bare-call check — the alias targets join the refused set,
    # and the refusal propagates one more hop (`g = f`)
    from research.guard import validate_source_ast

    validate_source_ast("model.eval()\n")  # the training idiom is still fine
    for source in (
        'f = exec\nf("x = 1")',
        'f = exec\ng = f\ng("x = 1")',
        "f = eval\nscore = f(payload)",
    ):
        with pytest.raises(PatchRejected, match="dynamic code execution"):
            validate_source_ast(source)


def test_dynamic_execution_bare_names_are_refused_in_any_syntax():
    # round-4 review finding 2: the two-generation alias walk missed walrus,
    # unpacking and container storage. One rule now covers every form that
    # makes the builtin appear as a bare ast.Name: Load or Store, any scope.
    from research.guard import validate_source_ast

    validate_source_ast("model.eval()\n")  # the training idiom is still fine
    validate_source_ast("def fit(self):\n    return self.model.eval()\n")
    for source in (
        'f = exec\nf("x = 1")',
        'f = exec\ng = f\ng("x = 1")',
        "f = eval\nscore = f(payload)",
        'if (f := exec):\n    f("x = 1")',
        "for f in [exec]:\n    f('x = 1')",
        "f, = [exec]\nf('x = 1')",
        "fs = [exec]\nfs[0]('x = 1')",
        "exec = print",           # even rebinding the name is refused
        'h = g\ng = f\nf = exec\nh("x = 1")',  # a 3-generation chain
    ):
        with pytest.raises(PatchRejected, match="bare name"):
            validate_source_ast(source)


def test_forbidden_symbols_cover_the_dynamic_execution_text():
    # the text-level scan still catches the quoted name in added lines where
    # no AST exists yet (patch validation runs before the patch is applied)
    for symbol in ("__import__", "compile(", "exec("):
        patch = GOOD_PATCH + f"+x = {symbol}('payload')\n"
        with pytest.raises(PatchRejected):
            validate_source_patch(patch, EDITABLE)


def test_unparseable_and_over_depth_string_payloads_price_by_length(tmp_path):
    # H-07: a string constant that cannot be parsed (or sits beyond the
    # string-recursion depth cap) used to cost one node while carrying
    # arbitrary payload — hiding code behind exec nesting had to stop paying.
    # The declared price: every _STRING_CHARS_PER_NODE chars is one node.
    from research.guard import _STRING_CHARS_PER_NODE

    # an unparseable blob is priced by length, not skipped
    (tmp_path / "blob.py").write_text("BLOB = " + repr("((((" * 500) + "\n")
    blob = editable_complexity(tmp_path, "blob.py")
    assert blob >= 2000 // _STRING_CHARS_PER_NODE

    # three exec levels of nesting hide a large payload past the parse-depth
    # cap: the growth over a trivial baseline must be clearly positive (the
    # priced string value is the payload itself, wrapper levels parse cheaply)
    payload = "v = 1\n" * 4000
    hidden = "data = " + repr(payload)
    wrapped = "exec(" + repr("exec(" + repr("exec(" + repr(hidden) + ")") + ")") + ")"
    (tmp_path / "before.py").write_text("WIDTH = 32\n")
    (tmp_path / "wrapped.py").write_text(wrapped + "\n")
    growth = (
        editable_complexity(tmp_path, "wrapped.py")
        - editable_complexity(tmp_path, "before.py")
    )
    assert growth >= len(payload) // _STRING_CHARS_PER_NODE


def test_simplicity_criterion_matches_upstream_intent():
    # the margin uses descriptive fold SD, not an independence-based SEM
    assert SourceResearchLoop.decide(0.30, 0.01, 0, None) == (True, "baseline 0.300000")
    # a gain inside the margin but with fewer lines is kept: upstream's
    # "removing something and getting equal or better results" is a win
    keep, reason = SourceResearchLoop.decide(0.2995, 0.01, -8, 0.30)
    assert keep and "simpler" in reason, reason
    # a large gain clears margin and the per-line floor
    assert SourceResearchLoop.decide(0.36, 0.01, 8, 0.30)[0]
    # The per-node floor is a FLOOR ON THE GAIN, not a price added to the
    # margin. With 20 nodes the floor (0.02) exceeds the margin (0.01), so a
    # gain of +0.015 clears the noise margin and is still refused: 20 nodes of
    # hack must buy more than 0.015.
    keep, reason = SourceResearchLoop.decide(0.315, 0.01, 20, 0.30)
    assert not keep and "added nodes" in reason, reason
    assert SourceResearchLoop.decide(0.325, 0.01, 20, 0.30)[0]
    # a small targeted edit only has to clear the noise margin
    assert SourceResearchLoop.decide(0.32, 0.01, 2, 0.30)[0]
    # a big sloppy patch must show a big gain
    assert not SourceResearchLoop.decide(0.35, 0.01, 400, 0.30)[0]
    assert SourceResearchLoop.decide(0.75, 0.01, 400, 0.30)[0]
    assert MIN_GAIN_PER_ADDED_NODE == 0.001


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
    assert editable_complexity(repo, EDITABLE) == 3

    wt.apply(GOOD_PATCH, EDITABLE)
    assert "WIDTH = 64" in editable.read_text()
    assert editable_complexity(repo, EDITABLE) == 6
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
    assert loop.state()["editable_complexity"] == 3
def _source_loop_for_candidate(tmp_path, source):
    from types import SimpleNamespace

    candidate = tmp_path / EDITABLE
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_bytes(source)
    loop = SourceResearchLoop.__new__(SourceResearchLoop)
    loop.worktree = SimpleNamespace(path=tmp_path)
    loop.editable = EDITABLE
    return loop, candidate


def test_loop_checks_candidate_size_before_reading_source(tmp_path, monkeypatch):
    from research.train_worker import MAX_CANDIDATE_SOURCE_BYTES

    loop, candidate = _source_loop_for_candidate(
        tmp_path, b"x" * (MAX_CANDIDATE_SOURCE_BYTES + 1)
    )
    original = Path.read_text

    def fail_candidate_read(path, *args, **kwargs):
        if path == candidate:
            pytest.fail("loop must check file size before reading candidate source")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_candidate_read)
    with pytest.raises((PatchRejected, RuntimeError), match="limit|exceeds"):
        loop.complexity()


def test_loop_parses_pathological_source_in_isolated_child(tmp_path, monkeypatch):
    from research import loop as loop_module

    source = (
        b"x = " + b"(" * 20_000 + b"1" + b")" * 20_000 + b"\n"
    )
    loop, _candidate = _source_loop_for_candidate(tmp_path, source)

    def host_parse_forbidden(*_args, **_kwargs):
        pytest.fail("candidate AST parsing must run outside the loop process")

    monkeypatch.setattr(
        loop_module, "editable_complexity", host_parse_forbidden, raising=False
    )
    with pytest.raises(PatchRejected):
        loop.complexity()
