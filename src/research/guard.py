"""Patch validation for the source-level loop — the whitelist, enforced.

The agent may change one file. `validate_source_patch` refuses:

1. any diff that touches a path other than the editable file — a candidate that
   can edit the loop, the runner or the frozen boundary can simply declare
   itself the winner;
2. any diff whose text mentions a `FORBIDDEN_SYMBOLS` name — reaching the
   evaluator, the split, the labels or the scaler is how a candidate would
   report a number of its own choosing;
3. any diff that does not apply to the current worktree, so a rejected patch
   never leaves the tree in a half-applied state;
4. a patch that empties the editable file — "delete the model" would otherwise
   score as a spectacular simplification.

Same evidence grade as the rest of the harness: it stops the careless and the
accidental. The structural protection is that the evaluator is imported and run
by the runner, in a different process, after the candidate has returned.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from research.frozen import FORBIDDEN_SYMBOLS

#: diff headers: "+++ b/path" or "--- a/path"; /dev/null for added files
_DIFF_PATH = re.compile(r"^(?:\+\+\+|---)\s+(?:[ab]/)?(\S+)", re.MULTILINE)

#: an import of the frozen module may only name the public API
_FROZEN_IMPORT = re.compile(
    r"from\s+research\.frozen\s+import\s+([^\n(]+)", re.MULTILINE
)


class PatchRejected(ValueError):
    """A patch the loop refuses to apply; the round is recorded as rejected."""


def touched_paths(patch: str) -> set[str]:
    return {
        p for p in _DIFF_PATH.findall(patch)
        if p not in ("/dev/null",)
    }


def validate_source_patch(patch: str, editable: str | Path) -> None:
    """Raise PatchRejected unless `patch` is a safe edit of `editable` alone."""
    editable = str(editable)
    if not patch.strip():
        raise PatchRejected("empty patch: a round must propose something")
    paths = touched_paths(patch)
    if not paths:
        raise PatchRejected("patch touches no file: nothing to apply")
    foreign = sorted(p for p in paths if p != editable)
    if foreign:
        raise PatchRejected(
            f"patch touches {foreign}, outside the editable file {editable!r}"
        )
    # only ADDED lines are judged: a patch is what it introduces, not what it
    # quotes as context or deletes. Scanning the whole diff rejected a
    # legitimate ensemble patch because its context lines contained the
    # baseline's own `batch.targets["category_labels"]` line.
    introduced = "\n".join(
        line for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    hits = sorted(s for s in FORBIDDEN_SYMBOLS if s in introduced)
    if hits:
        raise PatchRejected(
            f"patch references frozen research machinery: {hits}"
        )
    removals = sum(1 for line in patch.splitlines() if line.startswith("-")
                   and not line.startswith("---"))
    additions = sum(1 for line in patch.splitlines() if line.startswith("+")
                    and not line.startswith("+++"))
    if removals and not additions:
        raise PatchRejected(
            "patch only removes code: deleting the model is not a simplification"
        )


def validate_frozen_imports(source: str) -> None:
    """A candidate may import from `research.frozen` only via the public API."""
    from research.frozen import FROZEN_PUBLIC_API

    for names in _FROZEN_IMPORT.findall(source):
        for name in (n.strip() for n in names.split(",")):
            if name and name not in FROZEN_PUBLIC_API:
                raise PatchRejected(
                    f"candidate imports {name!r} from research.frozen, which is "
                    f"not part of the public API {FROZEN_PUBLIC_API}"
                )


def apply_patch(worktree: Path, patch: str, editable: str) -> None:
    """`git apply` the patch inside `worktree`, or raise PatchRejected.

    `--unidiff-zero` is deliberately NOT passed: a patch that does not carry
    enough context to prove it belongs to this file must be regenerated, not
    applied loosely.
    """
    result = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", "-"],
        cwd=worktree, input=patch, text=True, capture_output=True,
    )
    if result.returncode != 0:
        raise PatchRejected(
            f"git apply failed ({result.returncode}): {result.stderr.strip()[:300]}"
        )


def editable_lines(worktree: Path, editable: str) -> int:
    """Non-blank, non-comment lines in the editable file: the complexity proxy
    the simplicity criterion is priced against."""
    text = (worktree / editable).read_text(encoding="utf-8")
    return sum(
        1 for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    )