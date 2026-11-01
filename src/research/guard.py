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
accidental. The structural protection is that the score is computed by
`research.evaluator` in its own process, on a fold rebuilt from the frozen
manifest, after the interpreter that ran the candidate's code is done with
the model.
"""

from __future__ import annotations

import ast
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


def editable_complexity(worktree: Path, editable: str) -> int:
    """AST statement+expression node count: the proxy the simplicity criterion
    is priced against.

    The first implementation counted non-blank, non-comment text lines, which
    is gameable in both directions: a patch can compress many statements onto
    one dense line (fewer lines, same code — growth goes negative without
    anything being removed) or delete code and re-add an equivalent rewrite to
    reset a per-line floor. Counting AST statements plus expressions prices
    the semantics the file carries: densifying cannot reduce it, and only
    genuinely removing constructs reads as negative growth.

    String constants are parsed recursively and their nodes counted too
    (bounded depth/size): otherwise a patch could delete real constructs and
    re-add them wrapped in a string (`exec("...")`), buying the negative
    growth the simplification criterion rewards while actually hiding code.
    A constant that CANNOT be parsed — or that sits beyond the depth/length
    caps — is priced by length instead (`_STRING_CHARS_PER_NODE` chars per
    node): that is the declared cost of hiding payload where the parser
    cannot follow, not a semantic count (H-07).
    Together with `validate_source_ast`'s zero tolerance for exec/eval/
    compile/__import__ calls, the "simpler is better" branch is only
    reachable by genuinely removing AST constructs.
    """
    return _ast_node_count(
        ast.parse((worktree / editable).read_text(encoding="utf-8"))
    )


#: recursion bounds for string-constant parsing: a hostile file can nest
#: strings exponentially; these caps keep the count honest for real code and
#: finite for adversarial input
_MAX_STRING_PARSE_DEPTH = 3
_MAX_STRING_PARSE_LENGTH = 100_000

#: the declared price of unpriceable string payload: a constant that cannot
#: be parsed (or sits beyond the depth/length caps above) costs one node per
#: this many characters instead of the single node it used to cost
_STRING_CHARS_PER_NODE = 64


def _string_price(text: str) -> int:
    """The declared cost of string payload the parser cannot follow."""
    return max(1, len(text) // _STRING_CHARS_PER_NODE)


def _ast_node_count(tree: ast.AST, depth: int = 0) -> int:
    total = sum(isinstance(node, (ast.stmt, ast.expr)) for node in ast.walk(tree))
    parse_children = depth < _MAX_STRING_PARSE_DEPTH
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
        ):
            text = node.value
            if parse_children and len(text) <= _MAX_STRING_PARSE_LENGTH:
                try:
                    inner = ast.parse(text)
                except (SyntaxError, ValueError, RecursionError):
                    total += _string_price(text)
                    continue
                total += _ast_node_count(inner, depth + 1)
            else:
                total += _string_price(text)
    return total


#: Dynamic-execution builtins a patch may never CALL. The text-level symbol
#: scan catches the quoted name; this AST check is zero tolerance — any Call
#: whose function is one of these bare names is refused outright, whatever
#: the payload, however it was obfuscated.
DYNAMIC_EXECUTION_CALLS = frozenset({"exec", "eval", "compile", "__import__"})


def validate_source_ast(source: str) -> None:
    """Refuse any dynamic-execution call in the candidate's source.

    `exec`/`eval` string wrapping is how a patch could hide arbitrary logic
    from the complexity metric (one string node instead of the code it
    carries) and from the simplification criterion (delete real constructs,
    re-add them inside an `exec` string: the score's negative growth is then
    fiction, and a "simplification" KEEP can smuggle anything). A bare-name
    call to one of these builtins is refused unconditionally — zero
    tolerance, not a complexity price. `model.eval()` (an Attribute call) is
    the legitimate training idiom and is not touched.

    Aliasing is refused too, two generations deep (H-01): a target bound to a
    dynamic-execution builtin (`f = exec`) or to such a target (`g = f`)
    joins the refused call-name set, so `f("…")` and `g("…")` are refused
    like the direct call they are.
    """
    tree = ast.parse(source)
    banned = set(DYNAMIC_EXECUTION_CALLS)
    # two passes: the first binds names to the builtins themselves (`f = exec`
    # — a Name value, not a Call), the second binds names to those names
    for _generation in range(2):
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Name)
                and node.value.id in banned
            ):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        banned.add(target.id)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in banned
        ):
            raise PatchRejected(
                f"patch calls {node.func.id}(): dynamic code execution is "
                "refused outright — it hides logic from the complexity metric "
                "and the source guards"
            )