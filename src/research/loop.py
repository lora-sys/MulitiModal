"""The source-level AutoResearch loop: propose → apply → train → measure →
advance the branch or reset it.

This is the piece the previous, config-only loop was missing. Three properties
are taken from karpathy/autoresearch because they are what make the loop
research rather than a hyperparameter sweep:

1. **The agent edits source.** The patch may rewrite the model, the loss, the
   optimizer or the scheduling in `research/candidate.py`. Nothing about the
   architecture is fixed in advance.
2. **A fixed wall clock per fold.** Every candidate gets the same budget, so a
   higher score cannot be bought by training longer.
3. **keep/discard is a git operation.** A KEEP commits the patch on the run's
   branch and the next round starts from that commit; a REVERT resets the tree
   to the champion commit. The artifact of the research is code — the manifest
   is the evidence layer beside it, not a replacement for it.

Plus the criterion upstream states and this loop can measure: **simpler is
better**. A candidate must clear the noise margin PLUS a price for the net code
it adds, and a candidate that removes code while scoring at least as well is
kept outright.

Safety: all git operations happen in the run's own worktree, on a branch whose
name must start with `autoresearch/`. The loop refuses to operate anywhere else.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from runtime.manifest import MANIFEST_FILE, RunManifestStore
from runtime.tcm_eval import PRIMARY_METRIC

from research.guard import (
    PatchRejected,
    apply_patch,
    editable_lines,
    validate_frozen_imports,
    validate_source_patch,
)

#: Minimum score gain per net line added to the editable file.
#:
#: Upstream's rule is qualitative ("a 0.001 improvement that adds 20 lines of
#: hacky code? Probably not worth it; a 0.001 improvement from deleting code?
#: Definitely keep"). A search needs a number, and 0.001-per-line is that rule
#: read literally — so it is declared here rather than fitted to a result.
#:
#: It is a FLOOR on the gain, not a price added to the margin. Pricing lines at
#: a fixed absolute rate instead (the first cut used 0.002/line) makes any real
#: change unaffordable: a 15-line architecture edit would have needed +0.03 on
#: a selection whose standard error is 0.016, i.e. the rule would have vetoed
#: everything except micro-edits while looking like a complexity criterion.
#: As a floor it does what upstream means: a big sloppy patch must show a big
#: gain, while a targeted small one only has to clear the noise margin.
MIN_GAIN_PER_ADDED_LINE = 0.001

DEFAULT_EDITABLE = "src/research/candidate.py"


class GitWorktree:
    """The run's branch, in the run's worktree, and nowhere else."""

    def __init__(self, path: str | Path, branch: str) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise ValueError(f"{self.path} does not exist")
        if not self._git("rev-parse", "--git-dir", check=False):
            raise ValueError(f"{self.path} is not a git worktree")
        if not branch.startswith("autoresearch/"):
            raise ValueError(
                f"branch {branch!r} must start with 'autoresearch/' — the loop "
                "advances and resets commits and must not do that on a "
                "development branch"
            )
        self.branch = branch

    def _git(self, *args: str, check: bool = True) -> str:
        result = subprocess.run(
            ["git", *args], cwd=self.path, text=True, capture_output=True
        )
        if check and result.returncode != 0:
            raise RuntimeError(
                f"git {' '.join(args)} failed in {self.path}: {result.stderr.strip()}"
            )
        return result.stdout.strip()

    def head(self) -> str:
        return self._git("rev-parse", "HEAD")

    def ensure_branch(self) -> str:
        current = self._git("rev-parse", "--abbrev-ref", "HEAD")
        if current != self.branch:
            exists = self._git("rev-parse", "--verify", self.branch, check=False)
            if exists:
                self._git("checkout", self.branch)
            else:
                self._git("checkout", "-b", self.branch)
        return self.head()

    def commit(self, message: str) -> str:
        self._git("add", "-A")
        self._git("commit", "-m", message)
        return self.head()

    def reset_to(self, sha: str) -> None:
        self._git("reset", "--hard", sha)
        # untracked files a patch may have created must go too, or the next
        # round would inherit a rejected candidate's leftovers
        self._git("clean", "-fd", "--", "src/research")

    def apply(self, patch: str, editable: str) -> None:
        apply_patch(self.path, patch, editable)

    def sha_of(self, ref: str) -> str:
        return self._git("rev-parse", ref)

    def assert_clean(self) -> None:
        """Refuse to run with a dirty tree.

        A dirty worktree means the files being trained are not the commit the
        manifest will record as the round's parent, and a patch produced with
        `git diff` then carries someone else's uncommitted work into the round —
        which is exactly how the first attempt at this loop's round 0 was
        rejected for touching `frozen.py`. Failing here names the offending
        files instead of leaving a reader to reverse-engineer it from a guard
        error.
        """
        dirty = self._git("status", "--porcelain")
        if dirty:
            raise RuntimeError(
                "worktree is dirty; commit or discard before running a round:\n"
                + "\n".join(dirty.splitlines()[:10])
            )


class SourceResearchLoop:
    """Bounded rounds of source-level search over one frozen data contract."""

    def __init__(
        self,
        worktree: GitWorktree,
        out_dir: str | Path,
        manifest_path: str | Path,
        dataset_path: str | Path,
        editable: str = DEFAULT_EDITABLE,
        folds: int = 4,
        repeats: int = 2,
        margin_sem: float = 1.0,
        seed: int = 0,
        timeout_s: float | None = None,
        max_wall_clock_s: float = 6 * 3600,
    ) -> None:
        self.worktree = worktree
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = str(manifest_path)
        self.dataset_path = str(dataset_path)
        self.editable = editable
        self.folds = folds
        self.repeats = repeats
        self.margin_sem = margin_sem
        self.seed = seed
        from research.frozen import TIME_BUDGET_GRACE_SECONDS, TIME_BUDGET_SECONDS

        self.budget = TIME_BUDGET_SECONDS
        # a candidate that hangs is killed here: folds*repeats*(budget+grace)
        # plus room for data loading; the runner's own check is the honest
        # record, this is the enforcement
        self.timeout_s = timeout_s if timeout_s is not None else (
            folds * repeats * (TIME_BUDGET_SECONDS + TIME_BUDGET_GRACE_SECONDS) + 120
        )
        self.max_wall_clock_s = max_wall_clock_s
        self.store = RunManifestStore(self.out_dir / MANIFEST_FILE)

    # -- driver-facing state ------------------------------------------------

    def loc(self) -> int:
        return editable_lines(self.worktree.path, self.editable)

    def champion(self) -> dict | None:
        return self.store.champion()

    def state(self) -> dict:
        """Everything a proposer needs, and nothing it may not have: no holdout,
        no evaluator handle, no label file, no scaler."""
        entries = self.store.load()
        champion = self.champion()
        return {
            "round_index": len(entries),
            "editable_file": self.editable,
            "editable_lines": self.loc(),
            "time_budget_seconds": self.budget,
            "folds": self.folds,
            "repeats": self.repeats,
            "margin_sem": self.margin_sem,
            "min_gain_per_added_line": MIN_GAIN_PER_ADDED_LINE,
            "primary_metric": PRIMARY_METRIC,
            "primary_direction": "max",
            "champion_commit": champion.get("git_commit_sha") if champion else None,
            "best_metric": champion.get("best_so_far") if champion else None,
            "branch": self.worktree.branch,
            "history": [
                {
                    "run_id": e["run_id"],
                    "intent": e.get("intent"),
                    "decision": e.get("decision"),
                    "status": e.get("status"),
                    "reason": e.get("reason"),
                    "primary_value": (e.get("metrics") or {}).get("primary_value"),
                    "complexity": e.get("complexity"),
                }
                for e in entries
            ],
        }

    # -- the decision -------------------------------------------------------

    @staticmethod
    def decide(
        value: float, sem: float, growth: int, best: float | None,
        margin_sem: float = 1.0,
    ) -> tuple[bool, str]:
        """KEEP/REVERT, with the two upstream criteria made measurable.

        * A candidate must beat the champion by more than `margin_sem` standard
          errors of the measured mean — the margin is about the quantity being
          compared, so more folds sharpen it instead of leaving it fixed.
        * It must ALSO pay for the net code it adds. Upstream weighs a gain
          against the complexity that bought it, and a rule that ignores
          complexity will happily accept an unreadable patch for noise.
        * Removing code while scoring at least as well is kept outright: that is
          upstream's "removing something and getting equal or better results is
          a great outcome".
        """
        margin = margin_sem * sem
        if best is None:
            return True, f"baseline {value:.6f}"
        gain = value - best
        if grow_floor := MIN_GAIN_PER_ADDED_LINE * max(0, growth):
            pass
        if gain > margin and gain >= grow_floor:
            return True, (
                f"{value:.6f} beats {best:.6f}: +{gain:.4f} over a {margin:.4f} "
                f"margin, and >= {grow_floor:.4f} for {max(0, growth)} added lines"
            )
        if growth < 0 and value >= best - margin:
            return True, (
                f"simpler ({growth:+d} net lines) at {value:.6f} vs {best:.6f}, "
                f"inside the {margin:.4f} margin"
            )
        if gain <= margin:
            return False, (
                f"{value:.6f} vs best {best:.6f}: +{gain:.4f} does not clear the "
                f"{margin:.4f} noise margin"
            )
        return False, (
            f"{value:.6f} vs best {best:.6f}: +{gain:.4f} clears the {margin:.4f} "
            f"margin but not {grow_floor:.4f} for {max(0, growth)} added lines"
        )

    # -- one round ----------------------------------------------------------

    def run_round(
        self,
        patch: str,
        intent: str,
        proposer: str | None = None,
        proposal_mtime: float | None = None,
    ) -> dict:
        self.worktree.assert_clean()
        started = time.monotonic()
        entries = self.store.load()
        index = len(entries)
        run_id = f"tcm-{index:04d}"
        champion = self.champion()
        parent_sha = champion.get("git_commit_sha") if champion else self.worktree.head()
        loc_before = self.loc()
        entry: dict = {
            "run_id": run_id,
            "round_index": index,
            "intent": intent,
            "proposer": proposer,
            "editable_file": self.editable,
            "complexity": {"lines_before": loc_before},
            "started_at": _iso(),
        }

        def finish(decision: str, status: str, reason: str, **extra) -> dict:
            entry.update(
                decision=decision, status=status, reason=reason,
                elapsed_s=round(time.monotonic() - started, 2),
                best_so_far=(
                    (entry.get("metrics") or {}).get("primary_value")
                    if decision == "KEEP" else
                    (champion.get("best_so_far") if champion else None)
                ),
                finished_at=_iso(),
                **extra,
            )
            self.store.append(entry)
            return entry

        try:
            validate_source_patch(patch, self.editable)
            self.worktree.apply(patch, self.editable)
            validate_frozen_imports(
                (self.worktree.path / self.editable).read_text(encoding="utf-8")
            )
        except PatchRejected as exc:
            self.worktree.reset_to(parent_sha)
            return finish("REVERT", "rejected", f"guard: {exc}")

        loc_after = self.loc()
        growth = loc_after - loc_before
        entry["complexity"] = {
            "lines_before": loc_before, "lines_after": loc_after, "growth": growth,
            "min_gain_required": MIN_GAIN_PER_ADDED_LINE * max(0, growth),
        }
        entry["patch"] = patch

        config = {
            "data": {"manifest_path": self.manifest_path, "dataset_path": self.dataset_path},
            "train_provider": {"batch_size": 32},
            "trainer": {"seed": self.seed},
            "time_budget_seconds": self.budget,
        }
        config_path = self.out_dir / f"{run_id}-config.json"
        config_path.write_text(json.dumps(config, indent=2))
        try:
            result = self._run_runner(config_path)
        except TimeoutError as exc:
            self.worktree.reset_to(parent_sha)
            return finish("REVERT", "timeout", str(exc))
        except RuntimeError as exc:
            self.worktree.reset_to(parent_sha)
            return finish("REVERT", "crash", str(exc))

        if result.get("overran"):
            # the candidate exceeded the budget it was given; a score bought
            # that way is not comparable to the champion's
            self.worktree.reset_to(parent_sha)
            return finish(
                "REVERT", "overran",
                f"fold time exceeded the {self.budget:.0f}s budget "
                f"(fold_seconds {result['fold_seconds']})",
                metrics={"primary_metric": PRIMARY_METRIC,
                         "primary_value": result["primary_value"]},
            )

        value = float(result["primary_value"])
        sem = float(result["primary_sem"])
        previous_best = champion.get("best_so_far") if champion else None
        keep, reason = self.decide(value, sem, growth, previous_best, self.margin_sem)
        entry["metrics"] = {
            "primary_metric": PRIMARY_METRIC, "primary_value": value,
            "primary_std": result["primary_std"], "primary_sem": sem,
            "folds": result["folds"], "fold_seconds": result["fold_seconds"],
        }
        if keep:
            sha = self.worktree.commit(f"{run_id}: {intent}")
        else:
            self.worktree.reset_to(parent_sha)
            sha = parent_sha
        return finish(
            "KEEP" if keep else "REVERT", "completed", reason,
            git_commit_sha=sha, git_parent_sha=parent_sha,
            selection={"folds": self.folds, "repeats": self.repeats,
                       "margin_sem": self.margin_sem, "sem": sem,
                       "margin": self.margin_sem * sem},
        )

    def _run_runner(self, config_path: Path) -> dict:
        import os
        import sys

        command = [
            sys.executable, "-m", "research.runner", "--config", str(config_path),
            "--folds", str(self.folds), "--repeats", str(self.repeats),
        ]
        # the same interpreter that runs the loop, and only this worktree's src
        # on the import path: a candidate must not be able to reach the loop's
        # environment or another checkout's modules
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            "PYTHONPATH": str(self.worktree.path / "src"),
        }
        try:
            result = subprocess.run(
                command, cwd=self.worktree.path, text=True, capture_output=True,
                timeout=self.timeout_s, env=env,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"runner exceeded the hard timeout {self.timeout_s:.0f}s "
                "(a candidate that hangs is a crash, not a slow winner)"
            ) from exc
        if result.returncode != 0:
            detail = (result.stdout or result.stderr).strip().splitlines()
            last = detail[-1][:400] if detail else "no output"
            raise RuntimeError(f"runner failed ({result.returncode}): {last}")
        try:
            payload = json.loads(result.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError) as exc:
            raise RuntimeError(f"runner produced no JSON result: {result.stdout[-300:]!r}") from exc
        if not payload.get("ok"):
            raise RuntimeError(f"candidate crashed: {payload.get('error')}")
        return payload

    def run(self, proposals: list[dict]) -> dict:
        """Run rounds from a pre-supplied list of {patch, intent, proposer}.

        Supplying the list up front is how a batched sweep would run — and it is
        recorded as such: `proposal_mtime` and the round's start time let a
        reader see whether each proposal was written before the state it claims
        to answer (the v8 defect, where 22 of 24 rounds did exactly that).
        """
        deadline = time.monotonic() + self.max_wall_clock_s
        decisions = []
        for proposal in proposals:
            if time.monotonic() > deadline:
                break
            decisions.append(self.run_round(
                proposal["patch"], proposal.get("intent", ""),
                proposal.get("proposer"),
                proposal.get("proposal_mtime"),
            ))
        champion = self.champion()
        return {
            "rounds": len(decisions),
            "decisions": [
                {k: d[k] for k in ("run_id", "decision", "status", "reason")}
                for d in decisions
            ],
            "champion_run_id": champion.get("run_id") if champion else None,
            "champion_commit": champion.get("git_commit_sha") if champion else None,
            "best_metric": champion.get("best_so_far") if champion else None,
            "branch": self.worktree.branch,
        }


def _iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()