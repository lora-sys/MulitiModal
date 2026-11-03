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

from research.agent import (
    REQUIRED_HYPOTHESIS_FIELDS,
    AgentUnavailable,
    ProposalRejected,
)
from research.guard import (
    PatchRejected,
    apply_patch,
    editable_complexity,
    validate_frozen_imports,
    validate_source_ast,
    validate_source_patch,
)
from runtime.manifest import MANIFEST_FILE, RunManifestStore

#: Minimum score gain per net unit of complexity added to the editable file,
#: where the unit is one AST statement/expression node (see
#: `guard.editable_complexity`).
#:
#: Upstream's rule is qualitative ("a 0.001 improvement that adds 20 lines of
#: hacky code? Probably not worth it; a 0.001 improvement from deleting code?
#: Definitely keep"). A search needs a number, and 0.001-per-unit is that rule
#: read literally — so it is declared here rather than fitted to a result.
#:
#: The unit was the non-blank text line until the adversarial review of the
#: v10 run showed lines are gameable in both directions (densifying many
#: statements onto one line, or deleting and re-adding equivalent code to
#: reset a per-line floor). Nodes price the semantics: the rate stays 0.001,
#: declared, not fitted — raising it would re-price every archived round.
#:
#: It is a FLOOR on the gain, not a price added to the margin. Pricing
#: complexity at a fixed absolute rate instead (the first cut used 0.002/line)
#: makes any real change unaffordable: a 15-line architecture edit would have
#: needed +0.03 on a selection whose standard error is 0.016, i.e. the rule
#: would have vetoed everything except micro-edits while looking like a
#: complexity criterion. As a floor it does what upstream means: a big sloppy
#: patch must show a big gain, while a targeted small one only has to clear
#: the noise margin.
MIN_GAIN_PER_ADDED_NODE = 0.001

#: The complexity unit is one AST statement/expression node, with string
#: constants parsed recursively and their payload nodes counted too (guard.
#: editable_complexity). The v10 archived rounds used plain AST nodes without
#: string recursion, so the recorded metric changed name with the semantics:
#: complexity numbers are only comparable within one metric name.
COMPLEXITY_METRIC = "ast_statement_expr_nodes+string_payloads"

DEFAULT_EDITABLE = "src/research/candidate.py"

#: A round has a strict KEEP/REVERT decision (the champion logic needs exactly
#: two), and an `outcome` that says WHY, so an engineering refusal is never read
#: as a model result. The issue names these outcomes explicitly.
OUTCOMES = (
    "KEEP",
    "REVERT",
    "INVALID_PROTOCOL",
    "UNSUPPORTED_ARCHITECTURE",
    "CRASH",
    "BUDGET_EXHAUSTED",
)

_OUTCOME_BY_STATUS = {
    "rejected": "INVALID_PROTOCOL",
    "overran": "INVALID_PROTOCOL",
    "agent_error": "INVALID_PROTOCOL",
    "crash": "CRASH",
    "timeout": "CRASH",
    "unknown_architecture": "UNSUPPORTED_ARCHITECTURE",
}


def outcome_for(decision: str, status: str) -> str:
    """The detailed outcome of a round; `decision` stays KEEP/REVERT."""
    return _OUTCOME_BY_STATUS.get(status, decision if status == "completed" else decision)



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

    def ensure_identity(self) -> None:
        """Guarantee a committer identity INSIDE this worktree.

        The loop commits on every KEEP, so a run must not depend on the host's
        global git config: a fresh worktree on a machine with no `user.email`
        fails at the first commit, after training has already been paid for.
        `commit.gpgsign=false` keeps signing config from turning a KEEP into an
        exception mid-round.
        """
        if not self._git("config", "user.email", check=False):
            self._git("config", "user.email", "autoresearch@localhost")
            self._git("config", "user.name", "AutoResearch loop")
        self._git("config", "commit.gpgsign", "false", check=False)

    def ensure_branch(self) -> str:
        self.ensure_identity()
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
        folds: int = 4,
        repeats: int = 2,
        margin_sem: float = 1.0,
        seed: int = 0,
        timeout_s: float | None = None,
        max_wall_clock_s: float = 6 * 3600,
        task: str = "constitution",
        budget_s: float | None = None,
        editable: str | None = None,
        frozen_goal: str = "",
    ) -> None:
        from research.tasks import adapter_for

        self.worktree = worktree
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = str(manifest_path)
        self.dataset_path = str(dataset_path)
        # `editable` had been declared twice in this signature (DEFAULT_EDITABLE
        # and None), which is a SyntaxError: `compile()` refused the module even
        # though `ast.parse()` accepted it, so nothing that imports the engine
        # could run. The task adapter owns the default editable file, and an
        # explicit argument still wins.
        task_adapter = adapter_for(task)
        self.editable = (
            editable if editable is not None
            else task_adapter.default_editable or DEFAULT_EDITABLE
        )
        self.folds = folds
        self.repeats = repeats
        self.margin_sem = margin_sem
        self.seed = seed
        # the task selects the data/scoring seams the runner and evaluator
        # subprocesses will use, and the metric name the manifest records
        self.task = task
        self.primary_metric = task_adapter.primary_metric
        from research.frozen import TIME_BUDGET_GRACE_SECONDS, TIME_BUDGET_SECONDS

        # campaign-level choice, never a per-round fitted value: every
        # candidate in the campaign gets the SAME budget, so relative ranking
        # stays fair; the default keeps the constitution task's frozen 20 s
        self.budget = TIME_BUDGET_SECONDS if budget_s is None else float(budget_s)
        # a candidate that hangs is killed here: folds*repeats*(budget+grace)
        # plus room for data loading; the runner's own check is the honest
        # record, this is the enforcement
        self.timeout_s = timeout_s if timeout_s is not None else (
            folds * repeats * (self.budget + TIME_BUDGET_GRACE_SECONDS) + 120
        )
        self.max_wall_clock_s = max_wall_clock_s
        # run ids carry the task so two runs in one output directory stay
        # distinguishable in the manifest and on the dashboard
        self.run_prefix = str(task)[:16] or "tcm"
        #: the complexity/acceptance policy is DECLARED and versioned with the
        #: run. Re-pricing an archived round inside the same campaign is what the
        #: issue forbids, so this dict is recorded on every round instead of being
        #: recomputed from whatever the current constants happen to be.
        #: the frozen research objective, echoed to every proposer verbatim. The
        #: goal is fixed by the issue/operator; the proposer may choose hypotheses
        #: and their order, never the objective.
        self.frozen_goal = frozen_goal or (
            f"raise the trusted {self.primary_metric} of the {task} task on the "
            "frozen development selection protocol, without touching data, labels, "
            "splits, the evaluator, the metric or the holdout"
        )
        self.selection_policy = {
            "version": "autoresearch-policy-v2",
            "margin_sem": float(margin_sem),
            "min_gain_per_added_node": MIN_GAIN_PER_ADDED_NODE,
            "complexity_metric": COMPLEXITY_METRIC,
            "note": "declared for this run; a new policy is a new campaign",
        }
        self.store = RunManifestStore(self.out_dir / MANIFEST_FILE)

    # -- driver-facing state ------------------------------------------------

    def complexity(self) -> int:
        return editable_complexity(self.worktree.path, self.editable)

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
            "editable_complexity": self.complexity(),
            "time_budget_seconds": self.budget,
            "folds": self.folds,
            "repeats": self.repeats,
            "margin_sem": self.margin_sem,
            "min_gain_per_added_node": MIN_GAIN_PER_ADDED_NODE,
            "primary_metric": self.primary_metric,
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
        * It must ALSO pay for the net complexity it adds, in AST
          statement/expression nodes (see guard.editable_complexity). Upstream
          weighs a gain against the complexity that bought it, and a rule that
          ignores complexity will happily accept an unreadable patch for noise.
        * Removing constructs while scoring at least as well is kept outright:
          that is upstream's "removing something and getting equal or better
          results is a great outcome". This branch is only reachable by
          genuinely removing AST constructs: complexity counts string-
          wrapped payloads (see COMPLEXITY_METRIC) and dynamic-execution
          calls are refused outright, so deleting real code and re-adding it
          inside an `exec` string cannot buy negative growth.
        """
        margin = margin_sem * sem
        if best is None:
            return True, f"baseline {value:.6f}"
        gain = value - best
        if grow_floor := MIN_GAIN_PER_ADDED_NODE * max(0, growth):
            pass
        if gain > margin and gain >= grow_floor:
            return True, (
                f"{value:.6f} beats {best:.6f}: +{gain:.4f} over a {margin:.4f} "
                f"margin, and >= {grow_floor:.4f} for {max(0, growth)} added nodes"
            )
        if growth < 0 and value >= best - margin:
            return True, (
                f"simpler ({growth:+d} net nodes) at {value:.6f} vs {best:.6f}, "
                f"inside the {margin:.4f} margin"
            )
        if gain <= margin:
            return False, (
                f"{value:.6f} vs best {best:.6f}: +{gain:.4f} does not clear the "
                f"{margin:.4f} noise margin"
            )
        return False, (
            f"{value:.6f} vs best {best:.6f}: +{gain:.4f} clears the {margin:.4f} "
            f"margin but not {grow_floor:.4f} for {max(0, growth)} added nodes"
        )

    # -- one round ----------------------------------------------------------

    def run_round(
        self,
        patch: str,
        intent: str,
        proposer: str | None = None,
        proposal_mtime: float | None = None,
        batch_supplied: bool = False,
        hypothesis: dict | None = None,
        agent_kind: str | None = None,
        selection_policy: dict | None = None,
        doctor_context: dict | None = None,
        event_sink: dict | None = None,
    ) -> dict:
        self.worktree.assert_clean()
        started = time.monotonic()
        entries = self.store.load()
        index = len(entries)
        run_id = f"{self.run_prefix}-{index:04d}"
        champion = self.champion()
        parent_sha = champion.get("git_commit_sha") if champion else self.worktree.head()
        complexity_before = self.complexity()
        entry: dict = {
            "run_id": run_id,
            "round_index": index,
            "intent": intent,
            "proposer": proposer,
            # a batch-supplied list was written before this round's state
            # existed, so `proposal_mtime` predating the round start is a FACT
            # a reader can check rather than a claim this loop makes about the
            # proposer. The v8 run had 22/24 rounds in this position and the
            # manifest could not show it at the time.
            "proposal_mtime": proposal_mtime,
            "batch_supplied": batch_supplied,
            # who proposed, under what policy, and how the round was conducted.
            # `batch_supplied` is a FACT (the proposal predates this round's
            # state), so a fixed plan can never be described as adaptive.
            "agent_kind": agent_kind,
            "selection_policy": dict(selection_policy or self.selection_policy),
            "hypothesis": (hypothesis or {}).get("hypothesis"),
            "controlled_change": (hypothesis or {}).get("controlled_change"),
            "expected_observation": (hypothesis or {}).get("expected_observation"),
            "context_sha256": (hypothesis or {}).get("context_sha256"),
            "proposal_sha256": (hypothesis or {}).get("raw_reply_sha256"),
            "next_action": None,
            "editable_file": self.editable,
            "complexity": {"metric": COMPLEXITY_METRIC,
                           "nodes_before": complexity_before},
            "started_at": _iso(),
        }

        def finish(decision: str, status: str, reason: str, **extra) -> dict:
            entry.update(
                decision=decision, status=status, reason=reason,
                outcome=outcome_for(decision, status),
                decision_reason=reason,
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
            # zero tolerance for exec/eval/compile/__import__ calls: a dynamic
            # execution would hide logic from this file's own guards and from
            # the complexity metric
            validate_source_ast(
                (self.worktree.path / self.editable).read_text(encoding="utf-8")
            )
        except PatchRejected as exc:
            self.worktree.reset_to(parent_sha)
            return finish("REVERT", "rejected", f"guard: {exc}")

        complexity_after = self.complexity()
        growth = complexity_after - complexity_before
        entry["complexity"] = {
            "metric": COMPLEXITY_METRIC,
            "nodes_before": complexity_before, "nodes_after": complexity_after,
            "growth": growth,
            "min_gain_required": MIN_GAIN_PER_ADDED_NODE * max(0, growth),
        }
        entry["patch"] = patch

        config = {
            "data": {"manifest_path": self.manifest_path, "dataset_path": self.dataset_path},
            "train_provider": {"batch_size": 32},
            "trainer": {"seed": self.seed},
            "task": self.task,
            "time_budget_seconds": self.budget,
            # the runner and its evaluator subprocess bind the integrity
            # snapshot (and the candidate-file hash) to this worktree
            "worktree_path": str(self.worktree.path),
            "editable": self.editable,
        }
        config_path = self.out_dir / f"{run_id}-config.json"
        config_path.write_text(json.dumps(config, indent=2))
        try:
            result = self._run_runner(
                config_path,
                on_progress=(event_sink and event_sink.get("progress")),
                on_spawn=(event_sink and event_sink.get("spawn")),
            )
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
                metrics={"primary_metric": self.primary_metric,
                         "primary_value": result["primary_value"]},
            )

        value = float(result["primary_value"])
        sem = float(result["primary_sem"])
        entry["integrity_snapshot_scope"] = result.get(
            "integrity_snapshot_scope",
            "unrecorded — the runner did not report its snapshot coverage",
        )
        previous_best = champion.get("best_so_far") if champion else None
        keep, reason = self.decide(value, sem, growth, previous_best, self.margin_sem)
        entry["metrics"] = {
            "primary_metric": self.primary_metric, "primary_value": value,
            "primary_std": result["primary_std"], "primary_sem": sem,
            "folds": result["folds"], "fold_seconds": result["fold_seconds"],
        }
        if keep:
            sha = self.worktree.commit(f"{run_id}: {intent}")
        else:
            self.worktree.reset_to(parent_sha)
            sha = parent_sha
        entry["observed_evidence"] = {
            "primary_value": value,
            "previous_best": previous_best,
            "gain": None if previous_best is None else round(value - previous_best, 6),
            "sem": sem,
            "fold_std": result.get("primary_std"),
            "fold_seconds": result.get("fold_seconds"),
            "decision_basis": reason,
        }
        report = self._postflight(entry, doctor_context)
        entry["next_action"] = self._next_action(hypothesis, keep, report)
        return finish(
            "KEEP" if keep else "REVERT", "completed", reason,
            git_commit_sha=sha, git_parent_sha=parent_sha,
            selection={"folds": self.folds, "repeats": self.repeats,
                       "margin_sem": self.margin_sem, "sem": sem,
                       "margin": self.margin_sem * sem,
                       "policy_version": self.selection_policy.get("version")},
        )

    def _run_runner(self, config_path: Path, on_progress=None,
                    on_spawn=None) -> dict:
        from research.spawn import bootstrap_command, child_env

        command = bootstrap_command(
            str(self.worktree.path), "research.runner",
            "--config", str(config_path),
            "--folds", str(self.folds), "--repeats", str(self.repeats),
        )
        # the same interpreter that runs the loop; the child starts isolated
        # (no env injection, no user site, no cwd on sys.path) with the same
        # whitelist every other trusted child gets (H-02)
        env = child_env()
        # streamed, not buffered: the runner prints one progress line per fold so
        # the COORDINATOR (this process) can emit fold-level events while training
        # is still running. The candidate never writes an event; the runner
        # reports and the coordinator emits.
        stdout_lines: list[str] = []
        process = None
        try:
            process = subprocess.Popen(
                command, cwd=self.worktree.path, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env=env, start_new_session=True,
            )
            import os
            import selectors
            import signal

            if on_spawn is not None:
                try:
                    on_spawn(process.pid, os.getpgid(process.pid))
                except OSError:
                    pass
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ)
            deadline = time.monotonic() + self.timeout_s
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, self.timeout_s)
                if not selector.select(timeout=min(1.0, remaining)):
                    if process.poll() is not None:
                        break
                    continue
                line = process.stdout.readline()
                if line == "":
                    if process.poll() is not None:
                        break
                    continue
                stdout_lines.append(line)
                if on_progress is not None:
                    try:
                        record = json.loads(line).get("progress")
                    except json.JSONDecodeError:
                        record = None
                    if isinstance(record, dict):
                        on_progress(record)
            stderr = process.stderr.read() if process.stderr else ""
            process.wait(timeout=5)
        except subprocess.TimeoutExpired as exc:
            if process is not None and process.poll() is None:
                try:
                    import os
                    import signal

                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                except (ProcessLookupError, OSError, NameError):
                    pass
            raise TimeoutError(
                f"runner exceeded the hard timeout {self.timeout_s:.0f}s "
                "(a candidate that hangs is a crash, not a slow winner)"
            ) from exc
        output = "".join(stdout_lines)
        if process.returncode != 0:
            # stderr-first: a failed child's stdout may carry candidate
            # noise; the traceback is on stderr (round-4 review finding 3)
            detail = (stderr or output).strip().splitlines()
            last = detail[-1][:400] if detail else "no output"
            raise RuntimeError(f"runner failed ({process.returncode}): {last}")
        try:
            payload = json.loads(output.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError) as exc:
            raise RuntimeError(f"runner produced no JSON result: {output[-300:]!r}") from exc
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
                batch_supplied=True,
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




    # -- autonomous research (#61) -------------------------------------------

    def research_state(
        self,
        *,
        session=None,
        doctor_report: dict | None = None,
        agent_kind: str | None = None,
    ) -> dict:
        """The state a proposer is shown at the START of one round.

        It is re-read every round, so a proposal is always a response to the
        latest result rather than a step in a plan written in advance. It
        contains only development-side evidence: no holdout, no evaluator
        handle, no raw records, no scaler.
        """
        state = self.state()
        entries = self.store.load()
        last = None
        if entries:
            entry = entries[-1]
            last = {
                "run_id": entry.get("run_id"),
                "hypothesis": entry.get("hypothesis"),
                "patch_summary": entry.get("patch_summary") or None,
                "decision": entry.get("decision"),
                "outcome": entry.get("outcome"),
                "status": entry.get("status"),
                "reason": entry.get("reason"),
                "observed_evidence": entry.get("observed_evidence"),
                "next_action": entry.get("next_action"),
                "doctor": (entry.get("doctor") or {}).get("counts"),
            }
        state.update({
            "agent_kind": agent_kind,
            "frozen_goal": self.frozen_goal,
            "editable_file": self.editable,
            "editable_scope": (
                "only the editable file may change; data, labels, splits, the "
                "evaluator, the metric, the budget and the holdout are frozen"
            ),
            "selection_policy": dict(self.selection_policy),
            "last_round": last,
            "doctor": doctor_report,
            "remaining_budget": (
                None if session is None else {
                    "rounds_remaining": session.rounds_remaining(),
                    "max_rounds": int(session.spec["max_rounds"]),
                }
            ),
            "prohibited": [
                "modify data, labels, splits, evaluator, metric or budget",
                "read or unlock the holdout",
                "report your own score: only the trusted evaluator scores",
                "change the complexity/acceptance policy for this campaign",
            ],
            "proposal_requirements": list(REQUIRED_HYPOTHESIS_FIELDS),
        })
        return state

    def _postflight(self, entry: dict, doctor_context: dict | None) -> dict | None:
        """Diagnose a finished round from the trusted side, before it is stored."""
        if not doctor_context:
            entry["doctor"] = {"status": "NOT_AVAILABLE",
                              "reason": "no doctor context was provided for this round"}
            return None
        from research import doctor as doctor_mod

        view = {
            "decision": entry.get("decision"),
            "status": entry.get("status"),
            "reason": entry.get("reason"),
            "primary_metric": self.primary_metric,
            "primary_value": (entry.get("metrics") or {}).get("primary_value"),
            "primary_sem": (entry.get("metrics") or {}).get("primary_sem"),
            "previous_best": (entry.get("observed_evidence") or {}).get("previous_best"),
            "per_class": entry.get("per_class"),
            "diagnostics": entry.get("diagnostics"),
            "fold_seconds": (entry.get("metrics") or {}).get("fold_seconds"),
            "elapsed_s": entry.get("elapsed_s"),
        }
        report = doctor_mod.postflight(view, doctor_context.get("config") or {},
                                       doctor_context.get("adapter"))
        entry["doctor"] = report.to_dict()
        return report

    @staticmethod
    def _next_action(hypothesis: dict | None, keep: bool,
                     report) -> str:
        """What the loop decides to do next, from THIS round's evidence.

        A blocked postflight finding outranks the proposer's own suggestion: if
        the measurement itself is suspect, running the proposer's next idea would
        just repeat the mistake.
        """
        from research import doctor as doctor_mod

        if report is not None and report.blocked:
            blocked = [f.check for f in report.findings if f.severity == doctor_mod.BLOCKED]
            return (
                f"fix the protocol finding(s) {blocked} before any further "
                "experiment; the metric from this round is not a result"
            )
        fields = hypothesis or {}
        if keep:
            return str(fields.get("next_action_if_keep") or
                       "build on this change: vary one further controlled factor")
        return str(fields.get("next_action_if_revert") or
                   "diagnose why this failed (per-class breakdown, budget use, "
                   "loss trend) before proposing a different change")

    def run_autonomous(
        self,
        session,
        proposer,
        *,
        allow_stub: bool = False,
        doctor_context: dict | None = None,
        max_rounds: int | None = None,
    ) -> dict:
        """Rounds driven by a proposer that reads the LATEST result each time.

        The loop asks for exactly one hypothesis per round, applies it, trains,
        scores with the frozen evaluator, diagnoses, records, and then asks
        again. `None` from the proposer is a deliberate stop. A blocked
        preflight stops before any training. A blocked postflight stops the
        campaign rather than continuing on a suspect measurement, which is the
        behaviour the issue asks for: a weak result triggers diagnosis first.
        """
        from research import doctor as doctor_mod

        agent_kind = getattr(proposer, "kind", type(proposer).__name__)
        is_real = bool(getattr(proposer, "is_real_agent", False))
        if not is_real and not allow_stub:
            raise AgentUnavailable(
                f"proposer {agent_kind!r} is not a real Coding Agent. Autonomous "
                "research requires one (--agent-command); --allow-stub runs the "
                "engineering path and is recorded as autonomous_research=false."
            )

        rounds: list[dict] = []
        started = time.monotonic()
        stop_reason = "budget"
        doctor_report = None
        session.clear_stop()
        limit = int(max_rounds if max_rounds is not None else session.spec["max_rounds"])

        with session.writer() as writer:
            writer.emit("DOCTOR", "started", message="preflight diagnosis",
                        extra={"autonomous_research": is_real, "agent_kind": agent_kind})
            report = doctor_mod.preflight(
                (doctor_context or {}).get("config") or {},
                (doctor_context or {}).get("manifest") or {},
                (doctor_context or {}).get("records") or [],
                (doctor_context or {}).get("adapter"),
                run_id=session.run_id,
            )
            session.write_doctor(report.to_dict())
            writer.emit("DOCTOR", "blocked" if report.blocked else "finished",
                        metrics={"blocked": int(report.blocked)},
                        message=report.summary_line())
            if report.blocked:
                stop_reason = "blocked_protocol"
                counts = report.counts()
                return self._run_summary(rounds, session, is_real, agent_kind,
                                         stop_reason, counts, writer)

            while True:
                if session.stop_requested():
                    stop_reason = "operator_stop"
                    writer.emit("IDLE", "stopped", message="stop requested by operator")
                    break
                if session.rounds_remaining() <= 0 or len(rounds) >= limit:
                    stop_reason = "budget_exhausted"
                    writer.emit("IDLE", "info", metrics={"rounds": len(rounds)},
                                message="round budget exhausted")
                    break
                if not session.within_wall_clock(started):
                    stop_reason = "wall_clock_exhausted"
                    writer.emit("IDLE", "info", message="wall clock budget exhausted")
                    break

                index = session.next_round_index()
                writer.emit("PROPOSING", "started", round_index=index,
                            message=f"asking {agent_kind} for one hypothesis")
                state = self.research_state(
                    session=session,
                    doctor_report=(doctor_report.to_dict() if doctor_report else
                                   session.read_doctor()),
                    agent_kind=agent_kind,
                )
                try:
                    hypothesis = proposer.propose(state)
                except (AgentUnavailable, ProposalRejected, TimeoutError) as exc:
                    writer.emit("DECIDING", "failed", round_index=index,
                                message=f"proposal rejected: {exc}")
                    stop_reason = f"proposal_failed:{type(exc).__name__}"
                    break
                if hypothesis is None:
                    stop_reason = "proposer_stopped"
                    writer.emit("IDLE", "info", round_index=index,
                                message="proposer returned no further hypothesis")
                    break
                # a stop requested while the proposer was thinking must be
                # honoured BEFORE a new training run is launched: stopping is
                # meant to stop work, not to finish the round it interrupted
                if session.stop_requested():
                    stop_reason = "operator_stop"
                    writer.emit("IDLE", "stopped", round_index=index,
                                message="stop requested; proposal discarded before training")
                    break

                writer.emit("PROPOSING", "finished", round_index=index,
                            metrics={"complexity": hypothesis.patch.count("\n")},
                            message=hypothesis.hypothesis[:200])

                # fold-level events come from the coordinator's own callback, fed
                # by the runner's progress lines — never from candidate code
                event_sink = {
                    "progress": (lambda record, _index=index: writer.emit(
                        str(record.get("stage") or "TRAINING"), "started",
                        round_index=_index, repeat=record.get("repeat"),
                        fold=record.get("fold"), folds_total=record.get("folds"),
                        elapsed_seconds=record.get("seconds"),
                        budget_seconds=self.budget,
                        metrics=({"fold_score": record["score"]}
                                 if isinstance(record.get("score"), (int, float)) else None),
                        message=f"fold {record.get('fold')} {record.get('stage','')} "
                                "finished",
                    )),
                    "spawn": (lambda pid, pgid: session.mark_active_child(pid, pgid)),
                }
                entry = self.run_round(
                    hypothesis.patch, hypothesis.intent,
                    proposer=agent_kind if is_real else f"{agent_kind} (not an agent)",
                    hypothesis=hypothesis.to_dict(),
                    agent_kind=agent_kind,
                    batch_supplied=False,
                    doctor_context=doctor_context,
                    event_sink=event_sink,
                )
                session.clear_active_child()
                rounds.append(entry)
                writer.emit("DECIDING", "finished", round_index=index,
                            metrics={k: v for k, v in {
                                "primary_value": (entry.get("metrics") or {}).get("primary_value"),
                                "best_so_far": entry.get("best_so_far"),
                            }.items() if isinstance(v, (int, float))},
                            message=f"{entry.get('outcome')}: {str(entry.get('reason'))[:160]}")
                if entry.get("next_action"):
                    writer.emit("IDLE", "info", round_index=index,
                                message=str(entry["next_action"])[:200])

                doctor_report = None
                blocked_now = (entry.get("doctor") or {}).get("blocked")
                if blocked_now:
                    stop_reason = "blocked_postflight"
                    writer.emit("IDLE", "blocked", round_index=index,
                                message="postflight doctor blocked: stopping the campaign")
                    break

            writer.emit("DONE", "finished", metrics={"rounds": len(rounds)},
                        message=f"campaign finished: {stop_reason}")
            last_counts = None
            if rounds:
                last_counts = (rounds[-1].get("doctor") or {}).get("counts")
            return self._run_summary(
                rounds, session, is_real, agent_kind, stop_reason,
                last_counts, writer,
            )

    def _run_summary(self, rounds: list[dict], session, is_real: bool,
                     agent_kind: str, stop_reason: str, doctor_counts,
                     writer) -> dict:
        champion = self.champion()
        summary = {
            "run_id": session.run_id,
            "autonomous_research": is_real,
            "agent_kind": agent_kind,
            "rounds": len(rounds),
            "decisions": {o: sum(1 for r in rounds if r.get("outcome") == o)
                          for o in OUTCOMES},
            "stop_reason": stop_reason,
            "doctor_counts": doctor_counts,
            "champion_run_id": champion.get("run_id") if champion else None,
            "champion_commit": champion.get("git_commit_sha") if champion else None,
            "best_metric": champion.get("best_so_far") if champion else None,
            "branch": self.worktree.branch,
        }
        if not is_real:
            summary["note"] = (
                "deterministic stub, NOT a Coding Agent: this run exercises the "
                "engineering path and must not be reported as autonomous research"
            )
        return summary



def _iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()
