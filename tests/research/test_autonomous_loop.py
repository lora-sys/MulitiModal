"""The autonomous loop: one fresh hypothesis per round, driven by the last result.

Scope of these tests, stated plainly: they drive the REAL loop (guard, git
advance/reset, manifest, session, Doctor, proposer protocol) but stub the
training subprocess (`_run_runner`), so they run in milliseconds and assert loop
*behaviour*. Real training is exercised end to end by
`tests/research/test_toy_end_to_end.py`; these two are complementary and neither
replaces the other.

What must be true, from the issue:

* one hypothesis per round, re-read from the latest state — a fixed plan replayed
  from a list must be impossible to mistake for adaptive research;
* a stub is recorded as `autonomous_research: false`, and a run with no real
  agent fails closed unless the stub is explicitly allowed;
* stop, budget exhaustion and a blocked protocol each stop the campaign with a
  named reason and without replaying finished rounds.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from research.agent import (
    AgentUnavailable,
    DeterministicStubProposer,
    Hypothesis,
    ProposalRejected,
    SubprocessCodingAgent,
    UnavailableAgent,
    build_proposer,
    state_aware_step,
    stub_step,
)
from research.loop import SourceResearchLoop
from research.session import ResearchSession
from tests.research.test_doctor import ToyAdapter, build

pytestmark = pytest.mark.synthetic

EDITABLE = "src/research/candidate_toy.py"


def _git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "wt"
    repo.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    target = repo / EDITABLE
    target.parent.mkdir(parents=True, exist_ok=True)
    # The fixture must satisfy the guard's candidate contract (#59 refuses a file
    # that does not define build_model/train), and keep a stable 3-line shape so a
    # replacement hunk can carry context on BOTH sides. A hunk with no trailing
    # context asserts "the file ends here", which is why an append-only patch
    # applies to a one-line file exactly once and then fails — the same trap a real
    # proposer has to avoid.
    target.write_text(
        '"""Test candidate: satisfies the guard contract."""\n'
        "WIDTH = 32\n"
        "\n"
        "def build_model(fold):\n"
        "    raise NotImplementedError\n"
        "\n"
        "def train(model, fold, budget_seconds=1.0, seed=0):\n"
        "    return None\n"
    )
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=repo, check=True)
    return repo


def _patch(added_line: str) -> str:
    """Append a line before the stable `# end` marker (one-shot fixtures only)."""
    return (
        f"diff --git a/{EDITABLE} b/{EDITABLE}\n"
        f"--- a/{EDITABLE}\n"
        f"+++ b/{EDITABLE}\n"
        "@@ -1,3 +1,4 @@\n"
        " # config\n"
        " WIDTH = 32\n"
        f"+{added_line}\n"
        " # end\n"
    )


def _width_patch(repo: Path, value: int) -> str:
    """Replace the WIDTH line, reading its CURRENT value from the tree.

    Context lines on both sides keep the hunk position-independent, so the same
    shape applies on every round — which is what a proposer that reacts to the
    last result actually needs.
    """
    import re

    current = (repo / EDITABLE).read_text(encoding="utf-8")
    match = re.search(r"^WIDTH = (\d+)$", current, flags=re.MULTILINE)
    assert match, "fixture WIDTH line missing"
    return (
        f"diff --git a/{EDITABLE} b/{EDITABLE}\n"
        f"--- a/{EDITABLE}\n"
        f"+++ b/{EDITABLE}\n"
        "@@ -1,3 +1,3 @@\n"
        ' \"\"\"Test candidate: satisfies the guard contract.\"\"\"\n'
        f"-WIDTH = {match.group(1)}\n"
        f"+WIDTH = {value}\n"
        " \n"
    )


class FakeLoop(SourceResearchLoop):
    """The loop with the training subprocess replaced by a canned score."""

    def __init__(self, *args, scores: list[float], **kwargs):
        super().__init__(*args, **kwargs)
        self._scores = list(scores)
        self.runner_calls: list[Path] = []

    def _run_runner(self, config_path: Path, on_progress=None, on_spawn=None) -> dict:
        """Mirror the REAL runner's result contract.

        #59 made run_round consume `uncertainty_semantics` (and record the round
        as a crash when the key is missing), so a stub that returns only the old
        fields silently turns every round into REVERT. The constant is imported
        rather than copied, so a rename cannot drift the stub away from the engine.
        """
        from research.runner import CV_UNCERTAINTY_SEMANTICS

        self.runner_calls.append(config_path)
        value = self._scores.pop(0) if self._scores else 0.5
        # the real runner prints one progress line per fold; this stub does the
        # same so the tests cover the coordinator's fold-level event plumbing
        if on_progress is not None:
            for fold in range(self.folds):
                on_progress({"stage": "TRAINING", "repeat": 0, "fold": fold,
                             "folds": self.folds})
                on_progress({"stage": "EVALUATING", "repeat": 0, "fold": fold,
                             "folds": self.folds, "score": value, "seconds": 1.0})
        return {
            "primary_value": value, "primary_std": 0.02, "primary_sem": 0.01,
            "uncertainty_semantics": CV_UNCERTAINTY_SEMANTICS,
            "folds": [value], "n_folds": 1, "overran": False,
            "fold_seconds": [1.0], "worker_compute_seconds": [1.0],
            "budget_seconds": 1.0,
        }


def _setup(tmp_path: Path, *, scores, max_rounds=5, with_doctor=True):
    repo = _git_repo(tmp_path)
    config, manifest, records, _ = build()
    config["time_budget_seconds"] = 1.0
    run_dir = tmp_path / "run"
    session = ResearchSession.create(
        run_dir, task="toy", editable=EDITABLE, worktree=str(repo),
        branch="autoresearch/toy", folds=2, repeats=1,
        max_rounds=max_rounds, budget_seconds=1.0,
        primary_metric="toy/accuracy",
    )
    loop = FakeLoop(
        worktree=__import__("research.loop", fromlist=["GitWorktree"]).GitWorktree(
            repo, "autoresearch/toy"),
        out_dir=run_dir, manifest_path=str(tmp_path / "m.json"),
        dataset_path=str(tmp_path / "d.json"), task="toy",
        editable=EDITABLE, budget_s=1.0, scores=scores,
    )
    doctor_context = (
        {"config": config, "manifest": manifest, "records": records,
         "adapter": ToyAdapter()}
        if with_doctor else None
    )
    return repo, session, loop, doctor_context, config, manifest, records


def test_autonomous_rounds_read_the_latest_result_and_are_not_batched(tmp_path):
    repo, session, loop, ctx, *_ = _setup(tmp_path, scores=[0.30, 0.40, 0.35])

    seen_states: list[dict] = []

    def react(state: dict) -> Hypothesis:
        """The patch depends on what the PREVIOUS round decided."""
        seen_states.append(state)
        last = state.get("last_round")
        if last is None:
            value, label = 64, "start"
        elif last["decision"] == "KEEP":
            value, label = 96, "after-KEEP"
        else:
            value, label = 48, "after-REVERT"
        return Hypothesis(
            hypothesis=f"width -> {value} ({label})",
            controlled_change="only the trunk width; data, budget and loss are fixed",
            expected_observation="macro-F1 rises if the task is capacity-limited",
            patch=_width_patch(repo, value),
            intent=f"width -> {value}",
            next_action_if_keep="try an even wider trunk",
            next_action_if_revert="diagnose before widening again",
        )

    proposer = DeterministicStubProposer(steps=[react, react, react])
    summary = loop.run_autonomous(session, proposer, allow_stub=True,
                                  doctor_context=ctx, max_rounds=3)

    assert summary["rounds"] == 3
    assert summary["autonomous_research"] is False
    assert summary["agent_kind"] == "deterministic-stub"
    assert "must not be reported as autonomous research" in summary["note"]

    entries = session.entries()
    assert [e["run_id"] for e in entries] == ["toy-0000", "toy-0001", "toy-0002"]
    # the proposer was asked three times, and each time it saw the previous round
    assert len(seen_states) == 3
    assert seen_states[0]["last_round"] is None
    assert seen_states[1]["last_round"]["run_id"] == "toy-0000"
    assert seen_states[2]["last_round"]["run_id"] == "toy-0001"
    # ... and the patch really did depend on that state: round 1's recorded
    # hypothesis ("marker after-KEEP") exists only because round 0 was kept, and
    # round 2's patch carries the marker derived from round 1's decision
    assert seen_states[1]["last_round"]["hypothesis"] == "width -> 64 (start)"
    assert seen_states[2]["last_round"]["hypothesis"] == "width -> 96 (after-KEEP)"
    assert "WIDTH = 96" in entries[2]["patch"], "round 2's patch follows round 1's KEEP"
    # no round claims to be batch-supplied, and each carries the full record
    for entry in entries:
        assert entry["batch_supplied"] is False
        assert entry["proposal_mtime"] is None
        for field in ("hypothesis", "controlled_change", "expected_observation",
                      "observed_evidence", "decision_reason", "next_action",
                      "outcome", "agent_kind", "selection_policy"):
            assert entry[field] not in (None, ""), f"{field} missing from {entry['run_id']}"
    # the winner's commit advanced the branch; the losers were reset
    kinds = [e["outcome"] for e in entries]
    assert kinds[0] == "KEEP" and kinds[1] == "KEEP"
    assert entries[1]["git_commit_sha"] != entries[0]["git_commit_sha"]
    assert entries[2]["git_commit_sha"] == entries[1]["git_commit_sha"], "REVERT must not move HEAD"
    assert "WIDTH = 96" in (repo / EDITABLE).read_text(), "last KEEP is on the branch"


def test_stub_is_refused_unless_explicitly_allowed(tmp_path):
    _repo, session, loop, ctx, *_ = _setup(tmp_path, scores=[0.3])
    proposer = DeterministicStubProposer(steps=[stub_step(
        _patch("X = 1"), "h", "c", "e")])
    with pytest.raises(AgentUnavailable, match="not a real Coding Agent"):
        loop.run_autonomous(session, proposer, doctor_context=ctx)
    assert session.entries() == []


def test_unavailable_agent_fails_closed(tmp_path):
    proposer = build_proposer(agent_command=None)
    assert isinstance(proposer, UnavailableAgent) and proposer.is_real_agent is False
    with pytest.raises(AgentUnavailable, match="no real Coding Agent"):
        proposer.propose({})


def test_operator_stop_ends_the_campaign_without_extra_rounds(tmp_path):
    repo, session, loop, ctx, *_ = _setup(tmp_path, scores=[0.3, 0.4, 0.5])

    def stop_then_propose(state: dict) -> Hypothesis:
        session.request_stop("operator request")
        return Hypothesis(hypothesis="h", controlled_change="c",
                          expected_observation="e", patch=_patch("A = 1"))

    # the fixture is the contract-compliant candidate now, so the first round must
    # use the width patch (the old `_patch` context no longer exists in that file;
    # a rejected round would train nothing and make this assertion vacuous)
    proposer = DeterministicStubProposer(steps=[
        state_aware_step(lambda state: _width_patch(repo, 64), "h", "c", "e"),
        stop_then_propose,
    ])
    summary = loop.run_autonomous(session, proposer, allow_stub=True,
                                  doctor_context=ctx, max_rounds=5)
    # round 0 ran; the stop requested while round 1's proposal was being made is
    # honoured BEFORE round 1 trains, and round 0 was not replayed
    assert summary["stop_reason"] == "operator_stop"
    assert summary["rounds"] == 1
    assert len(session.entries()) == 1
    assert len(loop.runner_calls) == 1, "no training after the stop was requested"


def test_round_budget_exhaustion_is_recorded(tmp_path):
    _repo, session, loop, ctx, *_ = _setup(tmp_path, scores=[0.3, 0.31, 0.32], max_rounds=2)
    proposer = DeterministicStubProposer(steps=[stub_step(_patch(f"S = {i}"), "h", "c", "e")
                                                for i in range(5)])
    summary = loop.run_autonomous(session, proposer, allow_stub=True,
                                  doctor_context=ctx, max_rounds=2)
    assert summary["stop_reason"] == "budget_exhausted"
    assert summary["rounds"] == 2
    assert session.rounds_remaining() == 0


def test_blocked_preflight_prevents_any_round(tmp_path):
    _repo, session, loop, ctx, config, manifest, records = _setup(tmp_path, scores=[0.3])
    # inject the #1 leak: a dev record reusing a train user's identity
    dev_row = manifest["subject_row_by_id"][manifest["split"]["dev"][0]]
    records[dev_row].user_id = records[0].user_id
    proposer = DeterministicStubProposer(steps=[stub_step(_patch("A = 1"), "h", "c", "e")])
    summary = loop.run_autonomous(session, proposer, allow_stub=True,
                                  doctor_context=ctx, max_rounds=3)
    assert summary["stop_reason"] == "blocked_protocol"
    assert summary["rounds"] == 0
    assert session.entries() == []
    assert session.read_doctor()["blocked"] is True
    assert loop.runner_calls == [], "a blocked protocol must never reach training"


def test_rejected_proposal_is_invalid_protocol_and_stops(tmp_path):
    _repo, session, loop, ctx, *_ = _setup(tmp_path, scores=[0.3])
    foreign = (
        "diff --git a/src/research/frozen.py b/src/research/frozen.py\n"
        "--- a/src/research/frozen.py\n+++ b/src/research/frozen.py\n"
        "@@ -1 +1,2 @@\n TIME_BUDGET_SECONDS = 20.0\n+LEAK = True\n"
    )
    proposer = DeterministicStubProposer(steps=[stub_step(foreign, "h", "c", "e")])
    summary = loop.run_autonomous(session, proposer, allow_stub=True,
                                  doctor_context=ctx, max_rounds=2)
    entries = session.entries()
    assert entries, "a rejected round is still a recorded round"
    for entry in entries:
        # a guard refusal is an ENGINEERING outcome, recorded as such, and it is
        # never dressed up as evidence that the model is worse
        assert entry["outcome"] == "INVALID_PROTOCOL"
        assert entry["decision"] == "REVERT", "the strict decision stays KEEP/REVERT"
        assert "outside the editable file" in entry["reason"]
    assert loop.runner_calls == [], "a refused patch must never reach training"
    # a proposer that has nothing left is a deliberate stop, not a crash
    assert summary["stop_reason"] in ("budget_exhausted", "proposer_stopped")


def test_proposal_must_carry_all_four_fields():
    with pytest.raises(ProposalRejected, match="missing required field"):
        Hypothesis.from_dict({"hypothesis": "h", "patch": _patch("A = 1")})
    with pytest.raises(ProposalRejected, match="unified diff"):
        Hypothesis.from_dict({
            "hypothesis": "h", "controlled_change": "c",
            "expected_observation": "e", "patch": "just some prose",
        })


def test_subprocess_agent_parses_json_and_digests_the_reply(tmp_path):
    script = (
        "import json,sys;"
        "sys.stdin.read();"
        "print(json.dumps({'hypothesis':'h','controlled_change':'c',"
        "'expected_observation':'e','patch':'''" + _patch("Z = 1").replace("'", "\\'") + "'''}))"
    )
    agent = SubprocessCodingAgent(command=f"{sys.executable} -c \"{script}\"")
    hypothesis = agent.propose({"round_index": 0})
    assert hypothesis is not None and "Z = 1" in hypothesis.patch
    assert len(hypothesis.context_sha256) == 64
    assert hypothesis.raw_reply_sha256 == agent.last_reply_sha256


def test_subprocess_agent_stop_and_failure_are_distinguishable(tmp_path):
    stop = SubprocessCodingAgent(command=f"{sys.executable} -c \"print('STOP')\"")
    assert stop.propose({"round_index": 0}) is None

    broken = SubprocessCodingAgent(command=f"{sys.executable} -c \"import sys;sys.exit(3)\"")
    with pytest.raises(ProposalRejected, match="agent command failed"):
        broken.propose({"round_index": 0})

    silent = SubprocessCodingAgent(command=f"{sys.executable} -c \"print('hello')\"")
    with pytest.raises(ProposalRejected, match="no JSON object"):
        silent.propose({"round_index": 0})


def test_history_view_exposes_the_research_trail(tmp_path):
    repo, session, loop, ctx, *_ = _setup(tmp_path, scores=[0.30, 0.40])
    # The patch is built at PROPOSE time, not at setup time: after round 0 is
    # kept the file says WIDTH = 64, so a patch prepared in advance (expecting
    # 32) would not apply. That is the difference between a state-aware proposer
    # and a replayed plan, and it is why this uses state_aware_step.
    proposer = DeterministicStubProposer(steps=[
        state_aware_step(lambda state: _width_patch(repo, 64),
                         "first claim", "widen to 64", "A raises the metric"),
        state_aware_step(lambda state: _width_patch(repo, 96),
                         "second claim", "widen to 96", "B raises the metric"),
    ])
    loop.run_autonomous(session, proposer, allow_stub=True, doctor_context=ctx,
                        max_rounds=2)
    history = session.history()["rounds"]
    assert [row["hypothesis"] for row in history] == ["first claim", "second claim"]
    # the dashboard and the report both read `outcome`; it must be in the view
    assert history[0]["outcome"] == "KEEP"
    assert history[1]["outcome"] in ("KEEP", "REVERT")
    assert all("outcome" in row for row in history)
    payload = json.loads(json.dumps(session.status()))
    assert payload["rounds_completed"] == 2
    assert payload["best_metric"] == 0.40

def test_run_round_refuses_to_train_when_the_preflight_blocks(tmp_path):
    """The `step` path: a single externally-driven round must pass the gate too.

    The batch loop preflights before its rounds, but `step` calls run_round
    directly, so the diagnosis has to live in run_round - the one place every
    round passes - or an agent-driven round trains, and can be KEPT, on inputs
    the doctor would have refused.
    """
    repo, session, loop, ctx, *_ = _setup(tmp_path, scores=[0.30])
    # inject the #1 leak: a dev record reusing a train user's identity
    dev_row = ctx["manifest"]["subject_row_by_id"][ctx["manifest"]["split"]["dev"][0]]
    ctx["records"][dev_row].user_id = ctx["records"][0].user_id
    head_before = loop.worktree.head()

    entry = loop.run_round(
        _width_patch(repo, 64), "h",
        proposer="pi:test",
        hypothesis={"hypothesis": "h", "controlled_change": "c",
                    "expected_observation": "e"},
        agent_kind="external-coding-agent", doctor_context=ctx,
    )

    assert entry["status"] == "rejected"
    assert entry["outcome"] == "INVALID_PROTOCOL"
    assert entry["preflight_blocked"] is True
    assert entry["doctor"]["blocked"] is True
    assert loop.runner_calls == [], "a blocked preflight must never reach training"
    assert len(session.entries()) == 1, "the refusal is still a recorded round"
    assert loop.worktree.head() == head_before, "a refused round must not commit"


def test_a_blocked_postflight_never_keeps_the_champion(tmp_path, monkeypatch):
    """A blocked postflight must not leave a KEEP behind.

    The baseline round always KEEPs (there is no champion yet) and the postflight
    used to run AFTER that commit, so the campaign could stop on the blocking
    finding only once the champion had already advanced to a model measured
    under it.
    """
    repo, session, loop, ctx, *_ = _setup(tmp_path, scores=[0.30])
    from research import doctor as doctor_mod

    def blocked(_view, _config, _adapter):
        return doctor_mod.DoctorReport(
            "postflight", "toy",
            [doctor_mod.Finding("metric_integrity", doctor_mod.BLOCKED, "forced")],
        )

    monkeypatch.setattr(doctor_mod, "postflight", blocked)
    head_before = loop.worktree.head()

    entry = loop.run_round(
        _width_patch(repo, 64), "h",
        proposer="pi:test", agent_kind="external-coding-agent", doctor_context=ctx,
    )

    assert loop.runner_calls, "training happens; the postflight is after it"
    assert entry["doctor"]["blocked"] is True
    assert entry["doctor_blocked_keep"] is True
    assert entry["decision"] == "REVERT" and entry["outcome"] != "KEEP"
    assert "postflight doctor blocked" in entry["reason"]
    assert loop.worktree.head() == head_before, "the champion must not have advanced"
    assert loop.champion() is None, "a blocked measurement may not create a champion"
