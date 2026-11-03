"""Proposers: who suggests the next experiment, and what they must say.

The engine already knew how to *evaluate* a patch; what it lacked was a loop
that reads the latest result and asks for the next hypothesis. That request
lives here, behind one narrow interface so no agent framework is baked in:

    propose(state) -> Hypothesis | None

`state` is the freshly-read research state (frozen goal, champion, the previous
round's patch and metrics, the Doctor's findings, remaining budget). `None` means
"no further hypothesis", which is how a proposer stops deliberately rather than
by being killed.

Three proposers ship:

* `SubprocessCodingAgent` — a real Coding Agent, invoked as a subprocess with the
  state on stdin. The command is supplied by the operator (`--agent-command`), so
  this works with any tool that can read a prompt and print JSON; nothing here
  assumes Claude, Codex, or any provider.
* `DeterministicStubProposer` — a scripted stand-in for CI. It is explicitly NOT
  an agent, and runs driven by it are recorded as `autonomous_research: false`.
  The issue forbids passing a stub off as autonomous research; the flag is what
  makes that impossible to do by accident.
* `UnavailableAgent` — what `build_proposer` returns when no agent command is
  configured. Calling it raises, because "no agent" must fail closed: a run
  without an agent can still do the engineering smoke path, but it cannot claim
  to have done autonomous research.

Provenance: every accepted proposal records the SHA-256 of the state the agent
was shown and of the agent's raw reply. That is self-reported evidence of the
same grade as the rest of the harness — it shows the proposal matches the state
and the reply, not that the agent read it — and the loop records whether the
round was read-then-proposed or batch-supplied.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Protocol

#: fields every hypothesis must carry. The issue names exactly these, and a
#: proposal missing one is refused rather than defaulted: an experiment whose
#: expected observation was never stated cannot be falsified afterwards.
REQUIRED_HYPOTHESIS_FIELDS = (
    "hypothesis",
    "controlled_change",
    "expected_observation",
    "patch",
)
MAX_RAW_REPLY_CHARS = 200_000


class AgentUnavailable(RuntimeError):
    """No real Coding Agent is configured; autonomous research cannot run."""


class ProposalRejected(ValueError):
    """The agent's reply is not a usable, falsifiable proposal."""


@dataclass(frozen=True)
class Hypothesis:
    """One controlled experiment, in the form the issue requires."""

    hypothesis: str
    controlled_change: str
    expected_observation: str
    patch: str
    intent: str = ""
    next_action_if_keep: str = ""
    next_action_if_revert: str = ""
    raw_reply_sha256: str = ""
    context_sha256: str = ""

    def __post_init__(self) -> None:
        for name in REQUIRED_HYPOTHESIS_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ProposalRejected(f"hypothesis field {name!r} must be a non-empty string")
        if not _looks_like_unified_diff(self.patch):
            raise ProposalRejected(
                "patch must be a unified diff (a '--- a/<file>' and '+++ b/<file>' "
                "header plus at least one @@ hunk); free-form text cannot be applied "
                "and would be recorded as a protocol error"
            )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict, *, raw_reply_sha256: str = "",
                  context_sha256: str = "") -> Hypothesis:
        missing = [k for k in REQUIRED_HYPOTHESIS_FIELDS if not payload.get(k)]
        if missing:
            raise ProposalRejected(
                f"agent reply is missing required field(s) {missing}; a proposal "
                "must state what it changes, what it holds fixed and what it "
                "expects to observe"
            )
        return cls(
            hypothesis=str(payload["hypothesis"]),
            controlled_change=str(payload["controlled_change"]),
            expected_observation=str(payload["expected_observation"]),
            patch=str(payload["patch"]),
            intent=str(payload.get("intent") or payload["hypothesis"])[:200],
            next_action_if_keep=str(payload.get("next_action_if_keep") or ""),
            next_action_if_revert=str(payload.get("next_action_if_revert") or ""),
            raw_reply_sha256=raw_reply_sha256,
            context_sha256=context_sha256,
        )


class Proposer(Protocol):
    """Returns ONE next hypothesis per call, or None to stop deliberately."""

    kind: str
    is_real_agent: bool

    def propose(self, state: dict) -> Hypothesis | None:
        ...


def _looks_like_unified_diff(patch: str) -> bool:
    has_old = "\n--- a/" in "\n" + patch
    has_new = "\n+++ b/" in "\n" + patch
    has_hunk = "\n@@ " in "\n" + patch
    return bool(has_old and has_new and has_hunk)


def digest(payload) -> str:
    """SHA-256 of a JSON-serialisable payload or raw text, for provenance."""
    text = payload if isinstance(payload, str) else json.dumps(
        payload, ensure_ascii=False, sort_keys=True
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


PROPOSAL_PROMPT = """You are the research proposer for one AutoResearch round.

You are given the CURRENT research state as JSON. Propose exactly ONE next
experiment as a controlled, falsifiable hypothesis. You must react to the result
of the previous round that appears in the state; a proposal that repeats a
previous attempt without new reasoning is invalid.

Hard rules (violating any of them makes the proposal invalid):
- The patch may only modify the editable file named in `editable_file`.
- Do not touch data, labels, splits, the evaluator, the metric, the training
  budget, or the holdout. They are frozen and outside your scope.
- The training budget is fixed; you cannot buy score by training longer.
- The score comes from the trusted evaluator; you cannot report a score.

Reply with ONE JSON object and nothing else:

{
  "hypothesis": "a falsifiable claim: 'X raises metric M because Y'",
  "controlled_change": "what you change, and what you hold fixed",
  "expected_observation": "what the trusted metric should show if the claim holds",
  "patch": "a unified diff (--- a/<editable_file> / +++ b/<editable_file> / @@ ...)",
  "intent": "one line summary",
  "next_action_if_keep": "what to try next if this is kept",
  "next_action_if_revert": "what to diagnose or try next if this is rolled back"
}

RESEARCH STATE:
"""


def build_proposal_prompt(state: dict) -> str:
    return PROPOSAL_PROMPT + json.dumps(state, ensure_ascii=False, indent=2)


# -- real coding agents ---------------------------------------------------------


@dataclass
class SubprocessCodingAgent:
    """A real Coding Agent, invoked as a subprocess that prints JSON.

    The command is the operator's choice (`--agent-command "claude -p"`, a shell
    wrapper, anything). The state is passed on stdin as the prompt; the reply's
    last non-empty line must be the JSON object. `agent_kind` is recorded so a
    run can say which command produced its proposals, and the raw reply is
    digested for provenance.
    """

    command: str
    timeout_s: float = 600.0
    #: extra environment for the child; the parent's env is NOT inherited, so a
    #: research run cannot leak host secrets into an agent by accident
    env_allowlist: tuple[str, ...] = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR")
    kind: str = "coding-agent"
    is_real_agent: bool = True
    last_reply_sha256: str = field(default="", init=False)

    def __post_init__(self) -> None:
        if not str(self.command).strip():
            raise AgentUnavailable(
                "no --agent-command configured; autonomous research requires a real "
                "Coding Agent. Use --allow-stub only for the engineering smoke path, "
                "which is recorded as NOT autonomous research."
            )

    def propose(self, state: dict) -> Hypothesis | None:
        prompt = build_proposal_prompt(state)
        context_sha = digest(state)
        env = {key: os.environ[key] for key in self.env_allowlist if key in os.environ}
        try:
            proc = subprocess.run(
                ["/bin/sh", "-c", self.command],
                input=prompt, text=True, capture_output=True,
                timeout=self.timeout_s, env=env,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"agent command exceeded {self.timeout_s:.0f}s; a hung agent is a "
                "failed round, not a slow one"
            ) from exc
        raw = (proc.stdout or "")[-MAX_RAW_REPLY_CHARS:]
        self.last_reply_sha256 = digest(raw)
        if proc.returncode != 0:
            raise ProposalRejected(
                f"agent command failed (rc={proc.returncode}): "
                f"{(proc.stderr or '').strip()[-300:]}"
            )
        payload = _extract_json_object(raw)
        if payload is None:
            if "STOP" in raw.upper():
                return None
            raise ProposalRejected(
                "agent reply contained no JSON object; expected the schema in the "
                f"prompt. Tail of reply: {raw.strip()[-200:]!r}"
            )
        if str(payload.get("intent", "")).strip().upper() == "STOP":
            return None
        self._last_payload = payload
        return Hypothesis.from_dict(
            payload, raw_reply_sha256=self.last_reply_sha256, context_sha256=context_sha
        )


def _extract_json_object(text: str) -> dict | None:
    """The last JSON object in the reply, tolerating fenced blocks.

    Agents wrap JSON in prose and code fences; refusing that would waste rounds
    on formatting. The object still has to parse and carry the required fields,
    which is where the real validation happens.
    """
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.lower().startswith("json"):
            candidate = candidate[4:]
    for blob in (candidate, *reversed(candidate.splitlines())):
        try:
            parsed = json.loads(blob.strip())
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        try:
            parsed = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


# -- stub (explicitly not an agent) ---------------------------------------------


@dataclass
class DeterministicStubProposer:
    """A scripted proposer for CI: NOT an agent, and recorded as such.

    Each step is a callable `(state) -> Hypothesis | None` so a test can prove the
    proposer reacts to what the previous round produced; a fixed list of patches
    that ignores the state would be exactly the batching the issue rules out.
    """

    steps: list
    kind: str = "deterministic-stub"
    is_real_agent: bool = False
    calls: list = field(default_factory=list, init=False)

    def propose(self, state: dict) -> Hypothesis | None:
        self.calls.append(state)
        if not self.steps:
            return None
        step = self.steps.pop(0)
        return step(state) if callable(step) else Hypothesis.from_dict(step)


def stub_step(patch: str, hypothesis: str, change: str, expect: str,
              next_if_keep: str = "", next_if_revert: str = "", intent: str = ""):
    """A one-shot step that ignores the state but is still a valid hypothesis."""

    def run(_state: dict) -> Hypothesis:
        return Hypothesis(
            hypothesis=hypothesis, controlled_change=change,
            expected_observation=expect, patch=patch,
            intent=intent or hypothesis,
            next_action_if_keep=next_if_keep, next_action_if_revert=next_if_revert,
        )
    return run


def state_aware_step(patch_for, hypothesis, change, expect):
    """A step whose PATCH is derived from the freshly-read state.

    This is the shape a real proposer has, and the shape the tests use to prove
    the loop is not just replaying a plan: the patch depends on what the state
    says right now.
    """

    def run(state: dict) -> Hypothesis:
        return Hypothesis(
            hypothesis=hypothesis, controlled_change=change,
            expected_observation=expect, patch=patch_for(state),
            intent=hypothesis,
        )
    return run


@dataclass
class UnavailableAgent:
    """Returned when no agent was configured; every call fails closed."""

    reason: str = "no real Coding Agent configured"
    kind: str = "unavailable"
    is_real_agent: bool = False

    def propose(self, state: dict) -> Hypothesis | None:
        raise AgentUnavailable(
            f"{self.reason}. Autonomous research needs a real Coding Agent: pass "
            "--agent-command, or run with --allow-stub to exercise the engineering "
            "path (recorded as autonomous_research=false)."
        )


def build_proposer(agent_command: str | None, stub_steps: list | None = None,
                   timeout_s: float = 600.0) -> Proposer:
    """The proposer for a run: real agent if configured, stub only if asked."""
    if agent_command:
        return SubprocessCodingAgent(command=agent_command, timeout_s=timeout_s)
    if stub_steps is not None:
        return DeterministicStubProposer(steps=stub_steps)
    return UnavailableAgent()