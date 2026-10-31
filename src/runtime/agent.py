"""Agent adapters and patch validation for the #45 harness.

Two independent modes (#45 acceptance): a deterministic offline stub for
reproducible CI (mode ENGINEERING_SMOKE) and a configurable real Coding
Agent provider interface. Stub-only runs are ENGINEERING_SMOKE, never
AGENT_INTEGRATED; real-provider proposals additionally require verified
OS/container isolation (AGENT_ISOLATION_UNVERIFIED until then).

Static checks enforce the #45 boundary: patches may only touch the
editable scope; source patches containing evaluator/label/split/holdout
symbols are rejected before any training happens.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol

EDITABLE_SECTIONS = ("model", "trainer")
BATCH_SIZE_KEY = "trainer.batch_size"

#: any occurrence in a source patch rejects it before training (#45 hard rule)
FORBIDDEN_SOURCE_SYMBOLS = (
    "EvalManifest",
    "TCMTrustedEvaluator",
    "SyntheticBatchProvider",
    "trusted_score",
    "train_candidate",
    "target_mask",
    "category_labels",
    "label_manifest",
    "holdout",
    "standard_version",
    "MODEL_BUILDERS",
)


@dataclass
class Proposal:
    """One candidate from an agent: intent + bounded config patch.

    `source_patch` is reserved for real Coding Agent providers that edit
    model source; it is inert in this milestone and requires verified
    isolation before it can ever be applied.
    """

    intent: str
    config_patch: dict = field(default_factory=dict)
    source_patch: str | None = None


class AgentProvider(Protocol):
    """Returns a Proposal per round, or None when the plan is exhausted."""

    def propose(self, round_index: int, context: dict) -> Proposal | None:
        ...


class DeterministicStubProvider:
    """Offline deterministic provider: replays a fixed candidate plan.

    Plan position derives from the round_index the harness passes (equal to
    the number of recorded manifest entries), so a resumed run continues the
    plan where it left off instead of replaying already-tried candidates
    (review R-02). Tests inject custom plans for REVERT/rejection cases.
    """

    def __init__(self, plan: list[tuple[str, dict]] | None = None) -> None:
        self.plan = plan if plan is not None else [
            ("baseline run (no change)", {}),
            ("wider model", {"model.hidden_dim": 64, "model.fusion_dim": 64}),
            ("higher LR, one more epoch",
             {"trainer.learning_rate": 0.01, "trainer.epochs": 4}),
        ]

    def propose(self, round_index: int, context: dict) -> Proposal | None:
        if round_index >= len(self.plan):
            return None
        intent, patch = self.plan[round_index]
        return Proposal(intent=intent, config_patch=patch)


@dataclass
class FixedProposalProvider:
    """Yields one externally supplied proposal, then None.

    The research-step CLI driver uses this: the proposal comes from a real
    agent out-of-process, while the harness keeps every guarantee (scope
    validation, frozen seeds, subprocess isolation, append-only manifest).
    """

    proposal: Proposal | None

    def propose(self, round_index: int, context: dict) -> Proposal | None:
        proposal, self.proposal = self.proposal, None
        return proposal


def build_research_prompt(context: dict) -> str:
    """Render the research state for an LLM agent (#45 editable scope)."""
    return (
        "You are the AutoResearch agent optimizing a TCM constitution model. "
        "The research state is given below as JSON. Propose ONE next "
        "candidate by changing ONLY keys listed in editable_scope "
        "(model/trainer hyperparameters). Target: lower dev subject MAE. "
        "Respond with STRICT JSON of shape "
        '{"intent": "<one line>", "config_patch": {"<scope key>": <value>}} '
        '— or {"intent": "STOP", "config_patch": {}} if the budget should '
        "stop early. Keep changes small and targeted; explain the change in "
        "intent.\n\nRESEARCH STATE:\n" + json.dumps(context, ensure_ascii=False)
    )


def _extract_json(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    return text.strip()


class LLMProposalProvider:
    """AgentProvider backed by any LLM completion callable.

    `backend` receives the research prompt and returns raw text; the reply
    must be strict JSON with `intent` and `config_patch`. The provider only
    emits bounded config patches — no source patches — so the untrusted
    surface stays inside the scope-validated JSON the harness already
    checks before any training runs.
    """

    def __init__(self, backend) -> None:
        self.backend = backend

    def propose(self, round_index: int, context: dict) -> Proposal | None:
        raw = self.backend(build_research_prompt(context))
        try:
            data = json.loads(_extract_json(raw))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"agent reply is not valid JSON: {raw.strip()[:200]!r}"
            ) from exc
        if not isinstance(data, dict) or "intent" not in data \
                or "config_patch" not in data:
            raise ValueError(
                "agent proposal must be a JSON object with 'intent' and "
                f"'config_patch', got: {raw.strip()[:200]!r}"
            )
        if str(data["intent"]).strip().upper() == "STOP":
            return None
        patch = data["config_patch"]
        if not isinstance(patch, dict):
            raise ValueError("'config_patch' must be a JSON object")
        return Proposal(intent=str(data["intent"]), config_patch=dict(patch))


class ExternalAgentProvider:
    """Configurable interface for a real Coding Agent provider (#45).

    Deliberately non-functional in this milestone: real provider proposals
    execute arbitrary code, so they require an isolation verification
    (OS/container mounts, evaluator outside the agent-writable sandbox)
    that has not been performed. Until then every run must report
    AGENT_ISOLATION_UNVERIFIED / AGENT_NOT_VERIFIED, per #45.
    """

    def __init__(self, command: str | None = None) -> None:
        self.command = command
        if not command:
            raise ValueError(
                "ExternalAgentProvider needs a provider command; without one "
                "use DeterministicStubProvider (ENGINEERING_SMOKE)"
            )

    def propose(self, round_index: int, context: dict) -> Proposal | None:
        raise NotImplementedError(
            "AGENT_ISOLATION_UNVERIFIED: real Coding Agent proposals require "
            "verified OS/container isolation (#45); refusing to run in the "
            "current milestone-1 harness"
        )


def validate_config_patch(patch: dict, editable_scope: list[str]) -> None:
    """Reject any key outside the frozen editable scope (#45 hard rule)."""
    scope = set(editable_scope)
    illegal = sorted(set(patch) - scope)
    if illegal:
        raise ValueError(
            f"patch touches frozen keys {illegal}; editable scope is {sorted(scope)}"
        )


def validate_source_patch(source: str) -> None:
    """Source patches may never reference evaluator/label/split machinery."""
    hits = [s for s in FORBIDDEN_SOURCE_SYMBOLS if s in source]
    if hits:
        raise ValueError(
            f"source patch references frozen research machinery: {hits}"
        )


def apply_config_patch(base: dict, patch: dict) -> dict:
    """Return a deep-merged candidate config; keys were validated already.

    Deliberately whitelist-shaped: the result contains ONLY the five known
    sections (task/model/trainer/train_provider/eval_provider) plus the
    patch. Callers that need base keys outside this set (data, out_dir,
    data_mode) must re-add them after merging — the harness does.
    """
    cfg = {
        "task": base["task"],
        "model": dict(base.get("model", {})),
        "trainer": dict(base.get("trainer", {})),
        "train_provider": dict(base.get("train_provider", {})),
        "eval_provider": dict(base.get("eval_provider", {})),
    }
    for key, value in patch.items():
        if key == BATCH_SIZE_KEY:
            cfg["train_provider"]["batch_size"] = value
            continue
        section = key.split(".", 1)[0]
        name = key.split(".", 1)[1]
        if section not in EDITABLE_SECTIONS:
            raise ValueError(f"patch section {section!r} is not editable")
        cfg[section][name] = value
    return cfg
