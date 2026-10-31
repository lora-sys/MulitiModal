"""AutoResearch harness tests: KEEP/REVERT, rejections, budget, resume (#45)."""

import json

import pytest

from runtime.agent import (
    DeterministicStubProvider,
    ExternalAgentProvider,
    FixedProposalProvider,
    LLMProposalProvider,
    Proposal,
    apply_config_patch,
    build_research_prompt,
    validate_config_patch,
    validate_source_patch,
)
from runtime.harness import (
    TCM_ONLY_MESSAGE,
    AutoResearchHarness,
    default_tcm_runspec,
)
from runtime.manifest import RunManifestStore

pytestmark = pytest.mark.synthetic


def harness(tmp_path, provider=None, rounds=3, **runspec_overrides):
    runspec = default_tcm_runspec(rounds=rounds, timeout_s=120.0)
    runspec.update(runspec_overrides)
    return AutoResearchHarness(runspec, provider=provider, out_dir=tmp_path / "ar")


def test_full_stub_run_keeps_baseline_and_reverts_equal_or_worse(tmp_path):
    provider = DeterministicStubProvider(plan=[
        ("baseline", {}),
        # same patch + same frozen seed => identical score => not an improvement
        ("identical repeat", {}),
        ("wider model", {"model.hidden_dim": 64, "model.fusion_dim": 64}),
    ])
    summary = harness(tmp_path, provider).run()
    assert summary["mode"] == "ENGINEERING_SMOKE"
    assert summary["agent_status"] == "AGENT_NOT_VERIFIED"
    assert summary["isolation"] == "AGENT_ISOLATION_UNVERIFIED"
    assert summary["smoke_only"] is True
    assert summary["rounds_completed"] == 3
    assert summary["decisions"][0]["decision"] == "KEEP"  # baseline
    # equal-or-worse candidates must never replace the champion (#45)
    assert summary["decisions"][1]["decision"] == "REVERT"
    assert summary["decisions"][1]["status"] == "completed"
    assert summary["decisions"][1]["reason"].startswith("0.")
    champion = RunManifestStore(tmp_path / "ar" / "manifest.jsonl").champion()
    assert champion["run_id"] in ("tcm-0000", "tcm-0002")


def test_out_of_scope_patch_is_rejected_without_training(tmp_path):
    provider = DeterministicStubProvider(plan=[
        ("baseline", {}),
        ("tamper evaluator", {"frozen.evaluator": "my.own.Scorer"}),
    ])
    summary = harness(tmp_path, provider).run()
    rejected = summary["decisions"][1]
    assert rejected["decision"] == "REVERT"
    assert rejected["status"] == "rejected"
    assert "frozen keys" in rejected["reason"]

    entries = RunManifestStore(tmp_path / "ar" / "manifest.jsonl").load()
    assert "artifact_dir" not in entries[1]  # no training happened


def test_failing_training_is_recorded_as_crash_and_reverted(tmp_path):
    provider = DeterministicStubProvider(plan=[
        ("baseline", {}),
        ("impossible width", {"model.hidden_dim": -1}),  # nn.Linear raises
    ])
    summary = harness(tmp_path, provider).run()
    failed = summary["decisions"][1]
    assert failed["decision"] == "REVERT"
    assert failed["status"] == "crash"
    champion = RunManifestStore(tmp_path / "ar" / "manifest.jsonl").champion()
    assert champion["run_id"] == "tcm-0000"  # baseline stays champion


def test_plan_exhaustion_stops_cleanly(tmp_path):
    provider = DeterministicStubProvider(plan=[("baseline", {})])
    summary = harness(tmp_path, provider, rounds=3).run()
    assert summary["plan_exhausted"] is True
    assert summary["rounds_completed"] == 1


def test_resume_does_not_duplicate_finished_rounds(tmp_path):
    first = harness(tmp_path, rounds=1).run()
    assert first["rounds_completed"] == 1
    champion_after_first = first["champion_artifact"]

    second = harness(tmp_path, rounds=3).run()
    assert second["rounds_completed"] == 3
    entries = RunManifestStore(tmp_path / "ar" / "manifest.jsonl").load()
    assert len(entries) == 3
    assert [e["round_index"] for e in entries] == [0, 1, 2]
    # champion continuity: resume kept the earlier champion as the comparison
    # baseline, and the final champion is one of the persisted artifacts
    assert champion_after_first is not None
    assert second["champion_artifact"] is not None


def test_resume_rejects_changed_frozen_hashes(tmp_path):
    assert harness(tmp_path, rounds=1).run()["rounds_completed"] == 1
    tampered = harness(tmp_path, rounds=3,
                       **{"frozen": dict(default_tcm_runspec()["frozen"],
                                         standard_version="CHANGED-V9")})
    with pytest.raises(RuntimeError, match="frozen evaluation set changed"):
        tampered.run()


def test_physio_profile_is_refused(tmp_path):
    runspec = default_tcm_runspec()
    runspec["task_id"] = "physio_bpsi_efri"
    with pytest.raises(ValueError, match="TCM_ONLY"):
        AutoResearchHarness(runspec, out_dir=tmp_path / "physio")
    assert TCM_ONLY_MESSAGE


def test_external_agent_provider_refuses_unverified_isolation(tmp_path):
    provider = ExternalAgentProvider(command="fake-agent --round")
    with pytest.raises(NotImplementedError, match="AGENT_ISOLATION_UNVERIFIED"):
        provider.propose(0, {})
    h = harness(tmp_path, provider)
    assert h.mode == "AGENT_INTEGRATED"
    assert h.agent_status == "AGENT_ISOLATION_UNVERIFIED"


def test_run_context_carries_metrics_and_patches(tmp_path):
    seen: list[dict] = []

    class Recording:
        def __init__(self, inner):
            self.inner = inner

        def propose(self, round_index, context):
            seen.append(context)
            return self.inner.propose(round_index, context)

    provider = DeterministicStubProvider(plan=[
        ("baseline", {}),
        ("wider model", {"model.hidden_dim": 64}),
    ])
    harness(tmp_path, Recording(provider), rounds=2).run()
    second = seen[1]  # the context of round 1 must describe round 0 fully
    assert second["round_index"] == 1
    assert second["best_metric"] is not None
    assert second["editable_scope"] and second["base_config"]
    first = second["history"][0]
    assert first["decision"] == "KEEP"
    assert first["status"] == "completed"
    assert first["primary_value"] == pytest.approx(second["best_metric"])
    assert first["patch"] == {}


def test_step_runs_exactly_one_round(tmp_path):
    h = harness(tmp_path, rounds=2)
    entry = h.step(Proposal(intent="baseline", config_patch={}))
    assert entry["run_id"] == "tcm-0000"
    assert entry["decision"] == "KEEP"
    assert len(RunManifestStore(tmp_path / "ar" / "manifest.jsonl").load()) == 1
    entry2 = h.step(Proposal(intent="wider", config_patch={"model.hidden_dim": 64}))
    assert entry2["run_id"] == "tcm-0001"
    # the step driver reports the integrated-agent mode honestly
    assert h.mode == "AGENT_INTEGRATED"
    assert h.agent_status == "AGENT_ISOLATION_UNVERIFIED"


def test_step_refuses_when_budget_exhausted(tmp_path):
    h = harness(tmp_path, rounds=1)
    h.step(Proposal(intent="baseline", config_patch={}))
    with pytest.raises(RuntimeError, match="budget exhausted"):
        h.step(Proposal(intent="one more", config_patch={}))


def test_fixed_proposal_provider_yields_once():
    provider = FixedProposalProvider(Proposal(intent="only", config_patch={}))
    assert provider.propose(0, {}) is not None
    assert provider.propose(1, {}) is None


def test_llm_proposal_provider_parses_strict_json():
    def backend(_prompt):
        return '```json\n{"intent": "wider", "config_patch": {"model.hidden_dim": 64}}\n```'

    proposal = LLMProposalProvider(backend).propose(0, {"editable_scope": []})
    assert proposal == Proposal(intent="wider",
                                config_patch={"model.hidden_dim": 64})

    def stop(_prompt):
        return '{"intent": "STOP", "config_patch": {}}'

    assert LLMProposalProvider(stop).propose(0, {}) is None

    def garbage(_prompt):
        return "I suggest widening the model."

    with pytest.raises(ValueError, match="not valid JSON"):
        LLMProposalProvider(garbage).propose(0, {})

    def missing(_prompt):
        return '{"intent": "no patch here"}'

    with pytest.raises(ValueError, match="config_patch"):
        LLMProposalProvider(missing).propose(0, {})


def test_research_prompt_contains_scope_and_history():
    prompt = build_research_prompt({
        "editable_scope": ["model.hidden_dim"],
        "history": [{"run_id": "tcm-0000", "primary_value": 0.4}],
        "best_metric": 0.4,
    })
    assert "model.hidden_dim" in prompt
    assert "tcm-0000" in prompt and "0.4" in prompt


def test_source_patch_static_check_rejects_frozen_symbols():
    with pytest.raises(ValueError, match="frozen research machinery"):
        validate_source_patch("import runtime.tcm_eval\nuse holdout labels")
    validate_source_patch("# harmless refactor of the optimizer step")


def test_apply_config_patch_maps_sections():
    base = {
        "task": "tcm",
        "model": {"hidden_dim": 32},
        "trainer": {"epochs": 2},
        "train_provider": {"batch_size": 8},
        "eval_provider": {},
    }
    cfg = apply_config_patch(base, {
        "model.hidden_dim": 64, "trainer.epochs": 5, "trainer.batch_size": 16,
    })
    assert cfg["model"]["hidden_dim"] == 64
    assert cfg["trainer"]["epochs"] == 5
    assert cfg["train_provider"]["batch_size"] == 16
    with pytest.raises(ValueError, match="not editable"):
        apply_config_patch(base, {"eval_provider.n": 3})


def test_manifest_entries_carry_provenance(tmp_path):
    harness(tmp_path, rounds=1).run()
    entry = RunManifestStore(tmp_path / "ar" / "manifest.jsonl").load()[0]
    for key in ("run_id", "intent", "patch", "mode", "smoke_only",
                "source_commit", "frozen_hashes", "artifact_sha256",
                "metrics", "decision", "status"):
        assert key in entry, key
    assert entry["frozen_hashes"]["standard_version"] == "SYNTHETIC-CONTRACT-V0"
    assert len(entry["artifact_sha256"]) == 64


def test_validate_config_patch_and_scope():
    scope = ["model.hidden_dim", "trainer.epochs"]
    validate_config_patch({"model.hidden_dim": 8}, scope)
    with pytest.raises(ValueError, match="frozen keys"):
        validate_config_patch({"trainer.learning_rate": 0.5}, scope)


def test_cli_research_smoke(tmp_path, capsys):
    from runtime.cli import main

    assert main(["research", "--profile", "tcm", "--rounds", "2",
                 "--out", str(tmp_path / "runs")]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["rounds_completed"] == 2
    assert summary["smoke_only"] is True

    assert main(["research", "--profile", "physio", "--rounds", "1",
                 "--out", str(tmp_path / "runs2")]) == 2
    refusal = json.loads(capsys.readouterr().out)
    assert refusal["ok"] is False and "TCM_ONLY" in refusal["error"]


def test_cli_research_step_smoke(tmp_path, capsys):
    from runtime.cli import main

    out = str(tmp_path / "runs")
    assert main(["research-step", "--profile", "tcm", "--rounds", "1",
                 "--out", out, "--print-context"]) == 0
    context = json.loads(capsys.readouterr().out)
    assert context["round_index"] == 0 and context["history"] == []

    proposal = tmp_path / "proposal.json"
    proposal.write_text(json.dumps(
        {"intent": "baseline", "config_patch": {}}))
    assert main(["research-step", "--profile", "tcm", "--rounds", "1",
                 "--out", out, "--proposal-file", str(proposal)]) == 0
    entry = json.loads(capsys.readouterr().out)
    assert entry["run_id"] == "tcm-0000"
    assert entry["mode"] == "AGENT_INTEGRATED"

    assert main(["research-step", "--profile", "tcm", "--rounds", "1",
                 "--out", out, "--proposal-file", str(proposal)]) == 1
    assert "budget exhausted" in json.loads(capsys.readouterr().out)["error"]


# -- round-4 review fixes (R-01..R-04) -------------------------------------

def test_manifest_chain_detects_edit_deletion_and_reorder(tmp_path):
    store = RunManifestStore(tmp_path / "m.jsonl")
    store.append({"run_id": "tcm-0000", "decision": "KEEP", "status": "done"})
    store.append({"run_id": "tcm-0001", "decision": "REVERT", "status": "done"})
    store.append({"run_id": "tcm-0002", "decision": "KEEP", "status": "done"})
    assert len(store.load()) == 3  # intact chain loads

    lines = (tmp_path / "m.jsonl").read_text().splitlines()

    def rewrite(ls):
        (tmp_path / "m.jsonl").write_text("\n".join(ls) + "\n")

    # a hand-edit of one KEEP entry flips its best_so_far -> detected
    tampered = json.loads(lines[0])
    tampered["best_so_far"] = 0.0
    edited = [*lines]
    edited[0] = json.dumps(tampered, ensure_ascii=False)
    rewrite(edited)
    with pytest.raises(RuntimeError, match="integrity check failed"):
        RunManifestStore(tmp_path / "m.jsonl").load()

    # deleting a middle line breaks the link of the next one -> detected
    rewrite([lines[0], lines[2]])
    with pytest.raises(RuntimeError, match="integrity check failed"):
        RunManifestStore(tmp_path / "m.jsonl").load()

    # reordering two entries breaks their links -> detected
    rewrite([lines[1], lines[0], lines[2]])
    with pytest.raises(RuntimeError, match="integrity check failed"):
        RunManifestStore(tmp_path / "m.jsonl").load()

    (tmp_path / "m.jsonl").write_text("\n".join(lines) + "\n")
    assert len(store.load()) == 3


def test_manifest_chain_accepts_legacy_entries_and_extends_them(tmp_path):
    path = tmp_path / "m.jsonl"
    legacy = {"run_id": "tcm-0000", "decision": "KEEP", "status": "done"}
    path.write_text(json.dumps(legacy) + "\n")
    store = RunManifestStore(path)
    store.append({"run_id": "tcm-0001", "decision": "KEEP", "status": "done"})
    entries = store.load()
    assert len(entries) == 2
    assert "entry_sha256" not in entries[0]  # legacy line untouched
    assert entries[1]["prev_entry_sha256"] == ""
    assert len(entries[1]["entry_sha256"]) == 64


def test_resume_continues_plan_without_replay(tmp_path):
    plan = [
        ("baseline", {}),
        ("wider model", {"model.hidden_dim": 64, "model.fusion_dim": 64}),
        ("third candidate", {"trainer.epochs": 3}),
    ]
    first = harness(tmp_path, DeterministicStubProvider(plan=plan), rounds=1).run()
    assert first["rounds_completed"] == 1
    # a fresh provider on resume must propose plan[1], not replay plan[0]
    second = harness(
        tmp_path, DeterministicStubProvider(plan=plan), rounds=3
    ).run()
    assert second["rounds_completed"] == 3
    entries = RunManifestStore(tmp_path / "ar" / "manifest.jsonl").load()
    assert entries[1]["intent"] == "wider model"
    assert entries[2]["intent"] == "third candidate"


def test_agent_error_is_recorded_as_reverted_round(tmp_path):
    class Broken:
        def propose(self, round_index, context):
            raise ValueError("agent reply is not valid JSON: 'oops'")

    # each failed proposal consumes one round of the budget and is recorded
    summary = harness(tmp_path, Broken(), rounds=2).run()
    assert summary["rounds_completed"] == 2
    entries = RunManifestStore(tmp_path / "ar" / "manifest.jsonl").load()
    for entry in entries:
        assert entry["decision"] == "REVERT"
        assert entry["status"] == "agent_error"
        assert "not valid JSON" in entry["reason"]
    assert [e["run_id"] for e in entries] == ["tcm-0000", "tcm-0001"]


def test_scope_key_outside_known_sections_recorded_as_rejected(tmp_path):
    # editable_scope is runspec data; a custom scope can pass static
    # validation yet name a section apply_config_patch refuses — the round
    # must still be recorded as rejected, never crash the loop (R-03)
    provider = DeterministicStubProvider(plan=[
        ("baseline", {}),
        ("illegal section", {"eval_provider.n": 3}),
    ])
    summary = harness(
        tmp_path, provider,
        editable_scope=["model.hidden_dim", "eval_provider.n"],
    ).run()
    rejected = summary["decisions"][1]
    assert rejected["decision"] == "REVERT"
    assert rejected["status"] == "rejected"
    assert "not editable" in rejected["reason"]


def test_worker_env_drops_inherited_pythonpath(monkeypatch):
    from runtime.harness import SRC_DIR, _worker_env

    monkeypatch.setenv("PYTHONPATH", "/host/junk:/more/junk")
    env = _worker_env()
    assert env["PYTHONPATH"] == str(SRC_DIR)
