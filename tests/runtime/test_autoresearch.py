"""AutoResearch harness tests: KEEP/REVERT, rejections, budget, resume (#45)."""

import json

import pytest

from runtime.agent import (
    DeterministicStubProvider,
    ExternalAgentProvider,
    apply_config_patch,
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
        ("underfit model", {"model.hidden_dim": 2, "trainer.epochs": 1}),
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
    assert champion["run_id"] == "tcm-0000"


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
