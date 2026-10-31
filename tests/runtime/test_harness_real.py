"""End-to-end real-data AutoResearch test (#50): frozen manifest ->
harness round -> worker train (smoke_only=False) -> trusted score.

Requires the git-ignored Mendeley workbook on the host; skips elsewhere
(synthetic pipeline is covered by test_autoresearch.py).
"""

from pathlib import Path

import pytest

from runtime.agent import DeterministicStubProvider
from runtime.harness import REAL_TCM_PLAN, AutoResearchHarness, default_real_tcm_runspec
from runtime.manifest import RunManifestStore

REAL_XLS = Path("data/local/mendeley-cxtph3tjsg-v1/Share data.xls")

pytestmark = [
    pytest.mark.skipif(
        not REAL_XLS.exists(), reason="real Mendeley workbook not on this host"
    )
]


def test_real_data_harness_round(tmp_path):
    from data.mendeley import EXPECTED_SHA256, load_mendeley_scores
    from data.real_split import build_real_manifest

    records = load_mendeley_scores(REAL_XLS)
    saved = build_real_manifest(
        records, dataset_sha256=EXPECTED_SHA256
    )
    manifest_path = tmp_path / "manifest.json"
    saved.save(manifest_path)

    runspec = default_real_tcm_runspec(
        str(manifest_path), str(REAL_XLS), rounds=2, timeout_s=300.0
    )
    runspec["base_config"]["trainer"]["epochs"] = 2  # keep the e2e fast
    provider = DeterministicStubProvider(plan=REAL_TCM_PLAN[:2])
    harness = AutoResearchHarness(
        runspec, provider=provider, out_dir=tmp_path / "runs"
    )
    assert harness.mode == "REAL_DATA_TRAINING"
    assert harness.smoke_only is False

    summary = harness.run()
    assert summary["mode"] == "REAL_DATA_TRAINING"
    assert summary["smoke_only"] is False
    assert summary["rounds_completed"] == 2
    completed = [d for d in summary["decisions"] if d["status"] == "completed"]
    assert completed, f"no candidate completed: {summary['decisions']}"

    entries = RunManifestStore(tmp_path / "runs" / "manifest.jsonl").load()
    for entry in entries:
        assert entry["smoke_only"] is False
        assert entry["mode"] == "REAL_DATA_TRAINING"
    champion = RunManifestStore(tmp_path / "runs" / "manifest.jsonl").champion()
    # k-fold selection keeps no per-round model: a round trains `folds` models
    # and discards them, so the champion carries a fold-averaged score and no
    # artifact_dir. The deliverable comes from finalize(), one fit on the pool.
    assert champion is not None
    assert champion["artifact_dir"] is None
    assert champion["selection"]["folds"] == 4
    assert champion["primary_std"] is not None

    final = harness.finalize()
    assert final["champion_run_id"] == champion["run_id"]
    assert final["refit"]["refit_partition"] == "train+dev"

    from runtime.checkpoint import checkpoint_summary, load_training_checkpoint

    assert checkpoint_summary(final["refit"]["artifact_dir"])["smoke_only"] is False
    _, state, _ = load_training_checkpoint(final["refit"]["artifact_dir"])
    # the refit checkpoint must name the frozen manifest as its selection parent,
    # not merely descend from some manifest with the same row count
    assert state["selection_parent_manifest_sha256"] == saved.sha256
    assert state["manifest_sha256"] != saved.sha256
