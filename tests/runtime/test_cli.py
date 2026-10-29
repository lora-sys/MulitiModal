"""CLI and #45-protocol smoke tests (#49)."""

import json

import pytest

from runtime.cli import main
from runtime.protocols import train_candidate, trusted_score

pytestmark = pytest.mark.synthetic


def test_list_models(capsys):
    assert main(["list-models"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert set(out["registered_model_types"]) == {
        "tcm_constitution_v1", "physio_bpsi_efri_v0"
    }


@pytest.mark.parametrize("task", ["tcm", "physio"])
def test_validate_architecture(task, capsys):
    assert main(["validate-architecture", "--task", task]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True and out["smoke_only"] is True
    assert out["loss_parts"]


def test_smoke_tcm_end_to_end(tmp_path, capsys):
    assert main(["smoke", "--task", "tcm", "--epochs", "1", "--out",
                 str(tmp_path / "runs")]) == 0
    summary = json.loads(capsys.readouterr().out)
    ckpt_dir = summary["out_dir"]
    import os

    assert os.path.exists(os.path.join(ckpt_dir, "metadata.json"))
    assert os.path.exists(os.path.join(ckpt_dir, "training_state.pt"))
    assert summary["checkpoint"]["smoke_only"] is True

    assert main(["checkpoint-reload", "--dir", ckpt_dir]) == 0
    reload_out = json.loads(capsys.readouterr().out)
    assert reload_out["ok"] is True
    assert reload_out["epochs_completed"] == 1


def test_train_candidate_and_trusted_score(tmp_path):
    report = train_candidate({
        "task": "tcm",
        "model": {"hidden_dim": 16, "fusion_dim": 16},
        "trainer": {"epochs": 2, "seed": 0},
        "train_provider": {"n_batches": 2, "batch_size": 8, "seed": 0},
        "eval_provider": {"n_batches": 1, "batch_size": 8, "seed": 1},
        "out_dir": tmp_path / "candidate",
    })
    assert report["smoke_only"] is True
    assert report["epochs_run"] == 2

    scored_a = trusted_score(report["artifact_dir"])
    scored_b = trusted_score(report["artifact_dir"])
    assert scored_a["metric_version"] == scored_b["metric_version"]
    assert scored_a["eval_loss"] == pytest.approx(scored_b["eval_loss"])
    assert scored_a["smoke_only"] is True
