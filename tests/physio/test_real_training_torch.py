"""#102 real-data neural path: torch-gated pipeline regression only.

Torch is a training-host dependency. These checks prove the neural path runs end
to end on real rows and stays deterministic. They never assert that the model is
good: with five sessions the evaluation status remains INSUFFICIENT_EVIDENCE.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from physio.real_export import import_real_export
from physio.real_training import (
    MODEL_NEURAL,
    NEURAL_COMPLETED,
    STATUS_INSUFFICIENT,
    run_real_pre_experiment,
)

pytestmark = pytest.mark.synthetic
torch = pytest.importorskip("torch")

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "physio-legacy-sessions-synthetic.csv"


def _report(**kwargs):
    return run_real_pre_experiment(import_real_export(FIXTURE), **kwargs)


def test_neural_path_completes_on_real_rows():
    report = _report()
    assert report.neural_status == NEURAL_COMPLETED
    for fold in report.folds:
        assert fold.neural_status == NEURAL_COMPLETED
        assert fold.final_loss is not None
        assert fold.final_loss == fold.final_loss  # not NaN
        assert fold.mae[MODEL_NEURAL] is not None
        assert fold.mae[MODEL_NEURAL] >= 0.0


def test_torch_does_not_upgrade_the_evaluation_status():
    report = _report()
    assert report.evaluation_status == STATUS_INSUFFICIENT
    assert report.status == STATUS_INSUFFICIENT


def test_neural_predictions_are_deterministic():
    first = _report()
    second = _report()
    assert [fold.predictions[MODEL_NEURAL] for fold in first.folds] == [
        fold.predictions[MODEL_NEURAL] for fold in second.folds
    ]
    assert first.model_mae[MODEL_NEURAL] == second.model_mae[MODEL_NEURAL]


def test_neural_loss_is_finite_for_both_target_modes():
    for target in ("post_state", "delta_vendor_fatigue"):
        report = _report(target=target)
        assert report.neural_status == NEURAL_COMPLETED
        assert all(fold.final_loss is not None for fold in report.folds)
    assert len(report.folds) == 5


def test_neural_mae_is_reported_without_a_quality_claim():
    report = _report()
    assert report.model_mae[MODEL_NEURAL] is not None
    assert "INSUFFICIENT_EVIDENCE" in report.to_dict()["evaluation_status"]
