"""#102 real-data neural path: torch-gated pipeline regression only.

Torch is a training-host dependency. These checks prove the neural path runs end
to end on real rows and stays deterministic. They never assert that the model is
good: with five sessions the evaluation status remains INSUFFICIENT_EVIDENCE.
"""

from __future__ import annotations

import math
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
        assert math.isfinite(fold.final_loss)
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
    """A NaN loss on the *second* target must fail, not merely be non-None.

    ``final_loss is not None`` passes for NaN and inf, so finiteness is asserted per
    fold for both declared targets.
    """
    for target in ("post_state", "delta_vendor_fatigue"):
        report = _report(target=target)
        assert report.neural_status == NEURAL_COMPLETED
        assert len(report.folds) == 5
        for fold in report.folds:
            assert fold.final_loss is not None
            assert math.isfinite(fold.final_loss), f"{target} fold {fold.fold}: {fold.final_loss}"


def test_declared_targets_are_outside_the_model_output_scale():
    """The [0, 100] head (pinned by test_realtime_model) cannot express the targets.

    ``test_forward_returns_one_bounded_fatigue_state`` already proves the head emits
    within [0, 100]. This test pins the *target* side so the report's
    structural-mismatch disclosure cannot silently go stale if someone retargets
    the model to a domain it can express.
    """
    from physio.real_export import import_real_export as _import
    from physio.real_training import build_targets

    sessions = _import(FIXTURE)
    post_state = [v for v in build_targets(sessions, "post_state") if v is not None]
    delta = [v for v in build_targets(sessions, "delta_vendor_fatigue") if v is not None]
    assert post_state and delta

    # post_state fits the range numerically but sits three orders of magnitude
    # below its scale.
    assert all(0.0 <= v <= 100.0 for v in post_state)
    assert max(post_state) < 1.0

    # delta_vendor_fatigue is negative for every session here, i.e. entirely
    # outside the declared output range.
    assert all(v < 0.0 for v in delta)
    assert all(not 0.0 <= v <= 100.0 for v in delta)


def test_neural_mae_is_reported_without_a_quality_claim():
    report = _report()
    assert report.model_mae[MODEL_NEURAL] is not None
    assert "INSUFFICIENT_EVIDENCE" in report.to_dict()["evaluation_status"]


def test_each_fold_owns_its_checkpoint_directory(tmp_path):
    report = run_real_pre_experiment(
        import_real_export(FIXTURE), out_dir=tmp_path
    )
    assert report.neural_status == NEURAL_COMPLETED
    checkpoints = sorted(tmp_path.rglob("model_state.pt"))
    assert len(checkpoints) == len(report.folds)
    assert len({path.parent.name for path in checkpoints}) == len(report.folds)
    assert not (tmp_path / "model_state.pt").exists()
