"""#102 real-data pipeline pre-experiment: folds, baselines and honest status."""

from __future__ import annotations

from pathlib import Path

import pytest

from physio.dataset import IDENTITY_TRUSTED
from physio.real_export import import_real_export, session_feature_row
from physio.real_training import (
    MODEL_MEAN,
    MODEL_NEURAL,
    MODEL_RIDGE,
    NEURAL_COMPLETED,
    NEURAL_UNAVAILABLE,
    STATUS_INSUFFICIENT,
    TARGET_DELTA_FATIGUE,
    TARGET_POST_STATE,
    RealTrainingError,
    build_targets,
    run_real_pre_experiment,
)

pytestmark = pytest.mark.synthetic

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "physio-legacy-sessions-synthetic.csv"


def _sessions():
    return import_real_export(FIXTURE)


def _report(**kwargs):
    return run_real_pre_experiment(_sessions(), **kwargs)


def test_evaluation_status_stays_insufficient_for_five_sessions():
    report = _report()
    assert report.evaluation_status == STATUS_INSUFFICIENT
    assert "labeled samples 5 < required 30" in report.reason
    assert report.status == STATUS_INSUFFICIENT


def test_folds_are_leave_one_subject_out():
    report = _report()
    assert report.n_sessions == 5
    assert report.n_subjects == 5
    assert len(report.folds) == 5
    for fold in report.folds:
        assert fold.n_train == 4
        assert fold.n_test == 1
        assert not set(fold.train_subjects) & set(fold.test_subjects)


def test_shared_identity_is_reported_and_never_trusted():
    report = _report()
    assert report.n_members == 2
    assert report.identity_trust != IDENTITY_TRUSTED
    assert report.identity_trust == "SHARED_ACCOUNT_UNKNOWN"


def test_baselines_produce_finite_errors():
    report = _report()
    for fold in report.folds:
        assert fold.mae[MODEL_MEAN] is not None
        assert fold.mae[MODEL_RIDGE] is not None
        assert all(value >= 0.0 for value in fold.mae.values() if value is not None)
    assert report.model_mae[MODEL_MEAN] is not None
    assert report.model_mae[MODEL_RIDGE] is not None
    if report.neural_status != NEURAL_COMPLETED:
        assert report.model_mae[MODEL_NEURAL] is None


def test_mean_baseline_predicts_the_training_fold_mean():
    report = _report()
    fold = report.folds[0]
    train_targets = [
        value
        for other in report.folds
        if other.fold != fold.fold
        for value in other.targets
    ]
    expected = sum(train_targets) / len(train_targets)
    assert fold.predictions[MODEL_MEAN] == (expected,)


def test_scaler_is_fitted_on_training_rows_only():
    """A held-out subject's values must not enter the fold statistics."""
    from physio.real_training import _fit_scaler

    sessions = _sessions()
    rows = [session_feature_row(session) for session in sessions]
    train = rows[:4]
    fold_scaler = _fit_scaler(train)
    full_scaler = _fit_scaler(rows)
    assert fold_scaler.observed_counts != full_scaler.observed_counts
    assert fold_scaler.observed_counts[0] == sum(
        1 for row in train if row.observed_mask[0]
    )
    assert fold_scaler.observed_counts == tuple(
        sum(1 for row in train if row.observed_mask[index])
        for index in range(len(train[0].feature_names))
    )


def test_imputed_positions_never_enter_the_scaler_statistics():
    from physio.dataset import FoldImputer
    from physio.real_training import _fit_scaler

    sessions = _sessions()
    rows = [session_feature_row(session) for session in sessions]
    imputer = FoldImputer.fit(rows[:4])
    imputed = [imputer.transform(row) for row in rows]
    scaler = _fit_scaler(imputed[:4])
    assert all(imputed[1].imputed_mask[index] for index in (0, 1))
    assert all(imputed[4].observed_mask[index] for index in (0, 1))
    assert scaler.observed_counts == tuple(
        sum(1 for row in imputed[:4] if row.observed_mask[index])
        for index in range(len(rows[0].feature_names))
    )


def test_ridge_is_fitted_on_the_training_fold_only():
    from sklearn.linear_model import Ridge

    from physio.real_training import RIDGE_LAMBDA, _dense, _ridge_predict

    sessions = _sessions()
    rows = [session_feature_row(session) for session in sessions]
    targets = [0.3, 0.25, 0.28, 0.3, 0.26]
    fold_prediction = _ridge_predict(rows[:4], targets[:4], rows[4:])
    whole = Ridge(alpha=RIDGE_LAMBDA).fit(_dense(rows), targets)
    whole_prediction = tuple(float(v) for v in whole.predict(_dense(rows[4:])))
    assert fold_prediction != whole_prediction


def test_ridge_is_deterministic():
    first = _report()
    second = _report()
    assert [fold.predictions[MODEL_RIDGE] for fold in first.folds] == [
        fold.predictions[MODEL_RIDGE] for fold in second.folds
    ]


def test_neural_path_reports_torch_availability_honestly():
    report = _report()
    assert report.neural_status in {NEURAL_UNAVAILABLE, "completed", "refused"}
    for fold in report.folds:
        if fold.neural_status == NEURAL_UNAVAILABLE:
            assert fold.predictions[MODEL_NEURAL] == ()
            assert fold.mae[MODEL_NEURAL] is None


def test_missing_features_are_reported_per_feature():
    report = _report()
    assert report.feature_missing_counts["respiration_rate"] > 0
    assert report.feature_missing_counts["weight_kg"] > 0
    assert report.feature_missing_counts["bmi"] > 0
    assert set(report.imputed_features) == {"bmi", "respiration_rate", "weight_kg"}


def test_delta_target_mode_runs():
    report = _report(target=TARGET_DELTA_FATIGUE)
    assert report.target == TARGET_DELTA_FATIGUE
    assert report.evaluation_status == STATUS_INSUFFICIENT
    assert report.model_mae[MODEL_MEAN] is not None


def test_post_state_targets_stay_in_the_model_range():
    targets = build_targets(_sessions(), TARGET_POST_STATE)
    assert len(targets) == 5
    assert all(value is None or 0.0 <= value <= 1.0 for value in targets)


def test_delta_targets_preserve_missing_labels_as_none():
    targets = build_targets(_sessions(), TARGET_DELTA_FATIGUE)
    assert all(value is None or isinstance(value, float) for value in targets)
    assert all(value != 0.0 for value in targets if value is not None)


def test_unknown_target_mode_is_refused():
    with pytest.raises(RealTrainingError, match="unknown target mode"):
        run_real_pre_experiment(_sessions(), target="not-a-target")


def test_no_labelled_session_is_refused():
    sessions = _sessions()
    empty = tuple(
        type(session)(
            **{
                **{
                    name: getattr(session, name)
                    for name in (
                        "session_id",
                        "member_ref",
                        "subject_ref",
                        "device_ref",
                        "window_start",
                        "window_end",
                        "records",
                        "endpoints",
                        "static_values",
                        "static_origins",
                        "tcm_values",
                        "eeg_record_id",
                        "eeg_values",
                        "coded_values",
                        "identity_trust",
                        "issues",
                    )
                },
                "endpoints": {},
            }
        )
        for session in sessions
    )
    with pytest.raises(RealTrainingError, match="no labelled session"):
        run_real_pre_experiment(empty)


def test_report_serialises_to_json():
    import json

    report = _report()
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["evaluation_status"] == STATUS_INSUFFICIENT
    assert payload["identity_trust"] == "SHARED_ACCOUNT_UNKNOWN"
    assert len(payload["folds"]) == 5
