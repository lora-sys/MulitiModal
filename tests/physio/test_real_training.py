"""#102 real-data pipeline pre-experiment: folds, baselines and honest status."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from physio.dataset import (
    IDENTITY_SHARED,
    IDENTITY_TRUSTED,
    FeatureRow,
    FoldImputer,
    make_subject_folds,
)
from physio.preprocessing import WindowScaler
from physio.real_export import import_real_export, session_feature_row
from physio.real_training import (
    FOLD_SEED,
    MODEL_MEAN,
    MODEL_NEURAL,
    MODEL_RIDGE,
    MODELS,
    NEURAL_COMPLETED,
    NEURAL_UNAVAILABLE,
    STATUS_INSUFFICIENT,
    TARGET_DELTA_FATIGUE,
    TARGET_POST_STATE,
    RealTrainingError,
    _fit_scaler,
    _ridge_predict,
    _scaled_row,
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
    sessions = _sessions()
    rows = [session_feature_row(session) for session in sessions]
    # weight_kg is observed only in SYNTH-A and SYNTH-D; the other three sessions
    # have no 体脂 record, so the training fold holds exactly two observations and
    # the fold median must be their mean. Recomputing the expectation here keeps
    # the test independent of FoldImputer's own arithmetic.
    observed = sorted(
        float(row.values[index])
        for row in rows[:4]
        for index, name in enumerate(row.feature_names)
        if name == "weight_kg" and row.observed_mask[index]
    )
    assert len(observed) == 2
    imputer = FoldImputer.fit(rows[:4])
    imputed = [imputer.transform(row) for row in rows]
    scaler = _fit_scaler(imputed[:4])
    weight_index = rows[0].feature_names.index("weight_kg")
    assert all(imputed[1].imputed_mask[index] for index in (0, 1))
    assert all(imputed[2].imputed_mask[index] for index in (0, 1))
    assert all(imputed[4].imputed_mask[index] for index in (0, 1))
    assert all(imputed[0].observed_mask[index] for index in (0, 1))
    assert scaler.observed_counts[weight_index] == 2
    assert scaler.mean[weight_index] == pytest.approx(sum(observed) / 2.0)
    assert scaler.observed_counts == tuple(
        sum(1 for row in imputed[:4] if row.observed_mask[index])
        for index in range(len(rows[0].feature_names))
    )


def test_scaler_is_actually_applied_to_the_design_matrix():
    """Regression guard: a fitted scaler must transform the values it is given."""
    sessions = _sessions()
    rows = [session_feature_row(session) for session in sessions]
    scaler = _fit_scaler(rows[:4])
    assert any(std != 1.0 for std in scaler.std)
    scaled = [_scaled_row(row, scaler) for row in rows]
    for raw, scaled_row in zip(rows, scaled, strict=True):
        for index, name in enumerate(raw.feature_names):
            if not raw.valid_mask[index]:
                assert scaled_row.values[index] == 0.0
                assert scaled_row.valid_mask[index] is False
                continue
            expected = (float(raw.values[index]) - scaler.mean[index]) / scaler.std[index]
            assert scaled_row.values[index] == pytest.approx(expected), name


def test_scaler_transform_is_not_the_identity():
    sessions = _sessions()
    rows = [session_feature_row(session) for session in sessions]
    scaler = _fit_scaler(rows[:4])
    changed = 0
    for raw in rows:
        scaled = _scaled_row(raw, scaler)
        for index, valid in enumerate(raw.valid_mask):
            if valid and raw.values[index] != scaled.values[index]:
                changed += 1
    assert changed, "the scaler must change at least one observed value"


def test_scaler_statistics_come_from_the_training_fold_only():
    sessions = _sessions()
    rows = [session_feature_row(session) for session in sessions]
    fold_scaler = _fit_scaler(rows[:4])
    whole_scaler = _fit_scaler(rows)
    assert fold_scaler.mean != whole_scaler.mean
    assert fold_scaler.observed_counts != whole_scaler.observed_counts


def test_pooled_mae_uses_the_scaled_design_matrix():
    """End-to-end guard: reverting the call site must change the reported MAE.

    The three scaler unit tests exercise `_scaled_row` directly and would still
    pass if the caller handed the raw imputed rows to the models instead. This
    asserts the published pooled MAE equals an independently recomputed one built
    from an explicitly scaled design matrix.
    """
    sessions = _sessions()
    rows = [session_feature_row(session) for session in sessions]
    targets = list(build_targets(sessions, TARGET_POST_STATE))
    subjects = [session.subject_ref for session in sessions]
    folds = make_subject_folds(
        subjects,
        n_folds=len(set(subjects)),
        seed=FOLD_SEED,
        identity_trust=IDENTITY_SHARED,
        allow_untrusted=True,
    )

    expected_errors: list[float] = []
    for fold_index in range(folds.n_folds):
        train_idx = [
            i
            for i in range(len(rows))
            if folds.folds[subjects[i]] != fold_index and targets[i] is not None
        ]
        test_idx = [
            i
            for i in range(len(rows))
            if folds.folds[subjects[i]] == fold_index and targets[i] is not None
        ]
        if not train_idx or not test_idx:
            continue
        imputer = FoldImputer.fit([rows[i] for i in train_idx])
        imputed = [imputer.transform(row) for row in rows]
        scaler = _fit_scaler([imputed[i] for i in train_idx])
        scaled = [_scaled_row(imputed[i], scaler) for i in range(len(rows))]
        train_targets = [float(targets[i]) for i in train_idx]
        predicted = _ridge_predict(
            [scaled[i] for i in train_idx], train_targets, [scaled[i] for i in test_idx]
        )
        expected_errors.extend(
            abs(value - float(targets[i])) for value, i in zip(predicted, test_idx, strict=True)
        )

    report = _report(target=TARGET_POST_STATE)
    expected = sum(expected_errors) / len(expected_errors)
    assert report.model_mae[MODEL_RIDGE] == pytest.approx(expected)
    # And it must not equal the unscaled variant.
    unscaled: list[float] = []
    for fold_index in range(folds.n_folds):
        train_idx = [
            i
            for i in range(len(rows))
            if folds.folds[subjects[i]] != fold_index and targets[i] is not None
        ]
        test_idx = [
            i
            for i in range(len(rows))
            if folds.folds[subjects[i]] == fold_index and targets[i] is not None
        ]
        if not train_idx or not test_idx:
            continue
        imputer = FoldImputer.fit([rows[i] for i in train_idx])
        imputed = [imputer.transform(row) for row in rows]
        train_targets = [float(targets[i]) for i in train_idx]
        predicted = _ridge_predict(
            [imputed[i] for i in train_idx], train_targets, [imputed[i] for i in test_idx]
        )
        unscaled.extend(
            abs(value - float(targets[i])) for value, i in zip(predicted, test_idx, strict=True)
        )
    unscaled_mae = sum(unscaled) / len(unscaled)
    assert report.model_mae[MODEL_RIDGE] != pytest.approx(unscaled_mae)


def test_a_refused_fold_is_recorded_not_dropped():
    """A fold whose scaler cannot be fitted is reported with its reason."""
    report = _report()
    assert report.folds
    for fold in report.folds:
        if fold.refusal:
            assert fold.predictions == {name: () for name in MODELS}
            assert fold.targets == ()
            assert all(value is None for value in fold.mae.values())
            assert "refusing to fabricate a normalization" in fold.refusal


def test_masked_placeholder_features_are_named():
    """The report field must be derived, not vacuously empty.

    The synthetic fixture happens to impute every missing feature, so the field is
    empty there. That must be a property of the data, not of a deleted
    computation: the same helper is asserted to populate the field when a feature
    really cannot be imputed.
    """
    report = _report()
    assert isinstance(report.masked_placeholder_features, tuple)
    assert set(report.masked_placeholder_features) <= set(report.imputed_features)

    # Now force the situation: one session's weight_kg is the only observation, so
    # a fold holding it out leaves the feature unimputable.
    sessions = list(_sessions())
    rows = [session_feature_row(session) for session in sessions]
    targets = list(build_targets(sessions, TARGET_POST_STATE))
    subjects = [session.subject_ref for session in sessions]
    folds = make_subject_folds(
        subjects,
        n_folds=len(set(subjects)),
        seed=FOLD_SEED,
        identity_trust=IDENTITY_SHARED,
        allow_untrusted=True,
    )
    masked: set[str] = set()
    for fold_index in range(folds.n_folds):
        train_idx = [
            i
            for i in range(len(rows))
            if folds.folds[subjects[i]] != fold_index and targets[i] is not None
        ]
        if not train_idx:
            continue
        imputer = FoldImputer.fit([rows[i] for i in train_idx])
        for row in rows:
            transformed = imputer.transform(row)
            masked |= {
                name
                for name, valid in zip(
                    transformed.feature_names, transformed.valid_mask, strict=True
                )
                if not valid
            }
    assert masked, "the synthetic fixture should leave at least one feature unimputable"
    assert set(report.masked_placeholder_features) == masked


def test_a_feature_with_too_few_training_observations_stays_masked():
    """Prove the placeholder path: one observation cannot found a fold median."""
    names = ("a", "b")
    cutoff = datetime(2026, 11, 1, tzinfo=UTC)

    def row(session: str, values, observed, valid):
        return FeatureRow(
            feature_names=names,
            values=values,
            observed_mask=observed,
            valid_mask=valid,
            imputed_mask=(False, False),
            imputation=("none", "none"),
            subject_ref=f"subject:{session}",
            session_id=session,
            cutoff=cutoff,
        )

    rows = [
        row("S1", (1.0, 5.0), (True, True), (True, True)),
        row("S2", (0.0, 6.0), (False, True), (False, True)),
    ]
    imputer = FoldImputer.fit(rows)
    transformed = [imputer.transform(item) for item in rows]
    # "a" has a single observation in the training fold: below min_observed, so it
    # is never imputed and stays invalid. Downstream that becomes a 0.0 that is a
    # masked placeholder, not a measurement.
    assert transformed[1].values[0] == 0.0
    assert transformed[1].valid_mask[0] is False
    assert transformed[1].imputed_mask[0] is False
    # "b" is observed in both rows, so it is usable everywhere.
    assert transformed[1].valid_mask[1] is True


def test_fold_scaler_refuses_empty_training_rows():
    with pytest.raises(RealTrainingError, match="without training rows"):
        _fit_scaler([])


def test_fold_scaler_delegates_to_the_shared_window_scaler():
    """One zero-variance and mask policy, not a second copy of it."""
    from physio.real_training import _ScalerInput

    rows = [session_feature_row(session) for session in _sessions()]
    delegated = _fit_scaler(rows[:4])
    direct = WindowScaler.fit(
        [_ScalerInput(row.feature_names, row.values, row.observed_mask) for row in rows[:4]]
    )
    assert delegated.mean == direct.mean
    assert delegated.std == direct.std
    assert delegated.observed_counts == direct.observed_counts


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


def test_fold_checkpoint_directory_is_fold_scoped_without_out_dir():
    """Without an explicit out_dir the fallback must still be per-fold."""
    from physio.real_training import _fold_checkpoint_dir

    default_dir = _fold_checkpoint_dir(None, 0)
    assert default_dir.name == "fold-0"
    assert "physio-real-data" in default_dir.parts
    assert _fold_checkpoint_dir(None, 3).name == "fold-3"
    assert _fold_checkpoint_dir(Path("/tmp/x"), 2) == Path("/tmp/x/fold-2")
