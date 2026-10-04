from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from physio_autoresearch.contracts import ResearchError, candidate_id, load_contract, validate_candidate, validate_contract
from physio_autoresearch.data import check_fold, parse_data, subject_folds
from physio_autoresearch.trusted.worker import check_predictions, score_predictions


@pytest.fixture
def contract():
    return load_contract(Path("physio_autoresearch/constitution.synthetic.yaml"))


@pytest.fixture
def frame():
    return pd.DataFrame({"record_id": ["a1", "a2", "b1"], "subject_id": ["a", "a", "b"],
                         "age": [30, 30, 45], "pre_hr": [70, 72, 80], "pre_hrv": [30, np.nan, 45], "post_hr": [65, 66, 75]})


def raw(frame):
    return frame.to_csv(index=False).encode()


@pytest.mark.harness_bench
@pytest.mark.parametrize("change,code", [
    (lambda c: c["features"][0].update(name="post_hr"), "SCHEMA_COLLISION"),
    (lambda c: c["features"][0].update(name="post_fatigue"), "FUTURE_FEATURE_LEAKAGE"),
    (lambda c: c["features"][0].update(phase="post"), "FUTURE_FEATURE_LEAKAGE"),
    (lambda c: c.update(metric="accuracy"), "WRONG_METRIC"),
    (lambda c: c.update(source="real"), "UNKNOWN_PROVENANCE"),
    (lambda c: c["targets"][0].update(scale=0), "BAD_CONTRACT"),
    (lambda c: c["budget"].update(max_experiments=21), "BAD_BUDGET"),
])
def test_frozen_contract_rejects_bad_science(contract, change, code):
    change(contract)
    with pytest.raises(ResearchError, match=code):
        validate_contract(contract)


@pytest.mark.harness_bench
@pytest.mark.parametrize("spec,code", [
    ({"model": "transformer"}, "BAD_MODEL_CONFIG"),
    ({"model": "xgboost", "params": {"max_depth": 100}}, "BAD_MODEL_CONFIG"),
    ({"model": "random_forest", "params": {"max_depth": 1.2}}, "BAD_MODEL_CONFIG"),
    ({"model": "elastic_net", "features": ["post_hr"]}, "FUTURE_FEATURE_LEAKAGE"),
    ({"model": "elastic_net", "features": ["unknown_sensor"]}, "UNKNOWN_FEATURE"),
    ({"model": "elastic_net", "command": "touch evaluator.py"}, "BAD_MODEL_CONFIG"),
    ({"model": "elastic_net", "params": {"alpha": float("nan")}}, "BAD_MODEL_CONFIG"),
])
def test_candidate_boundary(contract, spec, code):
    with pytest.raises(ResearchError, match=code):
        validate_candidate(spec, contract)


def test_candidate_identity_ignores_hypothesis_and_feature_order(contract):
    a = validate_candidate({"model": "naive", "features": ["pre_hr", "age"], "hypothesis": "first"}, contract)
    b = validate_candidate({"model": "naive", "features": ["age", "pre_hr"], "hypothesis": "different"}, contract)
    assert candidate_id(a) == candidate_id(b)


@pytest.mark.parametrize("family,depth", [("random_forest", 6), ("xgboost", 3)])
def test_omitted_tree_depth_cannot_bypass_candidate_bounds(contract, family, depth):
    from physio_autoresearch.models import make_model
    proposal = {"model": family}
    candidate = validate_candidate(proposal, contract)
    assert candidate["params"]["max_depth"] == depth
    assert proposal == {"model": family}  # Canonicalization preserves the proposal.
    estimator = make_model(candidate, contract["seed"]).named_steps["model"]
    if family == "xgboost":
        estimator = estimator.estimator
    assert estimator.get_params()["max_depth"] == depth


@pytest.mark.harness_bench
@pytest.mark.parametrize("change,code", [
    (lambda f: f.drop(columns="age"), "SCHEMA_DRIFT"),
    (lambda f: f.assign(after_fatigue=1), "SCHEMA_DRIFT"),
    (lambda f: f.assign(record_id=["a", "a", "b"]), "DUPLICATE_RECORD"),
    (lambda f: f.assign(subject_id=["", "a", "b"]), "MISSING_ID"),
    (lambda f: f.assign(subject_id=[" a", "a", "b"]), "BAD_ID"),
    (lambda f: f.assign(age=["invalid", 30, 45]), "NONNUMERIC_DATA"),
    (lambda f: f.assign(pre_hr=[np.inf, 72, 80]), "NONFINITE_DATA"),
    (lambda f: f.assign(post_hr=[np.nan, 66, 75]), "MISSING_TARGET"),
    (lambda f: f.assign(pre_hr=[10000, 72, 80]), "OUT_OF_RANGE"),
])
def test_data_qc(frame, contract, change, code):
    with pytest.raises(ResearchError, match=code):
        parse_data(raw(change(frame)), contract)


@pytest.mark.harness_bench
def test_duplicate_headers(frame, contract):
    payload = raw(frame).replace(b"age,pre_hr", b"age,age")
    with pytest.raises(ResearchError, match="SCHEMA_DRIFT"):
        parse_data(payload, contract)


@pytest.mark.parametrize("row", ["r1,s1,30,70,40,65,66", "r1,s1,30,70,40", 'r1,s1,30,70,40,"65'])
def test_malformed_csv_cannot_shift_ids_or_targets(contract, row):
    payload = ("record_id,subject_id,age,pre_hr,pre_hrv,post_hr\n" + row + "\n").encode()
    with pytest.raises(ResearchError, match="CORRUPTED_DATA"):
        parse_data(payload, contract)


def test_csv_quoted_ids_and_blank_lines(contract):
    payload = b'record_id,subject_id,age,pre_hr,pre_hrv,post_hr\n\n"r,1","s,1",30,70,40,65\n'
    loaded = parse_data(payload, contract)
    assert loaded.record_id.tolist() == ["r,1"]
    assert loaded.subject_id.tolist() == ["s,1"]


@pytest.mark.harness_bench
def test_missing_features_are_preserved(frame, contract):
    loaded = parse_data(raw(frame), contract)
    assert np.isnan(loaded.pre_hrv.iloc[1])
    assert loaded.subject_id.iloc[0] == "a"


@pytest.mark.harness_bench
def test_subject_sessions_never_cross_folds(frame, contract):
    folds = subject_folds(frame, contract)
    assert len(folds) == 2
    for fold in folds:
        check_fold(frame, contract, fold)
    assert folds[0]["validation_ids"] == ["a1", "a2"]
    with pytest.raises(ResearchError, match="SUBJECT_LEAKAGE"):
        check_fold(frame, contract, {"train_ids": ["a1", "b1"], "validation_ids": ["a2"]})


@pytest.mark.harness_bench
def test_prediction_shape_and_nan():
    with pytest.raises(ResearchError, match="NAN_PREDICTION"):
        check_predictions([[np.nan]], 1, 1)
    with pytest.raises(ResearchError, match="INCOMPLETE_PREDICTIONS"):
        check_predictions([[1, 2]], 1, 1)


@pytest.mark.parametrize("prediction,rows,targets", [
    ([[1, 2, 3]], 3, 1),
    ([[[1]], [[2]], [[3]]], 3, 1),
    ([[1, 2, 3], [4, 5, 6]], 3, 2),
    ([1, 2, 3, 4], 2, 2),
    (1, 1, 1),
])
def test_predictions_cannot_reshape_wrong_sample_or_target_axes(prediction, rows, targets):
    with pytest.raises(ResearchError, match="INCOMPLETE_PREDICTIONS"):
        check_predictions(prediction, rows, targets)


def test_prediction_shape_preserves_supported_regressor_outputs():
    np.testing.assert_array_equal(check_predictions([1, 2, 3], 3, 1), [[1], [2], [3]])
    np.testing.assert_array_equal(check_predictions([[1, 2], [3, 4]], 2, 2), [[1, 2], [3, 4]])


@pytest.mark.parametrize("id_key", ["record_id", "subject_id"])
def test_generated_prediction_columns_cannot_overwrite_identifiers(contract, id_key):
    contract[id_key] = "prediction_post_hr"
    with pytest.raises(ResearchError, match="SCHEMA_COLLISION"):
        validate_contract(contract)


def test_generated_prediction_columns_cannot_collide_with_measured_targets(contract):
    contract["targets"].append({"name": "prediction_post_hr", "phase": "post", "unit": "bpm", "scale": 10})
    with pytest.raises(ResearchError, match="SCHEMA_COLLISION"):
        validate_contract(contract)


@pytest.mark.parametrize("row", ["r1,s1\x00suffix,30,70,40,65", "r1,s1,30,70,40,65\x0099"])
def test_csv_nul_bytes_cannot_silently_truncate_subjects_or_outcomes(contract, row):
    payload = ("record_id,subject_id,age,pre_hr,pre_hrv,post_hr\n" + row + "\n").encode()
    with pytest.raises(ResearchError, match="CORRUPTED_DATA"):
        parse_data(payload, contract)


@pytest.mark.harness_bench
def test_metric_is_subject_weighted_and_complete(frame, contract):
    predictions = pd.DataFrame({"record_id": ["a1", "a2", "b1"], "prediction_post_hr": [65, 66, 85]})
    scores = score_predictions(predictions, frame, contract)
    assert scores["metric"] == 0.5  # two subjects, not three rows
    assert scores["per_target_mae"]["post_hr"] == 5
    with pytest.raises(ResearchError, match="INCOMPLETE_PREDICTIONS"):
        score_predictions(predictions.iloc[:2], frame, contract)


def test_multiple_targets_use_frozen_units(frame, contract):
    contract = deepcopy(contract)
    contract["targets"].append({"name": "post_bp", "phase": "post", "unit": "mmHg", "scale": 20})
    frame["post_bp"] = [100, 100, 120]
    predictions = pd.DataFrame({"record_id": frame.record_id, "prediction_post_hr": [65, 66, 85], "prediction_post_bp": [120, 120, 120]})
    assert score_predictions(predictions, frame, contract)["metric"] == 0.5
