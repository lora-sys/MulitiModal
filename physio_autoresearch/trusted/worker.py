"""Bounded child process: fit candidate artifacts or independently score them."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..contracts import ResearchError, candidate_id, digest, file_hash, read_json, validate_candidate, validate_contract, write_json
from ..data import check_fold, load_data, subject_folds
from ..models import make_model


def context(candidate: Path):
    request = read_json(candidate / "request.json")
    contract = validate_contract(request["contract"])
    spec = validate_candidate(request["candidate"], contract)
    snapshot = Path(request["data_path"])
    if file_hash(snapshot) != request["data_sha256"]:
        raise ResearchError("DATA_CHANGED", "ingested snapshot hash mismatch")
    return request, contract, spec, load_data(snapshot, contract)


def train(candidate: Path):
    request, contract, spec, frame = context(candidate)
    targets = [t["name"] for t in contract["targets"]]
    indexed = frame.set_index(contract["record_id"])
    folds = subject_folds(frame, contract)
    artifacts = []
    for number, fold in enumerate(folds):
        check_fold(frame, contract, fold)
        model = make_model(spec, contract["seed"])
        train_frame = indexed.loc[fold["train_ids"]]
        model.fit(train_frame[spec["features"]], train_frame[targets].to_numpy())
        filename = f"fold-{number:03d}.joblib"
        joblib.dump(model, candidate / filename)
        artifacts.append({**fold, "filename": filename, "sha256": file_hash(candidate / filename)})
    full = make_model(spec, contract["seed"])
    full.fit(frame[spec["features"]], frame[targets].to_numpy())
    joblib.dump(full, candidate / "model.joblib")
    write_json(candidate / "artifacts.json", {"candidate_sha256": candidate_id(spec), "contract_sha256": digest(contract),
               "data_sha256": request["data_sha256"], "folds": artifacts, "model_sha256": file_hash(candidate / "model.joblib")})


def check_predictions(prediction, rows: int, targets: int) -> np.ndarray:
    result = np.asarray(prediction, dtype=float)
    if targets == 1 and result.shape == (rows,):
        result = result[:, None]  # sklearn's single-output regressors return 1-D.
    if result.shape != (rows, targets):
        raise ResearchError("INCOMPLETE_PREDICTIONS", str(result.shape))
    if not np.isfinite(result).all():
        raise ResearchError("NAN_PREDICTION", "predictions must be finite")
    return result


def score_predictions(predictions: pd.DataFrame, frame: pd.DataFrame, contract: dict) -> dict:
    record, subject = contract["record_id"], contract["subject_id"]
    if predictions[record].duplicated().any() or set(predictions[record]) != set(frame[record]):
        raise ResearchError("INCOMPLETE_PREDICTIONS", "every record must appear exactly once")
    joined = frame.merge(predictions, on=record, validate="one_to_one")
    per_target = {}
    per_subject = {}
    for target in contract["targets"]:
        name = target["name"]
        values = joined[f"prediction_{name}"].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ResearchError("NAN_PREDICTION", name)
        errors = np.abs(values - joined[name].to_numpy(dtype=float))
        means = pd.Series(errors).groupby(joined[subject]).mean()
        per_target[name] = float(means.mean())
        for key, error in means.items():
            per_subject.setdefault(str(key), []).append(float(error) / target["scale"])
    per_subject = {key: float(np.mean(values)) for key, values in per_subject.items()}
    return {"metric": float(np.mean(list(per_subject.values()))), "metric_name": contract["metric"],
            "per_target_mae": per_target, "per_subject_normalized_mae": per_subject}


def evaluate(candidate: Path):
    request, contract, spec, frame = context(candidate)
    artifacts = read_json(candidate / "artifacts.json")
    if artifacts["candidate_sha256"] != candidate_id(spec) or artifacts["contract_sha256"] != digest(contract) or artifacts["data_sha256"] != request["data_sha256"]:
        raise ResearchError("INVALID_CHECKPOINT", "artifact provenance mismatch")
    if file_hash(candidate / "model.joblib") != artifacts["model_sha256"]:
        raise ResearchError("INVALID_CHECKPOINT", "full-fit model hash mismatch")
    expected = subject_folds(frame, contract)
    actual = artifacts["folds"]
    if len(actual) != len(expected):
        raise ResearchError("INCOMPLETE_SPLIT", "wrong number of subject folds")
    if not expected:
        write_json(candidate / "evaluation.json", {"metric": None, "status": "FIT_ONLY_NOT_VALIDATED", "rows": len(frame)})
        return
    indexed = frame.set_index(contract["record_id"])
    targets = [t["name"] for t in contract["targets"]]
    records = []
    for number, (wanted, found) in enumerate(zip(expected, actual)):
        check_fold(frame, contract, found)
        if any(found[key] != wanted[key] for key in wanted) or found["filename"] != f"fold-{number:03d}.joblib":
            raise ResearchError("SUBJECT_LEAKAGE", "artifact split differs from evaluator-owned LOSO")
        artifact = candidate / found["filename"]
        if file_hash(artifact) != found["sha256"]:
            raise ResearchError("INVALID_CHECKPOINT", "fold model hash mismatch")
        # These pickles are created only by the fixed trainer. Never accept external pickles.
        model = joblib.load(artifact)
        features = indexed.loc[wanted["validation_ids"], spec["features"]]
        prediction = check_predictions(model.predict(features), len(features), len(targets))
        second = check_predictions(model.predict(features), len(features), len(targets))
        if not np.array_equal(prediction, second):
            raise ResearchError("UNREPRODUCIBLE", "artifact inference is nondeterministic")
        for key, values in zip(wanted["validation_ids"], prediction):
            records.append({contract["record_id"]: key, **{f"prediction_{name}": float(value) for name, value in zip(targets, values)}})
    predictions = pd.DataFrame(records)
    scores = score_predictions(predictions, frame, contract)
    predictions.to_csv(candidate / "predictions.csv", index=False)
    write_json(candidate / "evaluation.json", {**scores, "rows": len(frame), "subjects": len(expected),
               "status": "SMOKE_EVALUATION_NOT_SCIENTIFIC_PERFORMANCE" if contract["status"] == "smoke" else "DEVELOPMENT_EVALUATION_NOT_FINAL_TEST",
               "reloaded_artifacts": True, "deterministic_inference": True})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=["train", "evaluate"])
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    try:
        (train if args.operation == "train" else evaluate)(args.candidate.resolve())
    except Exception as exc:
        write_json(args.candidate / f"{args.operation}-error.json", {"code": getattr(exc, "code", "TRAIN_CRASH"), "detail": str(exc)})
        raise SystemExit(1)


if __name__ == "__main__":
    main()
