"""Reload the locally accepted full-fit model for pre-only inference."""

from pathlib import Path
import time

import joblib
import pandas as pd

from .contracts import ResearchError, file_hash, read_json
from .data import parse_data
from .harness import ROOT, champion_from, load_round, resolve_round
from .trusted.worker import check_predictions


def predict(round_path: Path, inputs: Path, output: Path) -> dict:
    round_path = resolve_round(round_path)
    if not output.resolve().is_relative_to(round_path / "inference") or output.resolve() == inputs.resolve():
        raise ResearchError("UNSAFE_OUTPUT_PATH", "prediction output must be inside the round's inference/ directory")
    _, contract, entries = load_round(round_path)
    champion = champion_from(entries)
    if champion is None:
        raise ResearchError("NO_CHAMPION", "no independently scored candidate is available")
    candidate = round_path / "candidates" / f"{champion['attempt']:03d}"
    artifacts = read_json(candidate / "artifacts.json")
    if artifacts["candidate_sha256"] != champion["candidate_sha256"] or file_hash(candidate / "model.joblib") != artifacts["model_sha256"]:
        raise ResearchError("INVALID_CHECKPOINT", "champion model integrity check failed")
    frame = parse_data(inputs.read_bytes(), contract, prediction_only=True)
    model = joblib.load(candidate / "model.joblib")
    started = time.perf_counter()
    prediction = check_predictions(model.predict(frame[champion["candidate"]["features"]]), len(frame), len(contract["targets"]))
    elapsed = time.perf_counter() - started
    result = frame[[contract["record_id"], contract["subject_id"]]].copy()
    for number, target in enumerate(contract["targets"]):
        result[f"prediction_{target['name']}"] = prediction[:, number]
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return {"rows": len(result), "output": str(output.resolve()), "inference_seconds": elapsed,
            "timing_scope": "batch prediction only; excludes loading and preprocessing validation",
            "model_bytes": (candidate / "model.joblib").stat().st_size,
            "status": "SMOKE_MODEL_NOT_VALIDATED" if contract["status"] == "smoke" else "DEVELOPMENT_MODEL_NOT_FINAL_TEST"}
