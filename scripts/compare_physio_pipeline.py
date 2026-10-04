"""Replay the fixed six-subject PR20 search and compare every held-out prediction."""

import hashlib
import json
import subprocess

import numpy as np
import pandas as pd

from physio_autoresearch.contracts import file_hash, read_json, write_json
from physio_autoresearch.harness import ROOT, environment, ingest, load_round, run_round
from physio_autoresearch.storage import history

BASELINE = "d6abaf1065d6e2359c37a80b96d5902e5baae4b8"


def main():
    baseline_store = ROOT / "runs/physio-v0"
    baseline = read_json(baseline_store / "experiment-report.json")["rounds"][1]
    baseline_round = baseline_store / "rounds" / baseline["data_version"]
    manifest = read_json(baseline_round / "manifest.json")
    assert manifest["source"] == "synthetic" and manifest["qc"]["subjects"] == baseline["subjects"] == 6
    for name, saved_hash in manifest["core_hashes"].items():
        filename = name.replace("\\", "/")
        source = subprocess.check_output(["git", "show", f"{BASELINE}:physio_autoresearch/{filename}"], cwd=ROOT)
        assert hashlib.sha256(source).hexdigest() == saved_hash, f"baseline source differs: {filename}"
    assert file_hash(baseline_round / "data.csv") == manifest["data_sha256"]
    before = {entry["candidate_sha256"]: entry for entry in history(baseline_round / "experiments.jsonl")
              if entry["kind"] == "result" and entry["evaluation"] is not None}
    assert len(before) == 12
    store = ROOT / "runs/optimization/pipeline-comparison"
    after_round = ingest(baseline_round / "data.csv", ROOT / "physio_autoresearch/constitution.synthetic.yaml", store)
    status = run_round(after_round)
    after_manifest, contract, entries = load_round(after_round)
    assert after_manifest["data_sha256"] == manifest["data_sha256"]
    assert after_manifest["contract_sha256"] == manifest["contract_sha256"]
    assert after_manifest["qc"]["split_sha256"] == manifest["qc"]["split_sha256"]
    assert after_manifest["environment"]["versions"] == manifest["environment"]["versions"]
    after = {entry["candidate_sha256"]: entry for entry in entries if entry["kind"] == "result" and entry["evaluation"] is not None}
    assert before.keys() == after.keys()
    cases = []
    for key, old in before.items():
        new = after[key]
        old_dir = baseline_round / "candidates" / f"{old['attempt']:03d}"
        new_dir = after_round / "candidates" / f"{new['attempt']:03d}"
        old_predictions = pd.read_csv(old_dir / "predictions.csv").set_index(contract["record_id"]).sort_index()
        new_predictions = pd.read_csv(new_dir / "predictions.csv").set_index(contract["record_id"]).sort_index()
        assert old_predictions.index.equals(new_predictions.index)
        np.testing.assert_allclose(new_predictions, old_predictions, rtol=0, atol=0)
        assert new["evaluation"] == old["evaluation"]
        assert new["decision"] == old["decision"]
        old_folds = read_json(old_dir / "artifacts.json")["folds"]
        new_folds = read_json(new_dir / "artifacts.json")["folds"]
        assert [{k: fold[k] for k in ("train_ids", "validation_ids", "validation_subject")} for fold in old_folds] == [
            {k: fold[k] for k in ("train_ids", "validation_ids", "validation_subject")} for fold in new_folds]
        cases.append({"candidate_sha256": key, "model": new["candidate"]["model"], "decision": new["decision"],
                      "metric_before": old["evaluation"]["metric"], "metric_after": new["evaluation"]["metric"],
                      "post_hr_mae_before": old["evaluation"]["per_target_mae"]["post_hr"],
                      "post_hr_mae_after": new["evaluation"]["per_target_mae"]["post_hr"], "max_prediction_delta": 0.0,
                      "model_bytes_before": (old_dir / "model.joblib").stat().st_size,
                      "model_bytes_after": (new_dir / "model.joblib").stat().st_size})
    assert run_round(after_round)["attempts"] == status["attempts"] == 12
    report = {"baseline_commit": BASELINE, "baseline_source_verified": True,
              "baseline_recorded_environment": manifest["environment"], "environment": environment(),
              "data_sha256": manifest["data_sha256"], "contract_sha256": manifest["contract_sha256"],
              "split_sha256": manifest["qc"]["split_sha256"], "seed": contract["seed"],
              "source": manifest["source"], "subjects": manifest["qc"]["subjects"], "candidates": cases, "resume_added_attempts": 0,
              "scope": "same fixed search, inputs, folds and predictions; no final test or physiological accuracy claim"}
    output = store / "comparison.json"
    write_json(output, report)
    print(json.dumps({"report": str(output), "candidates": len(cases), "max_prediction_delta": 0,
                      "champion_mae": status["champion"]["evaluation"]["per_target_mae"]["post_hr"]}))


if __name__ == "__main__":
    main()
