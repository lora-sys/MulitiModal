"""Reproducible synthetic V0 acceptance experiment, never a physiological study."""

from pathlib import Path
import json

import pandas as pd

from physio_autoresearch.__main__ import make_synthetic
from physio_autoresearch.contracts import candidate_id, validate_candidate, write_json
from physio_autoresearch.harness import ROOT, environment, fixed_search, ingest, load_round, now, results, run_round
from physio_autoresearch.inference import predict


def main():
    store = ROOT / "runs" / "physio-v0"
    protocol = ROOT / "physio_autoresearch" / "constitution.synthetic.yaml"
    report = {"created_at": now(), "environment": environment(), "source": "synthetic", "validated_physiology": False, "rounds": []}
    paths = []
    for subjects in (5, 6):
        data = store / f"synthetic-{subjects}.csv"
        make_synthetic(data, subjects)
        path = ingest(data, protocol, store)
        state = run_round(path)
        _, contract, entries = load_round(path)
        trials = results(entries)
        # Later AI proposals share this round and its budget. Identify the fixed
        # comparison by semantic candidate ID instead of assuming no later trials.
        fixed_ids = {candidate_id(validate_candidate(p, contract)) for p in fixed_search(contract)}
        fixed_trials = [t for t in trials if t["candidate_sha256"] in fixed_ids and t["evaluation"] is not None]
        assert len(fixed_trials) == len(fixed_ids) == 12
        assert {t["candidate"]["model"] for t in fixed_trials} == {"naive", "elastic_net", "random_forest", "xgboost"}
        assert state["champion"] and not state["pending_attempts"]
        # Resume the same data version without new fits or budget reset.
        assert run_round(path)["attempts"] == state["attempts"]
        report["rounds"].append({"subjects": subjects, "data_version": state["data_version"], "path": str(path),
          "attempts": state["attempts"], "elapsed_seconds": state["elapsed_seconds"],
          "kept": sum(t["decision"] == "KEEP" for t in trials), "reverted": sum(t["decision"] == "REVERT" for t in trials),
          "naive_mae": trials[0]["evaluation"]["per_target_mae"]["post_hr"],
          "champion_model": state["champion"]["candidate"]["model"],
          "champion_mae": state["champion"]["evaluation"]["per_target_mae"]["post_hr"],
          "fixed_search": {"completed": len(fixed_trials),
                           "best_mae": min(t["evaluation"]["per_target_mae"]["post_hr"] for t in fixed_trials)},
          "additional_attempts": state["attempts"] - len(fixed_trials),
          "status": state["model_status"], "resume_added_attempts": 0})
        paths.append(path)
        print(f"{subjects} synthetic subjects: {state['attempts']} attempts, {state['champion']['candidate']['model']}, SMOKE ONLY", flush=True)
    assert paths[0] != paths[1]
    tiny = store / "synthetic-1.csv"
    make_synthetic(tiny, 1)
    tiny_path = ingest(tiny, protocol, store)
    state = run_round(tiny_path, [{"model": "naive"}])
    assert state["champion"] is None and state["recent_results"][0]["evaluation"]["metric"] is None
    report["one_subject"] = {"path": str(tiny_path), "metric": None, "champion": None, "status": "FIT_ONLY_NOT_VALIDATED"}
    frame = pd.read_csv(store / "synthetic-6.csv").drop(columns="post_hr")
    inputs = store / "pre-only-input.csv"
    frame.to_csv(inputs, index=False)
    report["inference"] = predict(paths[1], inputs, paths[1] / "inference" / "predictions.csv")
    report["inference"]["input_scope"] = "six development rows, pipeline smoke only"
    # Preserve the initial acceptance snapshot after subsequent AI experiments.
    report_path = store / ("experiment-replay-report.json" if (store / "experiment-report.json").exists() else "experiment-report.json")
    write_json(report_path, report)
    print(json.dumps({"report": str(report_path), "one_subject_metric": None, "inference_rows": report["inference"]["rows"]}, indent=2))


if __name__ == "__main__":
    main()
