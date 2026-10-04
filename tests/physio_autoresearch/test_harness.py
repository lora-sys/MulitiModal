from copy import deepcopy
from pathlib import Path
import subprocess
import time

import joblib
import pandas as pd
import pytest
import yaml

from physio_autoresearch.__main__ import make_synthetic
from physio_autoresearch.contracts import ResearchError, load_contract, read_json, write_json
from physio_autoresearch import harness
from physio_autoresearch.harness import attempt, ingest, load_round, round_status, run_round, run_worker
from physio_autoresearch.storage import append, exclusive, history
from physio_autoresearch.trusted.worker import evaluate
from physio_autoresearch.models import MaskImputer
from physio_autoresearch.inference import predict


@pytest.fixture
def research(tmp_path):
    # Test command sets --basetemp below ignored runs/; no external data is read.
    source = tmp_path / "synthetic.csv"
    make_synthetic(source, 5, 2)
    contract = load_contract(Path("physio_autoresearch/constitution.synthetic.yaml"))
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(yaml.safe_dump(contract), encoding="utf-8")
    store = tmp_path / "store"
    return source, contract_path, store, ingest(source, contract_path, store)


@pytest.mark.harness_bench
def test_data_version_is_idempotent_and_new_subject_triggers_new_round(research):
    source, protocol, store, old = research
    assert ingest(source, protocol, store) == old
    make_synthetic(source, 6, 2)
    new = ingest(source, protocol, store)
    assert new != old
    assert round_status(new)["qc"]["subjects"] == 6
    assert round_status(old)["qc"]["subjects"] == 5
    assert round_status(new)["attempts"] == 0


@pytest.mark.harness_bench
@pytest.mark.parametrize("filename,code", [("data.csv", "DATA_CHANGED"), ("contract.json", "CONTRACT_CHANGED")])
def test_frozen_snapshot_mutation_is_rejected(research, filename, code):
    path = research[-1]
    if filename == "data.csv":
        (path / filename).write_text("changed", encoding="utf-8")
    else:
        contract = read_json(path / filename)
        contract["budget"]["max_experiments"] += 1
        write_json(path / filename, contract)
    with pytest.raises(ResearchError, match=code):
        round_status(path)


@pytest.mark.harness_bench
def test_core_code_change_requires_new_review(research, monkeypatch):
    monkeypatch.setattr(harness, "core_hashes", lambda: {"fake.py": "changed"})
    with pytest.raises(ResearchError, match="EVALUATOR_CHANGED"):
        load_round(research[-1])


@pytest.mark.harness_bench
def test_journal_tampering_is_detected(tmp_path):
    path = tmp_path / "events.jsonl"
    append(path, {"kind": "result", "metric": 5})
    append(path, {"kind": "result", "metric": 4})
    assert len(history(path)) == 2
    path.write_text(path.read_text().replace('"metric": 5', '"metric": 0'), encoding="utf-8")
    with pytest.raises(ResearchError, match="HISTORY_CORRUPTED"):
        history(path)


@pytest.mark.harness_bench
def test_os_lock_blocks_concurrent_writer(tmp_path):
    path = tmp_path / "round.lock"
    with exclusive(path):
        with pytest.raises(ResearchError, match="ROUND_BUSY"):
            with exclusive(path):
                pytest.fail("second writer entered")
    with exclusive(path):
        pass  # lock released independently of the persistent lock file


@pytest.mark.harness_bench
def test_failure_records_and_recovers_without_changing_champion(research, monkeypatch):
    path = research[-1]
    first = attempt(path, {"model": "naive"})
    assert first["decision"] == "KEEP"
    original = harness.run_worker
    def crash(*args):
        raise ResearchError("TRAIN_CRASH", "injected failure")
    monkeypatch.setattr(harness, "run_worker", crash)
    failure = attempt(path, {"model": "elastic_net", "params": {"alpha": 2}})
    assert failure["reason"] == "TRAIN_CRASH"
    assert round_status(path)["champion"]["attempt"] == first["attempt"]
    monkeypatch.setattr(harness, "run_worker", original)
    recovered = attempt(path, {"model": "elastic_net", "params": {"alpha": 1}})
    assert recovered["evaluation"]["reloaded_artifacts"]
    duplicate = attempt(path, {"model": "elastic_net", "params": {"alpha": 1}, "hypothesis": "same config again"})
    assert duplicate["reason"] == "DUPLICATE_EXPERIMENT"
    assert round_status(path)["attempts"] == 4


@pytest.mark.harness_bench
def test_budget_persists_across_resume(research):
    source, protocol, store, _ = research
    contract = load_contract(protocol)
    contract["budget"]["max_experiments"] = 2
    protocol.write_text(yaml.safe_dump(contract), encoding="utf-8")
    path = ingest(source, protocol, store)
    attempt(path, {"model": "naive"})
    attempt(path, {"model": "naive"})
    assert run_round(path)["attempts"] == 2
    with pytest.raises(ResearchError, match="OVER_BUDGET"):
        attempt(path, {"model": "elastic_net"})


@pytest.mark.harness_bench
def test_interrupted_reservation_fails_closed(research):
    path = research[-1]
    append(path / "experiments.jsonl", {"kind": "start", "attempt": 1})
    with pytest.raises(ResearchError, match="INTERRUPTED_ROUND"):
        attempt(path, {"model": "naive"})


@pytest.mark.harness_bench
def test_worker_timeout_terminates_child(tmp_path):
    start = time.monotonic()
    with pytest.raises(ResearchError, match="TRAIN_TIMEOUT"):
        run_worker("train", tmp_path, 0.001)
    assert time.monotonic() - start < 5


@pytest.mark.harness_bench
def test_locked_test_and_output_path_are_outside_agent_api(tmp_path):
    with pytest.raises(ResearchError, match="FORBIDDEN_TEST_ACCESS"):
        round_status(Path("physio_autoresearch/locked_test"))
    with pytest.raises(ResearchError, match="UNSAFE_OUTPUT_PATH"):
        harness.safe_store(Path("physio_autoresearch/private-data"))


@pytest.mark.harness_bench
def test_corrupt_checkpoint_is_not_scored(research):
    path = research[-1]
    result = attempt(path, {"model": "naive"})
    candidate = Path(result["artifact_directory"])
    (candidate / "fold-000.joblib").write_bytes(b"broken")
    with pytest.raises(ResearchError, match="INVALID_CHECKPOINT"):
        evaluate(candidate)


def test_all_four_families_and_fold_only_preprocessing(research):
    path = research[-1]
    plan = [{"model": "naive"}, {"model": "elastic_net", "params": {"alpha": 1}},
            {"model": "random_forest", "params": {"n_estimators": 10, "max_depth": 1}},
            {"model": "xgboost", "params": {"n_estimators": 5, "max_depth": 1}}]
    status = run_round(path, plan)
    assert status["attempts"] == 4
    assert all(row["evaluation"] is not None for row in status["recent_results"])
    assert all(row["evaluation"]["rows"] == 10 for row in status["recent_results"])
    result = status["recent_results"][1]
    candidate = Path(result["artifact_directory"])
    request = read_json(candidate / "request.json")
    artifact = read_json(candidate / "artifacts.json")["folds"][0]
    model = joblib.load(candidate / artifact["filename"])
    frame = pd.read_csv(path / "data.csv").set_index("record_id")
    expected = frame.loc[artifact["train_ids"], request["candidate"]["features"]].median().to_numpy()
    assert (model.named_steps["missing"].statistics_ == expected).all()
    # Repeating the same plan neither resets the budget nor repeats training.
    assert run_round(path, plan)["attempts"] == 4


def test_one_subject_never_receives_validation_metric(research):
    source, protocol, store, _ = research
    make_synthetic(source, 1)
    path = ingest(source, protocol, store)
    result = attempt(path, {"model": "naive"})
    assert result["evaluation"]["metric"] is None
    assert result["reason"] == "FIT_ONLY_NOT_VALIDATED"
    assert round_status(path)["champion"] is None


@pytest.mark.harness_bench
def test_validation_only_missingness_always_has_mask():
    transformer = MaskImputer().fit([[1, 2], [3, 4]])
    values = transformer.transform([[float("nan"), 4]])
    assert values.shape == (1, 4)
    assert values.tolist() == [[2, 4, 1, 0]]


@pytest.mark.harness_bench
def test_rejected_first_attempt_cannot_bypass_baseline(research):
    path = research[-1]
    assert attempt(path, {"model": "elastic_net"})["reason"] == "BASELINE_REQUIRED"
    assert attempt(path, {"model": "elastic_net", "params": {"alpha": 2}})["reason"] == "BASELINE_REQUIRED"
    assert round_status(path)["champion"] is None
    assert attempt(path, {"model": "naive"})["decision"] == "KEEP"
    assert attempt(path, {"model": "elastic_net", "params": {"alpha": 2}})["evaluation"] is not None


@pytest.mark.harness_bench
def test_failed_baseline_can_be_retried_once(research, monkeypatch):
    path = research[-1]
    original = harness.run_worker
    monkeypatch.setattr(harness, "run_worker", lambda *a: (_ for _ in ()).throw(ResearchError("TRAIN_CRASH", "injected")))
    assert attempt(path, {"model": "naive"})["reason"] == "TRAIN_CRASH"
    monkeypatch.setattr(harness, "run_worker", original)
    assert attempt(path, {"model": "naive"})["decision"] == "KEEP"


@pytest.mark.harness_bench
def test_reviewed_code_upgrade_preserves_budget_and_old_results(research, monkeypatch):
    source, protocol, store, path = research
    attempt(path, {"model": "naive"})
    actual = harness.environment
    def upgraded():
        value = actual()
        value["versions"]["review-test"] = "new"
        return value
    monkeypatch.setattr(harness, "environment", upgraded)
    with pytest.raises(ResearchError, match="EVALUATOR_CHANGED"):
        ingest(source, protocol, store)
    assert ingest(source, protocol, store, review_code_upgrade=True) == path
    assert round_status(path)["attempts"] == 1
    assert round_status(path)["champion"] is None
    assert len([e for e in history(path / "experiments.jsonl") if e["kind"] == "code_upgrade"]) == 1
    assert attempt(path, {"model": "naive"})["decision"] == "KEEP"
    assert round_status(path)["attempts"] == 2


def test_champion_pre_only_inference_and_artifact_size(research):
    source, _, _, path = research
    attempt(path, {"model": "naive"})
    frame = pd.read_csv(source).drop(columns="post_hr")
    inputs, output = source.parent / "pre-only.csv", path / "inference" / "predictions.csv"
    frame.to_csv(inputs, index=False)
    result = predict(path, inputs, output)
    assert result["rows"] == 10 and result["model_bytes"] > 0
    assert pd.read_csv(output).prediction_post_hr.notna().all()
    frame["post_hr"] = 1
    frame.to_csv(inputs, index=False)
    with pytest.raises(ResearchError, match="SCHEMA_DRIFT"):
        predict(path, inputs, output)


@pytest.mark.harness_bench
def test_model_time_and_failures_count_toward_total_runtime(research):
    path = research[-1]
    contract = read_json(path / "contract.json")
    usage = {"kind": "usage", "request": 1, "elapsed_seconds": contract["budget"]["max_seconds"],
             "usage": {"totalTokens": 10, "cost": {"total": 0.01}}}
    (path / "pi-usage.jsonl").write_text(__import__("json").dumps(usage) + "\n", encoding="utf-8")
    status = round_status(path)
    assert status["stop_reason"] == "OVER_BUDGET"
    assert status["model_usage"]["tokens"] == 10
    with pytest.raises(ResearchError, match="OVER_BUDGET"):
        attempt(path, {"model": "naive"})


def test_multi_outcome_artifacts_are_independently_scored(research):
    source, protocol, store, _ = research
    contract = load_contract(protocol)
    contract["targets"].append({"name": "post_bp", "phase": "post", "unit": "mmHg", "scale": 20})
    protocol.write_text(yaml.safe_dump(contract), encoding="utf-8")
    frame = pd.read_csv(source)
    frame["post_bp"] = 100 + frame["age"] * 0.1  # synthetic demonstration only
    frame.to_csv(source, index=False)
    path = ingest(source, protocol, store)
    state = run_round(path, [{"model": "naive"}, {"model": "elastic_net", "params": {"alpha": 1}}])
    assert state["attempts"] == 2
    for result in state["recent_results"]:
        assert set(result["evaluation"]["per_target_mae"]) == {"post_hr", "post_bp"}
        assert result["evaluation"]["reloaded_artifacts"]
