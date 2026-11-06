"""任务 03 契约测试：冻结清单、评估口径、排除计数与纯合成流程演练。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from syndrome import external_validation as ev

pytestmark = pytest.mark.synthetic


def _freeze(n_classes: int = 3) -> dict:
    freeze = ev.smoke_freeze()
    freeze["frozen"]["n_classes"] = n_classes
    return freeze


def _record(index: int, label: str, **overrides) -> dict:
    record = {
        "subject_id": f"smoke-{index:03d}",
        "symptom_text": "最近容易累、睡不好、肩膀发紧（合成文本）",
        "recruitment_stratum": "fatigue",
        "consent": True,
        "expert_status": "DETERMINED",
        "expert_labels": [label],
        "primary_label": label,
        "annotator_count": 2,
        "reference_source": "TCM_PROFESSIONAL",
        "form_version": "symptom-form-v1-synthetic",
    }
    return {**record, **overrides}


def _metrics_fixture():
    labels = ["A", "B", "C"]
    records = [_record(1, "A"), _record(2, "A"), _record(3, "B"), _record(4, "C")]
    score_rows = [[0.9, 0.05, 0.05], [0.35, 0.6, 0.05],
                  [0.2, 0.45, 0.35], [0.6, 0.3, 0.1]]
    predictions = {
        record["subject_id"]: ev.prediction_from_scores(row, 0.5, len(labels))
        for record, row in zip(records, score_rows, strict=True)
    }
    return records, predictions, labels


def test_metric_definitions_match_hand_computed_values():
    records, predictions, labels = _metrics_fixture()
    report = ev.evaluate(records, predictions, labels, _freeze(), freeze_verified=False)
    metrics = report["metrics"]
    assert metrics["top1_accuracy"] == 0.5
    assert metrics["top3_accuracy"] == 1.0
    assert metrics["macro_f1"] == 0.388889
    assert metrics["rejection_rate"] == 0.25
    assert metrics["decision_coverage"] == 0.75
    assert metrics["n_classes_with_effective_samples"] == 3
    per_class = report["per_class"]
    assert [row["syndrome"] for row in per_class] == labels
    assert per_class[0]["effective_sample_size"] == 2
    assert per_class[0]["tp"] == 1 and per_class[0]["fn"] == 1
    assert per_class[0]["recall"] == 0.5
    assert per_class[2]["recall"] == 0.0
    assert all(row["recall_ci95"][0] is not None for row in per_class)
    assert per_class[0]["top3_hit_rate"] == 1.0
    assert report["status"]["external_validation"] == "NOT_PERFORMED"
    assert report["status"]["use_conditions_met"] is False
    assert report["status"]["post_hoc_tuning"] is False
    assert "n_disagreement" in report["counts"]


def test_exclusions_are_counted_separately_and_not_silently_dropped():
    records, predictions, labels = _metrics_fixture()
    records += [
        _record(5, "A", consent=False),
        _record(6, "A", expert_status="UNDETERMINABLE"),
        _record(7, "A", expert_status="DISAGREEMENT"),
        _record(8, "A", primary_label="NOT_IN_TAXONOMY"),
        _record(9, "A", reference_source="MODEL"),
        _record(10, "A"),  # deliberately no prediction
    ]
    report = ev.evaluate(records, predictions, labels, _freeze(), freeze_verified=False)
    counts = report["counts"]
    assert counts["n_records"] == 10
    assert counts["n_no_consent"] == 1
    assert counts["n_undeterminable"] == 1
    assert counts["n_disagreement"] == 1
    assert counts["n_unlabeled"] == 1
    assert counts["n_invalid_reference"] == 1
    assert counts["n_missing_prediction"] == 1
    assert counts["n_excluded_total"] == 6
    assert counts["n_evaluated"] == 4
    assert counts["n_records"] == counts["n_evaluated"] + counts["n_excluded_total"]
    assert counts["status_all_with_predictions"]["UNCERTAIN"] == 1


def test_evaluate_rejects_model_reference_and_premapped_syndrome():
    records, predictions, labels = _metrics_fixture()
    with pytest.raises(ValueError, match="pre-map symptoms to a syndrome"):
        ev.evaluate([_record(1, "A", syndrome="气虚证")], predictions, labels, _freeze())
    with pytest.raises(ValueError, match="duplicate subject_id"):
        ev.evaluate([_record(1, "A"), _record(1, "A")], predictions, labels, _freeze())
    report = ev.evaluate([_record(11, "A", reference_source="LLM_SILVER")],
                         {}, labels, _freeze())
    assert report["counts"]["n_invalid_reference"] == 1
    assert report["metrics"]["top1_accuracy"] is None


def test_prediction_tie_break_is_deterministic_and_bounds_are_checked():
    row = [0.4, 0.4, 0.2]
    prediction = ev.prediction_from_scores(row, 0.5, 3)
    assert prediction["top1_index"] == 0
    assert prediction["top3_indices"] == [0, 1, 2]
    assert prediction["status"] == "UNCERTAIN"
    assert ev.prediction_from_scores([0.9, 0.05, 0.05], 0.5, 3)["status"] == "OK"
    with pytest.raises(ValueError, match="cover the frozen class taxonomy"):
        ev.prediction_from_scores([0.9, 0.1], 0.5, 3)
    with pytest.raises(ValueError, match="cover the frozen class taxonomy"):
        ev.prediction_from_scores([0.9] * 3, 0.5, 148)
    rows = [0.01] * 148
    rows[147] = 0.9
    assert ev.prediction_from_scores(rows, 0.5)["top1_index"] == 147


def test_wilson_interval_handles_zero_trials_and_full_success():
    assert ev.wilson_interval(0, 0) == [None, None]
    low, high = ev.wilson_interval(30, 30)
    assert low > 0.85 and high == 1.0
    low, high = ev.wilson_interval(1, 2)
    assert 0.0 < low < 0.5 < high < 1.0


def test_freeze_verification_rejects_artifact_and_code_drift(tmp_path, monkeypatch):
    artifact = tmp_path / "artifact"
    raw = tmp_path / "raw"
    artifact.mkdir()
    raw.mkdir()
    for name in ev.FROZEN_ARTIFACT_SHA256:
        target = raw / name if name == "syndrome_vocab.txt" else artifact / name
        target.write_text("synthetic bytes\n")
        monkeypatch.setitem(ev.FROZEN_ARTIFACT_SHA256, name, ev.sha256(target))
    freeze = ev.build_freeze(artifact, raw)
    assert ev.verify_freeze(freeze, artifact, raw)["verified"] is True
    assert freeze["frozen"]["threshold"] == ev.FROZEN_THRESHOLD
    assert freeze["frozen"]["n_classes"] == 148
    assert freeze["split"]["train"] == 0 and freeze["split"]["holdout"] == 0
    (artifact / "model.joblib").write_text("changed\n")
    with pytest.raises(ValueError, match="does not match the current artifact"):
        ev.verify_freeze(freeze, artifact, raw)
    (artifact / "model.joblib").write_text("synthetic bytes\n")
    with pytest.raises(ValueError, match="threshold drifted"):
        ev.verify_freeze({**freeze, "frozen": {**freeze["frozen"], "threshold": 0.9}},
                         artifact, raw)
    with pytest.raises(ValueError, match="evaluation_code_sha256"):
        ev.verify_freeze(
            {**freeze, "frozen": {**freeze["frozen"],
                                  "evaluation_code_sha256": "0" * 64}}, artifact, raw)
    with pytest.raises(ValueError, match="unsupported freeze manifest schema"):
        ev.verify_freeze({"schema_version": "wrong"}, artifact, raw)
    with pytest.raises(ValueError, match="does not pin the external dataset"):
        ev.verify_freeze(freeze, artifact, raw, tmp_path / "records.json")
    with pytest.raises(ValueError, match="model_id drifted"):
        ev.verify_freeze(ev.smoke_freeze(), artifact, raw)


def test_smoke_test_flow_has_no_real_patient_data(tmp_path):
    records, predictions, labels = ev.smoke_fixture()
    freeze = ev.smoke_freeze()
    report = ev.evaluate(records, predictions, labels, freeze,
                         freeze_verified=False, min_subjects=100)
    out = tmp_path / "smoke.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    assert report["status"]["evidence"] == "SMOKE_ONLY"
    assert report["status"]["go_research_claims"] == "NOT_GRANTED"
    assert report["counts"]["n_evaluated"] == 10
    assert report["conditions"]["freeze_verified"] is False
    assert report["status"]["use_conditions_met"] is False
    assert len(report["per_class"]) == 6
    assert all("smoke-" in record["subject_id"] for record in records)
    assert json.loads(out.read_text(encoding="utf-8"))["schema_version"] == ev.VALIDATION_SCHEMA


def test_only_inference_and_documented_frozen_constants(tmp_path):
    source = (Path(__file__).resolve().parents[2]
              / "src/syndrome/external_validation.py").read_text(encoding="utf-8")
    assert ".fit(" not in source
    assert "test.json" not in source
    assert "predict_proba" in source
    assert "NOT_GRANTED" in source
    assert ev.FROZEN_THRESHOLD == 0.27808433274565425
    assert ev.FROZEN_MODEL_ID == "syndrome-tfidf-charwb-1-4-c10-ar1"
    assert ev.N_CLASSES == 148
    assert ev.LOW_SAMPLE_THRESHOLD == 30
    assert set(ev.RECRUITMENT_STRATA) == {
        "fatigue", "sleep_problem", "shoulder_neck_discomfort", "other"}
    assert "TCM_PROFESSIONAL" in ev.REFERENCE_SOURCES
