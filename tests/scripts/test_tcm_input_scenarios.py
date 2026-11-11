"""任务 C 契约测试：场景表、SMOKE_ONLY、错误类型记录、不报告准确率。"""

from __future__ import annotations

import json

import pytest

from scripts import tcm_input_scenarios as scenarios

pytestmark = pytest.mark.synthetic


def test_scenarios_cover_all_four_required_categories():
    built = scenarios.build_scenarios()
    assert {s["category"] for s in built} == set(scenarios.CATEGORIES)
    assert {s["id"] for s in built} == {s["id"] for s in built}
    assert len({s["id"] for s in built}) == len(built)
    assert all(s["note"] for s in built)


def test_required_missing_body_never_zero_fills():
    built = {s["id"]: s for s in scenarios.build_scenarios()}
    body = built["missing_required_9"]["body"]
    assert len(body["features"]) == scenarios.FEATURE_DIM
    assert all(v is None for v in body["features"][:scenarios.REQUIRED_FEATURES])
    assert not any(body["feature_mask"][:scenarios.REQUIRED_FEATURES])
    assert all(body["feature_mask"][scenarios.REQUIRED_FEATURES:])


def test_extreme_bodies_use_observed_bounds_not_invented_values():
    bounds = [(lo, hi) for lo, hi in zip(range(9), range(10, 19), strict=True)]
    built = {s["id"]: s for s in scenarios.build_scenarios(bounds=bounds)}
    assert built["extreme_min"]["body"]["features"][:9] == [float(i) for i in range(9)]
    assert built["extreme_max"]["body"]["features"][:9] == [float(i) for i in range(10, 19)]


def test_observed_bounds_respects_mask(tmp_path):
    records = [
        {"features": [1.0, 5.0, None] + [0.0] * 60,
         "feature_mask": [True, True, False] + [True] * 60},
        {"features": [3.0, 7.0, None] + [0.0] * 60,
         "feature_mask": [True, True, False] + [True] * 60},
    ]
    path = tmp_path / "records.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    with pytest.raises(ValueError):
        scenarios.observed_bounds(path)  # 某维没有观测值


def test_run_scenarios_records_error_type_and_never_accuracy():
    def syndrome(text):
        if "头痛" in text:
            raise RuntimeError("stub failure")
        if "怕冷又怕热" in text:
            return {"status": "ABSTAIN", "candidates": [],
                    "backend_availability": "AVAILABLE"}
        return {"status": "OK", "candidates": [{"calibrated": False}],
                "backend_availability": "AVAILABLE"}

    def constitution(body):
        if body is None:
            return {"status": "ABSTAIN", "scores": [],
                    "backend_availability": "NOT_EVALUATED"}
        return {"status": "ABSTAIN", "scores": [],
                "reason": "特征缺失", "backend_availability": "AVAILABLE"}

    report = scenarios.run_scenarios(syndrome, constitution, scenarios.build_scenarios())
    assert report["accuracy_reported"] is False
    assert report["expert_labels"] is False
    assert report["purpose"] == "ROBUSTNESS_ONLY"
    assert report["synthetic_inputs"] == "SMOKE_ONLY"
    assert report["status"]["evidence"] == "LOCAL_EXPERIMENTAL"
    serialized = json.dumps(report, ensure_ascii=False).lower()
    assert '"accuracy"' not in serialized
    assert '"top3_accuracy"' not in serialized
    error_types = {r["models"]["syndrome"]["error_type"] for r in report["scenarios"]}
    assert "RuntimeError" in error_types
    assert any(r["models"]["syndrome"]["abstain"] is True for r in report["scenarios"])
    assert all("accuracy" not in r["models"]["syndrome"] for r in report["scenarios"])
    assert all("accuracy" not in r["models"]["constitution"] for r in report["scenarios"])


def test_report_has_no_expert_labels_or_predictions_accuracy_claim():
    report = scenarios.run_scenarios(
        lambda t: {"status": "ABSTAIN", "candidates": []},
        lambda b: {"status": "ABSTAIN", "scores": []},
        scenarios.build_scenarios())
    serialized = json.dumps(report, ensure_ascii=False).lower()
    assert '"f1"' not in serialized
    assert '"precision"' not in serialized
    assert '"accuracy"' not in serialized
    assert report["by_category"]["everyday_expression"]["n_scenarios"] == 5
    assert set(report["by_category"]) == set(scenarios.CATEGORIES)
