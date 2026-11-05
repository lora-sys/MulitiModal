"""最小中医演示的契约测试：形状、mask、负例、科研标注、拒判、生理-only 能力边界。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import tcm_minimal_demo as demo

pytestmark = pytest.mark.synthetic


def test_body_record_shape_and_mask():
    body = demo._body_record(0)
    assert len(body["features"]) == 63 and len(body["feature_mask"]) == 63
    assert all(isinstance(v, bool) for v in body["feature_mask"])
    assert "frozen subject" in body["source"]
    assert "未使用其标签分数" in body["source"]


def test_demo_body_features_research_estimate():
    result = demo.demo(0)
    assert result["schema_version"] == demo.DEMO_SCHEMA
    assert result["mode"] == "BODY_FEATURES"
    assert result["body_input"]["n_features"] == 63
    assert result["body_input"]["n_observed"] > 0
    c = result["constitution"]
    assert c["status"] == "OK"
    assert len(c["scores"]) == 8
    assert len(c["score_dims"]) == 8
    assert c["balanced"]["status"] == "ABSTAIN"
    assert c["balanced"]["value"] is None
    assert c["unobserved_dimensions"] == ["balanced"]
    assert c["evidence"] == "LOCAL_EXPERIMENTAL"
    assert c["label_tier"] == "PUBLISHED"
    assert c["champion_cv_masked_mae"] > 0
    assert all(s["calibrated"] is False for s in c["scores"])
    assert result["disclaimer"] == demo.RESEARCH_NOTE
    assert result["syndrome"] is None


def test_demo_with_symptom_text_shows_separate_candidates():
    result = demo.demo(0, "反复胸闷隐痛3年余，加重伴憋喘1周")
    s = result["syndrome"]
    assert s["status"] == "OK"
    assert 0 < len(s["candidates"]) <= 3
    assert all(c["calibrated"] is False for c in s["candidates"])
    assert "not a diagnosis" in s["note"]


def test_syndrome_candidates_reject_empty_and_out_of_domain():
    assert demo.syndrome_candidates(None)["status"] == "ABSTAIN"
    assert demo.syndrome_candidates("")["status"] == "ABSTAIN"
    assert demo.syndrome_candidates("I have a fever")["status"] == "OUT_OF_DOMAIN"
    assert demo.syndrome_candidates("气虚血瘀证")["status"] == "OUT_OF_DOMAIN"


def test_constitution_estimate_rejects_missing_and_bad_width():
    assert demo.constitution_estimate(None)["status"] == "ABSTAIN"
    all_missing = {"features": [None] * 63, "feature_mask": [False] * 63}
    assert demo.constitution_estimate(all_missing)["status"] == "ABSTAIN"
    wrong = {"features": [1.0] * 3, "feature_mask": [True] * 3}
    assert demo.constitution_estimate(wrong)["status"] == "OUT_OF_DOMAIN"
    zero_filled = {"features": [0.0] * 63, "feature_mask": [True] * 63}
    assert demo.constitution_estimate(zero_filled)["status"] == "OK"


def test_physiology_only_cannot_infer_constitution_or_syndrome():
    result = demo.physiology_demo()
    assert result["mode"] == "PHYSIOLOGY_ONLY"
    p = result["physiology"]
    assert p["status"] == "ABSTAIN"
    assert p["constitution"] == "UNAVAILABLE"
    assert p["syndrome"] == "UNAVAILABLE"
    assert "BLOCKED" in p["reason"]
    assert "NOT_AVAILABLE" in p["evidence"]
    assert "SMOKE_ONLY" in p["note"]


def test_physiology_only_never_calls_either_model(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("physiology-only must not call a model")

    monkeypatch.setattr(demo, "constitution_estimate", _boom)
    monkeypatch.setattr(demo, "syndrome_candidates", _boom)
    result = demo.physiology_demo()
    assert result["physiology"]["status"] == "ABSTAIN"


def test_no_holdout_or_training_in_demo_source():
    source = (Path(__file__).resolve().parents[2] / "scripts/tcm_minimal_demo.py").read_text(
        encoding="utf-8"
    )
    assert "test.json" not in source
    assert ".fit(" not in source
    assert "results.tsv" not in source


def test_demo_json_serializable():
    payload = json.dumps(demo.demo(0, "胸闷隐痛"), ensure_ascii=False)
    assert "health" not in payload or "LOCAL_EXPERIMENTAL" in payload
