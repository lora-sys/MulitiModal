"""任务 A 回归：证候推理的异常输入、缺失/损坏工件、直接构造坏请求。

全部 SYNTHETIC，不需要真实工件。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts import tcm_minimal_demo as demo
from syndrome.inference import SyndromeInference
from syndrome.labels import LabelSpace
from syndrome.service import (
    ModelRegistry,
    SymptomAnalysisRequest,
    SyndromeService,
)

pytestmark = pytest.mark.synthetic

LABELS = ("气虚血瘀证", "湿热蕴结证", "阳虚寒凝证")


class _StubPredictor:
    score_semantics = "stub scores, not probabilities"

    def predict_proba(self, texts):
        out = np.full((len(texts), len(LABELS)), 1.0 / len(LABELS))
        return out


def _inference():
    return SyndromeInference(
        _StubPredictor(), LabelSpace(LABELS), model_version="synthetic",
        provenance={}, threshold=0.0,
    )


@pytest.mark.parametrize("bad", [42, 1.5, True, b"abc", [], {}])
def test_non_string_text_is_explicitly_rejected(bad):
    result = _inference().predict_one(bad)  # 修复前 AttributeError/TypeError
    assert result.status == "OUT_OF_DOMAIN"
    assert "OUT_OF_DOMAIN" in result.message
    assert result.candidates == []


@pytest.mark.parametrize("bad_fields", [42, "x", [1, 2]])
def test_bad_fields_are_rejected(bad_fields):
    result = _inference().predict_one("胸闷隐痛", bad_fields)
    assert result.status == "OUT_OF_DOMAIN"


def test_predict_batch_mixed_bad_types_does_not_crash():
    inf = _inference()
    results = inf.predict_batch(["胸闷隐痛", 42, None, b"abc", "心悸失眠"])
    assert len(results) == 5
    assert results[1].status == "OUT_OF_DOMAIN"
    assert results[3].status == "OUT_OF_DOMAIN"
    assert results[0].status in ("OK", "UNCERTAIN")


def test_register_none_spec_raises_clear_error():
    with pytest.raises(ValueError, match="None"):
        ModelRegistry().register(None, default=True)


@pytest.mark.parametrize("bad", [42, 1.5, b"abc", []])
def test_service_direct_constructor_bad_text_fails_closed(bad):
    service = SyndromeService(registry=ModelRegistry())  # 无后端
    response = service.analyze_symptoms(SymptomAnalysisRequest(symptom_text=bad))
    assert response.status in ("ABSTAIN", "OUT_OF_DOMAIN")
    assert response.candidates == []


def test_demo_missing_syndrome_artifact_fails_closed(tmp_path):
    result = demo.syndrome_candidates("胸闷隐痛", str(tmp_path / "nope"))
    assert result["status"] == "ABSTAIN"
    assert result["candidates"] == []
    assert result["backend_availability"] == "NOT_AVAILABLE"
    assert "缺失" in result["reason"]


def test_demo_corrupt_syndrome_registry_fails_closed(tmp_path):
    art = tmp_path / "art"
    art.mkdir()
    (art / "registry_entry.json").write_text("NOT JSON", encoding="utf-8")
    result = demo.syndrome_candidates("胸闷隐痛", str(art))
    assert result["status"] == "ABSTAIN"
    assert result["backend_availability"] == "NOT_AVAILABLE"
    assert "损坏" in result["reason"]


def test_demo_missing_constitution_artifact_fails_closed(tmp_path):
    row = {"features": [1.0] * 63, "feature_mask": [True] * 63}
    result = demo.constitution_estimate(row, tmp_path / "nope")
    assert result["status"] == "ABSTAIN"
    assert result["scores"] == []
    assert result["backend_availability"] == "NOT_AVAILABLE"


def test_demo_bad_body_type_is_rejected():
    assert demo.constitution_estimate(42)["status"] == "OUT_OF_DOMAIN"
    assert demo.constitution_estimate("hello")["status"] == "OUT_OF_DOMAIN"
    assert demo.syndrome_candidates(42)["status"] == "OUT_OF_DOMAIN"


def test_demo_source_has_no_main_alias_shim():
    source = (Path(__file__).resolve().parents[2] / "scripts/tcm_minimal_demo.py").read_text(
        encoding="utf-8")
    assert "sys.modules[\"__main__\"]" not in source
    assert "_load_constitution_artifact" not in source
    assert "load_artifact(" in source
