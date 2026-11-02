"""#58 双模式服务契约测试：schema 往返、后端可替换、来源分区、fail-closed、
实时模式绝不调用分类器（结构性断言 + 行为断言）。全部合成数据。"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from syndrome.feedback import MeasurementSample
from syndrome.labels import STATUS_ABSTAIN, STATUS_OK, LabelSpace
from syndrome.service import (
    BACKEND_NOT_AVAILABLE,
    BackendSpec,
    LoadedBackend,
    ModelRegistry,
    RealtimeKnowledgeRequest,
    RealtimeKnowledgeResponse,
    SymptomAnalysisRequest,
    SymptomAnalysisResponse,
    SyndromeService,
    save_tfidf_logreg_artifact,
    tfidf_logreg_spec,
)

pytestmark = pytest.mark.synthetic

SRC_FEEDBACK = Path(__file__).resolve().parents[2] / "src" / "syndrome" / "feedback.py"


class HashPredictor:
    """确定性假模型：文本 crc32 → 词表内 argmax；两个后端用不同 scale 区分。"""

    score_semantics = "SYNTHETIC hash scores; uncalibrated"

    def __init__(self, label_space: LabelSpace, scale: float = 0.9) -> None:
        self.label_space = label_space
        self.scale = scale
        self.calls: list[list[str]] = []

    def predict_proba(self, texts: list[str]):
        import zlib

        import numpy as np

        self.calls.append(list(texts))
        rows = np.zeros((len(texts), len(self.label_space)), dtype=np.float64)
        for row, text in enumerate(texts):
            idx = zlib.crc32(text.encode("utf-8")) % len(self.label_space)
            rows[row, idx] = self.scale
            rows[row] += (1.0 - rows[row].sum()) / len(self.label_space)
        return rows


def make_spec(model_id: str, label_space: LabelSpace, scale: float = 0.9) -> BackendSpec:
    predictor = HashPredictor(label_space, scale=scale)

    def loader() -> LoadedBackend:
        return LoadedBackend(
            predictor=predictor,
            label_space=label_space,
            threshold=0.0,
            threshold_scope="UNVERIFIED",
            provenance={"source": "SYNTHETIC"},
        )

    return BackendSpec(
        model_id=model_id,
        model_version=f"{model_id}-v1",
        taxonomy_version=label_space.taxonomy_id,
        loader=loader,
        data_license="SYNTHETIC (no license grant)",
        provenance={"source": "SYNTHETIC"},
    )


def make_service(
    label_space: LabelSpace, topic_knowledge=None, n_backends: int = 1
) -> SyndromeService:
    registry = ModelRegistry()
    for i in range(n_backends):
        registry.register(make_spec(f"fake-{i}", label_space, scale=0.9 - 0.1 * i))
    return SyndromeService(registry=registry, topic_knowledge=topic_knowledge)


# ------------------------------------------------------------------- schema 往返


def test_symptom_request_roundtrip():
    request = SymptomAnalysisRequest(
        symptom_text="刺痛固定", fields={"age": 40}, model_id="fake-0"
    )
    assert SymptomAnalysisRequest.from_dict(request.to_dict()) == request


def test_realtime_request_roundtrip():
    request = RealtimeKnowledgeRequest(
        session_id="s-1",
        timestamp="2026-11-01T10:00:00+08:00",
        measurements=[
            MeasurementSample(
                source="band", metric="heart_rate", value=72.0, unit="bpm",
                timestamp="2026-11-01T10:00:00+08:00", quality="OK",
            )
        ],
        physio_summary={"n": 1},
    )
    assert RealtimeKnowledgeRequest.from_dict(request.to_dict()) == request


def test_symptom_response_roundtrip(label_space):
    service = make_service(label_space)
    response = service.analyze_symptoms(SymptomAnalysisRequest(symptom_text="刺痛固定"))
    assert SymptomAnalysisResponse.from_dict(response.to_dict()) == response


def test_realtime_response_roundtrip(label_space, fake_topic_knowledge):
    service = make_service(label_space, topic_knowledge=fake_topic_knowledge)
    request = RealtimeKnowledgeRequest(session_id="s-1", timestamp="t")
    response = service.realtime_knowledge(request)
    assert RealtimeKnowledgeResponse.from_dict(response.to_dict()) == response


def test_schema_version_mismatch_rejected(label_space):
    with pytest.raises(ValueError, match="schema_version"):
        SymptomAnalysisRequest.from_dict({"symptom_text": "x", "schema_version": "v0"})
    with pytest.raises(ValueError, match="schema_version"):
        RealtimeKnowledgeRequest.from_dict({"session_id": "s", "schema_version": "v0"})
    service = make_service(label_space)
    request = SymptomAnalysisRequest(symptom_text="刺痛固定", schema_version="v0")
    with pytest.raises(ValueError, match="schema_version"):
        service.analyze_symptoms(request)


# ------------------------------------------------------------------- 后端可替换


def test_backend_swap_through_one_api(label_space):
    """两个假后端经同一 API 调用：schema 相同，model_id/model_version/分数不同。"""
    registry = ModelRegistry()
    registry.register(make_spec("fake-linear", label_space, scale=0.9), default=True)
    registry.register(make_spec("fake-other", label_space, scale=0.7))
    service = SyndromeService(registry=registry)
    base = SymptomAnalysisRequest(symptom_text="刺痛固定")
    first = service.analyze_symptoms(base)
    second = service.analyze_symptoms(
        SymptomAnalysisRequest(symptom_text="刺痛固定", model_id="fake-other")
    )
    assert first.to_dict().keys() == second.to_dict().keys()
    assert first.status == second.status == STATUS_OK
    assert first.model_id == "fake-linear" and second.model_id == "fake-other"
    assert first.candidates[0]["score"] != second.candidates[0]["score"]


def test_unknown_model_id_fails_closed(label_space):
    service = make_service(label_space)
    response = service.analyze_symptoms(
        SymptomAnalysisRequest(symptom_text="刺痛固定", model_id="nope")
    )
    assert response.status == STATUS_ABSTAIN
    assert response.backend_availability == BACKEND_NOT_AVAILABLE
    assert response.candidates == []


def test_missing_backend_not_available_without_crash(label_space):
    service = SyndromeService()  # 空 registry
    response = service.analyze_symptoms(SymptomAnalysisRequest(symptom_text="刺痛固定"))
    assert response.status == STATUS_ABSTAIN
    assert response.backend_availability == BACKEND_NOT_AVAILABLE
    assert response.message


# ------------------------------------------------------------------- 来源分区


def test_symptom_source_tag_partition(label_space):
    service = make_service(label_space)
    response = service.analyze_symptoms(SymptomAnalysisRequest(symptom_text="刺痛固定"))
    assert response.candidates
    for candidate in response.candidates:
        assert candidate["source_tag"] == "PREDICTED"
        assert candidate["calibrated"] is False
    for ref in response.knowledge_reference:
        assert ref["source_tag"] == "KNOWLEDGE"
    assert response.provenance["uncertainty_threshold"] == 0.0


def test_realtime_source_tag_partition(label_space, fake_topic_knowledge):
    service = make_service(label_space, topic_knowledge=fake_topic_knowledge)
    request = RealtimeKnowledgeRequest(
        session_id="s-1",
        timestamp="t",
        measurements=[
            MeasurementSample(
                source="band", metric="heart_rate", value=72.0, unit="bpm",
                timestamp="2026-11-01T10:00:00+08:00", quality="OK",
            )
        ],
    )
    response = service.realtime_knowledge(request)
    assert response.status == STATUS_OK
    for item in response.measured:
        assert item["source_tag"] == "MEASURED"
    for ref in response.knowledge:
        assert ref["source_tag"] == "KNOWLEDGE"
        for label in label_space.labels:
            assert label not in ref["text"]  # 知识引用不暗示个人证候
    assert response.syndrome_judgment is None
    assert response.classifier_invoked is False
    assert "candidates" not in response.to_dict()


# ------------------------------------------------------------------- 实时模式隔离


def test_realtime_mode_never_loads_or_invokes_classifier(label_space):
    """实时知识模式绝不加载、绝不调用症状分类器（结构性保证）。"""
    registry = ModelRegistry()
    spec = make_spec("fake-0", label_space)
    registry.register(spec, default=True)
    service = SyndromeService(registry=registry)
    request = RealtimeKnowledgeRequest(
        session_id="s-1",
        timestamp="t",
        measurements=[
            MeasurementSample(
                source="band", metric="heart_rate", value=72.0, unit="bpm",
                timestamp="2026-11-01T10:00:00+08:00", quality="OK",
            )
        ],
    )
    response = service.realtime_knowledge(request)
    assert response.classifier_invoked is False
    assert response.syndrome_judgment is None
    assert service._inferences == {}  # 后端从未被构建


def test_feedback_module_never_imports_inference_or_models():
    """AST 结构性断言：feedback.py 不 import 任何推理/模型模块——
    生理数值在模块依赖层面无法进入证候分类器。"""
    tree = ast.parse(SRC_FEEDBACK.read_text(encoding="utf-8"))
    banned = {
        "syndrome.inference",
        "syndrome.baselines",
        "syndrome.bert_model",
        "syndrome.experiment",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(alias.name in banned for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module not in banned


# ------------------------------------------------------------------- fail-closed


def _measurement(**overrides) -> MeasurementSample:
    base = dict(
        source="band", metric="heart_rate", value=72.0, unit="bpm",
        timestamp="2026-11-01T10:00:00+08:00", quality="OK",
    )
    base.update(overrides)
    return MeasurementSample(**base)


def test_realtime_empty_measurements_abstain(label_space, fake_topic_knowledge):
    service = make_service(label_space, topic_knowledge=fake_topic_knowledge)
    response = service.realtime_knowledge(
        RealtimeKnowledgeRequest(session_id="s-1", timestamp="t")
    )
    assert response.status == STATUS_ABSTAIN
    assert response.knowledge == []
    assert response.notes


def test_realtime_empty_session_id_abstain(label_space):
    service = make_service(label_space)
    response = service.realtime_knowledge(
        RealtimeKnowledgeRequest(session_id="  ", timestamp="t")
    )
    assert response.status == STATUS_ABSTAIN


def test_realtime_without_topic_knowledge_fails_closed(label_space):
    """topic_knowledge 未提供：只回实测回执，不输出任何知识（不臆造内容）。"""
    service = make_service(label_space)
    response = service.realtime_knowledge(
        RealtimeKnowledgeRequest(
            session_id="s-1",
            timestamp="t",
            measurements=[_measurement()],
        )
    )
    assert response.status == STATUS_OK
    assert response.knowledge == []
    assert any("NOT_AVAILABLE" in n for n in response.notes)


def test_realtime_low_quality_and_regression_excluded_from_knowledge(
    label_space, fake_topic_knowledge
):
    service = make_service(label_space, topic_knowledge=fake_topic_knowledge)
    response = service.realtime_knowledge(
        RealtimeKnowledgeRequest(
            session_id="s-1",
            timestamp="t",
            measurements=[
                _measurement(quality="DEGRADED"),
                _measurement(metric="quantum_flux"),
                _measurement(value=999.0),
            ],
        )
    )
    assert response.status == STATUS_ABSTAIN
    assert response.knowledge == []
    dispositions = {m["disposition"] for m in response.measured}
    assert dispositions == {"low_quality", "unknown_metric", "out_of_range"}


# ------------------------------------------------------------------- TF-IDF 工件


def test_tfidf_artifact_roundtrip(label_space, corpus, tmp_path):
    from syndrome.baselines import TfidfNgramBaseline, TfidfNgramConfig

    texts, labels = corpus
    predictor = TfidfNgramBaseline(TfidfNgramConfig(min_df=1), n_classes=len(label_space))
    predictor.fit(texts, labels)
    spec = save_tfidf_logreg_artifact(
        predictor,
        label_space.labels,
        str(tmp_path),
        model_version="synthetic-tfidf-v1",
        threshold=0.0,
        threshold_scope="UNVERIFIED",
        data_license="SYNTHETIC (no license grant)",
        provenance={"source": "SYNTHETIC"},
    )
    registry = ModelRegistry()
    registry.register(spec, default=True)
    service = SyndromeService(registry=registry)
    response = service.analyze_symptoms(SymptomAnalysisRequest(symptom_text="刺痛固定"))
    assert response.status == STATUS_OK
    assert response.model_version == "synthetic-tfidf-v1"
    assert response.backend_availability == "AVAILABLE"
    assert response.candidates[0]["calibrated"] is False
    assert len(response.provenance["artifacts_sha256"]) == 2


def test_tfidf_artifact_missing_returns_none(tmp_path):
    assert tfidf_logreg_spec(str(tmp_path / "absent")) is None


def test_tfidf_artifact_tamper_fails_closed(label_space, corpus, tmp_path):
    from syndrome.baselines import TfidfNgramBaseline, TfidfNgramConfig

    texts, labels = corpus
    predictor = TfidfNgramBaseline(TfidfNgramConfig(min_df=1), n_classes=len(label_space))
    predictor.fit(texts, labels)
    save_tfidf_logreg_artifact(
        predictor,
        label_space.labels,
        str(tmp_path),
        model_version="synthetic-tfidf-v1",
        threshold=0.0,
        data_license="SYNTHETIC",
    )
    (tmp_path / "model.joblib").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        tfidf_logreg_spec(str(tmp_path)).loader()


def test_registry_rejects_foreign_taxonomy(label_space):
    spec = make_spec("fake-0", label_space)
    foreign = BackendSpec(
        model_id=spec.model_id,
        model_version=spec.model_version,
        taxonomy_version="some-other-taxonomy",
        loader=spec.loader,
    )
    with pytest.raises(ValueError, match="taxonomy"):
        ModelRegistry().register(foreign)
