"""推理接口契约：status 状态机、拒判、越界、缺失信息、provenance。

无信息/越界绝不产出证候标签（#53 Scope 2 硬约束）。
"""

from __future__ import annotations

import numpy as np
import pytest

from syndrome.inference import (
    DEFAULT_COVERAGE_FLOOR,
    SyndromeInference,
    is_out_of_domain,
    select_uncertainty_threshold,
)
from syndrome.labels import (
    STATUS_ABSTAIN,
    STATUS_OK,
    STATUS_OUT_OF_DOMAIN,
    STATUS_UNCERTAIN,
)

pytestmark = pytest.mark.synthetic


def make_inference(fixed_predictor, label_space, threshold=0.0) -> SyndromeInference:
    return SyndromeInference(
        fixed_predictor,
        label_space,
        model_version="test-model-v1",
        provenance={"artifact": "synthetic"},
        threshold=threshold,
    )


def test_ok_result_shape(fixed_predictor, label_space):
    inf = make_inference(fixed_predictor, label_space)
    r = inf.predict_one("患者近两周明显刺痛固定伴乏力").to_dict()
    assert r["status"] == STATUS_OK
    assert 0 < len(r["candidates"]) <= 3
    assert all(c["score"] >= 0 for c in r["candidates"])
    assert r["label_taxonomy_version"] == label_space.taxonomy_id
    assert r["model_version"] == "test-model-v1"
    assert "provenance" in r
    assert r["provenance"]["threshold_scope"] == "UNVERIFIED"
    assert r["calibrated"] is False  # 未校准必须显式声明


def test_candidates_always_within_frozen_vocab(fixed_predictor, label_space):
    """任何输入下候选都必须来自冻结词表——不存在凭空生成的标签。"""
    inf = make_inference(fixed_predictor, label_space)
    for text in ("刺痛固定", "完全无关的文本内容", "x"):
        r = inf.predict_one(text)
        assert all(c.label in label_space for c in r.candidates)


def test_empty_input_abstains(fixed_predictor, label_space):
    inf = make_inference(fixed_predictor, label_space)
    for text in ("", "   ", "。，、"):
        r = inf.predict_one(text)
        assert r.status == STATUS_ABSTAIN
        assert r.candidates == []  # 拒判时不给任何证候
        assert r.message


@pytest.mark.parametrize("text", ("", " ，。！？ "))
def test_empty_or_punctuation_text_with_fields_abstains_without_model_call(
    fixed_predictor, label_space, text
):
    inf = make_inference(fixed_predictor, label_space, threshold=0.0)
    single = inf.predict_one(text, fields={"age": 40})
    batch = inf.predict_batch([text], fields=[{"age": 40}])
    assert single.status == STATUS_ABSTAIN
    assert batch[0].status == STATUS_ABSTAIN
    assert single.candidates == batch[0].candidates == []
    assert fixed_predictor.calls == []


def test_out_of_domain_english_and_garbage(fixed_predictor, label_space):
    inf = make_inference(fixed_predictor, label_space)
    for text in ("I feel dizzy and cannot sleep", "@#$%^&*!!!", "1234567890"):
        r = inf.predict_one(text)
        assert r.status == STATUS_OUT_OF_DOMAIN
        assert r.candidates == []


def test_syndrome_name_or_suffix_omission_is_out_of_domain(fixed_predictor, label_space):
    """证候名及省略“证”后缀的已知别名都不作为症状分类。"""
    inf = make_inference(fixed_predictor, label_space)
    for text in (" 气虚血瘀证 ", "脾肾阳虚"):
        result = inf.predict_one(text)
        assert result.status == STATUS_OUT_OF_DOMAIN
        assert result.candidates == []
    assert fixed_predictor.calls == []


def test_is_out_of_domain_reasons(label_space):
    assert is_out_of_domain("hello world", label_space)[0] is True
    assert is_out_of_domain("失眠多梦心悸", label_space)[0] is False
    assert is_out_of_domain("", label_space)[0] is False  # 空文本走 ABSTAIN 分支


def test_low_confidence_is_uncertain(fixed_predictor, label_space):
    inf = make_inference(fixed_predictor, label_space, threshold=0.5)
    r = inf.predict_one("没有任何标记词的普通文本")  # 假模型给均匀低分
    assert r.status == STATUS_UNCERTAIN
    assert r.candidates  # UNCERTAIN 仍列出候选
    assert r.candidates[0].score < 0.5


@pytest.mark.parametrize("batch", (False, True))
@pytest.mark.parametrize(
    "malformation",
    ("wrong_rank", "wrong_row_count", "wrong_class_count", "nan", "negative", "over_one"),
)
def test_invalid_predictor_probabilities_fail_closed(
    label_space, batch, malformation
):
    class MalformedPredictor:
        def predict_proba(self, texts):
            rows = len(texts)
            probabilities = np.full(
                (rows, len(label_space)), 1.0 / len(label_space)
            )
            if malformation == "wrong_rank":
                return probabilities[0]
            if malformation == "wrong_row_count":
                return np.full((rows + 1, len(label_space)), 0.2)
            if malformation == "wrong_class_count":
                return np.full((rows, len(label_space) - 1), 0.2)
            if malformation == "nan":
                probabilities[0, 0] = np.nan
            elif malformation == "negative":
                probabilities[0, 0] = -0.01
            elif malformation == "over_one":
                probabilities[0, 0] = 1.01
            return probabilities

    inference = SyndromeInference(
        MalformedPredictor(),
        label_space,
        model_version="malformed-test-v1",
        provenance={"artifact": "synthetic"},
        threshold=0.0,
    )
    with pytest.raises(ValueError, match="predictor probabilities"):
        if batch:
            inference.predict_batch(["患者刺痛固定", "近期心悸失眠"])
        else:
            inference.predict_one("患者刺痛固定")


def test_missing_information_lists_unprovided_fields(fixed_predictor, label_space):
    inf = make_inference(fixed_predictor, label_space)
    r = inf.predict_one("患者刺痛固定，舌红苔薄白，脉细弦")
    assert "tongue" not in r.missing_information
    assert "pulse" not in r.missing_information
    assert "age" in r.missing_information and "gender" in r.missing_information


def test_evidence_only_from_known_inputs(fixed_predictor, label_space, fake_knowledge):
    """解释只能回显已输入内容或引用知识库，不得出现编造症状。"""
    inf = make_inference(fixed_predictor, label_space)
    inf.knowledge = fake_knowledge
    text = "患者近两周明显刺痛固定"
    r = inf.predict_one(text)
    assert any(text in e for e in r.evidence)
    for ref in r.knowledge_reference:
        assert ref["source"] == "syndrome_knowledge.json"
        assert ref["field"] in ("Definition", "Typical_performance")


def test_predict_batch_matches_single(fixed_predictor, label_space):
    inf = make_inference(fixed_predictor, label_space, threshold=0.5)
    texts = ["刺痛固定", "", "I cannot sleep", "胁肋胀闷叹气明显"]
    batch = inf.predict_batch(texts)
    single = [inf.predict_one(t) for t in texts]
    assert [b.status for b in batch] == [s.status for s in single]
    assert [b.candidates for b in batch] == [s.candidates for s in single]


def test_batch_length_mismatch_raises(fixed_predictor, label_space):
    inf = make_inference(fixed_predictor, label_space)
    with pytest.raises(ValueError):
        inf.predict_batch(["a", "b"], fields=[{"age": 1}])


def test_select_threshold_satisfies_coverage_floor():
    # 300 个样本：60% 高分正确、25% 低分正确、15% 低分错误
    max_probs = [0.9] * 180 + [0.3] * 75 + [0.3] * 45
    correct = [True] * 180 + [True] * 75 + [False] * 45
    info = select_uncertainty_threshold(max_probs, correct, coverage_floor=0.8)
    assert info["dev_coverage"] >= 0.8
    kept_acc = info["dev_accuracy_above"]
    full_acc = sum(correct) / len(correct)
    assert kept_acc >= full_acc  # 过滤后准确率不降
    assert 0.0 <= info["threshold"] <= 1.0


def test_select_threshold_is_safe_for_repeated_and_dense_probabilities():
    repeated = [0.12346] * 10
    dense = [0.12346 + i * 0.000001 for i in range(10)]
    for probs in (repeated, dense):
        info = select_uncertainty_threshold(probs, [True] * len(probs), coverage_floor=0.9)
        assert info["threshold"] == min(probs)
        assert info["dev_coverage"] >= 0.9


def test_select_threshold_impossible_floor_raises():
    with pytest.raises(ValueError):
        select_uncertainty_threshold([0.2] * 10, [True] * 10, coverage_floor=1.5)


@pytest.mark.parametrize(
    "probabilities",
    ([float("nan")], [-0.01], [1.01], [[0.5]]),
)
def test_select_threshold_rejects_invalid_probability_vectors(probabilities):
    with pytest.raises(ValueError):
        select_uncertainty_threshold(probabilities, [True])


def test_default_coverage_floor_value():
    assert DEFAULT_COVERAGE_FLOOR == 0.9

@pytest.mark.synthetic
def test_inference_provenance_preserves_tfidf_ovr_score_semantics(corpus):
    from syndrome.baselines import (
        TCM_SD_CLASS_COUNT,
        TfidfNgramBaseline,
        TfidfNgramConfig,
    )
    from syndrome.labels import LabelSpace

    texts, labels = corpus
    label_space = LabelSpace(
        tuple(f"synthetic-syndrome-{index}" for index in range(TCM_SD_CLASS_COUNT))
    )
    predictor = TfidfNgramBaseline(
        TfidfNgramConfig(min_df=1)
    ).fit(texts, labels)
    inference = SyndromeInference(
        predictor,
        label_space,
        model_version="synthetic-tfidf-v1",
        provenance={"source": "SYNTHETIC"},
        threshold=0.0,
    )
    result = inference.predict_one("患者近期明显刺痛固定伴乏力")
    assert result.provenance["score_semantics"] == predictor.score_semantics
    assert "not softmax" in result.provenance["score_semantics"]
    assert "uncalibrated" in predictor.provenance()["score_semantics"]
