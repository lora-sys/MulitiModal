"""鲁棒性套件（合成）：短输入、否定症状、口语化、越界、缺失字段。

真实数据上的 coverage / 拒判率 / 性能下降数字由
`python -m syndrome.experiment robustness` 产出并写入模型卡；
本套件锁的是接口行为契约。
"""

from __future__ import annotations

import pytest

from syndrome.baselines import TfidfNgramBaseline, TfidfNgramConfig
from syndrome.inference import SyndromeInference
from syndrome.labels import STATUS_ABSTAIN, STATUS_OK, STATUS_OUT_OF_DOMAIN, STATUS_UNCERTAIN

pytestmark = pytest.mark.synthetic


@pytest.fixture(scope="module")
def trained(label_space, make_corpus):
    texts, labels = make_corpus(n_per_class=12, seed=5)
    clf = TfidfNgramBaseline(TfidfNgramConfig(seed=5)).fit(texts, labels)
    inference = SyndromeInference(
        clf,
        label_space,
        model_version="tfidf-robustness-test",
        provenance={"suite": "synthetic-robustness"},
        threshold=0.0,
    )
    return clf, inference


def test_short_input_still_classifiable(trained):
    """短输入（<10 字）不崩、状态合法、候选在词表内。"""
    _clf, inference = trained
    r = inference.predict_one("畏寒肢冷")
    assert r.status in (STATUS_OK, STATUS_UNCERTAIN)
    assert all(c.label in inference.label_space for c in r.candidates)


def test_negated_symptom_does_not_crash(trained):
    """否定症状（「无发热」类）必须返回合法状态，不得编造阳性症状解释。"""
    _clf, inference = trained
    r = inference.predict_one("无发热，无畏寒，稍有乏力，舌淡")
    assert r.status in (STATUS_OK, STATUS_UNCERTAIN, STATUS_ABSTAIN)
    # 证据只能回显输入本身
    assert any("无发热" in e for e in r.evidence)


def test_colloquial_input_handled(trained):
    """口语化表达不崩，状态合法。"""
    _clf, inference = trained
    for text in ("老是怕冷拉肚子", "晚上睡不着心慌得很", "尿尿烫得慌还黄"):
        r = inference.predict_one(text)
        assert r.status in (STATUS_OK, STATUS_UNCERTAIN, STATUS_ABSTAIN, STATUS_OUT_OF_DOMAIN)


def test_out_of_domain_rejections(trained):
    """英文 / 乱码 / 证候名直接输入 → OUT_OF_DOMAIN，且不给候选。"""
    _clf, inference = trained
    for text in (
        "I have insomnia and palpitations for two weeks",
        "%$#@!!! ???",
        "qwerty asdf zxcv 12345",
        "湿热下注证",
    ):
        r = inference.predict_one(text)
        assert r.status == STATUS_OUT_OF_DOMAIN, text
        assert r.candidates == []


def test_missing_fields_reported(trained):
    _clf, inference = trained
    r = inference.predict_one("畏寒肢冷泄泻明显")
    assert set(r.missing_information) == {"age", "gender", "tongue", "pulse"}


def test_batch_mixed_statuses(trained):
    _clf, inference = trained
    results = inference.predict_batch(
        ["畏寒肢冷泄泻", "", "feeling cold", "心悸健忘失眠多梦"]
    )
    statuses = [r.status for r in results]
    assert statuses[0] in (STATUS_OK, STATUS_UNCERTAIN)
    assert statuses[1] == STATUS_ABSTAIN
    assert statuses[2] == STATUS_OUT_OF_DOMAIN
    assert statuses[3] in (STATUS_OK, STATUS_UNCERTAIN)


def test_scores_within_prob_range(trained):
    _clf, inference = trained
    r = inference.predict_one("刺痛固定明显")
    assert all(0.0 <= c.score <= 1.0 for c in r.candidates)
    if len(r.candidates) >= 2:
        assert r.candidates[0].score >= r.candidates[-1].score
