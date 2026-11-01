"""realtime_feedback 契约：只收结构化测量摘要，绝不输出个体证候判断（合成消息）。"""

from __future__ import annotations

import pytest

from syndrome.feedback import (
    ABSTAIN_REASON,
    MeasurementSample,
    realtime_feedback,
)

pytestmark = pytest.mark.synthetic


def sample(**overrides) -> MeasurementSample:
    base = dict(
        source="wearable-band",
        metric="heart_rate",
        value=88.0,
        unit="bpm",
        timestamp="2026-11-01T10:00:00+08:00",
        quality="OK",
    )
    base.update(overrides)
    return MeasurementSample(**base)


def test_never_returns_syndrome_judgment(fake_knowledge):
    """契约核心：无论输入什么测量，syndrome_judgment 恒为 None、status 恒为 ABSTAIN。"""
    results = [
        realtime_feedback([], fake_knowledge),
        realtime_feedback([sample()], fake_knowledge),
        realtime_feedback(
            [sample(metric="未知指标"), sample(quality="DEGRADED"), sample(value=None)],
            fake_knowledge,
        ),
    ]
    for r in results:
        assert r.syndrome_judgment is None
        assert r.status == "ABSTAIN"
        assert r.reason == ABSTAIN_REASON
        d = r.to_dict()
        assert d["syndrome_judgment"] is None


def test_references_come_from_knowledge_with_source(fake_knowledge):
    r = realtime_feedback([sample(metric="heart_rate", value=88.0)], fake_knowledge)
    assert r.references, "心悸/脉关键词应命中至少一条知识条目"
    for ref in r.references:
        assert ref["source"] == "syndrome_knowledge.json"
        assert ref["name"] in fake_knowledge
        assert ref["field"] in ("Definition", "Typical_performance")
        assert ref["text"]  # 逐字引用，非生成文本


def test_degraded_quality_noted_and_excluded(fake_knowledge):
    r = realtime_feedback([sample(quality="DEGRADED")], fake_knowledge)
    assert any("DEGRADED" in n for n in r.notes)
    assert r.references == []  # 质量异常的测量不参与知识检索


def test_unknown_metric_skipped_with_note(fake_knowledge):
    r = realtime_feedback([sample(metric="quantum_flux")], fake_knowledge)
    assert r.references == []
    assert any("quantum_flux" in n for n in r.notes)


def test_empty_samples_returns_empty_reference(fake_knowledge):
    r = realtime_feedback([], fake_knowledge)
    assert r.references == []
    assert r.notes


def test_no_fabricated_prescriptions(fake_knowledge):
    """引用文本必须逐字来自知识条目属性，不允许出现接口生成的处方/疗效文本。"""
    r = realtime_feedback([sample()], fake_knowledge)
    for ref in r.references:
        entry = fake_knowledge[ref["name"]]
        expected = getattr(entry, ref["field"])
        assert ref["text"] == expected
