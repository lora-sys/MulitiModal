"""realtime_feedback 契约（#58 主题知识版）：只收结构化测量摘要，
绝不输出个体证候判断；知识检索按指标名映射到非诊断性主题，与数值无关。"""

from __future__ import annotations

import pytest

from syndrome.feedback import (
    ABSTAIN_REASON,
    MeasurementSample,
    TopicKnowledge,
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


def test_never_returns_syndrome_judgment(fake_topic_knowledge):
    """契约核心：无论输入什么测量，syndrome_judgment 恒为 None。"""
    results = [
        realtime_feedback([], fake_topic_knowledge),
        realtime_feedback([sample()], fake_topic_knowledge),
        realtime_feedback(
            [sample(metric="未知指标"), sample(quality="DEGRADED"), sample(value=None)],
            fake_topic_knowledge,
        ),
    ]
    for r in results:
        assert r.syndrome_judgment is None
        assert r.reason == ABSTAIN_REASON
        assert r.to_dict()["syndrome_judgment"] is None


def test_accepted_measurement_yields_ok_with_topic_knowledge(fake_topic_knowledge):
    r = realtime_feedback([sample(metric="heart_rate", value=88.0)], fake_topic_knowledge)
    assert r.status == "OK"
    assert r.references, "心率指标应映射到 cardiovascular-general 主题"
    for ref in r.references:
        assert ref["source_tag"] == "KNOWLEDGE"
        assert ref["topic_id"] in fake_topic_knowledge
        entry = fake_topic_knowledge[ref["topic_id"]]
        assert ref["source"] == entry.source
        assert ref["license"] == entry.license
        assert ref["version"] == entry.version
        assert ref["text"] == entry.text  # 逐字引用，非生成文本


def test_references_decoupled_from_measurement_value(fake_topic_knowledge):
    """同一指标的不同数值得到相同主题知识：检索与测量数值和个人无关。"""
    low = realtime_feedback([sample(value=55.0)], fake_topic_knowledge)
    high = realtime_feedback([sample(value=140.0)], fake_topic_knowledge)
    assert [ref["topic_id"] for ref in low.references] == [
        ref["topic_id"] for ref in high.references
    ]


def test_references_never_contain_syndrome_names(fake_topic_knowledge, label_space):
    """主题知识检索不得把测量关联到任何证候条目（旧 METRIC_KEYWORDS 行为已移除）。"""
    r = realtime_feedback([sample()], fake_topic_knowledge)
    for ref in r.references:
        for label in label_space.labels:
            assert label not in ref["text"]
            assert label not in ref["title"]


def test_degraded_quality_noted_and_excluded(fake_topic_knowledge):
    r = realtime_feedback([sample(quality="DEGRADED")], fake_topic_knowledge)
    assert r.status == "ABSTAIN"
    assert r.references == []
    assert r.measured[0]["disposition"] == "low_quality"
    assert r.measured[0]["source_tag"] == "MEASURED"


def test_out_of_range_value_rejected(fake_topic_knowledge):
    r = realtime_feedback([sample(value=320.0)], fake_topic_knowledge)
    assert r.status == "ABSTAIN"
    assert r.references == []
    assert r.measured[0]["disposition"] == "out_of_range"


def test_unknown_metric_skipped_with_note(fake_topic_knowledge):
    r = realtime_feedback([sample(metric="quantum_flux")], fake_topic_knowledge)
    assert r.status == "ABSTAIN"
    assert r.references == []
    assert r.measured[0]["disposition"] == "unknown_metric"
    assert any("quantum_flux" in n for n in r.notes)


def test_missing_value_rejected(fake_topic_knowledge):
    r = realtime_feedback([sample(value=None)], fake_topic_knowledge)
    assert r.measured[0]["disposition"] == "missing_value"
    assert r.status == "ABSTAIN"


def test_timestamp_regression_rejected(fake_topic_knowledge):
    r = realtime_feedback(
        [
            sample(timestamp="2026-11-01T10:00:20+08:00"),
            sample(timestamp="2026-11-01T10:00:10+08:00"),
        ],
        fake_topic_knowledge,
    )
    assert r.measured[0]["disposition"] == "accepted"
    assert r.measured[1]["disposition"] == "timestamp_regression"
    assert r.references  # 第一条有效测量仍检索知识


def test_invalid_or_naive_timestamp_rejected(fake_topic_knowledge):
    r = realtime_feedback(
        [sample(timestamp="not-a-time"), sample(timestamp="2026-11-01T10:00:00")],
        fake_topic_knowledge,
    )
    assert r.measured[0]["disposition"] == "invalid_timestamp"
    assert r.measured[1]["disposition"] == "invalid_timestamp"  # naive 无时区
    assert r.status == "ABSTAIN"


def test_reconnect_after_gap_is_flagged_not_rejected(fake_topic_knowledge):
    r = realtime_feedback(
        [
            sample(timestamp="2026-11-01T10:00:00+08:00"),
            sample(timestamp="2026-11-01T10:15:00+08:00"),
        ],
        fake_topic_knowledge,
        reconnect_gap_seconds=300.0,
    )
    assert all(m["disposition"] == "accepted" for m in r.measured)
    assert r.measured[0]["reconnected"] is False
    assert r.measured[1]["reconnected"] is True
    assert any("重新连接" in n for n in r.notes)


def test_empty_samples_returns_empty_reference(fake_topic_knowledge):
    r = realtime_feedback([], fake_topic_knowledge)
    assert r.status == "ABSTAIN"
    assert r.references == []
    assert r.measured == []
    assert r.notes


def test_missing_topic_entry_noted(fake_topic_knowledge):
    """指标映射到的主题没有条目时：留实测回执，note 说明，不猜内容。"""
    knowledge = {k: v for k, v in fake_topic_knowledge.items() if k != "cardiovascular-general"}
    r = realtime_feedback([sample()], knowledge)
    assert r.status == "OK"
    assert r.references == []
    assert any("cardiovascular-general" in n for n in r.notes)


def test_knowledge_with_missing_provenance_fails_closed():
    """source / license / version 缺失的主题条目必须被排除（fail-closed）。"""
    broken = TopicKnowledge(
        topic_id="cardiovascular-general",
        title="无出处条目",
        text="无出处文本",
        source="",
        license="SYNTHETIC_MOCK",
        version="0.0.1",
    )
    r = realtime_feedback([sample()], {"cardiovascular-general": broken})
    assert r.references == []
    assert r.status == "OK"
    assert any("fail-closed" in n for n in r.notes)
