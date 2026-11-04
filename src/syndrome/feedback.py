"""realtime_feedback 接口（#53 Scope 2 / #58 实时知识模式基座）。

只接收未来生理模块的**结构化测量摘要**（source/metric/value/unit/timestamp/
quality），返回实测回执（MEASURED）与有出处的**非诊断性主题知识**（KNOWLEDGE），
或明确「不作证候判断」。

硬约束（结构性保证，不靠调用方自觉）：
- 本模块不接收、不持有任何模型对象——生理数值在类型层面无法进入证候分类器；
  本模块也不 import 任何推理/训练模块（tests 有 AST 结构性断言）；
- 返回结果的 syndrome_judgment 恒为 None；
- 知识检索是**主题级**的：设备指标名 → 一般健康主题（METRIC_TOPICS），与测量
  数值和个人无关。旧版 METRIC_KEYWORDS 按心率等指标关键词检索**证候条目**
  （如心率→「心悸/脉」），会被误读为个人证候关联，已按 #58 移除；
- 知识条目必须带 source / license / version；出处缺失的条目 fail-closed 排除，
  绝不输出无来源知识，也绝不生成处方或疗效承诺；
- 质量异常 / 越界 / 未知指标 / 缺值 / 时间戳回退的测量只留实测回执（MEASURED
  分区，附 disposition 标注），不参与知识检索。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime

ABSTAIN_REASON = "realtime_feedback 不作证候判断：生理测量不用于证候分类（#53/#58 契约）"

# 来源分区标签（#58：MEASURED / PREDICTED / KNOWLEDGE 严格分离）。
SOURCE_MEASURED = "MEASURED"
SOURCE_KNOWLEDGE = "KNOWLEDGE"

# 单条测量的处置结果：只有 accepted 参与知识检索，其余仅留实测回执。
DISPOSITION_ACCEPTED = "accepted"
DISPOSITION_LOW_QUALITY = "low_quality"
DISPOSITION_UNKNOWN_METRIC = "unknown_metric"
DISPOSITION_MISSING_VALUE = "missing_value"
DISPOSITION_INVALID_VALUE = "invalid_value"
DISPOSITION_OUT_OF_RANGE = "out_of_range"
DISPOSITION_INVALID_TIMESTAMP = "invalid_timestamp"
DISPOSITION_TIMESTAMP_REGRESSION = "timestamp_regression"

# 每条被拒收测量的可读原因（机器可读的 disposition 之外的人类可读面）。
_REJECTION_NOTES = {
    DISPOSITION_LOW_QUALITY: "质量异常：{sample.source}/{sample.metric}，不参与知识检索",
    DISPOSITION_UNKNOWN_METRIC: "未知指标 {sample.metric}：无主题映射，不参与知识检索",
    DISPOSITION_MISSING_VALUE: "缺值：{sample.source}/{sample.metric}，不参与知识检索",
    DISPOSITION_INVALID_VALUE: "value 非数值：{sample.source}/{sample.metric}，拒收",
    DISPOSITION_OUT_OF_RANGE: "越界：{sample.source}/{sample.metric}，不参与知识检索",
    DISPOSITION_INVALID_TIMESTAMP: "时间戳无效：{sample.source}/{sample.metric}，拒收",
    DISPOSITION_TIMESTAMP_REGRESSION: "时间戳回退：{sample.source}/{sample.metric}，拒收",
}

# 测量指标 → 非诊断性一般健康主题（设计选择，非医学映射）。
# 只按**指标名称**映射到主题级知识，与测量数值和个人无关。
METRIC_TOPICS: dict[str, tuple[str, ...]] = {
    "heart_rate": ("cardiovascular-general",),
    "heart_rate_variability": ("cardiovascular-general",),
    "blood_pressure": ("cardiovascular-general",),
    "respiration_rate": ("respiratory-general",),
    "spo2": ("respiratory-general",),
    "skin_temperature": ("thermal-comfort-general",),
    "skin_conductance": ("stress-general",),
    "sleep_duration": ("sleep-general",),
}

# 粗略合理性范围（工程护栏，拦截传感器错误值进入知识检索；非医学判定标准）。
# 单位约定（护栏假设的规范单位，unit 字段不做换算）：heart_rate bpm、
# heart_rate_variability ms、blood_pressure mmHg、respiration_rate breaths/min、
# spo2 %、skin_temperature °C、skin_conductance µS、sleep_duration h。
PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "heart_rate": (20.0, 250.0),
    "heart_rate_variability": (2.0, 500.0),
    "blood_pressure": (50.0, 260.0),
    "respiration_rate": (4.0, 60.0),
    "spo2": (50.0, 100.0),
    "skin_temperature": (25.0, 45.0),
    "skin_conductance": (0.0, 60.0),
    "sleep_duration": (0.0, 24.0),
}

TOPIC_NOTE = (
    "知识条目按设备指标名称映射到一般健康主题（METRIC_TOPICS），"
    "与测量数值和个人无关，不是个体证候判断"
)


@dataclass(frozen=True)
class TopicKnowledge:
    """一条非诊断性主题知识。source / license / version 缺一不可（fail-closed）。"""

    topic_id: str
    title: str
    text: str
    source: str
    license: str
    version: str


@dataclass(frozen=True)
class MeasurementSample:
    """一次生理测量的结构化摘要。quality 为测量端给出的质量状态。"""

    source: str
    metric: str
    value: float | None
    unit: str
    timestamp: str
    quality: str


@dataclass
class FeedbackResult:
    syndrome_judgment: None = None
    status: str = "ABSTAIN"
    reason: str = ABSTAIN_REASON
    measured: list[dict] = field(default_factory=list)
    references: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "syndrome_judgment": self.syndrome_judgment,
            "status": self.status,
            "reason": self.reason,
            "measured": list(self.measured),
            "references": list(self.references),
            "notes": list(self.notes),
        }


def _parse_timestamp(text: str) -> datetime | None:
    """解析 ISO 8601 时间戳；解析失败或 naive（无时区）一律返回 None（fail-closed，
    不臆断时区）。"""
    if not isinstance(text, str):
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed


def _reference_for(topic_id: str, entry: TopicKnowledge) -> dict:
    return {
        "topic_id": topic_id,
        "title": entry.title,
        "text": entry.text,
        "source": entry.source,
        "license": entry.license,
        "version": entry.version,
        "source_tag": SOURCE_KNOWLEDGE,
    }


def realtime_feedback(
    samples: Sequence[MeasurementSample],
    topic_knowledge: Mapping[str, TopicKnowledge] | None,
    max_references: int = 5,
    reconnect_gap_seconds: float = 300.0,
) -> FeedbackResult:
    """结构化测量摘要 → 实测回执 + 主题知识；绝不做个体证候预测。

    topic_knowledge 为 topic_id → TopicKnowledge 映射；None 或非 Mapping 表示
    知识库不可用（fail-closed：不输出任何知识条目，仅留实测回执与说明）。
    """
    result = FeedbackResult()
    result.notes.append(TOPIC_NOTE)
    if topic_knowledge is None or not isinstance(topic_knowledge, Mapping):
        if topic_knowledge is not None:
            result.notes.append("主题知识库类型无效：按 NOT_AVAILABLE 处理")
        result.notes.append("主题知识库不可用（NOT_AVAILABLE）：不输出任何知识条目")
        topic_knowledge = {}
    if not samples:
        result.notes.append("未提供测量数据，仅返回空知识参考")
        return result

    measured: list[dict] = []
    cited_topics: list[str] = []
    warned_topics: set[str] = set()
    last_ts_by_source: dict[str, datetime] = {}

    for sample in samples:
        ts = _parse_timestamp(sample.timestamp)
        previous = last_ts_by_source.get(sample.source)
        if ts is None:
            disposition = DISPOSITION_INVALID_TIMESTAMP
        elif previous is not None and ts < previous:
            # 时间戳回归按**设备来源**判定：多设备交错流的包间乱序不应误标。
            disposition = DISPOSITION_TIMESTAMP_REGRESSION
        elif sample.quality != "OK":
            disposition = DISPOSITION_LOW_QUALITY
        elif sample.metric not in METRIC_TOPICS:
            disposition = DISPOSITION_UNKNOWN_METRIC
        elif sample.value is None:
            disposition = DISPOSITION_MISSING_VALUE
        elif not isinstance(sample.value, (int, float)) or isinstance(sample.value, bool):
            disposition = DISPOSITION_INVALID_VALUE
        else:
            bounds = PLAUSIBLE_RANGES.get(sample.metric)
            if bounds is None:
                # METRIC_TOPICS / PLAUSIBLE_RANGES 的对齐由测试守护；缺表按未知指标拒收。
                disposition = DISPOSITION_UNKNOWN_METRIC
            else:
                low, high = bounds
                disposition = (
                    DISPOSITION_ACCEPTED
                    if low <= sample.value <= high
                    else DISPOSITION_OUT_OF_RANGE
                )

        reconnected = False
        if ts is not None and disposition != DISPOSITION_TIMESTAMP_REGRESSION:
            if previous is not None and (ts - previous).total_seconds() > reconnect_gap_seconds:
                reconnected = True
                result.notes.append(
                    f"测量中断后重新连接：{sample.source} 间隔 "
                    f"{(ts - previous).total_seconds():.0f}s，恢复正常记录"
                )
            last_ts_by_source[sample.source] = ts

        if disposition != DISPOSITION_ACCEPTED:
            result.notes.append(_REJECTION_NOTES[disposition].format(sample=sample))

        measured.append(
            {
                "metric": sample.metric,
                "value": sample.value,
                "unit": sample.unit,
                "quality": sample.quality,
                "source": sample.source,
                "timestamp": sample.timestamp,
                "source_tag": SOURCE_MEASURED,
                "disposition": disposition,
                "reconnected": reconnected,
            }
        )

        if disposition != DISPOSITION_ACCEPTED:
            continue
        for topic_id in METRIC_TOPICS[sample.metric]:
            if topic_id in cited_topics or len(result.references) >= max_references:
                continue
            entry = topic_knowledge.get(topic_id)
            if entry is None:
                if topic_id not in warned_topics:
                    warned_topics.add(topic_id)
                    result.notes.append(f"主题 {topic_id} 无知识条目（NOT_AVAILABLE），跳过")
                continue
            missing = [
                name
                for name in ("source", "license", "version")
                if not getattr(entry, name, "")
            ]
            if missing or not entry.title or not entry.text:
                if topic_id not in warned_topics:
                    warned_topics.add(topic_id)
                    result.notes.append(
                        f"主题 {topic_id} 知识条目出处缺失"
                        f"（{','.join(missing) if missing else 'title/text'}），fail-closed 跳过"
                    )
                continue
            result.references.append(_reference_for(topic_id, entry))
            cited_topics.append(topic_id)

    result.measured = measured
    if any(m["disposition"] == DISPOSITION_ACCEPTED for m in measured):
        result.status = "OK"
        if not result.references:
            result.notes.append("无可用主题知识条目；仅返回测量回执，不作证候判断")
    else:
        result.status = "ABSTAIN"
        result.notes.append("无有效测量（全部被拒收），仅返回实测回执与拒收原因")
    return result
