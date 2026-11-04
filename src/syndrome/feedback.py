"""realtime_feedback 接口（#53 Scope 2）。

只接收未来生理模块的**结构化测量摘要**（source/metric/value/unit/timestamp/
quality），返回带来源的知识参考或明确「不作证候判断」。

硬约束（结构性保证，不靠调用方自觉）：
- 本模块不接收、不持有任何模型对象——生理数值在类型层面无法进入证候分类器；
- 返回结果的 syndrome_judgment 恒为 None；
- 知识参考只逐字引用 syndrome_knowledge 的 Definition / Typical_performance，
  附来源字段，绝不生成处方或疗效承诺。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

ABSTAIN_REASON = "realtime_feedback 不作证候判断：生理测量不用于证候分类（#53 契约）"

# 测量指标 → 知识文本关键词（设计选择，非医学映射；只用于检索引用，不用于判断）。
# 命中即引用该证候条目，供展示"有来源的中医知识解释"。
METRIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "heart_rate": ("心悸", "脉"),
    "heart_rate_variability": ("心悸", "脉"),
    "blood_pressure": ("眩晕", "头痛", "脉"),
    "respiration_rate": ("气短", "喘", "呼吸"),
    "spo2": ("气短", "倦怠"),
    "skin_temperature": ("畏寒", "发热", "潮热", "寒热"),
    "skin_conductance": ("汗出", "自汗", "盗汗"),
    "sleep_duration": ("不寐", "失眠", "多寐"),
}


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
    references: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "syndrome_judgment": self.syndrome_judgment,
            "status": self.status,
            "reason": self.reason,
            "references": list(self.references),
            "notes": list(self.notes),
        }


def knowledge_field(entry: object, json_name: str) -> str | None:
    """按 JSON 键名取知识条目字段；兼容 snake_case 属性（如 data.tcm_sd 的
    SyndromeKnowledge.definition）与原始键名（如裸 JSONL dict 的 Definition）。"""
    for attr in (json_name, _SNAKE.get(json_name, json_name)):
        value = getattr(entry, attr, None)
        if value:
            return str(value)
    if isinstance(entry, dict):
        value = entry.get(json_name)
        if value:
            return str(value)
    return None


_SNAKE = {
    "Name": "name",
    "Definition": "definition",
    "Typical_performance": "typical_performance",
}


def _reference_for(entry: object, field_name: str) -> dict | None:
    text = knowledge_field(entry, field_name)
    name = knowledge_field(entry, "Name")
    if not text or not name:
        return None
    return {"name": name, "field": field_name, "text": text, "source": "syndrome_knowledge.json"}


def realtime_feedback(
    samples: Sequence[MeasurementSample],
    knowledge: Mapping[str, object],
    max_references: int = 5,
) -> FeedbackResult:
    """结构化测量摘要 → 知识参考；绝不做个体证候预测。

    knowledge 为 syndrome_knowledge.json 的按 Name 键映射（duck-typed：
    条目需有 Name / Definition / Typical_performance 属性）。
    """
    result = FeedbackResult()
    if not samples:
        result.notes.append("未提供测量数据，仅返回空知识参考")
        return result

    refs_by_name: dict[str, list[dict]] = {}
    for sample in samples:
        if sample.quality != "OK":
            result.notes.append(
                f"测量质量异常：{sample.source}/{sample.metric} quality={sample.quality}，"
                "该测量不用于知识检索"
            )
            continue
        keywords = METRIC_KEYWORDS.get(sample.metric)
        if keywords is None:
            result.notes.append(f"未知指标 {sample.metric}：无预置知识映射，跳过检索")
            continue
        for name, entry in knowledge.items():
            if name in refs_by_name:
                continue
            haystack = knowledge_field(entry, "Definition") or ""
            haystack += knowledge_field(entry, "Typical_performance") or ""
            if not any(k in haystack for k in keywords):
                continue
            refs = [
                r
                for r in (_reference_for(entry, f) for f in ("Definition", "Typical_performance"))
                if r
            ]
            if refs:
                refs_by_name[name] = refs
            if len(refs_by_name) >= max_references:
                break
        if len(refs_by_name) >= max_references:
            break

    result.references = [r for refs in refs_by_name.values() for r in refs]
    if not result.references:
        result.notes.append("无匹配的证候知识条目；仅返回测量回执，不作证候判断")
    return result
