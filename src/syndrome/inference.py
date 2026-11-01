"""标准化单条/批量推理接口（#53 Scope 2）。

输入 = 中文症状文本（可含人口学/体征结构化字段）；输出 = 候选证候 + 分数 +
依据的已输入要点 + 缺失信息 + label_taxonomy_version + status + model_version +
provenance。

硬约束：
- 空输入 / 无任何症状信息 → ABSTAIN 拒判，绝不输出证候标签；
- 英文 / 乱码 / 把证候名当症状输入 → OUT_OF_DOMAIN；
- 最高分低于 dev 上选定的阈值 → UNCERTAIN（仍列出候选）；
- 分数为模型原始 softmax 概率，未做校准——在结果里如实声明；
- 解释只能来自已输入内容或 syndrome_knowledge 引用，禁止编造症状/处方/疗效；
- 生理数值永不进入本接口（见 feedback.py）。
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from syndrome.feedback import knowledge_field
from syndrome.labels import (
    LABEL_TAXONOMY_ID,
    STATUS_ABSTAIN,
    STATUS_OK,
    STATUS_OUT_OF_DOMAIN,
    STATUS_UNCERTAIN,
    LabelSpace,
)

DEFAULT_TOP_K = 3

# 常规问诊期望出现的结构化字段；未提供项进入 missing_information（仅描述输入缺口，
# 不是医学要求）。
EXPECTED_FIELDS: tuple[str, ...] = ("age", "gender", "tongue", "pulse")

# 阈值选择的默认覆盖下限：dev 上非拒判覆盖率至少 90%，在满足覆盖的候选阈值里取
# 过滤后准确率最高者。
DEFAULT_COVERAGE_FLOOR = 0.9

_CJK_RE = re.compile(r"[一-鿿]")
_STRIP_RE = re.compile(r"[\s，。、；：,.;:!?！？()（）\[\]【】\"'“”]+")

# 仅当输入文本里出现这些关键词时，才认为对应望诊/切诊信息已在文本中
_TEXT_FIELD_HINTS: dict[str, tuple[str, ...]] = {
    "tongue": ("舌", "苔"),
    "pulse": ("脉",),
}


class ProbabilisticPredictor(Protocol):
    """推理接口对模型的最小要求：文本 → (n, C) 概率矩阵。"""

    def predict_proba(self, texts: list[str]) -> np.ndarray: ...


@dataclass(frozen=True)
class Candidate:
    label: str
    score: float


@dataclass
class InferenceResult:
    status: str
    candidates: list[Candidate] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    knowledge_reference: list[dict] = field(default_factory=list)
    message: str = ""
    label_taxonomy_version: str = LABEL_TAXONOMY_ID
    model_version: str = ""
    provenance: dict = field(default_factory=dict)
    calibrated: bool = False

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "candidates": [{"label": c.label, "score": round(c.score, 6)} for c in self.candidates],
            "evidence": list(self.evidence),
            "missing_information": list(self.missing_information),
            "knowledge_reference": list(self.knowledge_reference),
            "message": self.message,
            "label_taxonomy_version": self.label_taxonomy_version,
            "model_version": self.model_version,
            "provenance": dict(self.provenance),
            "calibrated": self.calibrated,
        }


def is_out_of_domain(text: str, label_space: LabelSpace) -> tuple[bool, str]:
    """越界判定：非中文（英文/乱码）或输入本身就是证候名称。

    这是保守的字符级启发式，只用于拒判，不用于任何正类判断；
    阈值 0.15 由 #53 鲁棒性套件的合成正反例校准，改动需重跑该套件。
    """
    if not text:
        return False, ""
    cjk = len(_CJK_RE.findall(text))
    if cjk / len(text) < 0.15:
        return True, "输入不是中文症状描述"
    if _STRIP_RE.sub("", text) in label_space:
        return True, "输入是证候名称而非症状描述"
    return False, ""


def has_any_information(text: str, fields: Mapping[str, object] | None) -> bool:
    """无信息判定：文本为空（或纯标点）且没有任何结构化字段。"""
    if fields:
        return True
    return bool(_STRIP_RE.sub("", text))


def select_uncertainty_threshold(
    max_probs: Sequence[float],
    correct: Sequence[bool],
    coverage_floor: float = DEFAULT_COVERAGE_FLOOR,
) -> dict[str, float]:
    """在 dev 上选定 UNCERTAIN 阈值：满足覆盖率下限的候选中取过滤后准确率最高者。

    返回 {"threshold", "dev_coverage", "dev_accuracy_above"}，全部入档。
    """
    probs = np.asarray(max_probs, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    if len(probs) != len(correct) or len(probs) == 0:
        raise ValueError("threshold selection needs non-empty paired dev outputs")
    candidates = np.unique(np.round(probs, 4))
    best: dict[str, float] | None = None
    for tau in candidates:
        kept = probs >= tau
        coverage = float(kept.mean())
        if coverage < coverage_floor or not kept.any():
            continue
        acc = float(correct[kept].mean())
        # 平票取更低阈值（更高覆盖）
        if best is None or acc > best["dev_accuracy_above"]:
            best = {
                "threshold": float(tau),
                "dev_coverage": coverage,
                "dev_accuracy_above": acc,
            }
    if best is None:
        raise ValueError(f"no threshold satisfies coverage floor {coverage_floor}")
    return best


def _artifact_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class SyndromeInference:
    """把任意 ProbabilisticPredictor 包装成 #53 的标准化推理契约。"""

    def __init__(
        self,
        predictor: ProbabilisticPredictor,
        label_space: LabelSpace,
        model_version: str,
        provenance: dict,
        threshold: float = 0.0,
        top_k: int = DEFAULT_TOP_K,
        knowledge: Mapping[str, object] | None = None,
        weight_path: str | None = None,
    ) -> None:
        self.predictor = predictor
        self.label_space = label_space
        self.threshold = float(threshold)
        self.top_k = top_k
        self.knowledge = knowledge
        self.model_version = model_version
        prov = dict(provenance)
        prov["uncertainty_threshold"] = self.threshold
        prov["threshold_scope"] = "selected on dev partition; holdout untouched"
        if weight_path:
            prov["weight_sha256"] = _artifact_sha256(weight_path)
        self.provenance = prov

    # ------------------------------------------------------------------ 输出构造

    def _knowledge_reference(self, label: str) -> list[dict]:
        """引用知识库中该证候的 Definition / Typical_performance（带来源，非生成）。"""
        if not self.knowledge:
            return []
        entry = self.knowledge.get(label)
        if entry is None:
            return []
        refs = []
        for field_name in ("Definition", "Typical_performance"):
            value = knowledge_field(entry, field_name)
            if value:
                refs.append(
                    {
                        "name": knowledge_field(entry, "Name") or label,
                        "field": field_name,
                        "text": value,
                        "source": "syndrome_knowledge.json",
                    }
                )
        return refs

    def _missing_information(self, text: str, fields: Mapping[str, object] | None) -> list[str]:
        provided = set(fields or {})
        missing = []
        for name in EXPECTED_FIELDS:
            if name in provided:
                continue
            hints = _TEXT_FIELD_HINTS.get(name, ())
            if hints and any(h in text for h in hints):
                continue
            missing.append(name)
        return missing

    def _base_result(self, text: str, fields: Mapping[str, object] | None) -> InferenceResult:
        return InferenceResult(
            status=STATUS_ABSTAIN,
            label_taxonomy_version=self.label_space.taxonomy_id,
            model_version=self.model_version,
            provenance=dict(self.provenance),
            evidence=[f"输入文本: {text}"] if text else [],
            missing_information=self._missing_information(text, fields),
        )

    # ------------------------------------------------------------------ 主入口

    def _decide(
        self, text: str, fields: Mapping[str, object] | None, scores: np.ndarray
    ) -> InferenceResult:
        """对一个已确认非空、非越界的样本给出判定（scores 为该样本的类概率向量）。"""
        result = self._base_result(text, fields)
        if fields:
            result.evidence.extend(f"{k}={v}" for k, v in sorted(fields.items()))
        order = np.argsort(-scores)[: self.top_k]
        result.candidates = [
            Candidate(self.label_space.name(int(i)), float(scores[int(i)])) for i in order
        ]
        result.knowledge_reference = self._knowledge_reference(result.candidates[0].label)
        result.calibrated = False
        result.provenance["score_semantics"] = "raw softmax probability; NOT calibrated"
        if float(scores.max()) < self.threshold:
            result.status = STATUS_UNCERTAIN
            result.message = (
                f"模型最高分 {scores.max():.3f} 低于阈值 {self.threshold:.3f}，"
                "候选仅供参考（UNCERTAIN）"
            )
        else:
            result.status = STATUS_OK
        return result

    def predict_one(
        self, text: str, fields: Mapping[str, object] | None = None
    ) -> InferenceResult:
        text = (text or "").strip()

        if not has_any_information(text, fields):
            result = self._base_result(text, fields)
            result.message = "未提供症状信息，不作证候判断（ABSTAIN）"
            return result

        ood, reason = is_out_of_domain(text, self.label_space)
        if ood:
            result = self._base_result(text, fields)
            result.status = STATUS_OUT_OF_DOMAIN
            result.message = f"{reason}，不作证候判断（OUT_OF_DOMAIN）"
            return result

        scores = np.asarray(self.predictor.predict_proba([text]), dtype=np.float64)[0]
        return self._decide(text, fields, scores)

    def predict_batch(
        self, texts: Sequence[str], fields: Sequence[Mapping[str, object] | None] | None = None
    ) -> list[InferenceResult]:
        fields = list(fields) if fields is not None else [None] * len(texts)
        if len(fields) != len(texts):
            raise ValueError(f"length mismatch: {len(texts)} texts vs {len(fields)} field sets")
        clean = [(t or "").strip() for t in texts]
        # 可判定样本的 probabilities 一次批量请求（GPU 批量前向）；
        # ABSTAIN / OUT_OF_DOMAIN 是逐样本前置分支，不消耗模型调用。
        decidable = [
            i
            for i, (t, f) in enumerate(zip(clean, fields, strict=False))
            if has_any_information(t, f) and not is_out_of_domain(t, self.label_space)[0]
        ]
        results: list[InferenceResult | None] = [None] * len(texts)
        if decidable:
            probs = np.asarray(self.predictor.predict_proba([clean[i] for i in decidable]))
            for row, i in enumerate(decidable):
                results[i] = self._decide(clean[i], fields[i], probs[row])
        for i, t in enumerate(clean):
            if results[i] is None:
                results[i] = self.predict_one(t, fields[i])
        return results  # type: ignore[return-value]


def save_threshold(threshold_info: dict[str, float], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(threshold_info, fh, ensure_ascii=False, indent=2)
