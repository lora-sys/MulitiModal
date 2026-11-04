"""tests/syndrome 共享 fixture：全部为显式 SYNTHETIC 数据（#26/#35 边界）。

标签/文本都是人工构造，不含任何真实病例内容。
"""

from __future__ import annotations

import numpy as np
import pytest

from syndrome.labels import LabelSpace

SYNTHETIC_VOCAB = ("气虚血瘀证", "肝郁气滞证", "脾肾阳虚证", "心脾两虚证", "湿热下注证")

# 每类一个可区分的合成症状词（仅测试用，非医学知识）
CLASS_MARKERS = {
    "气虚血瘀证": "刺痛固定",
    "肝郁气滞证": "胁肋胀闷叹气",
    "脾肾阳虚证": "畏寒肢冷泄泻",
    "心脾两虚证": "心悸健忘失眠多梦",
    "湿热下注证": "尿频灼热黄腻",
}


def synthetic_text(label: str, rng: np.random.Generator) -> str:
    marker = CLASS_MARKERS[label]
    filler = ["患者", "近期", "明显", "反复", "偶有", "伴", "一周"]
    parts = [rng.choice(filler) for _ in range(rng.integers(2, 5))]
    return "".join(parts) + marker + "".join(rng.choice(filler) for _ in range(rng.integers(0, 3)))


def synthetic_corpus(n_per_class: int = 12, seed: int = 7):
    rng = np.random.default_rng(seed)
    texts, labels = [], []
    for idx, label in enumerate(SYNTHETIC_VOCAB):
        for _ in range(n_per_class):
            texts.append(synthetic_text(label, rng))
            labels.append(idx)
    order = rng.permutation(len(texts))
    return [texts[i] for i in order], [labels[i] for i in order]


@pytest.fixture(scope="session")
def label_space() -> LabelSpace:
    return LabelSpace(SYNTHETIC_VOCAB)


@pytest.fixture(scope="session")
def corpus():
    return synthetic_corpus()


@pytest.fixture(scope="session")
def make_corpus():
    """供单个测试自定规模/seed 地生成合成语料。"""
    return synthetic_corpus


class FixedPredictor:
    """确定性假模型：按类别标记词给分，无标记时给低均匀分。

    用于推理接口的状态机测试，不涉及任何训练。
    """

    def __init__(self, label_space: LabelSpace, confidence: float = 0.9) -> None:
        self.label_space = label_space
        self.confidence = confidence
        self.calls: list[list[str]] = []

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        low = (1 - self.confidence) / (len(self.label_space) - 1)
        out = np.full((len(texts), len(self.label_space)), low)
        for row, text in enumerate(texts):
            hit = False
            for idx, label in enumerate(self.label_space.labels):
                if CLASS_MARKERS[label] in text:
                    out[row, idx] = self.confidence
                    hit = True
            if not hit:
                out[row] = 1.0 / len(self.label_space)
        return out


@pytest.fixture
def fixed_predictor(label_space) -> FixedPredictor:
    return FixedPredictor(label_space)


class FakeKnowledgeEntry:
    def __init__(self, name: str, definition: str = "", typical: str = "") -> None:
        self.Name = name
        self.Definition = definition
        self.Typical_performance = typical


@pytest.fixture
def fake_knowledge() -> dict:
    return {
        "心脾两虚证": FakeKnowledgeEntry(
            "心脾两虚证",
            definition="是指心血不足，脾气虚弱所表现的证候。",
            typical="心悸，失眠，多梦，健忘，食欲不振，倦怠乏力。",
        ),
        "脾肾阳虚证": FakeKnowledgeEntry(
            "脾肾阳虚证",
            definition="是指脾肾阳气亏虚，温化失权所表现的证候。",
            typical="畏寒肢冷，久泻不止，腰膝冷痛。",
        ),
    }
