"""Label space, evidence-state constants and label resolution for #53 Stage 1.

不引入任何"凭空补标签"路径：词表外的证候一律解析为 UNKNOWN（None），
由调用方显式决定 ABSTAIN / OUT_OF_DOMAIN 等证据状态。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

# 冻结的标签词表版本（与 src/data/tcm_sd.py 的 LABEL_TAXONOMY_ID 一致，见 test_labels.py）
LABEL_TAXONOMY_ID = "tcm-sd-syndrome-vocab-v1"

# 证据状态显式化（#53 Scope 2 / AGENTS.md 硬规则 7）
STATUS_OK = "OK"
STATUS_UNCERTAIN = "UNCERTAIN"
STATUS_ABSTAIN = "ABSTAIN"
STATUS_OUT_OF_DOMAIN = "OUT_OF_DOMAIN"

# 词表外标签的显式状态：None 表示 UNKNOWN，禁止映射到任何已知类别
UNKNOWN: int | None = None


class LabelSpace:
    """冻结的证候词表。构造后不可变；只接受词表内标签，缺失按 UNKNOWN 处理。"""

    def __init__(self, labels: Sequence[str]) -> None:
        labels = tuple(labels)
        if not labels:
            raise ValueError("label vocabulary must not be empty")
        if len(set(labels)) != len(labels):
            duplicates = sorted({x for x in labels if list(labels).count(x) > 1})
            raise ValueError(f"duplicate labels in vocabulary: {duplicates}")
        self._labels = labels
        self._index = {name: i for i, name in enumerate(labels)}

    @property
    def labels(self) -> tuple[str, ...]:
        return self._labels

    @property
    def taxonomy_id(self) -> str:
        return LABEL_TAXONOMY_ID

    def __len__(self) -> int:
        return len(self._labels)

    def __contains__(self, label: object) -> bool:
        return label in self._index

    def index_of(self, label: str) -> int:
        """词表内标签 → 索引；词表外抛 KeyError（用于数据侧断言，不得静默回退）。"""
        try:
            return self._index[label]
        except KeyError:
            raise KeyError(f"label {label!r} is not in taxonomy {LABEL_TAXONOMY_ID}") from None

    def resolve(self, label: str | None) -> int | None:
        """词表外标签解析为 UNKNOWN（None），绝不映射到相近类别。"""
        if label is None:
            return UNKNOWN
        return self._index.get(label, UNKNOWN)

    def resolve_all(self, labels: Iterable[str | None]) -> list[int | None]:
        return [self.resolve(x) for x in labels]

    def name(self, index: int) -> str:
        return self._labels[index]


def load_label_space(raw_dir: str) -> LabelSpace:
    """从 TCM-SD 原始目录加载冻结词表（依赖 #53 契约的 data.tcm_sd，惰性导入）。"""
    from data.tcm_sd import load_syndrome_vocab

    return LabelSpace(load_syndrome_vocab(raw_dir))
