"""LabelSpace 契约：冻结词表、UNKNOWN/OUT_OF_DOMAIN 处理、禁止凭空补标签。"""

from __future__ import annotations

import pytest

from syndrome.labels import (
    LABEL_TAXONOMY_ID,
    STATUS_ABSTAIN,
    STATUS_OK,
    STATUS_OUT_OF_DOMAIN,
    STATUS_UNCERTAIN,
    UNKNOWN,
    LabelSpace,
)

pytestmark = pytest.mark.synthetic


def test_taxonomy_id_is_frozen():
    assert LABEL_TAXONOMY_ID == "tcm-sd-syndrome-vocab-v1"


def test_status_values_cover_contract():
    assert {STATUS_OK, STATUS_UNCERTAIN, STATUS_ABSTAIN, STATUS_OUT_OF_DOMAIN} == {
        "OK",
        "UNCERTAIN",
        "ABSTAIN",
        "OUT_OF_DOMAIN",
    }


def test_label_space_basics(label_space):
    assert len(label_space) == 5
    assert label_space.index_of("气虚血瘀证") == 0
    assert label_space.name(0) == "气虚血瘀证"
    assert label_space.taxonomy_id == LABEL_TAXONOMY_ID
    assert "气虚血瘀证" in label_space
    assert "不存在的证" not in label_space


def test_duplicate_labels_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        LabelSpace(("甲证", "甲证", "乙证"))


def test_empty_vocab_rejected():
    with pytest.raises(ValueError):
        LabelSpace(())


def test_resolve_unknown_returns_none_never_fabricates(label_space):
    """词表外标签必须解析为 UNKNOWN（None），绝不映射到相近的已知类别。"""
    assert label_space.resolve("气虚不摄证") is UNKNOWN
    assert label_space.resolve(None) is UNKNOWN
    assert label_space.resolve_all(["气虚血瘀证", "编造证", None]) == [0, UNKNOWN, UNKNOWN]
    # UNKNOWN 不等于任何已知类别索引
    assert UNKNOWN not in range(len(label_space))


def test_index_of_unknown_raises_instead_of_fallback(label_space):
    with pytest.raises(KeyError):
        label_space.index_of("词表外证")


def test_label_space_is_immutable_view(label_space):
    labels = label_space.labels
    assert isinstance(labels, tuple)
