"""真实 TCM-SD 数据契约测试（#53）。

data/local/tcm-sd 缺失或 data.tcm_sd（D1 交付物）未落位时整体 skip——
不伪造真实数据结论。
"""

from __future__ import annotations

from pathlib import Path

import pytest

RAW_DIR = Path("data/local/tcm-sd")

pytest.importorskip("data.tcm_sd", reason="data.tcm_sd (#53 D1) 尚未落位")

pytestmark = pytest.mark.slow


def raw_dir() -> Path:
    if not RAW_DIR.exists():
        pytest.skip("data/local/tcm-sd 不存在：真实数据不入 Git")
    return RAW_DIR


def test_taxonomy_id_consistency():
    """labels.py 的冻结 taxonomy id 必须与 data.tcm_sd 的契约常量一致。"""
    import data.tcm_sd as tcm_sd
    from syndrome.labels import LABEL_TAXONOMY_ID

    assert tcm_sd.LABEL_TAXONOMY_ID == LABEL_TAXONOMY_ID


def test_vocab_size_and_label_space():
    from syndrome.labels import load_label_space

    space = load_label_space(str(raw_dir()))
    assert len(space) == 148


def test_load_records_and_knowledge():
    import data.tcm_sd as tcm_sd

    d = str(raw_dir())
    records = tcm_sd.load_tcm_sd_records(d)
    assert len(records) == 43180 + 5486 + 5486
    knowledge = tcm_sd.load_syndrome_knowledge(d)
    assert len(knowledge) == 1027


def test_subject_partitions_disjoint_and_total():
    """同一 user_id 只能属于一个 partition（#53 冻结契约）。"""
    from syndrome.experiment import load_partitions

    records, partitions, space, aux = load_partitions(str(raw_dir()))
    train, dev, test = partitions["train"], partitions["dev"], partitions["test"]
    assert not (train & dev) and not (train & test) and not (dev & test)
    assert len(train) + len(dev) + len(test) == len({r.user_id for r in records})
    # 官方文件边界仅作对照：记录对照数但分区必须与官方文件不同（user_id 跨文件重叠）
    assert aux["official_file_counts"]["train"] == 43180


def test_all_norm_syndromes_in_frozen_vocab():
    """真实数据的 norm_syndrome 必须全部落在冻结词表内（UNKNOWN 计数应为 0）。"""
    from syndrome.experiment import load_partitions, scenario_data

    records, partitions, space, _ = load_partitions(str(raw_dir()))
    data = scenario_data(records, partitions, space, "short")
    assert data["unknown_labels"] == {"train": 0, "dev": 0}


def test_text_builders_nonempty():
    import data.tcm_sd as tcm_sd
    from syndrome.experiment import load_partitions

    records, partitions, _space, _ = load_partitions(str(raw_dir()))
    train_records = [r for r in records if r.user_id in partitions["train"]][:50]
    assert all(tcm_sd.build_clinical_text(r).strip() for r in train_records)
    assert all(tcm_sd.build_short_text(r).strip() for r in train_records)
