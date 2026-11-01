"""experiment.py 管道胶水测试（SYNTHETIC）。

向 sys.modules 注入一个实现 #53 冻结契约的最小 fake data.tcm_sd，
在不依赖 D1 落位、不读任何真实数据的前提下验证：
load_partitions → scenario_data → 基线训练/评估 → 报告落盘 的完整链路，
以及 test partition 完全不被触碰。
"""

from __future__ import annotations

import json
import sys
import types

import pytest

from syndrome.labels import LABEL_TAXONOMY_ID

pytestmark = pytest.mark.synthetic

SYNTH_VOCAB = ("甲证", "乙证", "丙证", "丁证")

# user_id -> partition：train 6 / dev 2 / test 2
PARTITION_OF = {f"u{i}": ("train" if i < 6 else "dev" if i < 8 else "test") for i in range(10)}


def _fake_record(i: int) -> types.SimpleNamespace:
    marker = "标记" + "甲乙丙丁"[i % 4]
    return types.SimpleNamespace(
        user_id=f"u{i}",
        lcd_id=f"L{i}",
        lcd_name="病名",
        syndrome="原证",
        norm_syndrome=SYNTH_VOCAB[i % 4],
        chief_complaint=f"主诉{i}{marker}。",
        description=f"现病史{i}{marker}，病程一周。",
        detection="舌脉检查。",
        source_file=("train.json" if i < 7 else "dev.json" if i < 9 else "test.json"),
    )


def _fake_dedup(records, partitions):
    """与 data.tcm_sd.deduplicate_evaluation_records 相同契约的最小实现：
    dev 剔除与 train 临床全文逐字节相同者；test 对 train∪dev；train 全量保留。"""
    def text(r):
        return f"{r.chief_complaint}{r.description}{r.detection}"

    train_texts = {text(r) for r in records if r.user_id in partitions["train"]}
    dev_texts = {text(r) for r in records if r.user_id in partitions["dev"]}
    kept, dropped = [], {"dev": 0, "test": 0}
    for r in records:
        t = text(r)
        if r.user_id in partitions["train"]:
            kept.append(r)
        elif r.user_id in partitions["dev"]:
            if t in train_texts:
                dropped["dev"] += 1
            else:
                kept.append(r)
        elif r.user_id in partitions["test"]:
            if t in train_texts or t in dev_texts:
                dropped["test"] += 1
            else:
                kept.append(r)
        else:
            raise ValueError("record outside every partition")
    return kept, {
        "policy": "one-way",
        "text_basis": "build_clinical_text",
        "dropped": dropped,
        "kept": {"train": 0, "dev": 0, "test": 0},
    }


@pytest.fixture
def fake_tcm_sd(monkeypatch):
    fake = types.ModuleType("data.tcm_sd")
    fake.RAW_DIR_NAME = "data/local/tcm-sd"
    fake.SPLIT_SEED = 20261101
    fake.LABEL_TAXONOMY_ID = LABEL_TAXONOMY_ID
    fake.load_tcm_sd_records = lambda raw_dir: [_fake_record(i) for i in range(10)]
    fake.load_syndrome_vocab = lambda raw_dir: list(SYNTH_VOCAB)
    fake.load_syndrome_knowledge = lambda raw_dir: {}
    fake.subject_partitions = lambda records, seed=20261101: {
        name: frozenset(u for u, p in PARTITION_OF.items() if p == name)
        for name in ("train", "dev", "test")
    }
    fake.build_clinical_text = lambda r: f"{r.chief_complaint}{r.description}{r.detection}"
    fake.build_short_text = lambda r: r.chief_complaint
    fake.deduplicate_evaluation_records = _fake_dedup
    monkeypatch.setitem(sys.modules, "data.tcm_sd", fake)
    # 真实 data.tcm_sd 可能已被其他测试文件在收集期导入并挂到父包上；
    # "import data.tcm_sd as x" 优先取父包属性，必须一并替换
    data_pkg = sys.modules.get("data")
    if data_pkg is not None:
        monkeypatch.setattr(data_pkg, "tcm_sd", fake, raising=False)
    yield fake


def test_load_partitions_glue(fake_tcm_sd):
    from syndrome.experiment import load_partitions

    records, partitions, space, aux = load_partitions("data/local/tcm-sd")
    assert len(records) == 10
    assert len(space) == 4
    assert len(partitions["train"]) == 6 and len(partitions["test"]) == 2
    assert aux["official_file_counts"] == {"train": 7, "dev": 2, "test": 1}


def test_scenario_data_never_touches_test_partition(fake_tcm_sd):
    from syndrome.experiment import load_partitions, scenario_data

    records, partitions, space, _ = load_partitions("data/local/tcm-sd")
    for scenario in ("clinical", "short"):
        data = scenario_data(records, partitions, space, scenario)
        assert data["unknown_labels"] == {"train": 0, "dev": 0}
        assert len(data["train"]["texts"]) == 6
        assert len(data["dev"]["texts"]) == 2
        # 任何文本都不得来自 test partition 的用户（u8/u9）
        all_texts = data["train"]["texts"] + data["dev"]["texts"]
        assert not any("主诉8" in t or "主诉9" in t for t in all_texts)


def test_unknown_labels_excluded_not_fabricated(fake_tcm_sd):
    """词表外 norm_syndrome 被剔除并计数，绝不映射到相近类别。"""
    from syndrome.experiment import load_partitions, scenario_data

    records, partitions, space, _ = load_partitions("data/local/tcm-sd")
    records[0].norm_syndrome = "词表外证"
    data = scenario_data(records, partitions, space, "short")
    assert data["unknown_labels"] == {"train": 1, "dev": 0}
    assert len(data["train"]["texts"]) == 5


def test_scenario_data_dedups_dev_against_train(fake_tcm_sd):
    """R1 F1：dev 中与 train 病例临床全文逐字节相同者被剔除，train 全量保留。"""
    from syndrome.experiment import load_partitions, scenario_data

    records, partitions, space, _ = load_partitions("data/local/tcm-sd")
    # u0 在 train，u6 在 dev：把 u6 的临床文本改成与 u0 逐字节相同（标签冲突组同理）
    for field in ("chief_complaint", "description", "detection"):
        setattr(records[6], field, getattr(records[0], field))
    assert records[6].norm_syndrome != records[0].norm_syndrome  # 标签冲突组
    data = scenario_data(records, partitions, space, "clinical")
    assert len(data["train"]["texts"]) == 6  # train 侧不动
    assert len(data["dev"]["texts"]) == 1
    assert data["dedup"]["dropped"] == {"dev": 1, "test": 0}
    # 短文本口径用同一临床全文去重判定（文本基线不随场景切换）
    data_short = scenario_data(records, partitions, space, "short")
    assert len(data_short["dev"]["texts"]) == 1


def test_baseline_experiment_end_to_end(fake_tcm_sd, tmp_path):
    from syndrome.experiment import run_baseline_experiment

    report = run_baseline_experiment("data/local/tcm-sd", str(tmp_path), n_boot=30)
    assert report["taxonomy_id"] == LABEL_TAXONOMY_ID
    assert set(report["scenarios"]) == {"clinical", "short"}
    for summary in report["scenarios"].values():
        assert "macro_f1" in summary and "macro_f1_ci" in summary
        assert "vs_baseline" in summary
        assert summary["n_samples"] == 2  # dev partition 大小
    for name in ("baselines-report.json", "baselines-clinical.json", "baselines-short.json"):
        assert (tmp_path / name).exists()
    loaded = json.loads((tmp_path / "baselines-report.json").read_text(encoding="utf-8"))
    assert loaded["n_labels"] == 4


def test_robustness_suite_end_to_end(fake_tcm_sd, tmp_path):
    from syndrome.experiment import ADVERSARIAL_PROBES, run_robustness_suite

    report = run_robustness_suite("data/local/tcm-sd", str(tmp_path), n_boot=30)
    statuses = report["adversarial_statuses"]
    # 前 5 个探针（空/空白/英文/乱码）必须全部被拒判
    assert all(s != "OK" for s in statuses[:5])
    assert {statuses[0], statuses[1]} == {"ABSTAIN"}
    assert {statuses[2], statuses[3], statuses[4]} == {"OUT_OF_DOMAIN"}
    # 证候名探针的状态在真实词表下才可断言为 OUT_OF_DOMAIN，此处只要求状态合法
    assert all(s in {"OK", "UNCERTAIN", "ABSTAIN", "OUT_OF_DOMAIN"} for s in statuses[5:])
    assert report["adversarial_reject_rate"] >= 5 / len(ADVERSARIAL_PROBES)
    assert report["negation_robustness"] == "NOT_AVAILABLE"
    assert "degradation_truncated" in report
