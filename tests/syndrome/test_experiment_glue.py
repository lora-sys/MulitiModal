"""experiment.py 管道胶水测试（SYNTHETIC）。

向 sys.modules 注入一个实现 #53 冻结契约的最小 fake data.tcm_sd，
在不依赖 D1 落位、不读任何真实数据的前提下验证：
load_partitions → scenario_data → 基线训练/评估 → 报告落盘 的完整链路，
以及官方 test.json 不会被加载或传入候选路径。
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from syndrome.labels import LABEL_TAXONOMY_ID

pytestmark = pytest.mark.synthetic

SYNTH_VOCAB = tuple(f"合成证候{i:03}" for i in range(148))

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


def _fake_dedup(records, partitions, text_basis=None):
    """与 data.tcm_sd.deduplicate_evaluation_records 相同契约的最小实现：
    dev 剔除与 train 在 text_basis 下逐字节相同者；test 对 train∪dev；train
    全量保留。text_basis 缺省退化为拼接全文（真实默认是 build_clinical_text）。"""
    assert all(Path(r.source_file).name in {"train.json", "dev.json"} for r in records)
    if text_basis is None:
        def text(r):
            return f"{r.chief_complaint}{r.description}{r.detection}"
    else:
        text = text_basis

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


def _write_bound_threshold_artifact(
    model_dir: Path, aux: dict, scenario: str = "short"
) -> dict:
    from syndrome.bert_model import _preprocessing_binding
    from syndrome.experiment import _threshold_binding

    model_dir.mkdir(parents=True, exist_ok=True)
    weight_path = model_dir / "model.safetensors"
    weight_path.write_bytes(b"synthetic-only-model-weights")
    (model_dir / "vocab.txt").write_text(
        "[PAD]\n[UNK]\n合成词\n", encoding="utf-8"
    )
    (model_dir / "tokenizer_config.json").write_text(
        json.dumps({"do_lower_case": True}), encoding="utf-8"
    )
    max_len = 73
    (model_dir / "train_config.json").write_text(
        json.dumps({"max_len": max_len}), encoding="utf-8"
    )
    preprocessing = _preprocessing_binding(model_dir, max_len)
    (model_dir / "artifact_meta.json").write_text(
        json.dumps(preprocessing), encoding="utf-8"
    )
    threshold_info = {
        "threshold": 0.42,
        "dev_coverage": 0.9,
        "dev_accuracy_above": 0.8,
    }
    from syndrome.experiment import (
        _validated_threshold_metrics,
        load_partitions,
        scenario_data,
    )

    records, partitions, label_space, _ = load_partitions("synthetic")
    selection_identity = scenario_data(
        records, partitions, label_space, scenario
    )["threshold_selection_identity"]

    provenance = _threshold_binding(
        model_dir,
        aux,
        _validated_threshold_metrics(threshold_info),
        selection_identity=selection_identity,
    )
    evaluation = {
        "threshold": threshold_info,
        "threshold_provenance": provenance,
    }
    (model_dir / "dev-evaluation.json").write_text(
        json.dumps(evaluation), encoding="utf-8"
    )
    return provenance


def test_build_bert_inference_fails_closed_without_dev_threshold_artifact(
    tmp_path, fixed_predictor, monkeypatch
):
    from syndrome.bert_model import BertSyndromeClassifier
    from syndrome.experiment import build_bert_inference

    loaded = []

    def fake_load(cls, directory, expected_label_space=None, device=None):
        loaded.append(directory)
        return fixed_predictor

    monkeypatch.setattr(BertSyndromeClassifier, "load", classmethod(fake_load))
    with pytest.raises(FileNotFoundError, match="dev-evaluation.json"):
        build_bert_inference(str(tmp_path))
    assert loaded == []


def test_build_bert_inference_records_verified_dev_threshold_provenance(
    tmp_path, fixed_predictor, monkeypatch, fake_tcm_sd
):
    from syndrome.bert_model import BertSyndromeClassifier
    from syndrome.experiment import build_bert_inference, load_partitions

    _, _, _, aux = load_partitions("synthetic")
    expected_binding = _write_bound_threshold_artifact(tmp_path, aux)
    monkeypatch.setattr(
        BertSyndromeClassifier,
        "load",
        classmethod(lambda cls, directory, expected_label_space=None, device=None: fixed_predictor),
    )
    inference = build_bert_inference(
        str(tmp_path), knowledge={}, raw_dir="synthetic", scenario="short"
    )
    assert inference.threshold == 0.42
    assert inference.provenance["threshold_binding"] == expected_binding
    assert (
        inference.provenance["threshold_scope"]
        == "selected on dev partition; holdout untouched"
    )


def test_build_bert_inference_rejects_unverified_legacy_threshold(
    tmp_path, fixed_predictor, monkeypatch, fake_tcm_sd
):
    from syndrome.bert_model import BertSyndromeClassifier
    from syndrome.experiment import build_bert_inference

    (tmp_path / "model.safetensors").write_bytes(b"synthetic-only-model-weights")
    (tmp_path / "dev-evaluation.json").write_text(
        json.dumps(
            {"threshold": {
                "threshold": 0.42,
                "dev_coverage": 0.9,
                "dev_accuracy_above": 0.8,
            }}
        ),
        encoding="utf-8",
    )
    loaded = []
    monkeypatch.setattr(
        BertSyndromeClassifier,
        "load",
        classmethod(
            lambda cls, directory, expected_label_space=None, device=None: loaded.append(
                directory
            )
        ),
    )
    with pytest.raises(ValueError, match="UNVERIFIED"):
        build_bert_inference(
            str(tmp_path), knowledge={}, raw_dir="synthetic", scenario="short"
        )
    assert loaded == []


def test_build_bert_inference_rejects_stale_threshold_after_weight_change(
    tmp_path, fixed_predictor, monkeypatch, fake_tcm_sd
):
    from syndrome.bert_model import BertSyndromeClassifier
    from syndrome.experiment import build_bert_inference, load_partitions

    _, _, _, aux = load_partitions("synthetic")
    _write_bound_threshold_artifact(tmp_path, aux)
    (tmp_path / "model.safetensors").write_bytes(b"different synthetic weights")
    loaded = []
    monkeypatch.setattr(
        BertSyndromeClassifier,
        "load",
        classmethod(
            lambda cls, directory, expected_label_space=None, device=None: loaded.append(
                directory
            )
        ),
    )
    with pytest.raises(ValueError, match="model weights, preprocessing, and dev split"):
        build_bert_inference(
            str(tmp_path), knowledge={}, raw_dir="synthetic", scenario="short"
        )
    assert loaded == []


def test_build_bert_inference_rejects_changed_tokenizer_assets(
    tmp_path, fixed_predictor, monkeypatch, fake_tcm_sd
):
    from syndrome.bert_model import BertSyndromeClassifier
    from syndrome.experiment import build_bert_inference, load_partitions

    _, _, _, aux = load_partitions("synthetic")
    _write_bound_threshold_artifact(tmp_path, aux)
    (tmp_path / "vocab.txt").write_text("[PAD]\n[UNK]\nchanged\n", encoding="utf-8")
    loaded = []
    monkeypatch.setattr(
        BertSyndromeClassifier,
        "load",
        classmethod(
            lambda cls, directory, expected_label_space=None, device=None: (
                loaded.append(directory)
            )
        ),
    )
    with pytest.raises(ValueError, match="tokenizer assets or preprocessing"):
        build_bert_inference(
            str(tmp_path), knowledge={}, raw_dir="synthetic", scenario="short"
        )
    assert loaded == []


def test_build_bert_inference_rejects_max_len_drift(
    tmp_path, fixed_predictor, monkeypatch, fake_tcm_sd
):
    from syndrome.bert_model import BertSyndromeClassifier
    from syndrome.experiment import build_bert_inference, load_partitions

    _, _, _, aux = load_partitions("synthetic")
    _write_bound_threshold_artifact(tmp_path, aux)
    (tmp_path / "train_config.json").write_text(
        json.dumps({"max_len": 64}), encoding="utf-8"
    )
    loaded = []
    monkeypatch.setattr(
        BertSyndromeClassifier,
        "load",
        classmethod(
            lambda cls, directory, expected_label_space=None, device=None: (
                loaded.append(directory)
            )
        ),
    )
    with pytest.raises(ValueError, match="tokenizer assets or preprocessing"):
        build_bert_inference(
            str(tmp_path), knowledge={}, raw_dir="synthetic", scenario="short"
        )
    assert loaded == []


def test_build_bert_inference_rejects_stale_dev_manifest(
    tmp_path, fixed_predictor, monkeypatch, fake_tcm_sd
):
    from syndrome.bert_model import BertSyndromeClassifier
    from syndrome.experiment import build_bert_inference, load_partitions

    _, _, _, aux = load_partitions("synthetic")
    _write_bound_threshold_artifact(tmp_path, aux)
    fake_tcm_sd.EXPECTED_SHA256["dev.json"] = "f" * 64
    monkeypatch.setattr(
        BertSyndromeClassifier,
        "load",
        classmethod(lambda cls, directory, expected_label_space=None, device=None: fixed_predictor),
    )
    with pytest.raises(ValueError, match="model weights, preprocessing, and dev split"):
        build_bert_inference(
            str(tmp_path), knowledge={}, raw_dir="synthetic", scenario="short"
        )


def test_build_bert_inference_rejects_cross_split_threshold(
    tmp_path, fixed_predictor, monkeypatch, fake_tcm_sd
):
    from syndrome.bert_model import BertSyndromeClassifier
    from syndrome.experiment import build_bert_inference, load_partitions

    _, _, _, aux = load_partitions("synthetic")
    _write_bound_threshold_artifact(tmp_path, aux)
    fake_tcm_sd.subject_partitions = lambda records, seed=20261101: {
        "train": frozenset({f"u{i}" for i in range(7)}),
        "dev": frozenset({"u7"}),
        "test": frozenset({"u8", "u9"}),
    }
    monkeypatch.setattr(
        BertSyndromeClassifier,
        "load",
        classmethod(lambda cls, directory, expected_label_space=None, device=None: fixed_predictor),
    )
    with pytest.raises(ValueError, match="model weights, preprocessing, and dev split"):
        build_bert_inference(
            str(tmp_path), knowledge={}, raw_dir="synthetic", scenario="short"
        )


def test_build_bert_inference_rejects_cross_scenario_threshold(
    tmp_path, fixed_predictor, monkeypatch, fake_tcm_sd
):
    from syndrome.bert_model import BertSyndromeClassifier
    from syndrome.experiment import build_bert_inference, load_partitions

    _, _, _, aux = load_partitions("synthetic")
    _write_bound_threshold_artifact(tmp_path, aux, scenario="short")
    loaded = []
    monkeypatch.setattr(
        BertSyndromeClassifier,
        "load",
        classmethod(
            lambda cls, directory, expected_label_space=None, device=None: (
                loaded.append(directory)
            )
        ),
    )
    with pytest.raises(ValueError, match="threshold provenance"):
        build_bert_inference(
            str(tmp_path),
            knowledge={},
            raw_dir="synthetic",
            scenario="clinical",
        )
    assert loaded == []


def test_build_bert_inference_rejects_changed_threshold_selection_rows(
    tmp_path, monkeypatch, fake_tcm_sd
):
    from syndrome.bert_model import BertSyndromeClassifier
    from syndrome.experiment import build_bert_inference, load_partitions

    _, _, _, aux = load_partitions("synthetic")
    _write_bound_threshold_artifact(tmp_path, aux, scenario="clinical")
    original_loader = fake_tcm_sd.load_tcm_sd_development_records

    def changed_loader(raw_dir):
        records = original_loader(raw_dir)
        records[6].description += "合成病例文本变更"
        return records

    fake_tcm_sd.load_tcm_sd_development_records = changed_loader
    loaded = []
    monkeypatch.setattr(
        BertSyndromeClassifier,
        "load",
        classmethod(
            lambda cls, directory, expected_label_space=None, device=None: (
                loaded.append(directory)
            )
        ),
    )
    with pytest.raises(ValueError, match="threshold provenance"):
        build_bert_inference(
            str(tmp_path),
            knowledge={},
            raw_dir="synthetic",
            scenario="clinical",
        )
    assert loaded == []


@pytest.mark.parametrize("consumer", ("inference", "rescore"))
@pytest.mark.parametrize(
    ("metric", "replacement"),
    (
        ("threshold", 0.43),
        ("dev_coverage", 0.91),
        ("dev_accuracy_above", 0.81),
    ),
)
def test_threshold_consumers_reject_changed_selection_metrics(
    tmp_path, monkeypatch, fake_tcm_sd, consumer, metric, replacement
):
    from syndrome.bert_model import BertSyndromeClassifier
    from syndrome.experiment import (
        build_bert_inference,
        load_partitions,
        rescore_saved_bert,
    )

    _, _, _, aux = load_partitions("synthetic")
    model_dir = tmp_path if consumer == "inference" else tmp_path / "bert-short"
    _write_bound_threshold_artifact(model_dir, aux)
    evaluation_path = model_dir / "dev-evaluation.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["threshold"][metric] = replacement
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")

    loaded = []
    monkeypatch.setattr(
        BertSyndromeClassifier,
        "load",
        classmethod(
            lambda cls, directory, expected_label_space=None, device=None: loaded.append(
                directory
            )
        ),
    )
    with pytest.raises(ValueError, match="threshold provenance"):
        if consumer == "inference":
            build_bert_inference(
                str(model_dir), knowledge={}, raw_dir="synthetic", scenario="short"
            )
        else:
            rescore_saved_bert("synthetic", str(tmp_path), scenario="short")
    assert loaded == []


def test_bert_experiment_writes_bound_dev_threshold(
    tmp_path, monkeypatch, fake_tcm_sd
):
    import numpy as np

    from syndrome.bert_model import BertSyndromeClassifier
    from syndrome.experiment import run_bert_experiment

    def fake_init(self, label_space, model_path, seed=20261101):
        self.label_space = label_space
        self.seed = seed

    def fake_fit(self, train_texts, train_labels, dev_texts, dev_labels, config, output_dir):
        model_dir = Path(output_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "model.safetensors").write_bytes(b"synthetic-only-model-weights")
        (model_dir / "vocab.txt").write_text(
            "[PAD]\n[UNK]\n合成词\n", encoding="utf-8"
        )
        (model_dir / "tokenizer_config.json").write_text(
            json.dumps({"do_lower_case": True}), encoding="utf-8"
        )
        (model_dir / "train_config.json").write_text(
            json.dumps({"max_len": config.max_len}), encoding="utf-8"
        )
        from syndrome.bert_model import _preprocessing_binding

        preprocessing = _preprocessing_binding(model_dir, config.max_len)
        (model_dir / "artifact_meta.json").write_text(
            json.dumps(preprocessing), encoding="utf-8"
        )
        return {"best_dev_macro_f1": 0.0}

    def fake_predict_proba(self, texts):
        return np.full((len(texts), len(self.label_space)), 1.0 / len(self.label_space))

    monkeypatch.setattr(BertSyndromeClassifier, "__init__", fake_init)
    monkeypatch.setattr(BertSyndromeClassifier, "fit", fake_fit)
    monkeypatch.setattr(BertSyndromeClassifier, "predict_proba", fake_predict_proba)
    report = run_bert_experiment(
        "synthetic", "synthetic-base-model", str(tmp_path), epochs=1, n_boot=10
    )
    saved = json.loads(
        (tmp_path / "bert-short" / "dev-evaluation.json").read_text(encoding="utf-8")
    )
    assert saved["threshold_provenance"] == report["threshold_provenance"]
    assert saved["threshold_provenance"]["weights_sha256"]
    assert saved["threshold_provenance"]["development_manifest_sha256"]
    assert saved["threshold_provenance"]["split_identity_sha256"]
    assert saved["threshold_provenance"]["selection_scope"] == "dev_partition_only"
    assert saved["threshold_provenance"]["selection_partition"] == "dev"
    assert saved["threshold_provenance"]["selection_scenario"] == "short"
    assert saved["threshold_provenance"]["selection_text_builder"] == "build_short_text"
    assert saved["threshold_provenance"]["selection_record_count"] == 2
    assert len(saved["threshold_provenance"]["selection_dataset_sha256"]) == 64
    assert saved["threshold_provenance"]["max_len"] == 256
    assert saved["threshold_provenance"]["tokenizer_assets_sha256"]
    assert saved["threshold_provenance"]["preprocessing_sha256"]
    assert saved["threshold_provenance"]["preprocessing_contract"] == (
        "bert-fixed-padding-truncation-v1"
    )


@pytest.fixture
def fake_tcm_sd(monkeypatch):
    fake = types.ModuleType("data.tcm_sd")
    fake.RAW_DIR_NAME = "data/local/tcm-sd"
    fake.SPLIT_SEED = 20261101
    fake.LABEL_TAXONOMY_ID = LABEL_TAXONOMY_ID
    fake.DEVELOPMENT_RECORD_FILES = ("train.json", "dev.json")
    fake.VOCAB_FILE = "syndrome_vocab.txt"
    fake.EXPECTED_SHA256 = {
        "train.json": "a" * 64,
        "dev.json": "b" * 64,
        "syndrome_vocab.txt": "c" * 64,
    }
    fake.TRAIN_RATIO = 0.82
    fake.DEV_RATIO = 0.10
    fake.TEST_RATIO = 0.08
    fake.load_tcm_sd_records = lambda raw_dir: pytest.fail(
        "development experiment requested all official splits"
    )
    fake.load_tcm_sd_development_records = lambda raw_dir: [
        _fake_record(i) for i in range(9)
    ]
    fake.load_syndrome_vocab = lambda raw_dir: list(SYNTH_VOCAB)
    fake.load_syndrome_knowledge = lambda raw_dir: {}
    fake.subject_partitions = lambda records, seed=20261101: {
        name: frozenset(u for u, p in PARTITION_OF.items() if p == name)
        for name in ("train", "dev", "test")
    }

    def build_clinical_text(r):
        return f"{r.chief_complaint}{r.description}{r.detection}"

    def build_short_text(r):
        return r.chief_complaint

    # 场景身份记录的是被解析函数的 __name__（SCENARIOS 契约要求等于
    # "build_short_text"/"build_clinical_text"）；fake 必须提供同名具名函数
    fake.build_clinical_text = build_clinical_text
    fake.build_short_text = build_short_text
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
    assert len(records) == 9
    assert len(space) == 148
    assert len(partitions["train"]) == 6 and len(partitions["test"]) == 2
    assert all(Path(r.source_file).name != "test.json" for r in records)
    assert aux["development_file_counts"] == {"train": 7, "dev": 2}


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
    assert loaded["n_labels"] == 148


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
