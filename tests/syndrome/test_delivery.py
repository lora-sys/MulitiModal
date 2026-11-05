"""交付契约测试（全部 SYNTHETIC，不读真实数据 / 不读 holdout）。

覆盖：冻结冠军源码绑定、配置字段传递与向后兼容、选择集拆分与单向去重、
holdout 拒绝、工件保存/加载与哈希篡改、报告 schema、标签历史读取。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from syndrome.baselines import TfidfNgramBaseline, TfidfNgramConfig
from syndrome.delivery import (
    CHAMPION_COMMIT,
    CHAMPION_SOURCE_SHA256,
    N_CLASSES,
    REPORT_SCHEMA_VERSION,
    assert_holdout_untouched,
    champion_config,
    read_champion_result,
    selection_split,
    validate_report_schema,
    verify_champion_source,
)
from syndrome.labels import LABEL_TAXONOMY_ID
from syndrome.service import save_tfidf_logreg_artifact, tfidf_logreg_spec

pytestmark = pytest.mark.synthetic

REPO_ROOT = Path(__file__).resolve().parents[2]


def _frozen_hashes() -> tuple[str, str]:
    from research.prepare import SELECTION_DATASET_SHA256, SELECTION_MANIFEST_SHA256

    return SELECTION_MANIFEST_SHA256, SELECTION_DATASET_SHA256


def _synthetic_manifest(holdout=()) -> dict:
    manifest_sha, dataset_sha = _frozen_hashes()
    return {
        "manifest_sha256": manifest_sha,
        "dataset_sha256": dataset_sha,
        "split": {"train": ["s0", "s1", "s2"], "dev": ["s3", "s4"], "holdout": list(holdout)},
        "subject_row_by_id": {"s0": 1, "s1": 2, "s2": 3, "s3": 4, "s4": 5},
        "subject_user_by_id": {
            "s0": "u0", "s1": "u1", "s2": "u2", "s3": "u3", "s4": "u4",
        },
    }


def _synthetic_records() -> dict:
    return {
        1: SimpleNamespace(input_text="头痛发热", label=0),
        2: SimpleNamespace(input_text="腹痛泄泻", label=1),
        3: SimpleNamespace(input_text="心悸失眠", label=2),
        4: SimpleNamespace(input_text="头痛发热", label=0),  # 与 train 完全相同 → 剔除
        5: SimpleNamespace(input_text="胸闷气短", label=1),
    }


# ------------------------------------------------------------------ 冠军绑定


def test_frozen_champion_source_matches_delivery_constant():
    source = (REPO_ROOT / "src/research/train.py").read_bytes()
    assert hashlib.sha256(source).hexdigest() == CHAMPION_SOURCE_SHA256
    assert verify_champion_source(source) == CHAMPION_SOURCE_SHA256


def test_verify_champion_source_rejects_any_drift(monkeypatch):
    source = (REPO_ROOT / "src/research/train.py").read_bytes()
    tampered = source.replace(b"C=10.0", b"C=9.0")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        verify_champion_source(tampered)
    # 单独覆盖"hash 正确但字面量被改"的分支
    stripped = source.replace(b'analyzer="char_wb"', b'analyzer="char"')
    import syndrome.delivery as delivery
    monkeypatch.setattr(delivery, "CHAMPION_SOURCE_SHA256",
                        hashlib.sha256(stripped).hexdigest())
    with pytest.raises(ValueError, match="missing frozen literals"):
        verify_champion_source(stripped)


def test_champion_config_is_exact_and_does_not_change_defaults():
    cfg = champion_config()
    assert (cfg.analyzer, cfg.binary) == ("char_wb", True)
    assert (cfg.ngram_low, cfg.ngram_high, cfg.min_df) == (1, 4, 1)
    assert cfg.C == 10.0 and cfg.class_weight == "balanced" and cfg.max_iter == 2000
    assert cfg.solver == "ovr-liblinear" and cfg.sublinear_tf is True
    # #53 基线默认值保持不变（向后兼容）
    default = TfidfNgramConfig()
    assert (default.analyzer, default.binary) == ("char", False)
    assert (default.ngram_low, default.ngram_high, default.min_df) == (1, 3, 2)


def test_config_fields_reach_fitted_vectorizer_and_provenance():
    texts = ["甲类头晕气短", "甲类乏力自汗", "乙类胸闷刺痛", "乙类痛处固定", "丙类畏寒"]
    targets = [0, 0, 1, 1, 2]
    model = TfidfNgramBaseline(champion_config(seed=7), n_classes=3).fit(texts, targets)
    vectorizer = model._require().named_steps["tfidf"]
    assert vectorizer.analyzer == "char_wb"
    assert vectorizer.binary is True
    assert vectorizer.ngram_range == (1, 4)
    assert vectorizer.min_df == 1
    params = model.provenance()["params"]
    assert params["analyzer"] == "char_wb" and params["binary"] is True
    probs = model.predict_proba(["甲类头晕"])
    assert probs.shape == (1, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.all((probs >= 0) & (probs <= 1))


def test_predict_proba_masks_unseen_classes():
    """折内缺失类在 148 维输出中补 0，不凭空产生标签。"""
    model = TfidfNgramBaseline(
        TfidfNgramConfig(min_df=1, analyzer="char_wb", binary=True), n_classes=N_CLASSES
    ).fit(["甲类头晕", "甲类乏力", "乙类胸闷"], [2, 2, 7])
    probs = model.predict_proba(["甲类头晕"])
    assert probs.shape == (1, N_CLASSES)
    unseen = np.setdiff1d(np.arange(N_CLASSES), [2, 7])
    assert np.count_nonzero(probs[:, unseen]) == 0
    assert np.allclose(probs.sum(axis=1), 1.0)


# ------------------------------------------------------------------ 数据拆分


def test_selection_split_dedups_dev_against_train_and_keeps_train_duplicates():
    split = selection_split(_synthetic_manifest(), _synthetic_records())
    assert split["train_rows"] == 3
    assert split["dev_rows"] == 1
    assert split["dev_dedup_dropped"] == 1
    assert split["dev_texts"] == ["胸闷气短"]
    assert split["holdout_subjects"] == 0
    assert split["train_users"] == 3 and split["dev_users"] == 2


def test_assert_holdout_untouched_rejects_nonempty_holdout():
    with pytest.raises(ValueError, match="non-empty holdout"):
        assert_holdout_untouched(_synthetic_manifest(holdout=["s9"]))


def test_assert_holdout_untouched_rejects_wrong_frozen_hashes():
    manifest = _synthetic_manifest()
    manifest["manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="frozen protocol-v2 hashes"):
        assert_holdout_untouched(manifest)


def test_selection_split_rejects_shared_user_across_partitions():
    manifest = _synthetic_manifest()
    manifest["subject_user_by_id"]["s4"] = "u0"  # dev 与 train 同用户
    with pytest.raises(ValueError, match="share a user"):
        selection_split(manifest, _synthetic_records())


# ------------------------------------------------------------------ 工件往返


def test_artifact_save_load_roundtrip_and_hash_tamper(tmp_path):
    labels = ["甲证", "乙证", "丙证"]
    texts = ["甲类头晕气短", "甲类乏力自汗", "乙类胸闷刺痛", "乙类痛处固定", "丙类畏寒"]
    model = TfidfNgramBaseline(
        TfidfNgramConfig(min_df=1, analyzer="char_wb", binary=True), n_classes=3
    ).fit(texts, [0, 0, 1, 1, 2])
    spec = save_tfidf_logreg_artifact(
        model, labels, str(tmp_path / "artifact"),
        model_version="synthetic-delivery-v1", threshold=0.0,
        provenance={"source": "SYNTHETIC"}, model_id="synthetic-delivery",
    )
    loaded = tfidf_logreg_spec(str(tmp_path / "artifact")).loader()
    assert loaded.label_space.labels == tuple(labels)
    assert loaded.label_space.taxonomy_id == LABEL_TAXONOMY_ID
    assert loaded.predictor.predict(["甲类头晕"]) == model.predict(["甲类头晕"])
    assert spec.taxonomy_version == LABEL_TAXONOMY_ID

    # 负例：工件任一字节被改动，registry 哈希校验必须拒载
    tampered_dir = tmp_path / "tampered"
    tampered_dir.mkdir()
    for name in ("model.joblib", "labels.json", "registry_entry.json"):
        (tampered_dir / name).write_bytes((tmp_path / "artifact" / name).read_bytes())
    predictor_path = tampered_dir / "model.joblib"
    payload = bytearray(predictor_path.read_bytes())
    payload[0] ^= 0x01
    predictor_path.write_bytes(bytes(payload))
    with pytest.raises(ValueError, match="hash mismatch"):
        tfidf_logreg_spec(str(tampered_dir)).loader()


# ------------------------------------------------------------------ 报告契约


def test_validate_report_schema_accepts_complete_and_rejects_incomplete():
    complete = {key: {} for key in (
        "schema_version", "status", "champion", "inputs", "final_training",
        "acceptance", "threshold", "artifact", "inference", "robustness",
        "unverified_generalization",
    )}
    complete["schema_version"] = REPORT_SCHEMA_VERSION
    validate_report_schema(complete)
    del complete["robustness"]
    with pytest.raises(ValueError, match="missing required keys"):
        validate_report_schema(complete)
    complete["robustness"] = {}
    complete["schema_version"] = "wrong-v9"
    with pytest.raises(ValueError, match="unsupported schema_version"):
        validate_report_schema(complete)


def test_read_champion_result_reads_history_and_rejects_missing(tmp_path):
    header = ("commit\tmetric\tvalue\tdirection\tdecision\tdescription\t"
              "peak_vram_mb\tprotocol_sha256\tevidence\tpromotion\n")
    row = (f"{CHAMPION_COMMIT}\tsyndrome/macro_f1\t0.33332767910681194\tmaximize\tKEEP\t"
           "champion\tNOT_AVAILABLE\t" + "a" * 64 + "\tLOCAL_EXPERIMENTAL\tstrict\n")
    path = tmp_path / "results.tsv"
    path.write_text(header + row, encoding="utf-8")
    record = read_champion_result(path)
    assert record["commit"] == CHAMPION_COMMIT
    assert record["decision"] == "KEEP"
    assert record["history_rows"] == 1
    path.write_text(header, encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one champion row"):
        read_champion_result(path)


def test_champion_commit_is_full_hex():
    assert len(CHAMPION_COMMIT) == 40
    assert all(char in "0123456789abcdef" for char in CHAMPION_COMMIT)


def test_delivery_module_source_never_mentions_test_holdout_file():
    """源码不得出现对已耗尽解锁额度的历史 test split 的读取。"""
    source = (REPO_ROOT / "src/syndrome/delivery.py").read_text(encoding="utf-8")
    assert "test.json" not in source
    assert "load_tcm_sd_records" not in source
    assert "holdout" in source  # 显式断言/报告 holdout 未使用
