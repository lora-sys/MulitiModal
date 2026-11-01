"""BERT 模型契约（#53）：本地权重加载、predict_proba、save/load 往返。

transformers 未安装或本地权重缺失时 skip；真实训练在远程开发机 GPU 执行。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from syndrome.labels import LabelSpace

pytest.importorskip("transformers", reason="transformers 未安装")
pytest.importorskip("torch", reason="torch 未安装")

WEIGHTS = Path("data/local/models/bert-base-chinese")

pytestmark = pytest.mark.slow

SYNTH_VOCAB = ("甲证", "乙证", "丙证")


def weights_dir() -> Path:
    if not WEIGHTS.exists():
        pytest.skip(f"本地基座权重缺失：{WEIGHTS}（权重不入 Git，经 rsync 分发）")
    return WEIGHTS


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """bert-base-chinese + 3 类小词表；仅前向/保存，不做完整训练。"""
    from syndrome.bert_model import BertSyndromeClassifier

    out = tmp_path_factory.mktemp("bert-smoke")
    clf = BertSyndromeClassifier(LabelSpace(SYNTH_VOCAB), str(weights_dir()), seed=20261101)
    yield clf, out


def test_predict_proba_shape_and_rows(tiny):
    clf, _out = tiny
    probs = clf.predict_proba(["头晕心悸两年，加重一周", "无关文本"])
    assert probs.shape == (2, 3)
    assert all(abs(row.sum() - 1.0) < 1e-4 for row in probs)


def test_save_load_roundtrip(tiny, tmp_path):
    clf, out = tiny
    meta = clf.save(str(out))
    assert meta["model_version"] == "tcm-sd-syndrome-bert-v1"
    assert len(meta["weight_sha256"]) == 64
    assert (out / "label_space.json").exists()
    saved = json.loads((out / "label_space.json").read_text(encoding="utf-8"))
    assert saved["labels"] == list(SYNTH_VOCAB)

    from syndrome.bert_model import BertSyndromeClassifier

    loaded = BertSyndromeClassifier.load(str(out), device="cpu")
    a = clf.predict_proba(["畏寒肢冷"])
    b = loaded.predict_proba(["畏寒肢冷"])
    assert a.shape == b.shape == (1, 3)
    # 同权重同输入：确定性一致
    assert abs(a[0].argmax() - b[0].argmax()) == 0


def test_train_config_roundtrip_in_save(tiny, tmp_path):
    """超参与 provenance 随模型入档（#53: 记录全部超参与 seed）。"""
    from syndrome.bert_model import BertTrainConfig

    clf, _out = tiny
    out = tmp_path / "with-config"
    config = BertTrainConfig(model_path=str(WEIGHTS), epochs=1)
    clf.save(
        str(out),
        config=config,
        history=[{"epoch": 1, "dev_macro_f1": 0.5}],
        provenance={"seed": 20261101},
    )
    saved = json.loads((out / "train_config.json").read_text(encoding="utf-8"))
    assert saved["epochs"] == 1 and saved["seed"] == 20261101
    assert json.loads((out / "history.json").read_text(encoding="utf-8"))[0]["dev_macro_f1"] == 0.5
