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
    """bert-base-chinese + 3 类小词表；仅前向/保存，不做完整训练。

    predict_proba / save 的 #53 守卫（_require_fitted_max_len）要求模型带着已拟合的
    max_len；本 fixture 不做训练（真训练在远程开发机 GPU，见模块头），所以按训练
    配置的默认值声明该超参——与 test_save_load_restores_max_len_and_seed_without_weights
    里合成路径用的是同一种写法。
    """
    from syndrome.bert_model import BertSyndromeClassifier, BertTrainConfig

    out = tmp_path_factory.mktemp("bert-smoke")
    clf = BertSyndromeClassifier(LabelSpace(SYNTH_VOCAB), str(weights_dir()), seed=20261101)
    clf._fit_max_len = BertTrainConfig(model_path=str(weights_dir())).max_len
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

    loaded = BertSyndromeClassifier.load(
        str(out), expected_label_space=clf.label_space, device="cpu"
    )
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


@pytest.mark.synthetic
@pytest.mark.parametrize("malformation", ("nan", "wrong_class_count"))
def test_predict_tensors_rejects_malformed_probabilities(malformation):
    from types import SimpleNamespace

    import torch

    from syndrome.bert_model import BertSyndromeClassifier

    class MalformedModel:
        def eval(self):
            return None

        def __call__(self, input_ids, attention_mask):
            classes = len(SYNTH_VOCAB) + (malformation == "wrong_class_count")
            value = float("nan") if malformation == "nan" else 0.0
            logits = torch.full((len(input_ids), classes), value)
            return SimpleNamespace(logits=logits)

    classifier = BertSyndromeClassifier.__new__(BertSyndromeClassifier)
    classifier.label_space = LabelSpace(SYNTH_VOCAB)
    classifier.model = MalformedModel()
    classifier._device = "cpu"
    input_ids = torch.zeros((1, 4), dtype=torch.long)
    attention_mask = torch.ones((1, 4), dtype=torch.long)
    with pytest.raises(ValueError, match="model probabilities"):
        classifier._predict_tensors(input_ids, attention_mask, batch_size=1)


@pytest.mark.synthetic
def test_save_refuses_unknown_fitted_max_len(tmp_path):
    from syndrome.bert_model import BertSyndromeClassifier

    classifier = BertSyndromeClassifier.__new__(BertSyndromeClassifier)
    output_dir = tmp_path / "unknown-max-len"
    with pytest.raises(RuntimeError, match="fitted max_len"):
        classifier.save(str(output_dir))
    assert not output_dir.exists()


def test_save_load_restores_max_len_and_seed_without_weights(tmp_path, monkeypatch):
    """保存/加载应保留训练 max_len 和 seed，且测试不需要任何基座权重。"""
    from types import SimpleNamespace

    import torch
    import transformers

    from syndrome.bert_model import BertSyndromeClassifier, BertTrainConfig

    class FakeModel:
        def __init__(self):
            self.config = SimpleNamespace(id2label={}, label2id={})

        def save_pretrained(self, path):
            (Path(path) / "fake_config.json").write_text(
                json.dumps({
                    "id2label": self.config.id2label,
                    "label2id": self.config.label2id,
                }),
                encoding="utf-8",
            )

        def to(self, _device):
            return self

    class FakeTokenizer:
        def save_pretrained(self, path):
            (Path(path) / "vocab.txt").write_text(
                "[PAD]\n[UNK]\n合成词\n", encoding="utf-8"
            )

        def __call__(self, texts, *, max_length, **_kwargs):
            return {
                "input_ids": torch.zeros((len(texts), max_length), dtype=torch.long),
                "attention_mask": torch.ones((len(texts), max_length), dtype=torch.long),
            }

    def load_fake_model(_cls, path):
        config = json.loads((Path(path) / "fake_config.json").read_text(encoding="utf-8"))
        model = FakeModel()
        model.config = SimpleNamespace(
            id2label={int(key): value for key, value in config["id2label"].items()},
            label2id={key: int(value) for key, value in config["label2id"].items()},
        )
        return model

    monkeypatch.setattr(
        transformers.AutoModelForSequenceClassification,
        "from_pretrained",
        classmethod(load_fake_model),
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        classmethod(lambda _cls, _path: FakeTokenizer()),
    )

    config = BertTrainConfig(model_path="synthetic-model", max_len=73, seed=3187)
    classifier = BertSyndromeClassifier.__new__(BertSyndromeClassifier)
    classifier.label_space = LabelSpace(SYNTH_VOCAB)
    classifier.tokenizer = FakeTokenizer()
    classifier.model = FakeModel()
    classifier.seed = 20261101
    classifier._fit_max_len = config.max_len
    output_dir = tmp_path / "synthetic-artifact"
    saved_meta = classifier.save(str(output_dir), config=config)
    saved_labels = json.loads(
        (output_dir / "label_space.json").read_text(encoding="utf-8")
    )
    assert saved_labels["label_ids"] == [
        {"id": index, "label": label}
        for index, label in enumerate(SYNTH_VOCAB)
    ]
    assert saved_meta["label_ids"] == saved_labels["label_ids"]
    assert saved_meta["max_len"] == config.max_len
    assert saved_meta["tokenizer_assets_sha256"]
    assert saved_meta["preprocessing_sha256"]

    loaded = BertSyndromeClassifier.load(
        str(output_dir),
        expected_label_space=classifier.label_space,
        device="cpu",
    )
    assert loaded.model.config.id2label == dict(enumerate(SYNTH_VOCAB))
    assert loaded.seed == saved_meta["seed"] == config.seed
    assert loaded._require_fitted_max_len() == config.max_len
    encoded = loaded._encode(["合成症状文本"], loaded._require_fitted_max_len())
    assert encoded["input_ids"].shape == (1, config.max_len)

    resaved_dir = tmp_path / "synthetic-resaved"
    loaded.save(str(resaved_dir))
    resaved = BertSyndromeClassifier.load(
        str(resaved_dir), expected_label_space=loaded.label_space, device="cpu"
    )
    assert resaved._require_fitted_max_len() == config.max_len
    saved_config = json.loads(
        (resaved_dir / "train_config.json").read_text(encoding="utf-8")
    )
    assert saved_config["max_len"] == config.max_len

    saved_config.pop("max_len")
    (resaved_dir / "train_config.json").write_text(
        json.dumps(saved_config), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="max_len"):
        BertSyndromeClassifier.load(
            str(resaved_dir), expected_label_space=loaded.label_space, device="cpu"
        )

    saved_config["max_len"] = config.max_len
    (resaved_dir / "train_config.json").write_text(
        json.dumps(saved_config), encoding="utf-8"
    )
    (resaved_dir / "vocab.txt").write_text(
        "[PAD]\n[UNK]\nchanged-token\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="tokenizer assets or preprocessing"):
        BertSyndromeClassifier.load(
            str(resaved_dir), expected_label_space=loaded.label_space, device="cpu"
        )

@pytest.mark.synthetic
def test_load_rejects_label_order_that_disagrees_with_frozen_taxonomy(tmp_path, monkeypatch):
    import transformers

    class FakeModel:
        def to(self, _device):
            return self

    class FakeTokenizer:
        pass

    monkeypatch.setattr(
        transformers.AutoModelForSequenceClassification,
        "from_pretrained",
        classmethod(lambda _cls, _path: FakeModel()),
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        classmethod(lambda _cls, _path: FakeTokenizer()),
    )
    artifact = tmp_path / "reordered-artifact"
    artifact.mkdir()
    (artifact / "label_space.json").write_text(
        json.dumps({
            "taxonomy_id": LabelSpace(SYNTH_VOCAB).taxonomy_id,
            "labels": list(reversed(SYNTH_VOCAB)),
        }),
        encoding="utf-8",
    )
    (artifact / "artifact_meta.json").write_text(
        json.dumps({"seed": 3187}), encoding="utf-8"
    )
    (artifact / "train_config.json").write_text(
        json.dumps({"max_len": 32}), encoding="utf-8"
    )

    from syndrome.bert_model import BertSyndromeClassifier

    with pytest.raises(ValueError, match="label|taxonomy"):
        BertSyndromeClassifier.load(
            str(artifact),
            expected_label_space=LabelSpace(SYNTH_VOCAB),
            device="cpu",
        )
