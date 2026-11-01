"""bert-base-chinese 微调（#53 Scope 2 神经模型）。

- 权重从本地目录加载（远程机无外网，权重经 rsync 放 data/local/models/，
  git-ignored），绝不在线下载；
- 每 epoch 在 dev partition 上评估 macro-F1 选最佳 epoch（模型选择只用
  train/dev，holdout 不参与）；
- 保存：模型权重 + 冻结词表 + 全部超参 + seed + torch/transformers 版本 +
  权重 SHA-256，供 provenance 入档与复现；
- torch/transformers 惰性导入：未安装时仅影响本模块，不影响包内其他接口。
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from syndrome.baselines import environment_record
from syndrome.labels import LabelSpace
from syndrome.metrics import macro_f1

MODEL_VERSION = "tcm-sd-syndrome-bert-v1"


@dataclass
class BertTrainConfig:
    """全部超参入档；改动须经 dev macro-F1 选择并记录在 history 里。"""

    model_path: str  # 本地 bert-base-chinese 权重目录
    max_len: int = 256
    lr: float = 2e-5
    batch_size: int = 32
    eval_batch_size: int = 128
    epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    seed: int = 20261101


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _artifact_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pick_device(device: str | None) -> str:
    import torch

    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


class BertSyndromeClassifier:
    """LabelSpace + 本地 bert-base-chinese 的封装。"""

    def __init__(self, label_space: LabelSpace, model_path: str, seed: int = 20261101) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        _seed_everything(seed)
        self.label_space = label_space
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=len(label_space)
        )
        self.seed = seed
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self._device)

    # ------------------------------------------------------------------ 数据

    def _encode(self, texts: list[str], max_len: int):
        return self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )

    # ------------------------------------------------------------------ 训练

    def fit(
        self,
        train_texts: list[str],
        train_labels: list[int],
        dev_texts: list[str],
        dev_labels: list[int],
        config: BertTrainConfig,
        output_dir: str,
    ) -> dict:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from transformers import get_linear_schedule_with_warmup

        if len(train_texts) != len(train_labels) or len(dev_texts) != len(dev_labels):
            raise ValueError("texts/labels length mismatch")
        self._device = _pick_device(None)
        _seed_everything(config.seed)

        train_enc = self._encode(train_texts, config.max_len)
        train_ds = TensorDataset(
            train_enc["input_ids"],
            train_enc["attention_mask"],
            torch.tensor(train_labels, dtype=torch.long),
        )
        train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)

        dev_input = self._encode(dev_texts, config.max_len)
        dev_ids = dev_input["input_ids"]
        dev_mask = dev_input["attention_mask"]
        dev_labels_t = torch.tensor(dev_labels, dtype=torch.long)

        self.model.to(self._device)
        self._fit_max_len = config.max_len
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        total_steps = len(train_dl) * config.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            int(total_steps * config.warmup_ratio),
            total_steps,
        )

        history: list[dict] = []
        best_f1 = -1.0
        best_state: dict | None = None
        for epoch in range(1, config.epochs + 1):
            self.model.train()
            total_loss = 0.0
            for batch in train_dl:
                ids, mask, y = (t.to(self._device) for t in batch)
                optimizer.zero_grad()
                logits = self.model(input_ids=ids, attention_mask=mask).logits
                loss = torch.nn.functional.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += float(loss.item()) * len(y)
            train_loss = total_loss / len(train_ds)

            dev_pred, dev_scores = self._predict_tensors(dev_ids, dev_mask, config.eval_batch_size)
            dev_f1 = macro_f1(dev_labels_t.tolist(), dev_pred, len(self.label_space))
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 6),
                    "dev_macro_f1": round(dev_f1, 6),
                }
            )
            if dev_f1 > best_f1:
                best_f1 = dev_f1
                best_state = {
                    k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                }

        if best_state is not None:
            self.model.load_state_dict(best_state)

        provenance = environment_record(
            config.seed,
            extra={
                "torch_version": torch.__version__,
                "transformers_version": __import__("transformers").__version__,
                "device": self._device,
                "base_model_path": config.model_path,
                "train_size": len(train_texts),
                "dev_size": len(dev_texts),
            },
        )
        self.save(output_dir, config=config, history=history, provenance=provenance)
        return {
            "history": history,
            "best_dev_macro_f1": round(best_f1, 6),
            "provenance": provenance,
        }

    # ------------------------------------------------------------------ 推理

    def _predict_tensors(self, input_ids, attention_mask, batch_size: int):
        import torch

        self.model.eval()
        preds, all_scores = [], []
        with torch.no_grad():
            for start in range(0, len(input_ids), batch_size):
                ids = input_ids[start : start + batch_size].to(self._device)
                mask = attention_mask[start : start + batch_size].to(self._device)
                logits = self.model(input_ids=ids, attention_mask=mask).logits
                scores = torch.softmax(logits, dim=-1)
                preds.append(scores.argmax(dim=-1).cpu())
                all_scores.append(scores.cpu())
        scores = torch.cat(all_scores).numpy()
        return torch.cat(preds).tolist(), scores

    def predict_proba(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        cfg_batch = batch_size or 128
        enc = self._encode(texts, self._max_len_or_default())
        _, scores = self._predict_tensors(enc["input_ids"], enc["attention_mask"], cfg_batch)
        return scores

    def _max_len_or_default(self) -> int:
        return getattr(self, "_fit_max_len", 256)

    # ------------------------------------------------------------------ 持久化

    def save(
        self,
        output_dir: str,
        config: BertTrainConfig | None = None,
        history: list[dict] | None = None,
        provenance: dict | None = None,
    ) -> dict:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(out)
        self.tokenizer.save_pretrained(out)
        with open(out / "label_space.json", "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "taxonomy_id": self.label_space.taxonomy_id,
                    "labels": list(self.label_space.labels),
                },
                fh,
                ensure_ascii=False,
                indent=2,
            )
        if config is not None:
            with open(out / "train_config.json", "w", encoding="utf-8") as fh:
                json.dump(asdict(config), fh, ensure_ascii=False, indent=2)
        if history is not None:
            with open(out / "history.json", "w", encoding="utf-8") as fh:
                json.dump(history, fh, ensure_ascii=False, indent=2)
        if provenance is not None:
            with open(out / "provenance.json", "w", encoding="utf-8") as fh:
                json.dump(provenance, fh, ensure_ascii=False, indent=2)

        weight_file = out / "model.safetensors"
        weight_sha = _artifact_sha256(weight_file) if weight_file.exists() else "NOT_AVAILABLE"
        meta = {
            "model_version": MODEL_VERSION,
            "weight_sha256": weight_sha,
            "seed": self.seed,
        }
        with open(out / "artifact_meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)
        return meta

    @classmethod
    def load(cls, directory: str, device: str | None = None) -> BertSyndromeClassifier:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        d = Path(directory)
        with open(d / "label_space.json", encoding="utf-8") as fh:
            meta = json.load(fh)
        label_space = LabelSpace(meta["labels"])
        cls_new = cls.__new__(cls)
        import torch

        cls_new.label_space = label_space
        cls_new.tokenizer = AutoTokenizer.from_pretrained(str(d))
        cls_new.model = AutoModelForSequenceClassification.from_pretrained(str(d))
        cls_new.seed = meta.get("seed", -1)
        cls_new._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cls_new.model.to(cls_new._device)
        return cls_new
