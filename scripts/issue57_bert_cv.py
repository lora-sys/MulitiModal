"""#57 C line: bert-base-chinese per-fold fine-tune on the short scenario.

Recipe (frozen in the #57 design): max_len 128, batch 32, lr in
{2e-5, 5e-5}, loss in {plain CE, class-balanced CE}, best epoch selected on
the deduplicated fold-dev (exact input_text dedup applied by
``research.task_sd.build_fold``), warmup 0.1, weight decay 0.01, AdamW with
a linear schedule.

Modes:
  rank — the 4 config points (lr x loss) at seed 20261101 on ONE fold
         (repeat 0, fold 0) to rank configs before committing GPU time;
  cv   — the winner's full 4-fold x 2-repeat CV at the requested seeds.

Fold membership, dedup, and metric reporting are imported (not forked) via
``scripts.issue57_common``. Weight SHA-256 is computed over the in-memory
state_dict serialization; full per-fold checkpoints are not written.

Usage (acam-lab, repo root):
    PYTHONPATH=src python -m scripts.issue57_bert_cv rank \
        --model data/local/models/bert-base-chinese \
        --out runs/issue57/c-bert-rank
    PYTHONPATH=src python -m scripts.issue57_bert_cv cv \
        --model data/local/models/bert-base-chinese --lr 2e-5 --loss balanced \
        --seeds 20261101,20261102 --out runs/issue57/c-bert-cv
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import time
from pathlib import Path

import numpy as np

from scripts.evaluate_sd_v2_cv import _metric_report
from scripts.issue57_bert_sanity import load_tokenizer
from scripts.issue57_common import (
    head_tail_strata,
    iter_cv_folds,
    load_selection_pool,
    summarize_folds,
    write_results,
)
from syndrome.baselines import environment_record
from syndrome.metrics import topk_accuracy

LOSS_CHOICES = ("plain", "balanced")


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_weights(y: list[int], n_classes: int) -> np.ndarray:
    counts = np.bincount(np.asarray(y), minlength=n_classes).astype(np.float64)
    observed = counts > 0
    weights = np.zeros(n_classes, dtype=np.float32)
    weights[observed] = len(y) / (observed.sum() * counts[observed])
    return weights


def state_dict_sha256(state: dict) -> str:
    import torch

    buffer = io.BytesIO()
    torch.save({key: value.cpu() for key, value in state.items()}, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def load_sequence_classifier(model_path: str, num_labels: int) -> tuple[object, str]:
    """Load a local encoder checkpoint for classification.

    Tries the Auto* resolution first; on failure falls back to the explicit
    BERT classes, which read config.json directly and do not need a
    ``model_type`` key (2019-era checkpoints like mc-bert lack one).
    Returns (model, loader_name).
    """
    from transformers import AutoModelForSequenceClassification

    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=num_labels
        )
        return model, "AutoModelForSequenceClassification"
    except (ValueError, OSError):
        from transformers import BertConfig, BertForSequenceClassification

        config = BertConfig.from_pretrained(model_path, num_labels=num_labels)
        model = BertForSequenceClassification.from_pretrained(
            model_path, config=config
        )
        return model, "BertForSequenceClassification"


def load_encoder(model_path: str) -> tuple[object, str]:
    """Frozen-encoder load with the same Auto -> explicit-BERT fallback."""
    from transformers import AutoModel

    try:
        return AutoModel.from_pretrained(model_path), "AutoModel"
    except (ValueError, OSError):
        from transformers import BertModel

        return BertModel.from_pretrained(model_path), "BertModel"


def train_bert_fold(
    fold_info: dict,
    model_path: str,
    lr: float,
    loss: str,
    seed: int,
    device: str,
    max_len: int = 128,
    batch_size: int = 32,
    epochs: int = 3,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
) -> dict:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import get_linear_schedule_with_warmup

    from syndrome.metrics import macro_f1

    seed_everything(seed)
    tokenizer = load_tokenizer(model_path)
    train_texts = [r.input_text for r in fold_info["train_labeled"]]
    train_labels = [int(r.label) for r in fold_info["train_labeled"]]
    dev_texts = [r.input_text for r in fold_info["dev_labeled"]]
    dev_labels = [int(r.label) for r in fold_info["dev_labeled"]]

    train_enc = tokenizer(
        train_texts, padding="max_length", truncation=True,
        max_length=max_len, return_tensors="pt",
    )
    dev_enc = tokenizer(
        dev_texts, padding="max_length", truncation=True,
        max_length=max_len, return_tensors="pt",
    )
    train_dl = DataLoader(
        TensorDataset(
            train_enc["input_ids"],
            train_enc["attention_mask"],
            torch.tensor(train_labels, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    model, _loader = load_sequence_classifier(model_path, 148)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    total_steps = len(train_dl) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * warmup_ratio), total_steps
    )
    class_weight = (
        torch.tensor(class_weights(train_labels, 148), device=device)
        if loss == "balanced"
        else None
    )

    dev_ids = dev_enc["input_ids"]
    dev_mask = dev_enc["attention_mask"]

    def dev_predict() -> tuple[list[int], np.ndarray]:
        model.eval()
        scores = []
        with torch.no_grad():
            for start in range(0, len(dev_ids), 256):
                logits = model(
                    input_ids=dev_ids[start : start + 256].to(device),
                    attention_mask=dev_mask[start : start + 256].to(device),
                ).logits
                scores.append(torch.softmax(logits, dim=-1).cpu().numpy())
        all_scores = np.concatenate(scores)
        return all_scores.argmax(-1).tolist(), all_scores

    history = []
    best_f1 = -1.0
    best_state: dict | None = None
    fit_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for ids, mask, y in train_dl:
            ids = ids.to(device)
            mask = mask.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(input_ids=ids, attention_mask=mask).logits
            loss_value = torch.nn.functional.cross_entropy(
                logits, y, weight=class_weight
            )
            loss_value.backward()
            optimizer.step()
            scheduler.step()
            total_loss += float(loss_value.item()) * len(y)
        y_pred, _ = dev_predict()
        f1 = macro_f1(dev_labels, y_pred, 148)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(total_loss / len(train_labels), 6),
                "dev_macro_f1": round(f1, 6),
            }
        )
        if f1 > best_f1:
            best_f1 = f1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    fit_seconds = time.perf_counter() - fit_start

    assert best_state is not None
    model.load_state_dict(best_state)
    predict_start = time.perf_counter()
    y_pred, scores = dev_predict()
    predict_seconds = time.perf_counter() - predict_start

    param_count = sum(p.numel() for p in model.parameters())
    return {
        "metrics_report": _metric_report(dev_labels, y_pred),
        "top3_accuracy": round(float(topk_accuracy(dev_labels, scores, 3)), 6),
        "strata": head_tail_strata(dev_labels, y_pred, train_labels),
        "history": history,
        "best_dev_macro_f1": round(best_f1, 6),
        "n_params": param_count,
        "model_bytes": param_count * 4,
        "weight_sha256": state_dict_sha256(best_state),
        "fit_seconds": round(fit_seconds, 1),
        "predict_seconds": round(predict_seconds, 1),
    }


def _fold_result_payload(fold_info: dict, result: dict) -> dict:
    return {
        "repeat": fold_info["repeat"],
        "fold": fold_info["fold"],
        "n_train_records": fold_info["n_train_records"],
        "n_dev_records_before_dedup": fold_info["n_dev_records_before_dedup"],
        "n_dev_text_duplicates_dropped": fold_info["n_dev_text_duplicates_dropped"],
        "n_dev_records_scored": result["metrics_report"]["n_examples"],
        "metrics": result["metrics_report"],
        "top3_accuracy": result["top3_accuracy"],
        "strata": result["strata"],
        "history": result["history"],
        "best_dev_macro_f1": result["best_dev_macro_f1"],
        "n_params": result["n_params"],
        "model_bytes": result["model_bytes"],
        "weight_sha256": result["weight_sha256"],
        "fit_seconds": result["fit_seconds"],
        "predict_seconds": result["predict_seconds"],
    }


def run_rank(args, repo_root: Path) -> dict:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest, records_by_row = load_selection_pool(repo_root, args.scenario)
    fold_filter = (0, args.fold)
    config_results = []
    for lr in (2e-5, 5e-5):
        for loss in LOSS_CHOICES:
            fold_info = next(
                iter_cv_folds(
                    manifest, records_by_row, args.scenario, args.seed,
                    folds=args.folds, repeats=1, fold_filter=fold_filter,
                )
            )
            result = train_bert_fold(
                fold_info, args.model, lr, loss, args.seed, device,
                max_len=args.max_len, batch_size=args.batch_size,
                epochs=args.epochs,
            )
            config_results.append(
                {
                    "lr": lr,
                    "loss": loss,
                    "seed": args.seed,
                    "repeat": fold_info["repeat"],
                    "fold": fold_info["fold"],
                    "best_dev_macro_f1": result["best_dev_macro_f1"],
                    "final_dev_macro_f1": result["metrics_report"]["macro_f1"],
                    "history": result["history"],
                    "fit_seconds": result["fit_seconds"],
                }
            )
            print(
                json.dumps(
                    {
                        "progress": True,
                        "lr": lr,
                        "loss": loss,
                        "best_dev_macro_f1": result["best_dev_macro_f1"],
                        "history": result["history"],
                    }
                ),
                flush=True,
            )
    ranked = sorted(
        config_results, key=lambda item: item["best_dev_macro_f1"], reverse=True
    )
    return {
        "issue": 57,
        "line": "C",
        "stage": "config-rank",
        "scenario": args.scenario,
        "status": "OK",
        "model_path": args.model,
        "ranked_configs": [
            {"lr": item["lr"], "loss": item["loss"], "best_dev_macro_f1": item[
                "best_dev_macro_f1"
            ]}
            for item in ranked
        ],
        "winner": {"lr": ranked[0]["lr"], "loss": ranked[0]["loss"]},
        "training_config": {
            "max_len": args.max_len,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "seed": args.seed,
            "fold_filter": list(fold_filter),
            "environment": environment_record(
                args.seed,
                extra={
                    "torch_version": torch.__version__,
                    "transformers_version": __import__("transformers").__version__,
                    "device": device,
                    "gpu_name": (
                        torch.cuda.get_device_name(0)
                        if device == "cuda"
                        else "NOT_AVAILABLE"
                    ),
                },
            ),
        },
        "config_results": config_results,
    }


def run_cv(args, repo_root: Path) -> dict:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [int(value) for value in args.seeds.split(",")]
    manifest, records_by_row = load_selection_pool(repo_root, args.scenario)

    fold_results = []
    for seed in seeds:
        for fold_info in iter_cv_folds(
            manifest, records_by_row, args.scenario, seed,
            folds=args.folds, repeats=args.repeats,
        ):
            result = train_bert_fold(
                fold_info, args.model, args.lr, args.loss, seed, device,
                max_len=args.max_len, batch_size=args.batch_size,
                epochs=args.epochs,
            )
            payload = _fold_result_payload(fold_info, result)
            payload["seed"] = seed
            fold_results.append(payload)
            print(
                json.dumps(
                    {
                        "progress": True,
                        "seed": seed,
                        "repeat": fold_info["repeat"],
                        "fold": fold_info["fold"],
                        "macro_f1": result["metrics_report"]["macro_f1"],
                        "best_epoch": max(
                            result["history"],
                            key=lambda item: item["dev_macro_f1"],
                        )["epoch"],
                    }
                ),
                flush=True,
            )

    return {
        "issue": 57,
        "line": "C",
        "stage": "cv",
        "scenario": args.scenario,
        "status": "OK",
        "model_path": args.model,
        "cv": {
            "protocol": (
                "same selection_fold_ids user-grouped CV as the frozen "
                "evaluator (imported via scripts.issue57_common)"
            ),
            "folds": args.folds,
            "repeats": args.repeats,
            "seeds": seeds,
            **summarize_folds(fold_results),
        },
        "training_config": {
            "model": "bert-base-chinese-finetune",
            "max_len": args.max_len,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "loss": args.loss,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "best_epoch_selection": "deduped fold-dev macro_f1",
            "environment": environment_record(
                seeds[0],
                extra={
                    "torch_version": torch.__version__,
                    "transformers_version": __import__("transformers").__version__,
                    "device": device,
                    "gpu_name": (
                        torch.cuda.get_device_name(0)
                        if device == "cuda"
                        else "NOT_AVAILABLE"
                    ),
                },
            ),
        },
        "folds": fold_results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    def add_common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--model", default="data/local/models/bert-base-chinese")
        sub.add_argument("--scenario", choices=("short", "clinical"), default="short")
        sub.add_argument("--seed", type=int, default=20261101)
        sub.add_argument("--folds", type=int, default=4)
        sub.add_argument("--max-len", type=int, default=128)
        sub.add_argument("--batch-size", type=int, default=32)
        sub.add_argument("--epochs", type=int, default=3)

    rank_parser = subparsers.add_parser("rank", help="4 config points on one fold")
    add_common(rank_parser)
    rank_parser.add_argument("--fold", type=int, default=0)
    rank_parser.add_argument("--out", default="runs/issue57/c-bert-rank")

    cv_parser = subparsers.add_parser("cv", help="winner full 4x2 CV")
    add_common(cv_parser)
    cv_parser.add_argument("--lr", type=float, required=True)
    cv_parser.add_argument("--loss", choices=LOSS_CHOICES, required=True)
    cv_parser.add_argument("--seeds", default="20261101,20261102")
    cv_parser.add_argument("--repeats", type=int, default=2)
    cv_parser.add_argument("--out", default="runs/issue57/c-bert-cv")

    args = parser.parse_args(argv)
    repo_root = Path.cwd().resolve()
    payload = run_rank(args, repo_root) if args.mode == "rank" else run_cv(
        args, repo_root
    )
    out_path = write_results(repo_root / args.out / "results.json", payload)
    if args.mode == "rank":
        print(
            json.dumps(
                {
                    "ok": True,
                    "results": str(out_path),
                    "ranked_configs": payload["ranked_configs"],
                    "winner": payload["winner"],
                },
                ensure_ascii=False,
            )
        )
    else:
        print(
            json.dumps(
                {
                    "ok": True,
                    "results": str(out_path),
                    "mean_fold_macro_f1": payload["cv"]["mean_fold_macro_f1"],
                    "population_std_fold_macro_f1": payload["cv"][
                        "population_std_fold_macro_f1"
                    ],
                },
                ensure_ascii=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
