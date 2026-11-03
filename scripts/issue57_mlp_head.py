"""#57 B line: hashed char n-gram features + MLP head on the short scenario.

Feature contract is the frozen stateless HashingVectorizer parameter set
imported verbatim from ``research.task_sd.HASHED_TEXT_PARAMS`` (char_wb
1-3 grams, 2^16 buckets, alternate_sign off, L2 norm) — the same contract
as the sd-v1 champion. Head: 512 -> 256 ReLU MLP with dropout 0.2 over the
65,536 hashed features, class-balanced weighted CE, Adam lr 1e-3, batch
256, at most 10 epochs, early stop on deduped fold-dev macro-F1.

CV protocol: the same 4-fold x 2-repeat user-grouped CV as the frozen
evaluator — fold membership, scenario-text dedup, and metric reporting are
imported (not forked) via ``scripts.issue57_common`` (which reuses
``research.task_sd.build_fold`` and ``scripts.evaluate_sd_v2_cv``); the
only per-fold difference is that fit sees the deduped fold-dev rows for
early stopping.

Usage (acam-lab, repo root):
    PYTHONPATH=src python -m scripts.issue57_mlp_head \
        --seed 20261101 --out runs/issue57/b-mlp-short
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np

from research.task_sd import HASHED_TEXT_PARAMS, _hashing_vectorizer
from scripts.evaluate_sd_v2_cv import _metric_report
from scripts.issue57_common import (
    head_tail_strata,
    iter_cv_folds,
    load_selection_pool,
    summarize_folds,
    write_results,
)
from syndrome.baselines import environment_record
from syndrome.metrics import topk_accuracy

EARLY_STOP_PATIENCE = 3


def build_mlp(n_features: int, n_classes: int):
    import torch

    return torch.nn.Sequential(
        torch.nn.Linear(n_features, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(256, n_classes),
    )


def class_weights(y: list[int], n_classes: int) -> np.ndarray:
    counts = np.bincount(np.asarray(y), minlength=n_classes).astype(np.float64)
    observed = counts > 0
    weights = np.zeros(n_classes, dtype=np.float32)
    weights[observed] = len(y) / (observed.sum() * counts[observed])
    return weights


def state_dict_sha256(state: dict) -> str:
    buffer = io.BytesIO()
    import torch

    torch.save({key: value.cpu() for key, value in state.items()}, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def train_fold(
    fold_info: dict,
    seed: int,
    device: str,
    n_classes: int,
    batch_size: int = 256,
    max_epochs: int = 10,
) -> dict:
    import torch

    torch.manual_seed(seed * 1000 + fold_info["repeat"] * 100 + fold_info["fold"])
    vectorizer = _hashing_vectorizer(HASHED_TEXT_PARAMS)
    x_train = vectorizer.transform(
        [r.input_text for r in fold_info["train_labeled"]]
    )
    y_train = np.asarray([int(r.label) for r in fold_info["train_labeled"]])
    x_dev = vectorizer.transform([r.input_text for r in fold_info["dev_labeled"]])
    y_dev = np.asarray([int(r.label) for r in fold_info["dev_labeled"]])

    model = build_mlp(x_train.shape[1], n_classes).to(device)
    weights = torch.tensor(
        class_weights(y_train.tolist(), n_classes), device=device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    def dev_batches(dense_fn):
        outputs = []
        with torch.no_grad():
            for start in range(0, x_dev.shape[0], 1024):
                dense = torch.tensor(
                    x_dev[start : start + 1024].toarray(), dtype=torch.float32,
                    device=device,
                )
                outputs.append(dense_fn(model(dense)).cpu().numpy())
        return np.concatenate(outputs)

    from syndrome.metrics import macro_f1

    def dev_macro_f1(state: dict | None = None) -> float:
        if state is not None:
            model.load_state_dict(state)
        model.eval()
        preds = dev_batches(lambda logits: logits.argmax(-1))
        return macro_f1(y_dev.tolist(), preds.tolist(), n_classes)

    history = []
    best_f1 = -1.0
    best_state: dict | None = None
    epochs_since_best = 0
    fit_start = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(x_train.shape[0])
        total_loss = 0.0
        for start in range(0, x_train.shape[0], batch_size):
            batch_idx = order[start : start + batch_size].numpy()
            dense = torch.tensor(
                x_train[batch_idx].toarray(), dtype=torch.float32, device=device
            )
            target = torch.tensor(
                y_train[batch_idx], dtype=torch.long, device=device
            )
            optimizer.zero_grad()
            loss = loss_fn(model(dense), target)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_idx)
        f1 = dev_macro_f1()
        history.append(
            {"epoch": epoch, "train_loss": round(total_loss / len(y_train), 6),
             "dev_macro_f1": round(f1, 6)}
        )
        if f1 > best_f1:
            best_f1 = f1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= EARLY_STOP_PATIENCE:
                break
    fit_seconds = time.perf_counter() - fit_start

    assert best_state is not None
    model.load_state_dict(best_state)
    predict_start = time.perf_counter()
    y_pred = dev_batches(lambda logits: logits.argmax(-1))
    predict_seconds = time.perf_counter() - predict_start

    param_count = sum(p.numel() for p in model.parameters())
    return {
        "metrics_report": _metric_report(y_dev.tolist(), y_pred.tolist()),
        "top3_accuracy": round(
            float(topk_accuracy(
                y_dev.tolist(), _dev_scores(model, x_dev, device), 3
            )), 6
        ),
        "strata": head_tail_strata(
            y_dev.tolist(), y_pred.tolist(), y_train.tolist()
        ),
        "history": history,
        "best_dev_macro_f1": round(best_f1, 6),
        "n_params": param_count,
        "model_bytes": param_count * 4,
        "weight_sha256": state_dict_sha256(best_state),
        "fit_seconds": round(fit_seconds, 1),
        "predict_seconds": round(predict_seconds, 1),
    }


def _dev_scores(model, x_dev, device: str):
    import torch

    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, x_dev.shape[0], 1024):
            dense = torch.tensor(
                x_dev[start : start + 1024].toarray(), dtype=torch.float32,
                device=device,
            )
            scores.append(torch.softmax(model(dense), dim=-1).cpu().numpy())
    return np.concatenate(scores)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("short", "clinical"), default="short")
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--out", default="runs/issue57/b-mlp-short")
    args = parser.parse_args(argv)

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    repo_root = Path.cwd().resolve()
    manifest, records_by_row = load_selection_pool(repo_root, args.scenario)

    fold_results = []
    for fold_info in iter_cv_folds(
        manifest, records_by_row, args.scenario, args.seed,
        folds=args.folds, repeats=args.repeats,
    ):
        result = train_fold(fold_info, args.seed, device, 148)
        fold_results.append(
            {
                "repeat": fold_info["repeat"],
                "fold": fold_info["fold"],
                "n_train_records": fold_info["n_train_records"],
                "n_dev_records_before_dedup": fold_info["n_dev_records_before_dedup"],
                "n_dev_text_duplicates_dropped": fold_info[
                    "n_dev_text_duplicates_dropped"
                ],
                "n_dev_records_scored": result["metrics_report"][
                    "n_examples"
                ],
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
        )
        print(
            json.dumps(
                {
                    "progress": True,
                    "repeat": fold_info["repeat"],
                    "fold": fold_info["fold"],
                    "macro_f1": result["metrics_report"]["macro_f1"],
                    "best_epoch": max(
                        result["history"], key=lambda h: h["dev_macro_f1"]
                    )["epoch"],
                }
            ),
            flush=True,
        )

    payload = {
        "issue": 57,
        "line": "B",
        "scenario": args.scenario,
        "status": "OK",
        "cv": {
            "protocol": (
                "same selection_fold_ids user-grouped CV as the frozen "
                "evaluator (imported via scripts.issue57_common)"
            ),
            "folds": args.folds,
            "repeats": args.repeats,
            **summarize_folds(fold_results),
        },
        "training_config": {
            "model": "hashed-char-1-3-65536-mlp-512-256",
            "feature_params": HASHED_TEXT_PARAMS,
            "hidden": [512, 256],
            "dropout": 0.2,
            "optimizer": "adam",
            "lr": 1e-3,
            "batch_size": 256,
            "max_epochs": 10,
            "early_stop": {
                "metric": "deduped fold-dev macro_f1",
                "patience_epochs": EARLY_STOP_PATIENCE,
            },
            "loss": "class-balanced weighted cross-entropy",
            "seed": args.seed,
            "environment": environment_record(
                args.seed,
                extra={
                    "torch_version": torch.__version__,
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
    out_path = write_results(repo_root / args.out / "cv-results.json", payload)
    summary = payload["cv"]
    print(
        json.dumps(
            {
                "ok": True,
                "results": str(out_path),
                "mean_fold_macro_f1": summary["mean_fold_macro_f1"],
                "population_std_fold_macro_f1": summary[
                    "population_std_fold_macro_f1"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
