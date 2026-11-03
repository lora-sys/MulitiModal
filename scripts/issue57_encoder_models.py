"""#57 D line: mc-bert and chinese-roberta-wwm-ext comparison, short scenario.

For each encoder:
  check    — verify it loads under transformers 5.22.1 (mc-bert is a 2019
             checkpoint; on load failure record NOT_AVAILABLE with the
             exact traceback and move on);
  frozen   — frozen encoder + logistic head: one CLS-embedding pass per
             fold, then LogisticRegression (class_weight balanced) fit on
             the fold-train embeddings only;
  finetune — end-to-end fine-tune inheriting the C line's winning recipe
             (pass --lr/--loss from the C rank outcome).

Short scenario only, seed 20261101 for the ranking pass. Fold membership,
dedup, and metric reporting are imported (not forked) via
``scripts.issue57_common``; fine-tune reuses ``train_bert_fold`` verbatim.

Usage (acam-lab, repo root):
    PYTHONPATH=src python -m scripts.issue57_encoder_models check \
        --models mc-bert chinese-roberta-wwm-ext --out runs/issue57/d-check
    PYTHONPATH=src python -m scripts.issue57_encoder_models frozen \
        --model mc-bert --out runs/issue57/d-mcbert-frozen
    PYTHONPATH=src python -m scripts.issue57_encoder_models finetune \
        --model mc-bert --lr 2e-5 --loss balanced --out runs/issue57/d-mcbert-ft
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np

from scripts.evaluate_sd_v2_cv import _metric_report
from scripts.issue57_bert_cv import (
    _fold_result_payload,
    load_encoder,
    load_sequence_classifier,
    train_bert_fold,
)
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

MODEL_ROOT = "data/local/models"


def check_model(model_path: str) -> dict:
    """Load-verify one encoder; record NOT_AVAILABLE with the traceback on failure."""
    record: dict = {"model_path": model_path}
    try:
        tokenizer = load_tokenizer(model_path)
        record["tokenizer_class"] = type(tokenizer).__name__
        model, loader = load_sequence_classifier(model_path, 148)
        record["loader"] = loader
        record["model_class"] = type(model).__name__
        record["vocab_size"] = int(model.config.vocab_size)
        record["hidden_size"] = int(model.config.hidden_size)
        record["status"] = "OK"
        del model
    except Exception:
        record["status"] = "NOT_AVAILABLE"
        record["error"] = traceback.format_exc()
    return record


#: memo of CLS embeddings keyed by (model_path, max_len, text).
#:
#: Every fold re-embeds its train+dev texts, and the folds overlap almost
#: completely, so the un-cached frozen run spent ~8x the necessary encoder work
#: (measured: 1500s did not finish 2 of 8 folds, i.e. ~2h per model-seed, ~9h
#: for the four frozen combinations the D line needs). The encoder runs in eval
#: mode with no gradients, so a memo returns bit-identical vectors and only
#: removes recomputation.
_EMBED_MEMO: dict[tuple[str, int, str], np.ndarray] = {}


def embed_texts(texts: list[str], tokenizer, model, device: str, max_len: int,
                memo_key: str | None = None) -> np.ndarray:
    """One CLS-embedding pass (last_hidden_state[:, 0, :]) in eval mode.

    `memo_key` is the encoder path; when given, each text's vector is remembered
    so later folds reuse it instead of re-running the encoder.
    """
    import torch

    cached: list[np.ndarray | None] = [None] * len(texts)
    if memo_key is not None:
        for i, text in enumerate(texts):
            hit = _EMBED_MEMO.get((memo_key, max_len, text))
            if hit is not None:
                cached[i] = hit
    pending = [i for i, row in enumerate(cached) if row is None]
    if pending:
        embeddings = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(pending), 256):
                batch_ids = pending[start : start + 256]
                enc = tokenizer(
                    [texts[i] for i in batch_ids],
                    padding="max_length",
                    truncation=True,
                    max_length=max_len,
                    return_tensors="pt",
                )
                output = model.base_model(
                    input_ids=enc["input_ids"].to(device),
                    attention_mask=enc["attention_mask"].to(device),
                )
                embeddings.append(output.last_hidden_state[:, 0, :].cpu().numpy())
        vectors = np.concatenate(embeddings)
        for offset, i in enumerate(pending):
            vector = vectors[offset]
            cached[i] = vector
            if memo_key is not None:
                _EMBED_MEMO[(memo_key, max_len, texts[i])] = vector
    if any(row is None for row in cached):
        raise RuntimeError("embedding memo left a row unfilled")
    return np.stack(cached)


def frozen_logreg_fold(fold_info: dict, model_path: str, seed: int, device: str,
                       max_len: int = 128) -> dict:
    # `liblinear` is binary-only, and the frozen head crashed on the real
    # taxonomy (148 classes): the D-line run recorded a ValueError, not a
    # result. The A-line TF-IDF baseline answers the same problem with
    # OneVsRestClassifier(liblinear); the same family here keeps the
    # frozen-head comparison against A on ONE classifier instead of two.
    # n_jobs=1: joblib serialisation of the embedding matrices trips
    # "WRITEBACKIFCOPY base is read-only" on this numpy 2.5 / joblib ABI, the
    # same reason recorded in src/syndrome/baselines.py.
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

    tokenizer = load_tokenizer(model_path)
    encoder, encoder_loader = load_encoder(model_path)
    encoder = encoder.to(device)
    encoder_class = type(encoder).__name__
    train_texts = [r.input_text for r in fold_info["train_labeled"]]
    train_labels = [int(r.label) for r in fold_info["train_labeled"]]
    dev_texts = [r.input_text for r in fold_info["dev_labeled"]]
    dev_labels = [int(r.label) for r in fold_info["dev_labeled"]]

    embed_start = time.perf_counter()
    x_train = embed_texts(train_texts, tokenizer, encoder, device, max_len,
                          memo_key=model_path)
    x_dev = embed_texts(dev_texts, tokenizer, encoder, device, max_len,
                        memo_key=model_path)
    embed_seconds = time.perf_counter() - embed_start
    del encoder

    fit_start = time.perf_counter()
    classifier = OneVsRestClassifier(
        LogisticRegression(
            C=4.0,
            class_weight="balanced",
            # dense 768-float rows: liblinear's cost is per non-zero, so a
            # dense matrix is ~38x the work of the sparse TF-IDF baseline at
            # the same row count. With 33,814 train rows per fold the default
            # 2000-iteration cap took ~20 minutes per fold and no longer
            # finished. 200 is declared here, and how many of the 148 classes
            # hit the cap is recorded in the result rather than hidden.
            max_iter=200,
            dual=False,
            solver="liblinear",
            random_state=seed,
        ),
        n_jobs=1,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        classifier.fit(x_train, train_labels)
    not_converged = sum(
        1 for w in caught if issubclass(w.category, ConvergenceWarning)
    )
    fit_seconds = time.perf_counter() - fit_start
    predict_start = time.perf_counter()
    y_pred = [int(value) for value in classifier.predict(x_dev)]
    predict_seconds = time.perf_counter() - predict_start
    scores = classifier.decision_function(x_dev)

    return {
        "metrics_report": _metric_report(dev_labels, y_pred),
        "top3_accuracy": round(
            float(topk_accuracy(dev_labels, scores, 3)), 6
        ),
        "strata": head_tail_strata(dev_labels, y_pred, train_labels),
        "embed_seconds": round(embed_seconds, 1),
        "fit_seconds": round(fit_seconds, 1),
        "predict_seconds": round(predict_seconds, 1),
        "n_train_embeddings": int(x_train.shape[0]),
        "embedding_dim": int(x_train.shape[1]),
        "encoder_class": encoder_class,
        "encoder_loader": encoder_loader,
        "head": "OneVsRestClassifier(LogisticRegression)",
        "head_max_iter": 200,
        "head_classes_not_converged": not_converged,
    }


def run_check(args, repo_root: Path) -> dict:
    payload = {
        "issue": 57,
        "line": "D",
        "stage": "load-check",
        "models": [check_model(str(Path(args.model_root) / name))
                   for name in args.models],
    }
    return payload


def run_frozen(args, repo_root: Path) -> dict:
    import sklearn
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = str(Path(args.model_root) / args.model)
    manifest, records_by_row = load_selection_pool(repo_root, args.scenario)

    fold_results = []
    for fold_info in iter_cv_folds(
        manifest, records_by_row, args.scenario, args.seed,
        folds=args.folds, repeats=args.repeats,
    ):
        result = frozen_logreg_fold(
            fold_info, model_path, args.seed, device, max_len=args.max_len
        )
        fold_results.append(
            {
                "repeat": fold_info["repeat"],
                "fold": fold_info["fold"],
                "n_train_records": fold_info["n_train_records"],
                "n_dev_records_scored": result["metrics_report"]["n_examples"],
                "metrics": result["metrics_report"],
                "top3_accuracy": result["top3_accuracy"],
                "strata": result["strata"],
                "embed_seconds": result["embed_seconds"],
                "fit_seconds": result["fit_seconds"],
                "predict_seconds": result["predict_seconds"],
                "n_train_embeddings": result["n_train_embeddings"],
                "embedding_dim": result["embedding_dim"],
                "encoder_class": result["encoder_class"],
                "encoder_loader": result["encoder_loader"],
            }
        )
        print(
            json.dumps(
                {
                    "progress": True,
                    "repeat": fold_info["repeat"],
                    "fold": fold_info["fold"],
                    "macro_f1": result["metrics_report"]["macro_f1"],
                }
            ),
            flush=True,
        )

    return {
        "issue": 57,
        "line": "D",
        "stage": "frozen-encoder-logreg",
        "scenario": args.scenario,
        "status": "OK",
        "model_path": model_path,
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
            "encoder": args.model,
            "encoder_training": "frozen (eval mode, no gradients)",
            "pooling": "cls (last_hidden_state[:, 0, :])",
            "head": "OneVsRestClassifier(LogisticRegression(C=4.0, "
                    "class_weight=balanced, solver=liblinear, dual=False, "
                    "max_iter=200), n_jobs=1)",
            "max_len": args.max_len,
            "seed": args.seed,
            "sklearn_version": sklearn.__version__,
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
        "folds": fold_results,
    }


def run_finetune(args, repo_root: Path) -> dict:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [int(value) for value in args.seeds.split(",")]
    model_path = str(Path(args.model_root) / args.model)
    manifest, records_by_row = load_selection_pool(repo_root, args.scenario)

    fold_results = []
    for seed in seeds:
        for fold_info in iter_cv_folds(
            manifest, records_by_row, args.scenario, seed,
            folds=args.folds, repeats=args.repeats,
        ):
            result = train_bert_fold(
                fold_info, model_path, args.lr, args.loss, seed, device,
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
                    }
                ),
                flush=True,
            )

    return {
        "issue": 57,
        "line": "D",
        "stage": "end-to-end-finetune",
        "scenario": args.scenario,
        "status": "OK",
        "model_path": model_path,
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
            "encoder": args.model,
            "recipe_source": "C-line winner (lr/loss inherited)",
            "max_len": args.max_len,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "loss": args.loss,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
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

    def add_model_root(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--model-root",
            default=MODEL_ROOT,
            help="directory containing the encoder weight directories",
        )

    check_parser = subparsers.add_parser("check", help="load-verify encoders")
    check_parser.add_argument(
        "--models", nargs="+",
        default=["mc-bert", "chinese-roberta-wwm-ext"],
    )
    add_model_root(check_parser)
    check_parser.add_argument("--out", default="runs/issue57/d-check")

    frozen_parser = subparsers.add_parser("frozen", help="frozen encoder + logreg")
    add_model_root(frozen_parser)
    frozen_parser.add_argument("--model", required=True)
    frozen_parser.add_argument("--scenario", choices=("short", "clinical"),
                               default="short")
    frozen_parser.add_argument("--seed", type=int, default=20261101)
    frozen_parser.add_argument("--folds", type=int, default=4)
    frozen_parser.add_argument("--repeats", type=int, default=2)
    frozen_parser.add_argument("--max-len", type=int, default=128)
    frozen_parser.add_argument("--out", required=True)

    ft_parser = subparsers.add_parser("finetune", help="end-to-end fine-tune")
    add_model_root(ft_parser)
    ft_parser.add_argument("--model", required=True)
    ft_parser.add_argument("--scenario", choices=("short", "clinical"),
                           default="short")
    ft_parser.add_argument("--seed", type=int, default=20261101)
    ft_parser.add_argument("--seeds", default="20261101")
    ft_parser.add_argument("--folds", type=int, default=4)
    ft_parser.add_argument("--repeats", type=int, default=2)
    ft_parser.add_argument("--max-len", type=int, default=128)
    ft_parser.add_argument("--batch-size", type=int, default=32)
    ft_parser.add_argument("--epochs", type=int, default=3)
    ft_parser.add_argument("--lr", type=float, required=True)
    ft_parser.add_argument("--loss", choices=("plain", "balanced"), required=True)
    ft_parser.add_argument("--out", required=True)

    args = parser.parse_args(argv)
    repo_root = Path.cwd().resolve()
    if args.mode == "check":
        payload = run_check(args, repo_root)
    elif args.mode == "frozen":
        payload = run_frozen(args, repo_root)
    else:
        payload = run_finetune(args, repo_root)
    out_path = write_results(repo_root / args.out / "results.json", payload)
    print(json.dumps({"ok": True, "results": str(out_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
