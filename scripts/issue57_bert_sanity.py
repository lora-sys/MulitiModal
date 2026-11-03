"""#57 C-line GPU sanity checks (run before any BERT training on acam-lab).

(a) Overfit a 64-sample batch of real short-scenario training rows to
    ~100% train accuracy with bert-base-chinese — a broken pipeline cannot
    overfit, so this is the go/no-go for the C line.
(b) Truncation audit: tokenize every short and clinical selection record
    and report the fraction exceeding max_len 128 / 256.
(c) Save→reload determinism using the existing ``syndrome.bert_model``
    bindings: fit a tiny model, save, reload with ``load``, and require
    byte-identical predictions.

Usage (acam-lab, repo root):
    PYTHONPATH=src python -m scripts.issue57_bert_sanity \
        --model data/local/models/bert-base-chinese \
        --raw-dir /home/lora/workspace/projects/MulitiModal/data/local/tcm-sd \
        --out runs/issue57/c-sanity
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from scripts.issue57_common import load_selection_pool, write_results


def load_tokenizer(model_path: str):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model_path)
    except Exception:
        from transformers import BertTokenizer

        return BertTokenizer.from_pretrained(model_path)


def _percentile(sorted_values: list[int], fraction: float) -> int:
    if not sorted_values:
        return 0
    index = min(int(fraction * len(sorted_values)), len(sorted_values) - 1)
    return sorted_values[index]


def overfit_64(model_path: str, texts: list[str], labels: list[int]) -> dict:
    import torch
    from transformers import AutoModelForSequenceClassification

    tokenizer = load_tokenizer(model_path)
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=148
    ).to(device)
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    history = []
    start = time.perf_counter()
    final_acc = 0.0
    for epoch in range(1, 101):
        optimizer.zero_grad()
        logits = model(input_ids=ids, attention_mask=mask).logits
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        final_acc = float((logits.argmax(-1) == y).float().mean().item())
        history.append({"epoch": epoch, "loss": round(float(loss.item()), 4),
                        "train_acc": round(final_acc, 4)})
        if final_acc == 1.0:
            break
    return {
        "n_samples": len(texts),
        "max_len": 128,
        "lr": 1e-4,
        "epochs_ran": len(history),
        "final_train_acc": final_acc,
        "overfit_ok": final_acc >= 0.99,
        "wall_seconds": round(time.perf_counter() - start, 1),
        "last_5_epochs": history[-5:],
        "device": device,
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "tokenizer_class": type(tokenizer).__name__,
    }


def truncation_audit(model_path: str, repo_root: Path) -> dict:
    tokenizer = load_tokenizer(model_path)
    report: dict = {"tokenizer_class": type(tokenizer).__name__, "scenarios": {}}
    for scenario in ("short", "clinical"):
        _manifest, records_by_row = load_selection_pool(repo_root, scenario)
        texts = [record.input_text for record in records_by_row.values()]
        lengths = sorted(
            len(ids) for ids in tokenizer(texts, add_special_tokens=True)["input_ids"]
        )
        n = len(lengths)
        report["scenarios"][scenario] = {
            "n_records": n,
            "n_chars_mean": round(
                sum(len(text) for text in texts) / max(n, 1), 1
            ),
            "token_len_p50": _percentile(lengths, 0.50),
            "token_len_p90": _percentile(lengths, 0.90),
            "token_len_p99": _percentile(lengths, 0.99),
            "token_len_max": lengths[-1] if lengths else 0,
            "frac_over_128": round(
                sum(1 for value in lengths if value > 128) / max(n, 1), 6
            ),
            "frac_over_256": round(
                sum(1 for value in lengths if value > 256) / max(n, 1), 6
            ),
        }
    return report


def save_reload_determinism(
    model_path: str, raw_dir: str, texts: list[str], labels: list[int], tmp_dir: Path
) -> dict:
    from syndrome.bert_model import BertSyndromeClassifier, BertTrainConfig
    from syndrome.labels import load_label_space

    label_space = load_label_space(raw_dir)
    if len(label_space) != 148:
        raise ValueError(f"expected the 148-class taxonomy, got {len(label_space)}")
    classifier = BertSyndromeClassifier(label_space, model_path, seed=20261101)
    config = BertTrainConfig(model_path=model_path, max_len=128, epochs=1)
    fit_report = classifier.fit(
        texts, labels, texts, labels, config, str(tmp_dir / "issue57-sanity-model")
    )
    before = classifier.predict_proba(texts, batch_size=16)
    reloaded = BertSyndromeClassifier.load(
        str(tmp_dir / "issue57-sanity-model"), label_space
    )
    after = reloaded.predict_proba(texts, batch_size=16)
    max_abs_diff = float(np.abs(before - after).max())
    return {
        "label_space_taxonomy": label_space.taxonomy_id,
        "n_label_space": len(label_space),
        "fit_report": fit_report["history"],
        "n_probe_texts": len(texts),
        "predictions_identical": bool((before == after).all()),
        "max_abs_prob_diff": max_abs_diff,
        "determinism_ok": bool((before == after).all()) and max_abs_diff == 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="data/local/models/bert-base-chinese"
    )
    parser.add_argument(
        "--raw-dir",
        default="/home/lora/workspace/projects/MulitiModal/data/local/tcm-sd",
    )
    parser.add_argument("--out", default="runs/issue57/c-sanity")
    args = parser.parse_args(argv)

    repo_root = Path.cwd().resolve()
    _manifest, records_by_row = load_selection_pool(repo_root, "short")
    labeled = [r for r in records_by_row.values() if r.label is not None]
    sample = labeled[:64]
    sample_texts = [r.input_text for r in sample]
    sample_labels = [int(r.label) for r in sample]

    overfit = overfit_64(args.model, sample_texts, sample_labels)
    truncation = truncation_audit(args.model, repo_root)
    determinism = save_reload_determinism(
        args.model,
        args.raw_dir,
        sample_texts[:16],
        sample_labels[:16],
        repo_root / args.out,
    )

    payload = {
        "issue": 57,
        "stage": "c-sanity",
        "model_path": args.model,
        "overfit_64": overfit,
        "truncation_audit": truncation,
        "save_reload_determinism": determinism,
        "verdict": {
            "overfit_ok": overfit["overfit_ok"],
            "determinism_ok": determinism["determinism_ok"],
        },
    }
    out_path = write_results(repo_root / args.out / "sanity-results.json", payload)
    print(json.dumps({"ok": True, "results": str(out_path),
                      "overfit_ok": overfit["overfit_ok"],
                      "determinism_ok": determinism["determinism_ok"],
                      "truncation": truncation["scenarios"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
