"""Shared machinery for the #57 neural-model comparison experiments.

This module imports the frozen protocol pieces instead of forking them:

- fold membership: ``research.frozen.selection_fold_ids`` (as driven by
  ``research.task_sd.build_fold``, which also applies the one-way exact
  ``input_text`` dedup on the CV dev rows);
- manifest validation and the per-fold metric report:
  ``scripts.evaluate_sd_v2_cv._validate_manifest`` / ``._metric_report``;
- selection-input loading with hash verification:
  ``research.evaluator.load_selection_inputs``.

The per-fold loop below mirrors ``scripts.evaluate_sd_v2_cv.evaluate_cv``
with one deliberate difference: ``fit`` receives the deduplicated fold-dev
rows so neural models can do model selection (early stop / best epoch) on
fold-dev, exactly as the #57 design specifies.
"""
from __future__ import annotations

import json
import statistics
from collections.abc import Iterator
from pathlib import Path

from research.evaluator import load_selection_inputs
from research.task_sd import build_fold
from scripts.evaluate_sd_v2_cv import (
    FIXED_LABEL_COUNT,
    _validate_manifest,
)
from syndrome.metrics import per_class_metrics

#: head/tail class strata over the frozen 148-class taxonomy, ranked by
#: per-fold TRAIN support: head = top quartile (37 classes), tail = bottom
#: half (74 classes). Stratum macro-F1 averages per-class F1 (zero_division
#: stays 0, same semantics as the primary metric) within each stratum.
N_HEAD = 37
N_TAIL = 74


def load_selection_pool(repo_root: Path, scenario: str) -> tuple[dict, dict[int, object]]:
    """Load and validate the frozen protocol-v2 selection inputs."""
    input_dir = repo_root / "runs" / "sd-v2" / scenario
    input_config = {
        "task": "syndrome",
        "data": {
            "manifest_path": str(input_dir / "selection_manifest.json"),
            "dataset_path": str(input_dir / "selection_records.json"),
        },
    }
    manifest, records, _hashes = load_selection_inputs(
        input_config,
        manifest_validator=lambda value: _validate_manifest(value, scenario),
    )
    records_by_row = {record.subject_index: record for record in records}
    return manifest, records_by_row


def iter_cv_folds(
    manifest: dict,
    records_by_row: dict[int, object],
    scenario: str,
    seed: int,
    folds: int = 4,
    repeats: int = 2,
    fold_filter: tuple[int, int] | None = None,
) -> Iterator[dict]:
    """Yield one deduplicated CV fold at a time, mirroring evaluate_cv."""
    row_by_subject = manifest["subject_row_by_id"]
    for repeat in range(repeats):
        for fold in range(folds):
            if fold_filter is not None and (repeat, fold) != fold_filter:
                continue
            _fold_data, fold_manifest = build_fold(
                {
                    "train_provider": {"batch_size": 32},
                    "trainer": {"seed": seed},
                },
                manifest,
                fold=fold,
                folds=folds,
                repeat=repeat,
                records=list(records_by_row.values()),
            )
            train_ids = fold_manifest["split"]["train"]
            dev_ids = fold_manifest["split"]["dev"]
            train_rows = [
                records_by_row[row_by_subject[subject_id]] for subject_id in train_ids
            ]
            dev_rows = [
                records_by_row[row_by_subject[subject_id]] for subject_id in dev_ids
            ]
            train_labeled = [r for r in train_rows if r.label is not None]
            dev_labeled = [r for r in dev_rows if r.label is not None]
            if not train_labeled or not dev_labeled:
                raise ValueError("CV fold has no labeled train or deduped dev rows")
            dedup = fold_manifest["selection_fold"]["dev_text_dedup"]
            yield {
                "repeat": repeat,
                "fold": fold,
                "train_labeled": train_labeled,
                "dev_labeled": dev_labeled,
                "n_train_records": len(train_rows),
                "n_dev_records_before_dedup": dedup["n_before"],
                "n_dev_text_duplicates_dropped": dedup["n_dropped"],
            }


def head_tail_strata(
    y_true: list[int], y_pred: list[int], train_labels: list[int]
) -> dict:
    """Macro-F1 over head (top-quartile train support) and tail (bottom-half) classes."""
    counts = [0] * FIXED_LABEL_COUNT
    for label in train_labels:
        counts[label] += 1
    ranked = sorted(range(FIXED_LABEL_COUNT), key=lambda c: (-counts[c], c))
    head = set(ranked[:N_HEAD])
    tail = set(ranked[len(ranked) - N_TAIL :])
    classes = per_class_metrics(
        y_true, y_pred, [str(index) for index in range(FIXED_LABEL_COUNT)]
    )
    strata = {}
    for name, members in (("head", head), ("tail", tail)):
        f1s = [classes[c]["f1"] for c in members]
        strata[name] = {
            "definition": (
                f"top-{N_HEAD}" if name == "head" else f"bottom-{N_TAIL}"
            )
            + " of 148 classes ranked by fold-train label count (ties by class id)",
            "macro_f1": round(sum(f1s) / len(f1s), 6),
            "n_dev_true_examples": sum(
                classes[c]["support"] for c in members
            ),
        }
    return strata


def summarize_folds(fold_results: list[dict]) -> dict:
    values = [item["metrics"]["macro_f1"] for item in fold_results]
    return {
        "mean_fold_macro_f1": round(statistics.fmean(values), 6),
        "population_std_fold_macro_f1": round(statistics.pstdev(values), 6),
        "n_fold_runs": len(fold_results),
        "fold_macro_f1s": values,
    }


def write_results(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return path
