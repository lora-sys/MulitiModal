"""User-grouped development CV for #56 protocol-v2 selection artifacts.

Only selection_records.json and selection_manifest.json are read from
runs/sd-v2/<scenario>. The selection manifest freezes train/dev rows and their
user mappings while omitting holdout rows. No mixed record bundle, finalizer,
ledger, test split, or candidate patch is opened or invoked.

Run from the repository root with a new output directory:
    PYTHONPATH=src python -m scripts.evaluate_sd_v2_cv --mode cv --scenario short \
        --out runs/sd-v2/<new-id>/short
    PYTHONPATH=src python -m scripts.evaluate_sd_v2_cv --mode cv --scenario clinical \
        --out runs/sd-v2/<new-id>/clinical
    PYTHONPATH=src python -m scripts.evaluate_sd_v2_cv --mode dev --scenario short \
        --out runs/sd-v2/<new-id>/dev-short
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from data.tcm_sd import LABEL_TAXONOMY_ID
from research.evaluator import load_selection_inputs
from research.frozen import selection_fold_ids
from research.task_sd import (
    EVAL_METRIC_VERSION,
    PRIMARY_METRIC_SD,
    SyndromeRecord,
    build_fold,
)
from research.tasks import validate_manifest_contract
from syndrome.baselines import (
    NGRAM_BASELINE_NAME,
    TCM_SD_CLASS_COUNT,
    TfidfNgramBaseline,
    TfidfNgramConfig,
    environment_record,
)
from syndrome.metrics import macro_f1, per_class_metrics

FIXED_LABEL_COUNT = 148
CV_FOLDS = 4
CV_REPEATS = 2
BASELINE_CONFIG = TfidfNgramConfig(
    ngram_low=1,
    ngram_high=3,
    min_df=2,
    sublinear_tf=True,
    C=4.0,
    class_weight="balanced",
    max_iter=2000,
    seed=20261101,
    solver="ovr-liblinear",
)


def _new_baseline() -> TfidfNgramBaseline:
    return TfidfNgramBaseline(TfidfNgramConfig(**asdict(BASELINE_CONFIG)))


def _validate_manifest(manifest: dict, scenario: str) -> None:
    validate_manifest_contract("syndrome", manifest, selection_pool=True)
    if manifest["split"]["holdout"] != []:
        raise ValueError("selection manifest must have an empty holdout partition")
    if manifest.get("task") != "syndrome":
        raise ValueError("manifest task must be syndrome")
    if manifest.get("task_type") != "syndrome_classification":
        raise ValueError("manifest task_type must be syndrome_classification")
    if manifest.get("n_classes") != FIXED_LABEL_COUNT:
        raise ValueError("v2 CV requires the frozen 148-class taxonomy")
    if TCM_SD_CLASS_COUNT != FIXED_LABEL_COUNT:
        raise RuntimeError("baseline class count no longer matches the frozen 148")
    if manifest.get("label_taxonomy_version") != LABEL_TAXONOMY_ID:
        raise ValueError("manifest label taxonomy does not match TCM-SD v1")
    if manifest.get("modalities", {}).get("text", {}).get("kind") != "hashed_text":
        raise ValueError("v2 CV requires the hashed_text modality")
    notes = manifest.get("notes")
    if not isinstance(notes, str) or not notes.startswith(f"scenario={scenario};"):
        raise ValueError("manifest scenario does not match --scenario")

    row_by_subject = manifest["subject_row_by_id"]
    user_by_subject = manifest.get("subject_user_by_id")
    user_partitions = manifest.get("user_partition_by_id")
    user_strata = manifest.get("user_stratum_by_id")
    if not all(
        isinstance(value, dict)
        for value in (user_by_subject, user_partitions, user_strata)
    ):
        raise ValueError("selection manifest lacks frozen pool user metadata")
    if set(user_by_subject) != set(row_by_subject):
        raise ValueError("selection user mapping does not match selection rows")
    present_users = set(user_by_subject.values())
    frozen_users = set(user_partitions)
    if set(user_strata) != frozen_users:
        raise ValueError("selection user partitions and strata have different users")
    if not present_users <= frozen_users:
        raise ValueError("selection rows reference an unfrozen user")
    if any(partition not in {"train", "dev"} for partition in user_partitions.values()):
        raise ValueError("selection user metadata must not contain holdout users")
    if any(not isinstance(stratum, str) or not stratum for stratum in user_strata.values()):
        raise ValueError("selection manifest contains an invalid frozen user stratum")
    for partition in ("train", "dev"):
        if any(
            user_partitions[user_by_subject[subject_id]] != partition
            for subject_id in manifest["split"][partition]
        ):
            raise ValueError(
                f"selection user partitions disagree with split {partition!r}"
            )

    parent_sha = manifest.get("selection_parent_manifest_sha256")
    if (
        not isinstance(parent_sha, str)
        or len(parent_sha) != 64
        or any(char not in "0123456789abcdef" for char in parent_sha)
    ):
        raise ValueError("selection manifest lacks its parent manifest SHA-256")


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_run_paths(
    scenario: str, out_arg: str | Path, repo_root: Path | None = None
) -> tuple[Path, Path, Path]:
    root = (repo_root or Path.cwd()).resolve()
    sd_root = (root / "runs" / "sd-v2").resolve(strict=True)
    if not _within(sd_root, root):
        raise ValueError("runs/sd-v2 resolves outside the repository")
    input_dir = (sd_root / scenario).resolve(strict=True)
    if not _within(input_dir, sd_root):
        raise ValueError("scenario input directory resolves outside runs/sd-v2")
    records_path = input_dir / "selection_records.json"
    manifest_path = input_dir / "selection_manifest.json"

    out_candidate = Path(out_arg)
    if not out_candidate.is_absolute():
        out_candidate = root / out_candidate
    out_dir = out_candidate.resolve(strict=False)
    if out_dir == sd_root or not _within(out_dir, sd_root):
        raise ValueError("output must be a new directory under runs/sd-v2")
    if _within(out_dir, input_dir) or _within(input_dir, out_dir):
        raise ValueError("output directory must be separate from scenario inputs")
    if os.path.lexists(out_candidate):
        raise FileExistsError("output directory already exists")
    return records_path, manifest_path, out_dir


def _metric_report(
    y_true: list[int],
    y_pred: list[int],
    *,
    support_definition: str = (
        "count of true labels in this fold's post-dedup CV dev rows"
    ),
    absent_class_scope: str = "CV dev",
) -> dict:
    labels = [str(index) for index in range(FIXED_LABEL_COUNT)]
    classes = per_class_metrics(y_true, y_pred, labels)
    return {
        "primary_metric": PRIMARY_METRIC_SD,
        "metric_version": EVAL_METRIC_VERSION,
        "fixed_label_count": FIXED_LABEL_COUNT,
        "macro_denominator": FIXED_LABEL_COUNT,
        "support_definition": support_definition,
        "zero_division": 0,
        "empty_or_unseen_class_definition": (
            "all 148 frozen label ids remain in the macro denominator; "
            "zero_division=0 sets undefined per-class precision, recall, and F1 "
            f"to 0; support is 0 when the class is absent from {absent_class_scope}"
        ),
        "n_examples": len(y_true),
        "n_true_classes_observed": len(set(y_true)),
        "n_predicted_classes_observed": len(set(y_pred)),
        "macro_f1": round(macro_f1(y_true, y_pred, FIXED_LABEL_COUNT), 6),
        "per_class": [
            {
                "label_id": index,
                "support": item["support"],
                "precision": item["precision"],
                "recall": item["recall"],
                "f1": item["f1"],
            }
            for index, item in enumerate(classes)
        ],
    }


def evaluate_cv(
    manifest: dict,
    manifest_path: Path,
    records_path: Path,
    records_by_row: dict[int, SyndromeRecord],
    scenario: str,
    *,
    baseline_factory: Callable[[], TfidfNgramBaseline] | None = None,
) -> dict:
    _validate_manifest(manifest, scenario)
    row_by_subject = manifest["subject_row_by_id"]
    pool_subjects = set(manifest["split"]["train"]) | set(manifest["split"]["dev"])
    model_factory = baseline_factory or _new_baseline
    fold_results: list[dict] = []

    for repeat in range(CV_REPEATS):
        for fold in range(CV_FOLDS):
            expected_train_ids, expected_dev_ids = selection_fold_ids(
                manifest, fold=fold, folds=CV_FOLDS, repeat=repeat
            )
            if (
                set(expected_train_ids) & set(expected_dev_ids)
                or set(expected_train_ids) | set(expected_dev_ids) != pool_subjects
            ):
                raise ValueError("selection_fold_ids returned invalid CV partitions")
            user_by_subject = manifest["subject_user_by_id"]
            train_users = {user_by_subject[subject_id] for subject_id in expected_train_ids}
            dev_users = {user_by_subject[subject_id] for subject_id in expected_dev_ids}
            if train_users & dev_users:
                raise ValueError("selection fold split a user across train and dev")

            # Reuse the task fold builder for exact scenario-text dedup. It
            # receives only selection records and creates an empty holdout.
            _fold_data, fold_manifest = build_fold(
                {
                    "train_provider": {"batch_size": 32},
                    "trainer": {"seed": BASELINE_CONFIG.seed},
                },
                manifest,
                fold=fold,
                folds=CV_FOLDS,
                repeat=repeat,
                records=list(records_by_row.values()),
            )
            if fold_manifest["split"]["holdout"] != []:
                raise ValueError("CV fold builder returned a non-empty holdout")
            train_ids = fold_manifest["split"]["train"]
            dev_ids = fold_manifest["split"]["dev"]
            if set(train_ids) != set(expected_train_ids):
                raise ValueError("fold builder changed CV train membership")
            if not set(dev_ids) <= set(expected_dev_ids):
                raise ValueError("fold builder added rows to CV dev")

            train_rows = [
                records_by_row[row_by_subject[subject_id]] for subject_id in train_ids
            ]
            dev_rows = [
                records_by_row[row_by_subject[subject_id]] for subject_id in dev_ids
            ]
            if not train_rows or not dev_rows:
                raise ValueError("CV fold has no train rows or no deduplicated dev rows")
            model = model_factory()
            model.fit(
                [record.input_text for record in train_rows],
                [int(record.label) for record in train_rows],
            )
            y_true = [int(record.label) for record in dev_rows]
            y_pred = [
                int(value)
                for value in model.predict([record.input_text for record in dev_rows])
            ]
            if len(y_true) != len(y_pred):
                raise ValueError("baseline prediction count does not match CV dev")

            metric = _metric_report(y_true, y_pred)
            dedup = fold_manifest["selection_fold"]["dev_text_dedup"]
            fold_results.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "n_train_records": len(train_rows),
                    "n_train_users": len(train_users),
                    "n_dev_records_before_dedup": dedup["n_before"],
                    "n_dev_records_scored": len(dev_rows),
                    "n_dev_text_duplicates_dropped": dedup["n_dropped"],
                    "n_dev_users_scored": len({record.user_id for record in dev_rows}),
                    "metrics": metric,
                }
            )

    values = [item["metrics"]["macro_f1"] for item in fold_results]
    return {
        "scenario": scenario,
        "status": "OK",
        "selection_parent_manifest_sha256": manifest[
            "selection_parent_manifest_sha256"
        ],
        "selection_parent_dataset_sha256": manifest[
            "selection_parent_dataset_sha256"
        ],
        "selection_manifest_sha256": manifest["manifest_sha256"],
        "dataset_sha256": manifest["dataset_sha256"],
        "label_tier": manifest["label_tier"],
        "label_taxonomy_version": manifest["label_taxonomy_version"],
        "n_selection_records_loaded": len(records_by_row),
        "n_selection_users_with_rows": len(
            {record.user_id for record in records_by_row.values()}
        ),
        "n_selection_frozen_pool_users": len(manifest["user_partition_by_id"]),
        "cv": {
            "protocol": "selection_fold_ids user-grouped CV over frozen train+dev",
            "folds": CV_FOLDS,
            "repeats": CV_REPEATS,
            "seed_source": (
                "manifest.split.seed + selection fold version + repeat + "
                "frozen user stratum"
            ),
            "text_dedup": (
                "research.task_sd.build_fold exact input_text; CV dev against "
                "CV train, one-way"
            ),
            "mean_fold_macro_f1": round(statistics.fmean(values), 6),
            "population_std_fold_macro_f1": round(statistics.pstdev(values), 6),
            "n_fold_runs": len(fold_results),
        },
        "training_config": {
            "baseline": NGRAM_BASELINE_NAME,
            "parameters": asdict(BASELINE_CONFIG),
            "fit_scope": "selection_fold_ids train rows only",
            "environment": environment_record(seed=BASELINE_CONFIG.seed),
        },
        "metric_contract": {
            "primary_metric": PRIMARY_METRIC_SD,
            "metric_version": EVAL_METRIC_VERSION,
            "fixed_label_count": FIXED_LABEL_COUNT,
            "macro_denominator": FIXED_LABEL_COUNT,
            "support_definition": (
                "per-fold count of true labels in the post-dedup CV dev rows"
            ),
            "zero_division": 0,
            "empty_or_unseen_class_definition": (
                "all 148 frozen ids remain in the denominator; zero_division=0 "
                "sets undefined per-class precision, recall, and F1 to 0; "
                "support is 0 when absent from CV dev"
            ),
        },
        "input_contract": {
            "records_file": "selection_records.json",
            "manifest_file": "selection_manifest.json",
            "selection_records_scope": "original frozen split.train and split.dev",
            "holdout_partition_empty": True,
            "holdout_user_mapping_present": False,
            "holdout_records_fit_or_scored": 0,
        },
        "folds": fold_results,
    }



def evaluate_dev(
    manifest: dict,
    records_by_row: dict[int, SyndromeRecord],
    scenario: str,
    *,
    input_hashes: dict[str, str],
    baseline_factory: Callable[[], TfidfNgramBaseline] | None = None,
) -> dict:
    """Fit the frozen baseline on train and score exact-text-deduped dev only."""
    if manifest.get("split", {}).get("holdout") != []:
        raise ValueError("dev evaluation requires an empty holdout partition")
    _validate_manifest(manifest, scenario)

    row_by_subject = manifest["subject_row_by_id"]
    train_ids = list(manifest["split"]["train"])
    dev_ids = list(manifest["split"]["dev"])
    train_subjects = set(train_ids)
    dev_subjects = set(dev_ids)
    if train_subjects & dev_subjects:
        raise ValueError("frozen train and dev subject IDs overlap")
    if train_subjects | dev_subjects != set(row_by_subject):
        raise ValueError("frozen train/dev rows do not match the selection mapping")

    expected_rows = {int(row) for row in row_by_subject.values()}
    if set(records_by_row) != expected_rows:
        raise ValueError("dev records do not match frozen train/dev row mapping")

    subject_by_row = {
        int(row): subject_id for subject_id, row in row_by_subject.items()
    }
    user_by_subject = manifest["subject_user_by_id"]
    for row, record in records_by_row.items():
        subject_id = subject_by_row[row]
        if record.user_id != user_by_subject[subject_id]:
            raise ValueError(
                f"selection record user does not match frozen mapping for row {row}"
            )

    train_users = {user_by_subject[subject_id] for subject_id in train_ids}
    dev_users = {user_by_subject[subject_id] for subject_id in dev_ids}
    if train_users & dev_users:
        raise ValueError("frozen dev users overlap train users")

    train_rows = [
        records_by_row[int(row_by_subject[subject_id])]
        for subject_id in train_ids
    ]
    dev_rows_before_dedup = [
        records_by_row[int(row_by_subject[subject_id])]
        for subject_id in dev_ids
    ]
    if not train_rows or not dev_rows_before_dedup:
        raise ValueError("frozen train or dev partition has no selection rows")

    train_texts = {record.input_text for record in train_rows}
    dev_rows_after_dedup = [
        record
        for record in dev_rows_before_dedup
        if record.input_text not in train_texts
    ]
    n_text_duplicates_dropped = (
        len(dev_rows_before_dedup) - len(dev_rows_after_dedup)
    )
    train_labeled = [record for record in train_rows if record.label is not None]
    dev_labeled = [
        record for record in dev_rows_after_dedup if record.label is not None
    ]
    if not train_labeled or not dev_labeled:
        raise ValueError("dev evaluation has no labeled train or deduplicated dev rows")

    y_train = [int(record.label) for record in train_labeled]
    y_true = [int(record.label) for record in dev_labeled]
    if any(label < 0 or label >= FIXED_LABEL_COUNT for label in y_train + y_true):
        raise ValueError("train/dev labels fall outside the frozen 148-class taxonomy")

    model_factory = baseline_factory or _new_baseline
    model = model_factory()
    model.fit(
        [record.input_text for record in train_labeled],
        y_train,
    )
    y_pred = [
        int(value)
        for value in model.predict([record.input_text for record in dev_labeled])
    ]
    if len(y_true) != len(y_pred):
        raise ValueError("baseline prediction count does not match deduplicated dev")

    metrics = _metric_report(
        y_true,
        y_pred,
        support_definition=(
            "count of true labels in frozen split.dev rows after exact "
            "scenario-text dedup against frozen split.train"
        ),
        absent_class_scope="frozen dev",
    )
    expected_hashes = {"manifest_sha256", "dataset_sha256"}
    if set(input_hashes) != expected_hashes:
        raise ValueError("selection input hashes are incomplete")
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
        for value in input_hashes.values()
    ):
        raise ValueError("selection input hashes must be lowercase SHA-256 values")
    if input_hashes["dataset_sha256"] != manifest["dataset_sha256"]:
        raise ValueError("selection records hash does not match the frozen manifest")
    return {
        "scenario": scenario,
        "mode": "dev",
        "status": "OK",
        "selection_parent_manifest_sha256": manifest[
            "selection_parent_manifest_sha256"
        ],
        "selection_parent_dataset_sha256": manifest[
            "selection_parent_dataset_sha256"
        ],
        "selection_manifest_sha256": manifest["manifest_sha256"],
        "dataset_sha256": manifest["dataset_sha256"],
        "input_file_sha256": {
            "selection_manifest.json": input_hashes["manifest_sha256"],
            "selection_records.json": input_hashes["dataset_sha256"],
        },
        "label_tier": manifest["label_tier"],
        "label_taxonomy_version": manifest["label_taxonomy_version"],
        "dev": {
            "n_train_records": len(train_rows),
            "n_train_labeled_records": len(train_labeled),
            "n_train_users": len(train_users),
            "n_dev_records_before_dedup": len(dev_rows_before_dedup),
            "n_dev_text_duplicates_dropped": n_text_duplicates_dropped,
            "n_dev_records_after_dedup": len(dev_rows_after_dedup),
            "n_dev_unlabeled_records_excluded": (
                len(dev_rows_after_dedup) - len(dev_labeled)
            ),
            "n_dev_records_scored": len(dev_labeled),
            "n_dev_users_scored": len({record.user_id for record in dev_labeled}),
            "metrics": metrics,
        },
        "training_config": {
            "baseline": NGRAM_BASELINE_NAME,
            "parameters": asdict(BASELINE_CONFIG),
            "fit_scope": "frozen split.train rows with observed labels only",
            "environment": environment_record(seed=BASELINE_CONFIG.seed),
        },
        "metric_contract": {
            "primary_metric": PRIMARY_METRIC_SD,
            "metric_version": EVAL_METRIC_VERSION,
            "fixed_label_count": FIXED_LABEL_COUNT,
            "macro_denominator": FIXED_LABEL_COUNT,
            "support_definition": metrics["support_definition"],
            "zero_division": 0,
            "empty_or_unseen_class_definition": (
                "all 148 frozen ids remain in the denominator; zero_division=0 "
                "sets undefined per-class precision, recall, and F1 to 0; "
                "support is 0 when absent from dev"
            ),
        },
        "input_contract": {
            "records_file": "selection_records.json",
            "manifest_file": "selection_manifest.json",
            "training_partition": "frozen split.train",
            "evaluation_partition": "frozen split.dev",
            "dev_text_dedup": "exact scenario input_text against train",
            "holdout_partition_empty": True,
            "train_dev_users_disjoint": True,
            "holdout_records_fit_or_scored": 0,
        },
    }


def _write_report(
    out_dir: Path, report: dict, filename: str = "cv-report.json"
) -> Path:
    report_path = out_dir / filename
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="#56 dev and user-grouped CV on protocol-v2 selection inputs"
    )
    parser.add_argument("--mode", choices=("cv", "dev"), default="cv")
    parser.add_argument("--scenario", choices=("short", "clinical"), required=True)
    parser.add_argument(
        "--out",
        required=True,
        help="new, unused output directory under runs/sd-v2",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        records_path, manifest_path, out_dir = resolve_run_paths(
            args.scenario, args.out
        )
        input_config = {
            "task": "syndrome",
            "data": {
                "manifest_path": str(manifest_path),
                "dataset_path": str(records_path),
            },
        }
        manifest, selection_records, input_hashes = load_selection_inputs(
            input_config,
            manifest_validator=lambda value: _validate_manifest(
                value, args.scenario
            ),
        )
        records_by_row = {
            record.subject_index: record for record in selection_records
        }
        if args.mode == "dev":
            report = evaluate_dev(
                manifest,
                records_by_row,
                args.scenario,
                input_hashes=input_hashes,
            )
            report_name = "dev-report.json"
        else:
            report = evaluate_cv(
                manifest, manifest_path, records_path, records_by_row, args.scenario
            )
            report_name = "cv-report.json"
        out_dir.mkdir(parents=True, exist_ok=False)
        report_path = _write_report(out_dir, report, report_name)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 2

    print(
        json.dumps(
            {
                "ok": True,
                "scenario": args.scenario,
                "report": str(report_path),
                "selection_manifest_sha256": report["selection_manifest_sha256"],
                "mode": args.mode,
                **(
                    {
                        "dev_macro_f1": report["dev"]["metrics"]["macro_f1"],
                        "n_dev_records_scored": report["dev"]["n_dev_records_scored"],
                    }
                    if args.mode == "dev"
                    else {
                        "mean_fold_macro_f1": report["cv"]["mean_fold_macro_f1"],
                        "n_fold_runs": report["cv"]["n_fold_runs"],
                    }
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
