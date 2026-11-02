"""Synthetic regression coverage for frozen selection-only dev scoring."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from data.tcm_sd import LABEL_TAXONOMY_ID
from research.task_sd import (
    MODALITY_KIND_HASHED_TEXT,
    SyndromeRecord,
    build_syndrome_manifest,
    build_syndrome_selection_artifacts,
    records_to_json,
)
from scripts import evaluate_sd_v2_cv as evaluator

pytestmark = pytest.mark.synthetic


def _full_manifest(records, user_parts, scenario):
    user_ids = set().union(*user_parts.values())
    user_strata = {
        user_id: next(
            (
                record.norm_syndrome
                for record in records
                if record.user_id == user_id
            ),
            "y0",
        )
        for user_id in user_ids
    }
    payload = records_to_json(records)
    return build_syndrome_manifest(
        records,
        dataset_sha256=hashlib.sha256(payload.encode()).hexdigest(),
        n_classes=148,
        label_taxonomy_version=LABEL_TAXONOMY_ID,
        label_tier="SMOKE_ONLY",
        modality_kind=MODALITY_KIND_HASHED_TEXT,
        notes=f"scenario={scenario}; synthetic dev fixture",
        frozen_user_partitions=user_parts,
        frozen_user_strata=user_strata,
    )


def _build_inputs(root: Path, scenario: str, *, duplicate_dev_text: bool = False):
    user_parts = {"train": set(), "dev": set(), "test": set()}
    records = []
    for partition in ("train", "dev", "test"):
        for index in range(8):
            user_id = f"synthetic-{partition}-{index}"
            user_parts[partition].add(user_id)
            records.append(
                SyndromeRecord(
                    subject_index=len(records) + 1,
                    user_id=user_id,
                    features={},
                    label=index % 2,
                    norm_syndrome=f"y{index % 2}",
                    input_text=(
                        f"{scenario} {partition} symptom phrase {index} "
                        f"label-{index % 2}"
                    ),
                    provenance={"source": "synthetic-fixture"},
                )
            )

    if duplicate_dev_text:
        initial_manifest = _full_manifest(records, user_parts, scenario)
        initial_rows, initial_manifest_obj = build_syndrome_selection_artifacts(
            initial_manifest, records
        )
        train_ids = initial_manifest_obj.data["split"]["train"]
        dev_ids = initial_manifest_obj.data["split"]["dev"]
        row_by_subject = initial_manifest_obj.data["subject_row_by_id"]
        rows = {record.subject_index: record for record in initial_rows}
        shared_text = rows[row_by_subject[train_ids[0]]].input_text
        records_by_row = {record.subject_index: record for record in records}
        records_by_row[row_by_subject[dev_ids[0]]].input_text = shared_text

    full_manifest = _full_manifest(records, user_parts, scenario)
    selection_records, selection_manifest_obj = build_syndrome_selection_artifacts(
        full_manifest, records
    )
    selection_manifest = selection_manifest_obj.data
    input_dir = root / "runs" / "sd-v2" / scenario
    input_dir.mkdir(parents=True)
    records_path = input_dir / "selection_records.json"
    manifest_path = input_dir / "selection_manifest.json"
    records_path.write_bytes(records_to_json(selection_records).encode("utf-8"))
    manifest_path.write_text(
        json.dumps(selection_manifest, ensure_ascii=False), encoding="utf-8"
    )
    return selection_manifest, selection_records, records_path, manifest_path


class _RecordingBaseline:
    def __init__(self, calls):
        self.calls = calls

    def fit(self, texts, labels):
        self.calls.append(("fit", list(texts), list(labels)))
        return self

    def predict(self, texts):
        self.calls.append(("predict", list(texts), []))
        return [0] * len(texts)


def _run_dev(tmp_path, monkeypatch, *, duplicate_dev_text=False):
    manifest, records, records_path, manifest_path = _build_inputs(
        tmp_path, "short", duplicate_dev_text=duplicate_dev_text
    )
    calls = []
    monkeypatch.setattr(
        evaluator,
        "TfidfNgramBaseline",
        lambda _config=None: _RecordingBaseline(calls),
    )
    monkeypatch.chdir(tmp_path)
    assert evaluator.main(
        [
            "--mode",
            "dev",
            "--scenario",
            "short",
            "--out",
            "runs/sd-v2/dev-short",
        ]
    ) == 0
    report = json.loads(
        (tmp_path / "runs" / "sd-v2" / "dev-short" / "dev-report.json").read_text(
            encoding="utf-8"
        )
    )
    return manifest, records, records_path, manifest_path, calls, report


def test_dev_mode_fits_train_only_and_reports_hashes_and_148_class_support(
    tmp_path, monkeypatch
):
    manifest, records, records_path, manifest_path, calls, report = _run_dev(
        tmp_path, monkeypatch
    )
    fit_call = next(call for call in calls if call[0] == "fit")
    predict_call = next(call for call in calls if call[0] == "predict")
    row_by_subject = manifest["subject_row_by_id"]
    rows = {record.subject_index: record for record in records}
    expected_train = [
        rows[int(row_by_subject[subject_id])].input_text
        for subject_id in manifest["split"]["train"]
    ]
    expected_dev = [
        rows[int(row_by_subject[subject_id])].input_text
        for subject_id in manifest["split"]["dev"]
    ]

    assert fit_call[1] == expected_train
    assert predict_call[1] == expected_dev
    assert report["mode"] == "dev"
    assert report["input_contract"]["training_partition"] == "frozen split.train"
    assert report["input_contract"]["evaluation_partition"] == "frozen split.dev"
    assert report["input_contract"]["train_dev_users_disjoint"] is True
    assert report["input_contract"]["holdout_partition_empty"] is True
    assert report["input_file_sha256"]["selection_records.json"] == hashlib.sha256(
        records_path.read_bytes()
    ).hexdigest()
    assert report["input_file_sha256"]["selection_manifest.json"] == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    metric = report["dev"]["metrics"]
    assert metric["fixed_label_count"] == 148
    assert metric["macro_denominator"] == 148
    assert len(metric["per_class"]) == 148
    assert sum(item["support"] for item in metric["per_class"]) == len(
        predict_call[1]
    )


def test_dev_mode_drops_exact_cross_partition_scenario_text(tmp_path, monkeypatch):
    _manifest, _records, _records_path, _manifest_path, calls, report = _run_dev(
        tmp_path, monkeypatch, duplicate_dev_text=True
    )
    fit_call = next(call for call in calls if call[0] == "fit")
    predict_call = next(call for call in calls if call[0] == "predict")
    dev = report["dev"]

    assert dev["n_dev_text_duplicates_dropped"] == 1
    assert dev["n_dev_records_before_dedup"] == (
        dev["n_dev_records_after_dedup"] + 1
    )
    assert len(predict_call[1]) == dev["n_dev_records_scored"]
    assert set(fit_call[1]).isdisjoint(predict_call[1])


def test_dev_mode_rejects_nonempty_holdout_before_scoring(tmp_path):
    manifest, records, records_path, _manifest_path = _build_inputs(tmp_path, "short")
    bad_manifest = copy.deepcopy(manifest)
    bad_manifest["split"]["holdout"] = ["sd-r99999"]
    records_by_row = {record.subject_index: record for record in records}
    with pytest.raises(ValueError, match="empty holdout"):
        evaluator.evaluate_dev(
            bad_manifest,
            records_by_row,
            "short",
            input_hashes={
                "manifest_sha256": "0" * 64,
                "dataset_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
            },
        )


def test_dev_mode_rejects_record_user_mismatch(tmp_path):
    manifest, records, records_path, _manifest_path = _build_inputs(tmp_path, "short")
    records_by_row = {record.subject_index: record for record in records}
    train_subject = manifest["split"]["train"][0]
    train_row = int(manifest["subject_row_by_id"][train_subject])
    records_by_row[train_row].user_id = "mismatched-user"
    with pytest.raises(ValueError, match="user does not match frozen mapping"):
        evaluator.evaluate_dev(
            manifest,
            records_by_row,
            "short",
            input_hashes={
                "manifest_sha256": "0" * 64,
                "dataset_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
            },
        )
