from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from data.tcm_sd import LABEL_TAXONOMY_ID
from research.frozen import selection_fold_ids
from research.task_sd import (
    EVAL_METRIC_VERSION,
    MODALITY_KIND_HASHED_TEXT,
    SyndromeRecord,
    build_syndrome_manifest,
    build_syndrome_selection_artifacts,
    records_to_json,
)
from scripts import evaluate_sd_v2_cv as evaluator

pytestmark = pytest.mark.synthetic


def _full_manifest(records, user_parts, scenario):
    users = set().union(*user_parts.values())
    user_strata = {
        user_id: next(
            (
                record.norm_syndrome
                for record in records
                if record.user_id == user_id
            ),
            "y0",
        )
        for user_id in users
    }
    payload = records_to_json(records)
    return build_syndrome_manifest(
        records,
        dataset_sha256=hashlib.sha256(payload.encode()).hexdigest(),
        n_classes=148,
        label_taxonomy_version=LABEL_TAXONOMY_ID,
        label_tier="SMOKE_ONLY",
        modality_kind=MODALITY_KIND_HASHED_TEXT,
        notes=f"scenario={scenario}; synthetic fixture",
        frozen_user_partitions=user_parts,
        frozen_user_strata=user_strata,
    )


def _selection_artifact(records, full_manifest):
    return build_syndrome_selection_artifacts(full_manifest, records)


def _write_inputs(root: Path, scenario: str, *, remove_dev_user: bool = False):
    records = []
    user_parts = {"train": set(), "dev": set(), "test": set()}
    removed_user_id = None
    for partition in ("train", "dev", "test"):
        for index in range(8):
            user_id = f"synthetic-{partition}-{index}"
            user_parts[partition].add(user_id)
            if remove_dev_user and partition == "dev" and index == 7:
                # This user's only scenario row was removed by scenario dedup.
                removed_user_id = user_id
                continue
            label = index % 2
            text = (
                f"{scenario} shared clinical phrase class-{label} "
                f"record-{partition}-{index}"
            )
            if partition == "test":
                text = f"HOLDOUT_ONLY_SECRET_TOKEN_{index}"
            records.append(
                SyndromeRecord(
                    subject_index=len(records) + 1,
                    user_id=user_id,
                    features={},
                    label=label,
                    norm_syndrome=f"y{label}",
                    input_text=text,
                    provenance={"source": "synthetic-fixture"},
                )
            )

    full_manifest = _full_manifest(records, user_parts, scenario)
    selection_records, selection_manifest_obj = _selection_artifact(
        records, full_manifest
    )
    selection_manifest = selection_manifest_obj.data
    train_ids, dev_ids = selection_fold_ids(
        selection_manifest, fold=0, folds=4, repeat=0
    )
    row_by_subject = selection_manifest["subject_row_by_id"]
    records_by_row = {record.subject_index: record for record in selection_records}
    train_record = records_by_row[row_by_subject[train_ids[0]]]
    dev_record = records_by_row[row_by_subject[dev_ids[0]]]
    dev_record.input_text = train_record.input_text

    full_manifest = _full_manifest(records, user_parts, scenario)
    selection_records, selection_manifest_obj = _selection_artifact(
        records, full_manifest
    )
    selection_manifest = selection_manifest_obj.data
    selection_payload = records_to_json(selection_records).encode("utf-8")

    input_dir = root / "runs" / "sd-v2" / scenario
    input_dir.mkdir(parents=True)
    selection_records_path = input_dir / "selection_records.json"
    selection_manifest_path = input_dir / "selection_manifest.json"
    selection_records_path.write_bytes(selection_payload)
    selection_manifest_path.write_text(
        json.dumps(selection_manifest, ensure_ascii=False), encoding="utf-8"
    )

    # Synthetic mixed artifacts are forbidden paths for the evaluator.
    mixed_records_path = input_dir / "records.json"
    mixed_manifest_path = input_dir / "manifest.json"
    holdout_path = input_dir / "holdout.json"
    mixed_records_path.write_text(records_to_json(records), encoding="utf-8")
    mixed_manifest_path.write_text(
        json.dumps(full_manifest.data, ensure_ascii=False), encoding="utf-8"
    )
    holdout_path.write_text(
        json.dumps(
            {
                "records": [
                    record.input_text
                    for record in records
                    if record.user_id.startswith("synthetic-test-")
                ]
            }
        ),
        encoding="utf-8",
    )
    return (
        records,
        selection_manifest,
        selection_records,
        selection_records_path,
        selection_manifest_path,
        {mixed_records_path, mixed_manifest_path, holdout_path},
        removed_user_id,
    )


class _RecordingBaseline:
    def __init__(self, calls):
        self.calls = calls

    def fit(self, texts, labels):
        self.calls.append(("fit", list(texts), list(labels)))
        return self

    def predict(self, texts):
        self.calls.append(("predict", list(texts), []))
        return [0] * len(texts)


def _run_cv(tmp_path, monkeypatch, scenario, calls, output_name):
    monkeypatch.setattr(
        evaluator,
        "TfidfNgramBaseline",
        lambda _config=None: _RecordingBaseline(calls),
    )
    monkeypatch.chdir(tmp_path)
    assert evaluator.main(
        [
            "--scenario",
            scenario,
            "--out",
            f"runs/sd-v2/{output_name}",
        ]
    ) == 0
    return json.loads(
        (
            tmp_path
            / "runs"
            / "sd-v2"
            / output_name
            / "cv-report.json"
        ).read_text(encoding="utf-8")
    )


@pytest.mark.parametrize("scenario", ["short", "clinical"])
def test_v2_cv_uses_selection_folds_deduplicates_and_never_opens_holdout_paths(
    tmp_path, monkeypatch, scenario
):
    (
        _records,
        manifest,
        selection_records,
        selection_records_path,
        selection_manifest_path,
        forbidden_paths,
        _removed_user_id,
    ) = _write_inputs(tmp_path, scenario)
    calls = []
    opened_paths = []
    descriptor_opens = []
    original_path_open = Path.open
    original_os_open = os.open

    def recording_path_open(path, *args, **kwargs):
        opened_paths.append(path.resolve())
        return original_path_open(path, *args, **kwargs)

    def recording_os_open(path, flags, *args, **kwargs):
        descriptor_opens.append((Path(os.fspath(path)).absolute(), flags))
        return original_os_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(Path, "open", recording_path_open)
    monkeypatch.setattr(os, "open", recording_os_open)
    report = _run_cv(
        tmp_path,
        monkeypatch,
        scenario,
        calls,
        f"cv-{scenario}-new",
    )

    forbidden_resolved = {path.resolve() for path in forbidden_paths}
    assert forbidden_resolved.isdisjoint(opened_paths)
    assert forbidden_resolved.isdisjoint(
        {path.resolve() for path, _flags in descriptor_opens}
    )
    selection_file_opens = {
        path.name: flags
        for path, flags in descriptor_opens
        if path in {selection_records_path.absolute(), selection_manifest_path.absolute()}
    }
    assert set(selection_file_opens) == {
        "selection_records.json",
        "selection_manifest.json",
    }
    assert all(flags & os.O_NOFOLLOW for flags in selection_file_opens.values())

    assert report["scenario"] == scenario
    assert report["metric_contract"]["metric_version"] == EVAL_METRIC_VERSION
    assert report["metric_contract"]["fixed_label_count"] == 148
    assert report["n_selection_records_loaded"] == len(selection_records)
    assert report["input_contract"]["holdout_partition_empty"] is True
    assert report["input_contract"]["holdout_user_mapping_present"] is False
    assert report["input_contract"]["holdout_records_fit_or_scored"] == 0

    fold_zero = next(
        item
        for item in report["folds"]
        if item["repeat"] == 0 and item["fold"] == 0
    )
    assert fold_zero["n_dev_text_duplicates_dropped"] >= 1
    assert fold_zero["n_dev_records_before_dedup"] == (
        fold_zero["n_dev_records_scored"]
        + fold_zero["n_dev_text_duplicates_dropped"]
    )

    pool_subjects = set(manifest["split"]["train"]) | set(manifest["split"]["dev"])
    assert manifest["split"]["holdout"] == []
    assert not any(
        partition == "holdout"
        for partition in manifest["user_partition_by_id"].values()
    )
    user_map = manifest["subject_user_by_id"]
    for repeat in range(evaluator.CV_REPEATS):
        for fold in range(evaluator.CV_FOLDS):
            train_ids, dev_ids = selection_fold_ids(
                manifest,
                fold=fold,
                folds=evaluator.CV_FOLDS,
                repeat=repeat,
            )
            assert not set(train_ids) & set(dev_ids)
            assert set(train_ids) | set(dev_ids) == pool_subjects
            assert {user_map[subject_id] for subject_id in train_ids}.isdisjoint(
                {user_map[subject_id] for subject_id in dev_ids}
            )

    observed_texts = [
        text
        for _kind, texts, _labels in calls
        for text in texts
    ]
    assert all("HOLDOUT_ONLY_SECRET_TOKEN" not in text for text in observed_texts)
    assert all(
        "HOLDOUT_ONLY_SECRET_TOKEN" not in record.input_text
        for record in selection_records
    )


def test_builder_selection_artifact_with_empty_dev_user_reaches_evaluator(
    tmp_path, monkeypatch
):
    (
        _records,
        manifest,
        selection_records,
        _selection_records_path,
        _selection_manifest_path,
        _forbidden_paths,
        removed_user_id,
    ) = _write_inputs(tmp_path, "short", remove_dev_user=True)
    assert removed_user_id is not None
    assert removed_user_id in manifest["user_partition_by_id"]
    assert removed_user_id in manifest["user_stratum_by_id"]
    assert removed_user_id not in manifest["subject_user_by_id"].values()
    calls = []
    report = _run_cv(
        tmp_path,
        monkeypatch,
        "short",
        calls,
        "cv-short-removed-dev-user",
    )
    assert report["n_selection_records_loaded"] == len(selection_records)
    assert (
        report["n_selection_frozen_pool_users"]
        == report["n_selection_users_with_rows"] + 1
    )
    assert report["input_contract"]["holdout_partition_empty"] is True
    assert len(report["folds"]) == evaluator.CV_FOLDS * evaluator.CV_REPEATS


def test_removing_a_single_row_user_fails_without_rebalancing_frozen_folds():
    users = [f"stable-user-{index}" for index in range(6)]
    user_parts = {
        # the internal holdout user keeps rows so every frozen partition of
        # the parent manifest is non-empty; the CV pool is train+dev only
        "train": set(users[:3]),
        "dev": set(users[3:5]),
        "test": set(users[5:]),
    }
    # two scenario rows per user so the development bundle clears the
    # 10-case minimum; user-level fold identity is what matters here
    records = []
    for user_id in users:
        for row in range(2):
            records.append(
                SyndromeRecord(
                    subject_index=len(records) + 1,
                    user_id=user_id,
                    features={},
                    label=0,
                    norm_syndrome="y0",
                    input_text=f"synthetic unique text for {user_id} row-{row}",
                    provenance={"source": "synthetic-fixture"},
                )
            )
    parent_manifest = _full_manifest(records, user_parts, "short")
    selection_records, selection_manifest_obj = _selection_artifact(
        records, parent_manifest
    )
    selection_manifest = selection_manifest_obj.data

    user_by_subject = selection_manifest["subject_user_by_id"]
    fold_users = {fold: set() for fold in range(evaluator.CV_FOLDS)}
    for fold in range(evaluator.CV_FOLDS):
        _train_ids, dev_ids = selection_fold_ids(
            selection_manifest,
            fold=fold,
            folds=evaluator.CV_FOLDS,
            repeat=0,
        )
        fold_users[fold].update(user_by_subject[subject_id] for subject_id in dev_ids)

    singleton_folds = [
        fold for fold, observed_users in fold_users.items()
        if len(observed_users) == 1
    ]
    assert singleton_folds
    emptied_fold = singleton_folds[0]
    removed_user = next(iter(fold_users[emptied_fold]))
    # Scenario filtering removes the user's rows from the frozen selection
    # artifact without touching user metadata — exactly what the builder's
    # empty-text drop does. The frozen parent manifest is not rebuilt.
    from research.task_sd import _rehash_manifest_data

    reduced_records = [
        record for record in selection_records if record.user_id != removed_user
    ]
    reduced_manifest = json.loads(json.dumps(selection_manifest))
    removed_subjects = {
        subject_id
        for subject_id, user_id in reduced_manifest["subject_user_by_id"].items()
        if user_id == removed_user
    }
    for partition in ("train", "dev"):
        reduced_manifest["split"][partition] = [
            subject_id
            for subject_id in reduced_manifest["split"][partition]
            if subject_id not in removed_subjects
        ]
    for mapping in ("subject_row_by_id", "subject_user_by_id"):
        reduced_manifest[mapping] = {
            subject_id: value
            for subject_id, value in reduced_manifest[mapping].items()
            if subject_id not in removed_subjects
        }
    reduced_manifest.pop("manifest_sha256")
    reduced_manifest["dataset_sha256"] = hashlib.sha256(
        records_to_json(reduced_records).encode("utf-8")
    ).hexdigest()
    reduced_manifest["manifest_sha256"] = _rehash_manifest_data(reduced_manifest)

    assert removed_user in reduced_manifest["user_partition_by_id"]
    assert removed_user in reduced_manifest["user_stratum_by_id"]
    assert removed_user not in reduced_manifest["subject_user_by_id"].values()
    with pytest.raises(
        ValueError,
        match="frozen selection fold assignment has no observed users",
    ):
        selection_fold_ids(
            reduced_manifest,
            fold=emptied_fold,
            folds=evaluator.CV_FOLDS,
            repeat=0,
        )


@pytest.mark.parametrize(
    ("selected_name", "target_name"),
    [
        ("selection_records.json", "records.json"),
        ("selection_manifest.json", "manifest.json"),
    ],
)
def test_selection_artifact_symlinks_are_rejected_before_following_targets(
    tmp_path, monkeypatch, selected_name, target_name
):
    (
        _records,
        _manifest,
        _selection_records,
        _selection_records_path,
        _selection_manifest_path,
        forbidden_paths,
        _removed_user_id,
    ) = _write_inputs(tmp_path, "short")
    input_dir = tmp_path / "runs" / "sd-v2" / "short"
    selected_path = input_dir / selected_name
    target_path = input_dir / target_name
    selected_path.unlink()
    selected_path.symlink_to(target_path)
    opened_paths = []
    original_os_open = os.open

    def recording_os_open(path, flags, *args, **kwargs):
        opened_paths.append(Path(os.fspath(path)).absolute())
        return original_os_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", recording_os_open)
    monkeypatch.chdir(tmp_path)
    assert evaluator.main(
        [
            "--scenario",
            "short",
            "--out",
            "runs/sd-v2/cv-symlink-rejection",
        ]
    ) == 2
    assert not any(path in forbidden_paths for path in opened_paths)


def test_output_path_refuses_overwrite_and_external_directories(tmp_path):
    _write_inputs(tmp_path, "short")
    existing = tmp_path / "runs" / "sd-v2" / "existing-output"
    existing.mkdir()
    marker = existing / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        evaluator.resolve_run_paths(
            "short", "runs/sd-v2/existing-output", repo_root=tmp_path
        )
    assert marker.read_text(encoding="utf-8") == "preserve"
    with pytest.raises(ValueError, match="under runs/sd-v2"):
        evaluator.resolve_run_paths(
            "short", "outside-output", repo_root=tmp_path
        )


def test_macro_f1_keeps_all_148_labels_and_zero_support_classes():
    result = evaluator._metric_report([0], [0])
    assert result["fixed_label_count"] == 148
    assert result["macro_denominator"] == 148
    assert len(result["per_class"]) == 148
    assert result["per_class"][0]["support"] == 1
    assert result["per_class"][147]["support"] == 0
    assert result["per_class"][147]["f1"] == 0.0
    assert result["macro_f1"] == round(1 / 148, 6)
