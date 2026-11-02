from __future__ import annotations

import json

import pytest

from data.tcm_sd import TcmSdRecord
from research.task_sd import records_from_json
from scripts import build_sd_campaign_inputs as builder

pytestmark = pytest.mark.synthetic


def _record(user_id: str, lcd_id: str, complaint: str, history: str) -> TcmSdRecord:
    return TcmSdRecord(
        user_id=user_id,
        lcd_id=lcd_id,
        lcd_name="synthetic case",
        syndrome="synthetic syndrome",
        norm_syndrome="气虚证",
        chief_complaint=complaint,
        description=history,
        detection="synthetic tongue and pulse",
        source_file="train.json",
    )


def _synthetic_records() -> list[TcmSdRecord]:
    return [
        _record("train-copy", "lcd-1", "shared complaint", "shared history"),
        _record("train-other", "lcd-2", "train only complaint", "train history"),
        _record("dev-copy", "lcd-3", "shared complaint", "shared history"),
        _record("dev-other", "lcd-4", "dev only complaint", "dev history"),
        # filler users clear the development bundle's 10-case minimum
        *(
            _record(f"filler-{index}", f"lcd-fill-{index}", f"filler {index}", "h")
            for index in range(6)
        ),
        # sd-v1's frozen-seed replay unblinded these users' test partition
        _record("exposed-1", "lcd-e1", "exposed complaint one", "history"),
        _record("exposed-2", "lcd-e2", "exposed complaint two", "history"),
    ]


def _frozen_partitions() -> dict[str, frozenset[str]]:
    return {
        "train": frozenset(
            {"train-copy", "train-other"} | {f"filler-{i}" for i in range(4)}
        ),
        "dev": frozenset({"dev-copy", "dev-other", "filler-4", "filler-5"}),
        "test": frozenset({"exposed-1", "exposed-2"}),
    }


@pytest.mark.parametrize("scenario", ["short", "clinical"])
def test_builder_preserves_dev_duplicates_until_the_selection_fold_is_built(
    tmp_path, monkeypatch, scenario
):
    records = _synthetic_records()
    frozen_partitions = _frozen_partitions()
    monkeypatch.setattr(
        builder,
        "load_tcm_sd_development_records",
        lambda _raw_dir: records,
    )
    monkeypatch.setattr(
        builder,
        "load_syndrome_vocab",
        lambda _raw_dir: ["气虚证", "血瘀证"],
    )
    monkeypatch.setattr(
        builder,
        "subject_partitions",
        lambda _records, seed: frozen_partitions,
    )
    out = tmp_path / f"selection-{scenario}"

    assert builder.main(
        [
            "--raw-dir",
            str(tmp_path / "synthetic-no-data-access"),
            "--out",
            str(out),
            "--scenario",
            scenario,
        ]
    ) == 0

    selected = records_from_json(
        (out / "selection_records.json").read_text(encoding="utf-8")
    )
    selection_manifest = json.loads(
        (out / "selection_manifest.json").read_text(encoding="utf-8")
    )
    summary = json.loads((out / "build-summary.json").read_text(encoding="utf-8"))
    selected_users = {record.user_id for record in selected}
    duplicate_rows = [
        record
        for record in selected
        if record.user_id in {"train-copy", "dev-copy"}
    ]

    assert {"train-copy", "dev-copy"} <= selected_users
    assert len(duplicate_rows) == 2
    assert duplicate_rows[0].input_text == duplicate_rows[1].input_text
    assert selection_manifest["split"]["holdout"] == []
    assert selection_manifest["user_partition_by_id"]["dev-copy"] == "dev"
    assert summary["evaluation_dedup_policy"] == (
        "preserve selection rows; exact scenario text dedup occurs on each "
        "frozen evaluation side"
    )
    assert summary["empty_text_dropped"] == 0


def test_builder_excludes_sd_v1_exposed_test_users_from_the_selection_pool(
    tmp_path, monkeypatch
):
    records = _synthetic_records()
    frozen_partitions = _frozen_partitions()
    monkeypatch.setattr(
        builder,
        "load_tcm_sd_development_records",
        lambda _raw_dir: records,
    )
    monkeypatch.setattr(
        builder,
        "load_syndrome_vocab",
        lambda _raw_dir: ["气虚证", "血瘀证"],
    )
    monkeypatch.setattr(
        builder,
        "subject_partitions",
        lambda _records, seed: frozen_partitions,
    )
    out = tmp_path / "selection-exclusion"

    assert builder.main(
        [
            "--raw-dir",
            str(tmp_path / "synthetic-no-data-access"),
            "--out",
            str(out),
            "--scenario",
            "short",
        ]
    ) == 0

    selected = records_from_json(
        (out / "selection_records.json").read_text(encoding="utf-8")
    )
    selection_manifest = json.loads(
        (out / "selection_manifest.json").read_text(encoding="utf-8")
    )
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((out / "build-summary.json").read_text(encoding="utf-8"))
    selected_users = {record.user_id for record in selected}

    assert selected_users.isdisjoint(frozen_partitions["test"])
    assert "exposed-1" not in selection_manifest["user_partition_by_id"]
    # the complete frozen assignment stays auditable in the manifest hash
    assert "user_partition_sha256" in manifest
    assert "user_partition_sha256" in selection_manifest
    exclusion = summary["sd_v1_exposed_user_exclusion"]
    assert exclusion["n_users"] == 2
    assert exclusion["n_rows"] == 2
    assert "frozen split seed" in exclusion["policy"]
