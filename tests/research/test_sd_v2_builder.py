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


def _patched_subject_partitions(monkeypatch, frozen_partitions):
    """Stand-in for the internal clean-population split the builder makes.

    The builder calls subject_partitions over the clean (non-exposed)
    records and folds the returned internal test slice back into train, so
    this stand-in must only ever return train/dev over the records it is
    given — never the exposed users.
    """
    monkeypatch.setattr(
        builder,
        "subject_partitions",
        lambda records, seed: {
            "train": frozenset(frozen_partitions["train"] & {
                r.user_id for r in records
            }),
            "dev": frozenset(frozen_partitions["dev"] & {
                r.user_id for r in records
            }),
            "test": frozenset(),
        },
    )


def _exposed_users_file(tmp_path, users: list[str]):
    import hashlib

    payload = json.dumps(
        {
            "source": "synthetic sd-v1 unlock delivery artifacts",
            "n_rows": len(users),
            "n_users": len(users),
            "users": sorted(users),
        },
        indent=1,
    )
    path = tmp_path / "sd-v1-exposed-users.json"
    path.write_text(payload, encoding="utf-8")
    return path, hashlib.sha256(payload.encode()).hexdigest()


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
    _patched_subject_partitions(monkeypatch, frozen_partitions)
    exposed_file, _ = _exposed_users_file(
        tmp_path, ["exposed-1", "exposed-2"]
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
            "--exposed-users-file",
            str(exposed_file),
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
    _patched_subject_partitions(monkeypatch, frozen_partitions)
    exposed_file, exposed_sha = _exposed_users_file(
        tmp_path, ["exposed-1", "exposed-2"]
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
            "--exposed-users-file",
            str(exposed_file),
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

    assert selected_users.isdisjoint({"exposed-1", "exposed-2"})
    assert "exposed-1" not in selection_manifest["user_partition_by_id"]
    # the complete frozen assignment stays auditable in the manifest hash
    assert "user_partition_sha256" in manifest
    assert "user_partition_sha256" in selection_manifest
    exclusion = summary["sd_v1_exposed_user_exclusion"]
    assert exclusion["n_users"] == 2
    assert exclusion["n_users_in_population"] == 2
    assert exclusion["n_rows"] == 2
    assert exclusion["file_sha256"] == exposed_sha
    assert "not a seed replay" in exclusion["policy"]


def test_builder_rejects_a_missing_or_tampered_exposed_users_file(
    tmp_path, monkeypatch
):
    records = _synthetic_records()
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
    good_file, _ = _exposed_users_file(tmp_path, ["exposed-1"])
    tampered = tmp_path / "tampered.json"
    tampered.write_text(
        json.dumps(
            {
                "source": "tampered",
                "n_rows": 1,
                "n_users": 2,
                "users": ["exposed-1"],
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "selection-fail-closed"
    argv = [
        "--raw-dir",
        str(tmp_path / "synthetic-no-data-access"),
        "--out",
        str(out),
        "--scenario",
        "short",
    ]
    with pytest.raises(ValueError, match="n_users does not match"):
        builder.main(argv + ["--exposed-users-file", str(tampered)])
    with pytest.raises(FileNotFoundError):
        builder.main(argv + ["--exposed-users-file", str(tmp_path / "absent.json")])
    assert not out.exists()
    assert good_file.exists()
