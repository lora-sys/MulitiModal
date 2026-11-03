"""Fold-path isolation tests: what the candidate's FoldData is allowed to hold.

The selection folds are the candidate's whole world. These tests pin the two
leaks an adversarial review found in the first implementation:

1. the fold manifest deep-copied the parent's full subject_row_by_id, so the
   holdout identity was recoverable by set difference;
2. RealBatchProvider held records for EVERY subject, including holdout rows
   with their scores, reachable as `fold.train.records_by_row`.

Both are closed structurally: the fold manifest keeps only train∪dev rows, and
the providers are built from records filtered to those rows.
"""

from __future__ import annotations

import json
import random

import pytest

from data.mendeley import FEATURE_GROUP_SIZES, SubjectRecord
from data.real_split import build_real_manifest, fold_manifest, pool_manifest
from data.tcm_sd import subject_partitions
from research.frozen import build_fold, selection_fold_ids
from research.task_sd import FEATURE_GROUP, SyndromeRecord, build_syndrome_manifest

pytestmark = pytest.mark.synthetic

DATASET_SHA = "a" * 64


def fake_records(n=30, seed=7):
    rng = random.Random(seed)
    records = []
    for i in range(1, n + 1):
        scores = [None] + [float(rng.randint(10, 90)) for _ in range(8)]
        records.append(
            SubjectRecord(
                subject_index=i,
                features={
                    name: [rng.random() for _ in range(size)]
                    for name, size in FEATURE_GROUP_SIZES.items()
                },
                scores=scores,
                score_mask=[s is not None for s in scores],
                provenance={"source": "test-fixture"},
            )
        )
    return records


def _manifest_path(tmp_path, records):
    base = build_real_manifest(records, dataset_sha256=DATASET_SHA).data
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(base, ensure_ascii=False))
    return path, base


def test_constitution_selection_pool_preserves_frozen_split_seed():
    records = fake_records()
    base = build_real_manifest(records, dataset_sha256=DATASET_SHA).data
    selection = pool_manifest(base, records).data
    assert selection["split"]["seed"] == base["split"]["seed"]


def test_selection_fold_keeps_repeat_cases_of_each_user_together():
    records = [
        SyndromeRecord(
            subject_index=i,
            user_id=f"su-{i:04d}",
            features={FEATURE_GROUP: [float(i % 7), float((i + 1) % 7)]},
            label=i % 5,
            norm_syndrome="y0",
            provenance={"source": "test-fixture"},
        )
        for i in range(1, 61)
    ]
    initial_parts = subject_partitions(records)
    shared_user = sorted(initial_parts["train"] | initial_parts["dev"])[0]
    source = next(record for record in records if record.user_id == shared_user)
    records.append(
        SyndromeRecord(
            subject_index=61,
            user_id=shared_user,
            features=dict(source.features),
            label=source.label,
            norm_syndrome=source.norm_syndrome,
            provenance={"source": "test-fixture", "repeat": True},
        )
    )
    frozen_parts = subject_partitions(records)
    manifest = build_syndrome_manifest(
        records,
        dataset_sha256="b" * 64,
        n_classes=5,
        label_taxonomy_version="tcm-sd-fold-test-v0",
        label_tier="SMOKE_ONLY",
        frozen_user_partitions=frozen_parts,
    ).data

    train_ids, dev_ids = selection_fold_ids(manifest, fold=1, folds=5, repeat=0)
    shared_ids = {
        subject_id
        for subject_id, user_id in manifest["subject_user_by_id"].items()
        if user_id == shared_user
    }

    assert len(shared_ids) == 2
    assert shared_ids <= set(train_ids) or shared_ids <= set(dev_ids)
    assert not (shared_ids & set(train_ids) and shared_ids & set(dev_ids))


def test_build_fold_hands_the_candidate_only_fold_records(tmp_path):
    records = fake_records()
    path, base = _manifest_path(tmp_path, records)
    config = {
        "data": {"manifest_path": str(path), "dataset_path": "unused"},
        "train_provider": {"batch_size": 8},
        "trainer": {"seed": 0},
    }
    selection = pool_manifest(base, records).data
    selection_rows = set(selection["subject_row_by_id"].values())
    selection_records = [
        record for record in records if record.subject_index in selection_rows
    ]
    fold_data, fold = build_fold(
        config, selection, fold=0, folds=4, repeat=0, records=selection_records
    )
    fold_rows = set(fold["subject_row_by_id"].values())
    held = set(fold_data.train.records_by_row) | set(fold_data.dev.records_by_row)
    assert held == fold_rows
    holdout_rows = {
        base["subject_row_by_id"][sid] for sid in base["split"]["holdout"]
    }
    assert holdout_rows and not held & holdout_rows


def test_fold_manifest_does_not_leak_holdout_identity(tmp_path):
    records = fake_records()
    path, base = _manifest_path(tmp_path, records)
    config = {
        "data": {"manifest_path": str(path), "dataset_path": "unused"},
        "train_provider": {"batch_size": 8},
        "trainer": {"seed": 0},
    }
    selection = pool_manifest(base, records).data
    selection_rows = set(selection["subject_row_by_id"].values())
    selection_records = [
        record for record in records if record.subject_index in selection_rows
    ]
    _, fold = build_fold(
        config, selection, fold=1, folds=4, repeat=0, records=selection_records
    )
    assert fold["split"]["holdout"] == []
    expected = set(fold["split"]["train"]) | set(fold["split"]["dev"])
    assert set(fold["subject_row_by_id"]) == expected


def test_fold_manifest_derivation_is_directly_pruned(tmp_path):
    records = fake_records()
    _, base = _manifest_path(tmp_path, records)
    train_ids, dev_ids = base["split"]["train"], base["split"]["dev"]
    fold = fold_manifest(base, records, train_ids[:-1], dev_ids[:1]).data
    expected = set(train_ids[:-1]) | set(dev_ids[:1])
    assert set(fold["subject_row_by_id"]) == expected
    assert not set(fold["subject_row_by_id"]) & set(base["split"]["holdout"])


def test_pool_manifest_carries_no_holdout_at_all(tmp_path):
    """F-05: the pool manifest is written to disk (champion/pool-manifest.json)
    where anything that can read the champion directory can diff its subject
    ids against a fold manifest and recover holdout identity. Both the holdout
    list and the row mapping must be empty of holdout rows; the trusted holdout
    scoring path rebuilds the view from the frozen parent manifest."""
    from data.real_split import holdout_view, pool_manifest

    records = fake_records()
    _, base = _manifest_path(tmp_path, records)
    pool = pool_manifest(base, records, notes="test").data
    assert pool["split"]["holdout"] == []
    holdout_ids = set(base["split"]["holdout"])
    assert holdout_ids and not holdout_ids & set(pool["subject_row_by_id"])
    # the pool scaler is the pool's own; the view re-attaches only the holdout
    # partition and rows, never a different scaler
    view = holdout_view(pool, base)
    assert view["split"]["holdout"] == base["split"]["holdout"]
    assert view["subject_row_by_id"] == base["subject_row_by_id"]
    assert view["scaler"] == pool["scaler"]
