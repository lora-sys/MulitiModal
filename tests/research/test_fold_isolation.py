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
from data.real_split import build_real_manifest, fold_manifest
from research.frozen import build_fold

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


def test_build_fold_hands_the_candidate_only_fold_records(tmp_path):
    records = fake_records()
    path, base = _manifest_path(tmp_path, records)
    config = {
        "data": {"manifest_path": str(path), "dataset_path": "unused"},
        "train_provider": {"batch_size": 8},
        "trainer": {"seed": 0},
    }
    fold_data, fold = build_fold(
        config, str(path), "unused", fold=0, folds=4, repeat=0, records=records
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
    _, fold = build_fold(
        config, str(path), "unused", fold=1, folds=4, repeat=0, records=records
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
