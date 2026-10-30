"""Frozen real-data manifest tests (#50): split, fold-only scaler, labels, hashes."""

import json
import random

import pytest

from data.mendeley import SubjectRecord
from data.real_split import (
    LABEL_VERSION_CLASSIFICATION,
    LABEL_VERSION_REGRESSION,
    build_real_manifest,
    derive_classification_targets,
    load_real_manifest,
)

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
                    "demographics": [float(rng.randint(20, 70)), float(i % 2)],
                    "anthropometrics": [float(rng.randint(150, 190)),
                                        float(rng.randint(45, 95)), 22.5],
                    "meridian": [rng.random() for _ in range(3)],
                    "pulse_wave": [rng.random() for _ in range(2)],
                    "tongue": [rng.random()],
                },
                scores=scores,
                score_mask=[s is not None for s in scores],
                provenance={"source": "test-fixture"},
            )
        )
    return records


def test_split_is_subject_disjoint_and_deterministic():
    m1 = build_real_manifest(fake_records(), dataset_sha256=DATASET_SHA)
    m2 = build_real_manifest(fake_records(), dataset_sha256=DATASET_SHA)
    assert m1.sha256 == m2.sha256
    ids = {p: set(m1.data["split"][p]) for p in ("train", "dev", "holdout")}
    assert not ids["train"] & ids["dev"]
    assert not ids["train"] & ids["holdout"]
    assert not ids["dev"] & ids["holdout"]
    assert sum(len(v) for v in ids.values()) == 30


def test_scaler_is_fit_on_train_rows_only():
    records = fake_records()
    m = build_real_manifest(records, dataset_sha256=DATASET_SHA)
    by_row = {r.subject_index: r for r in records}
    train_rows = [m.data["subject_row_by_id"][sid] for sid in m.data["split"]["train"]]
    cols = [[v for g in ("demographics", "anthropometrics")
             for v in by_row[r].features[g]] for r in train_rows]
    for j in range(len(cols[0])):
        mean = sum(c[j] for c in cols) / len(cols)
        assert m.data["scaler"]["body_features"]["mean"][j] == pytest.approx(mean)


def test_dev_outlier_cannot_shift_the_scaler():
    records = fake_records()
    baseline = build_real_manifest(records, dataset_sha256=DATASET_SHA)
    dev_id = baseline.data["split"]["dev"][0]
    dev_row = baseline.data["subject_row_by_id"][dev_id]
    tampered = [
        r if r.subject_index != dev_row else
        SubjectRecord(
            subject_index=r.subject_index,
            features={**r.features,
                      "anthropometrics": [r.features["anthropometrics"][0], 1e9, 1e9]},
            scores=r.scores, score_mask=r.score_mask, provenance=r.provenance,
        )
        for r in records
    ]
    after = build_real_manifest(tampered, dataset_sha256=DATASET_SHA)
    assert after.data["scaler"] == baseline.data["scaler"]


def test_excluded_subject_is_absent_from_every_partition():
    m = build_real_manifest(fake_records(), dataset_sha256=DATASET_SHA,
                            exclude_subject_rows=(3,))
    every = set(m.data["split"]["train"]) | set(m.data["split"]["dev"]) \
        | set(m.data["split"]["holdout"])
    assert "mendeley-r003" not in every
    assert m.data["excluded_subject_rows"] == [3]


def test_label_contract_per_task_type():
    cls = build_real_manifest(fake_records(), dataset_sha256=DATASET_SHA)
    assert cls.data["label_tier"] == "RULE"
    assert cls.data["label_version"] == LABEL_VERSION_CLASSIFICATION
    reg = build_real_manifest(fake_records(), dataset_sha256=DATASET_SHA,
                              task_type="score_regression")
    assert reg.data["label_tier"] == "PUBLISHED"
    assert reg.data["label_version"] == LABEL_VERSION_REGRESSION


def test_invalid_task_type_and_modality_set_are_rejected():
    with pytest.raises(ValueError, match="task_type"):
        build_real_manifest(fake_records(), dataset_sha256=DATASET_SHA,
                            task_type="bogus")
    with pytest.raises(ValueError, match="modality_set"):
        build_real_manifest(fake_records(), dataset_sha256=DATASET_SHA,
                            modality_set="bogus")


def test_derive_classification_targets_threshold_and_abstain():
    labels, mask = derive_classification_targets(
        [None, 39.9, 40.0, 30.0, 10.0, 100.0]
    )
    assert labels == [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
    assert mask == [False, True, True, True, True, True]


def test_load_real_manifest_rejects_tampering(tmp_path):
    m = build_real_manifest(fake_records(), dataset_sha256=DATASET_SHA)
    path = tmp_path / "manifest.json"
    m.save(path)
    assert load_real_manifest(path)["manifest_sha256"] == m.sha256
    data = json.loads(path.read_text())
    data["label_version"] = "tampered"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_real_manifest(path)
