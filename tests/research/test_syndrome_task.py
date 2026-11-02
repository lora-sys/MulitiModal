"""Syndrome-task AutoResearch adaptation tests (issue #53, task="syndrome").

Everything here is SYNTHETIC (SMOKE_ONLY, AGENTS.md rule 7): the assertions
pin the ENGINEERING seams — task dispatch, the syndrome data contract, the
trusted evaluator, guard coverage of candidate_sd.py, the export round-trip,
the task-level unlock ledger, and the full runner chain (train_worker
subprocess + evaluator subprocess). None of it claims anything about real
TCM-SD performance; the real data path lands via #53 scope 1.
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
from pathlib import Path

import pytest

from data.real_split import RealManifest, load_real_manifest
from data.tcm_sd import subject_partitions, user_majority_strata
from research.frozen import selection_fold_ids
from research.task_sd import (
    FEATURE_GROUP,
    PRIMARY_METRIC_SD,
    SyndromeBatchProvider,
    SyndromeRecord,
    SyndromeTrustedEvaluator,
    _macro_f1,
    build_syndrome_manifest,
    build_syndrome_selection_artifacts,
    records_from_json,
    records_to_json,
    syndrome_input_spec,
)
from research.tasks import adapter_for

pytestmark = pytest.mark.synthetic

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DATASET_SHA = "d" * 64
TAXONOMY = "tcm-sd-taxonomy-test-v0"
N_CLASSES = 5
_MINIMAL_CANDIDATE = (
    "def build_model(fold):\n"
    "    return None\n"
    "\n"
    "def train(model, fold, budget_seconds, seed):\n"
    "    return None\n"
)


def fake_sd_records(n=30, seed=17, n_classes=N_CLASSES, labeled=True):
    rng = random.Random(seed)
    records = []
    for i in range(1, n + 1):
        label = None if (not labeled or i % 7 == 0) else rng.randrange(n_classes)
        records.append(
            SyndromeRecord(
                subject_index=i,
                user_id=f"su-{i:04d}",
                norm_syndrome=f"y{i % 3}",
                features={FEATURE_GROUP: [rng.random() for _ in range(12)]},
                label=label,
                provenance={"source": "test-fixture"},
            )
        )
    return records


def _frozen(tmp_path, records, **kwargs):
    dataset_sha256 = kwargs.pop("dataset_sha256", DATASET_SHA)
    manifest = build_syndrome_manifest(
        records, dataset_sha256=dataset_sha256, n_classes=N_CLASSES,
        label_taxonomy_version=TAXONOMY, label_tier="SMOKE_ONLY", **kwargs,
    )
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps(manifest.data, ensure_ascii=False))
    return manifest.data, path


def _records_path(tmp_path, records):
    path = tmp_path / "records.json"
    path.write_text(records_to_json(records))
    return path


class _ConstantModel:
    """Model-like stub: softmax one-hot at one class, contract-carrying."""

    def __init__(self, manifest, n_classes, cls, version=None):
        self.input_spec = syndrome_input_spec(manifest)
        if version is not None:
            from dataclasses import replace

            self.input_spec = replace(self.input_spec, version=version)
        self.n_classes = n_classes
        self.cls = cls

    def __call__(self, batch):
        return self.forward(batch)

    def forward(self, batch):
        import torch

        from core.batch import ModelOutput

        b = next(iter(batch.features.values())).shape[0]
        probs = torch.zeros(b, self.n_classes)
        probs[:, self.cls] = 1.0
        return ModelOutput(predictions={"syndrome_probs": probs})


# -- task dispatch -------------------------------------------------------------


def test_adapter_dispatch_and_defaults():
    constitution = adapter_for("constitution")
    syndrome = adapter_for("syndrome")
    assert constitution.primary_metric == "category_probs/macro_ap"
    assert syndrome.primary_metric == PRIMARY_METRIC_SD == "syndrome/macro_f1"
    assert constitution.default_editable == "src/research/candidate.py"
    assert syndrome.default_editable == "src/research/candidate_sd.py"
    with pytest.raises(ValueError, match="unknown research task"):
        adapter_for("physio")


# -- data contract ---------------------------------------------------------------


def test_manifest_round_trips_through_the_hash_check(tmp_path):
    records = fake_sd_records()
    base, path = _frozen(tmp_path, records)
    loaded = load_real_manifest(path)
    assert loaded["task"] == "syndrome"
    assert loaded["task_type"] == "syndrome_classification"
    assert loaded["label_taxonomy_version"] == TAXONOMY
    assert loaded["n_classes"] == N_CLASSES
    assert loaded["label_tier"] == "SMOKE_ONLY"
    assert loaded["modalities"]["text"]["feature_groups"] == [FEATURE_GROUP]
    assert loaded["split"]["holdout"]
    # subject-level disjointness is the #27 contract, verified on the frozen file
    parts = [set(loaded["split"][p]) for p in ("train", "dev", "holdout")]
    assert parts[0] | parts[1] | parts[2] == set(loaded["subject_row_by_id"])
    for a in parts:
        for b in parts:
            if a < b:
                assert not a & b


def test_published_manifest_requires_frozen_user_partitions():
    with pytest.raises(ValueError, match="frozen_user_partitions is required"):
        build_syndrome_manifest(
            fake_sd_records(),
            dataset_sha256=DATASET_SHA,
            n_classes=N_CLASSES,
            label_taxonomy_version=TAXONOMY,
            label_tier="PUBLISHED",
        )


def test_manifest_builder_refuses_bad_labels_and_tiny_datasets(tmp_path):
    good = fake_sd_records(12)
    with pytest.raises(ValueError, match="outside the frozen taxonomy"):
        bad_label = [
            SyndromeRecord(
                subject_index=i, user_id=f"su-{i:04d}", norm_syndrome="y0",
                features={FEATURE_GROUP: [0.0]}, label=0, provenance={},
            )
            for i in range(1, 12)
        ]
        bad_label[3].label = 99
        build_syndrome_manifest(
            bad_label,
            dataset_sha256=DATASET_SHA, n_classes=N_CLASSES,
            label_taxonomy_version=TAXONOMY, label_tier="PUBLISHED",
        )
    with pytest.raises(ValueError, match="duplicate subject_index"):
        build_syndrome_manifest(
            [
                SyndromeRecord(
                    subject_index=1, user_id="su-0001", norm_syndrome="y0",
                    features={FEATURE_GROUP: [0.0]}, label=0, provenance={},
                )
            ]
            * 2,
            dataset_sha256=DATASET_SHA, n_classes=N_CLASSES,
            label_taxonomy_version=TAXONOMY, label_tier="PUBLISHED",
        )
    with pytest.raises(ValueError, match="too few cases"):
        build_syndrome_manifest(
            good[:5], dataset_sha256=DATASET_SHA, n_classes=N_CLASSES,
            label_taxonomy_version=TAXONOMY, label_tier="PUBLISHED",
        )


def test_records_json_round_trip():
    records = fake_sd_records(6)
    parsed = records_from_json(records_to_json(records))
    assert [(r.subject_index, r.label) for r in parsed] == [
        (r.subject_index, r.label) for r in records
    ]
    assert parsed[0].features == records[0].features
    assert parsed[0].provenance == records[0].provenance
    assert [r.user_id for r in parsed] == [r.user_id for r in records]
    assert [r.norm_syndrome for r in parsed] == [r.norm_syndrome for r in records]


# -- subject-level manifest split (R1 F3: data.tcm_sd is the single source) ----


def test_manifest_keeps_every_case_of_a_user_in_one_partition(tmp_path):
    records = fake_sd_records()
    # one user owns three cases (repeat visits): they must not be split
    shared_rows = [3, 11, 27]
    for row in shared_rows:
        records[row - 1].user_id = "su-shared"
    base, _path = _frozen(tmp_path, records)
    parts = {p: set(base["split"][p]) for p in ("train", "dev", "holdout")}
    for a in parts:
        for b in parts:
            if a < b:
                assert not parts[a] & parts[b]
    owners = {
        p for p, ids in parts.items()
        if {f"sd-r{row:05d}" for row in shared_rows} & ids
    }
    assert len(owners) == 1


def test_manifest_split_matches_subject_partitions_single_source(tmp_path):
    from data.tcm_sd import subject_partitions

    records = fake_sd_records()
    base, _path = _frozen(tmp_path, records)
    expected = subject_partitions(records, seed=20261101)
    for part_name, users in (
        ("train", expected["train"]),
        ("dev", expected["dev"]),
        ("holdout", expected["test"]),
    ):
        want_rows = sorted(
            f"sd-r{r.subject_index:05d}"
            for r in records if r.user_id in users
        )
        assert base["split"][part_name] == want_rows
    # the internal 0.70/0.15/0.15 draw is gone; ratios come from the frozen
    # data-layer contract
    assert base["split"]["fractions"] == {
        "train": 0.82, "dev": 0.10, "holdout": 0.08,
    }


def test_deleted_dev_user_rows_keep_all_original_user_partitions():
    from data.tcm_sd import subject_partitions

    records = [
        SyndromeRecord(
            subject_index=i,
            user_id=f"su-{i:04d}",
            features={FEATURE_GROUP: [float(i % 11), float(i % 5)]},
            label=i % N_CLASSES,
            norm_syndrome="y0",
            provenance={"source": "test-fixture"},
        )
        for i in range(1, 61)
    ]
    initial_parts = subject_partitions(records)
    deleted_user = sorted(initial_parts["dev"])[0]
    source = next(record for record in records if record.user_id == deleted_user)
    records.append(
        SyndromeRecord(
            subject_index=61,
            user_id=deleted_user,
            features=dict(source.features),
            label=source.label,
            norm_syndrome=source.norm_syndrome,
            provenance={"source": "test-fixture", "repeat": True},
        )
    )
    frozen_parts = subject_partitions(records)
    frozen_by_user = {
        user_id: partition
        for partition, users in frozen_parts.items()
        for user_id in users
    }
    assert frozen_by_user[deleted_user] == "dev"
    frozen_strata = user_majority_strata(records)

    def build(records_to_freeze, user_parts):
        payload = records_to_json(records_to_freeze)
        return build_syndrome_manifest(
            records_to_freeze,
            dataset_sha256=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            n_classes=N_CLASSES,
            label_taxonomy_version=TAXONOMY,
            label_tier="SMOKE_ONLY",
            frozen_user_partitions=user_parts,
            frozen_user_strata=frozen_strata,
        ).data

    full = build(records, frozen_parts)
    filtered = [record for record in records if record.user_id != deleted_user]
    recomputed = subject_partitions(filtered)
    recomputed_by_user = {
        user_id: partition
        for partition, users in recomputed.items()
        for user_id in users
    }
    remaining_users = {record.user_id for record in filtered}
    assert any(
        recomputed_by_user[user_id] != frozen_by_user[user_id]
        for user_id in remaining_users
    )

    without_deleted_rows = build(filtered, frozen_parts)

    def manifest_user_parts(manifest):
        assigned = {}
        split_name = {"train": "train", "dev": "dev", "holdout": "test"}
        for manifest_part, data_part in split_name.items():
            for subject_id in manifest["split"][manifest_part]:
                user_id = manifest["subject_user_by_id"][subject_id]
                assert user_id not in assigned or assigned[user_id] == data_part
                assigned[user_id] = data_part
        return assigned

    before = manifest_user_parts(full)
    after = manifest_user_parts(without_deleted_rows)
    assert all(before[user_id] == after[user_id] for user_id in remaining_users)
    assert without_deleted_rows["user_partition_by_id"] == full["user_partition_by_id"]
    assert deleted_user in without_deleted_rows["user_partition_by_id"]
    assert deleted_user not in without_deleted_rows["subject_user_by_id"].values()


def test_provider_masks_unlabeled_and_declares_the_syndrome_spec(tmp_path):
    records = fake_sd_records()
    base, _path = _frozen(tmp_path, records)
    provider = SyndromeBatchProvider(base, records, "train", batch_size=8)
    assert "tcm-sd-text-v1:" in provider.input_spec().version
    assert TAXONOMY in provider.input_spec().version
    mask_seen = 0
    for item in provider:
        batch = item.batch
        targets = batch.targets["syndrome_labels"]
        mask = batch.target_mask["syndrome_labels"]
        assert targets.shape[1] == 1 and mask.shape[1] == 1
        mask_seen += int(mask.sum())
        # every labeled row is inside the frozen taxonomy
        assert bool(((targets[mask] >= 0) & (targets[mask] < N_CLASSES)).all())
        # unlabeled rows are 0.0 + mask False (ABSTAIN, never guessed)
        assert bool((targets[~mask] == 0).all())
    train_ids = set(base["split"]["train"])
    n_unlabeled = sum(
        1 for r in records
        if r.label is None and f"sd-r{r.subject_index:05d}" in train_ids
    )
    assert mask_seen == len(base["split"]["train"]) - n_unlabeled


# -- hashed_text modality (real-scale path, ~44k cases) -------------------------


def fake_sd_text_records(n=30, seed=23, n_classes=N_CLASSES, labeled=True):
    """Synthetic text cases (SMOKE_ONLY): label-discriminative keyword
    tokens so a learner has signal; features are empty per the hashed_text
    contract (raw text travels in input_text, never in features)."""
    rng = random.Random(seed)
    records = []
    for i in range(1, n + 1):
        label = None if (not labeled or i % 7 == 0) else rng.randrange(n_classes)
        tokens = " ".join(f"synd{label}kw" for _ in range(3))
        records.append(
            SyndromeRecord(
                subject_index=i,
                user_id=f"su-{i:04d}",
                norm_syndrome=f"y{i % 3}",
                features={},
                label=label,
                input_text=f"{tokens} chief complaint case {i}",
                provenance={"source": "test-fixture"},
            )
        )
    return records


def _frozen_text(tmp_path, records, **kwargs):
    manifest = build_syndrome_manifest(
        records, dataset_sha256=DATASET_SHA, n_classes=N_CLASSES,
        label_taxonomy_version=TAXONOMY, label_tier="SMOKE_ONLY",
        modality_kind="hashed_text", **kwargs,
    )
    path = tmp_path / "frozen-text.json"
    path.write_text(json.dumps(manifest.data, ensure_ascii=False))
    return manifest.data, path
def _selection_artifacts(path, records):
    frozen_data = load_real_manifest(path)
    frozen = RealManifest(
        data=frozen_data, sha256=frozen_data["manifest_sha256"],
    )
    return build_syndrome_selection_artifacts(frozen, records)


def test_selection_fold_deduplicates_dev_text_before_manifest_and_matches_evaluator(tmp_path):
    from research.frozen import selection_fold_ids
    from research.task_sd import build_fold

    records = fake_sd_text_records(n=60)
    initial = build_syndrome_manifest(
        records,
        dataset_sha256=hashlib.sha256(records_to_json(records).encode()).hexdigest(),
        n_classes=N_CLASSES,
        label_taxonomy_version=TAXONOMY,
        label_tier="SMOKE_ONLY",
        modality_kind="hashed_text",
    ).data
    train_ids, dev_ids = selection_fold_ids(initial, fold=0, folds=4, repeat=0)
    records_by_row = {record.subject_index: record for record in records}
    row_by_id = initial["subject_row_by_id"]
    train_record = records_by_row[row_by_id[train_ids[0]]]
    duplicate_dev_id = next(
        subject_id
        for subject_id in dev_ids
        if records_by_row[row_by_id[subject_id]].user_id != train_record.user_id
    )
    duplicate_dev_record = records_by_row[row_by_id[duplicate_dev_id]]
    assert duplicate_dev_record.user_id != train_record.user_id
    duplicate_dev_record.input_text = train_record.input_text

    frozen = build_syndrome_manifest(
        records,
        dataset_sha256=hashlib.sha256(records_to_json(records).encode()).hexdigest(),
        n_classes=N_CLASSES,
        label_taxonomy_version=TAXONOMY,
        label_tier="SMOKE_ONLY",
        modality_kind="hashed_text",
    )
    config = {
        "train_provider": {"batch_size": 8},
        "trainer": {"seed": 0},
    }
    selection_records, selection_manifest = build_syndrome_selection_artifacts(
        frozen, records
    )
    fold_data, fold_manifest = build_fold(
        config,
        selection_manifest.data,
        fold=0,
        folds=4,
        repeat=0,
        records=selection_records,
    )

    assert fold_manifest["split"]["train"] == train_ids
    assert duplicate_dev_id not in fold_manifest["split"]["dev"]
    assert fold_manifest["selection_fold"]["dev_text_dedup"]["n_dropped"] == 1
    assert not fold_manifest["split"]["holdout"]
    assert "user_partition_by_id" not in fold_manifest
    assert set(fold_manifest["subject_user_by_id"]) == set(
        fold_manifest["subject_row_by_id"]
    )

    fold_rows = fold_manifest["subject_row_by_id"]
    train_texts = {
        records_by_row[fold_rows[subject_id]].input_text
        for subject_id in fold_manifest["split"]["train"]
    }
    dev_texts = {
        records_by_row[fold_rows[subject_id]].input_text
        for subject_id in fold_manifest["split"]["dev"]
    }
    assert not train_texts & dev_texts
    train_users = {
        fold_manifest["subject_user_by_id"][subject_id]
        for subject_id in fold_manifest["split"]["train"]
    }
    dev_users = {
        fold_manifest["subject_user_by_id"][subject_id]
        for subject_id in fold_manifest["split"]["dev"]
    }
    assert not train_users & dev_users
    assert set(fold_data.dev.subject_ids) == set(fold_manifest["split"]["dev"])

    evaluator = SyndromeTrustedEvaluator(fold_manifest)
    model = _ConstantModel(fold_manifest, N_CLASSES, 0)
    report = evaluator.evaluate(model, fold_data.dev, "dev")
    assert report.n_subjects == len(fold_manifest["split"]["dev"])

    fold_records = [records_by_row[row] for row in fold_rows.values()]
    incomplete_provider = SyndromeBatchProvider(
        fold_manifest,
        fold_records,
        "dev",
        subject_ids=fold_data.dev.subject_ids[:-1],
    )
    with pytest.raises(ValueError, match="provider sample set mismatch"):
        evaluator.evaluate(model, incomplete_provider, "dev")


def test_hashed_text_manifest_freezes_the_vectorization(tmp_path):
    records = fake_sd_text_records()
    base, _path = _frozen_text(tmp_path, records)
    params = base["modalities"]["text"]
    assert params["kind"] == "hashed_text"
    assert params["analyzer"] == "char_wb"
    assert params["ngram_range"] == [1, 3]
    assert params["n_features"] == 65536
    assert params["alternate_sign"] is False
    assert params["norm"] == "l2"
    assert params["dim"] == 65536
    # stateless features: nothing fit anywhere, so the scaler is identity
    assert base["scaler"] == {"text": {"kind": "identity"}}
    assert base["scaler_fit_rows"] == 0
    version = syndrome_input_spec(base).version
    assert version.startswith("tcm-sd-text-v2:")
    assert TAXONOMY in version and "hashed_text:65536" in version
    # fail-closed: a record with neither features nor input_text is refused
    broken = fake_sd_text_records(12)
    broken[3].input_text = ""
    with pytest.raises(ValueError, match="neither features nor input_text"):
        build_syndrome_manifest(
            broken, dataset_sha256=DATASET_SHA, n_classes=N_CLASSES,
            label_taxonomy_version=TAXONOMY, label_tier="SMOKE_ONLY",
            modality_kind="hashed_text",
        )
    with pytest.raises(ValueError, match="unknown modality_kind"):
        build_syndrome_manifest(
            fake_sd_text_records(12), dataset_sha256=DATASET_SHA,
            n_classes=N_CLASSES, label_taxonomy_version=TAXONOMY,
            label_tier="SMOKE_ONLY", modality_kind="tfidf_idf",
        )


def test_hashed_text_provider_is_deterministic_exact_and_masks(tmp_path):
    import torch
    from sklearn.feature_extraction.text import HashingVectorizer

    records = fake_sd_text_records()
    base, _path = _frozen_text(tmp_path, records)
    by_row = {r.subject_index: r for r in records}
    fresh = HashingVectorizer(
        analyzer="char_wb", ngram_range=(1, 3), n_features=65536,
        alternate_sign=False, norm="l2",
    )
    p1 = SyndromeBatchProvider(base, records, "dev", batch_size=8)
    p2 = SyndromeBatchProvider(base, records, "dev", batch_size=8)
    seen_labeled = 0
    for i1, i2 in zip(p1, p2, strict=True):
        assert i1.subject_ids == i2.subject_ids
        x1 = i1.batch.features["text"]
        assert torch.equal(x1, i2.batch.features["text"])
        want = torch.from_numpy(fresh.transform(
            [by_row[base["subject_row_by_id"][s]].input_text
             for s in i1.subject_ids]
        ).toarray()).to(torch.float32)
        assert torch.equal(x1, want)  # exact sklearn HashingVectorizer semantics
        norms = x1.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms))
        mask = i1.batch.target_mask["syndrome_labels"].squeeze(-1)
        targets = i1.batch.targets["syndrome_labels"].squeeze(-1)
        assert bool((targets[~mask] == 0).all())  # ABSTAIN, never guessed
        seen_labeled += int(mask.sum())
    dev_ids = set(base["split"]["dev"])
    n_unlabeled = sum(
        1 for r in records
        if r.label is None and f"sd-r{r.subject_index:05d}" in dev_ids
    )
    assert seen_labeled == len(base["split"]["dev"]) - n_unlabeled
    # train shuffle: same manifest + batch_size -> same epoch order and
    # matching (order-dependent) features
    t1 = SyndromeBatchProvider(base, records, "train", batch_size=8)
    t2 = SyndromeBatchProvider(base, records, "train", batch_size=8)
    for i1, i2 in zip(t1, t2, strict=True):
        assert i1.subject_ids == i2.subject_ids
        assert torch.equal(
            i1.batch.features["text"], i2.batch.features["text"]
        )
    # a fitted scaler on a hashed manifest is refused (stateless contract)
    tampered = json.loads(json.dumps(base))
    tampered["scaler"] = {
        "text": {"mean": [0.0], "std": [1.0], "degenerate": [False]},
    }
    with pytest.raises(ValueError, match="identity"):
        SyndromeBatchProvider(tampered, records, "dev", batch_size=8)
    # and text stripped after manifest build is a fail-closed breach
    stripped = list(records)
    for r in stripped:
        r.input_text = ""
    with pytest.raises(ValueError, match="neither features nor input_text"):
        SyndromeBatchProvider(base, stripped, "dev", batch_size=8)


def test_hashed_text_evaluator_refuses_a_foreign_spec(tmp_path):
    records = fake_sd_text_records()
    base, _path = _frozen_text(tmp_path, records)
    evaluator = SyndromeTrustedEvaluator(base)
    provider = SyndromeBatchProvider(base, records, "dev", batch_size=8)
    # the float_groups spec of the same taxonomy is a DIFFERENT contract
    float_base, _float_path = _frozen(tmp_path, fake_sd_records())
    foreign = _ConstantModel(float_base, N_CLASSES, 0)
    with pytest.raises(ValueError, match="input spec mismatch"):
        evaluator.evaluate(foreign, provider, partition="dev")
    report = evaluator.evaluate(
        _ConstantModel(base, N_CLASSES, 0), provider, partition="dev"
    )
    assert report.n_subjects == len(base["split"]["dev"])
    assert report.input_spec_version.startswith("tcm-sd-text-v2:")


def test_hashed_text_records_json_round_trip():
    records = fake_sd_text_records(6)
    parsed = records_from_json(records_to_json(records))
    assert [r.input_text for r in parsed] == [r.input_text for r in records]
    assert all(r.features == {} for r in parsed)
    assert [(r.subject_index, r.label) for r in parsed] == [
        (r.subject_index, r.label) for r in records
    ]


def test_hashed_text_fold_pool_scaler_and_export_round_trip(tmp_path):
    """Fold/pool derivation keeps the identity scaler (no refit over
    feature-less records), and the skeleton model exports + rebuilds on
    65536-dim hashed inputs through the frozen-architecture chain."""
    import torch

    import research.candidate_sd as candidate_sd
    from research.model_export import load_eval_model, save_eval_model
    from research.task_sd import build_fold as build_sd_fold

    records = fake_sd_text_records()
    base, path = _frozen_text(tmp_path, records)
    selection_records, selection_manifest = _selection_artifacts(path, records)
    config = {
        "train_provider": {"batch_size": 8},
        "trainer": {"seed": 0},
    }
    fold_data, fold = build_sd_fold(
        config, selection_manifest.data, fold=0, folds=4, repeat=0,
        records=selection_records,
    )
    assert fold_data.dims["modalities"]["text"] == 65536
    assert fold["scaler"] == {"text": {"kind": "identity"}}
    assert fold["split"]["holdout"] == []
    identity = {"text": {"kind": "identity"}}
    pool = adapter_for("syndrome").pool_manifest(
        selection_manifest.data, selection_records, notes="pool"
    )
    assert pool.data["scaler"] == identity
    # the pool manifest still verifies against its own hash and keeps the
    # v2 input spec
    assert pool.data["manifest_sha256"]
    assert syndrome_input_spec(pool.data).version.startswith("tcm-sd-text-v2:")
    assert set(pool.data["split"]["train"]) == (
        set(base["split"]["train"]) | set(base["split"]["dev"])
    )

    model = candidate_sd.build_model(fold_data)
    candidate_sd.train(model, fold_data, budget_seconds=0.05, seed=0)
    batch = next(iter(fold_data.train)).batch
    saved = save_eval_model(model, tmp_path, "ht0", task="syndrome")
    loaded = load_eval_model(
        saved["weights_path"], saved["architecture_path"], task="syndrome",
    )
    with torch.no_grad():
        original = model(batch)
        copy = loaded(batch)
    assert torch.equal(
        original.predictions["syndrome_probs"], copy.predictions["syndrome_probs"]
    )
    assert loaded.input_spec.version == model.input_spec.version


# -- trusted evaluator ------------------------------------------------------------


def test_macro_f1_oracle():
    # hand-computed: class0 tp2 fp1 fn1 -> 2/3; class1 tp0 fp0 fn1 -> 0
    assert _macro_f1(
        [(0, 0), (0, 0), (0, 1), (1, 0)], n_classes=2,
    ) == pytest.approx(1 / 3)
    assert _macro_f1([(1, 1), (1, 1)], n_classes=2) == 0.5
    assert _macro_f1([], n_classes=2) is None


def test_evaluator_scores_a_constant_predictor_exactly(tmp_path):
    records = fake_sd_records()
    base, _path = _frozen(tmp_path, records)
    provider = SyndromeBatchProvider(base, records, "dev", batch_size=8)
    report = SyndromeTrustedEvaluator(base).evaluate(
        _ConstantModel(base, N_CLASSES, 0), provider, partition="dev"
    )
    # the oracle: a constant class-0 predictor over the dev labels
    by_id = {f"sd-r{r.subject_index:05d}": r for r in records}
    pairs = [(0, by_id[s].label) for s in base["split"]["dev"]
             if by_id[s].label is not None]
    classes = range(N_CLASSES)
    f1s = []
    for c in classes:
        tp = sum(1 for p, lbl in pairs if p == c and lbl == c)
        fp = sum(1 for p, lbl in pairs if p == c and lbl != c)
        fn = sum(1 for p, lbl in pairs if p != c and lbl == c)
        f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    assert report.primary_metric == "syndrome/macro_f1"
    assert report.macro_f1 == pytest.approx(sum(f1s) / N_CLASSES)
    assert report.n_subjects == len(base["split"]["dev"])
    assert report.n_labeled == len(pairs)
    assert report.smoke_only is True
    assert report.to_dict()["partition"] == "dev"


def test_evaluator_gives_a_perfect_predictor_one_and_fail_closed_empty(tmp_path):
    records = fake_sd_records(labeled=True)
    for r in records:
        r.label = 1
    base, _path = _frozen(tmp_path, records)
    report = SyndromeTrustedEvaluator(base).evaluate(
        _ConstantModel(base, N_CLASSES, 1),
        SyndromeBatchProvider(base, records, "dev", batch_size=8),
        partition="dev",
    )
    assert report.macro_f1 == pytest.approx(1 / N_CLASSES)

    empty = fake_sd_records(labeled=False)
    base_empty, _ = _frozen(tmp_path, empty, notes="all-unlabeled")
    failed = SyndromeTrustedEvaluator(base_empty).evaluate(
        _ConstantModel(base_empty, N_CLASSES, 0),
        SyndromeBatchProvider(base_empty, empty, "dev", batch_size=8),
        partition="dev",
    )
    assert failed.macro_f1 is None
    assert failed.primary_value is None
    assert failed.not_computable


def test_evaluator_refuses_leakage_spec_mismatch_and_wrong_keys(tmp_path):
    records = fake_sd_records()
    base, _path = _frozen(tmp_path, records)
    evaluator = SyndromeTrustedEvaluator(base)
    train_provider = SyndromeBatchProvider(base, records, "train", batch_size=8)
    with pytest.raises(ValueError, match="does not match"):
        evaluator.evaluate(_ConstantModel(base, N_CLASSES, 0),
                           train_provider, partition="dev")
    dev_provider = SyndromeBatchProvider(base, records, "dev", batch_size=8)
    with pytest.raises(ValueError, match="input spec mismatch"):
        evaluator.evaluate(
            _ConstantModel(base, N_CLASSES, 0, version="some-other-spec-v9"),
            dev_provider, partition="dev",
        )

    import torch

    from core.batch import ModelOutput

    class WrongKey:
        input_spec = syndrome_input_spec(base)

        def __call__(self, batch):
            return self.forward(batch)

        def forward(self, batch):
            return ModelOutput(predictions={"category_probs": torch.zeros(1, 9)})

    with pytest.raises(ValueError, match="did not predict"):
        evaluator.evaluate(WrongKey(), dev_provider, partition="dev")


# -- folds and the isolation chain -------------------------------------------------


def test_syndrome_build_fold_prunes_like_the_constitution_fold(tmp_path):
    from research.task_sd import build_fold as build_sd_fold

    records = fake_sd_records()
    base, path = _frozen(tmp_path, records)
    selection_records, selection_manifest = _selection_artifacts(path, records)
    config = {
        "train_provider": {"batch_size": 8},
        "trainer": {"seed": 0},
    }
    fold_data, fold = build_sd_fold(
        config, selection_manifest.data, fold=0, folds=4, repeat=0,
        records=selection_records,
    )
    assert fold_data.dims["targets"] == N_CLASSES
    assert fold_data.dev is not None
    assert fold["split"]["holdout"] == []
    fold_rows = set(fold["subject_row_by_id"].values())
    assert fold_rows == (
        set(fold_data.train.records_by_row) | set(fold_data.dev.records_by_row)
    )
    holdout_rows = {base["subject_row_by_id"][s] for s in base["split"]["holdout"]}
    assert holdout_rows and not fold_rows & holdout_rows


def test_import_candidate_resolves_the_editable_path():
    from research.train_worker import import_candidate
    def candidate_config(editable):
        source = (WORKTREE_ROOT / editable).read_bytes()
        return {
            "worktree_path": str(WORKTREE_ROOT),
            "editable": editable,
            "candidate_sha256": hashlib.sha256(source).hexdigest(),
        }
    assert import_candidate(
        candidate_config("src/research/candidate.py")
    ).__name__ == "research.candidate"
    assert import_candidate(
        candidate_config("src/research/candidate_sd.py")
    ).__name__ == "research.candidate_sd"
    with pytest.raises(RuntimeError, match="src/<package>/<module>"):
        import_candidate({
            "worktree_path": str(WORKTREE_ROOT),
            "editable": "research/evil.py",
        })

def test_guard_covers_the_syndrome_editable(tmp_path):
    from research.guard import (
        PatchRejected,
        editable_complexity,
        validate_source_ast,
        validate_source_patch,
    )

    sd = "src/research/candidate_sd.py"
    # the real skeleton file is priced by the same AST+string-payload metric
    assert editable_complexity(WORKTREE_ROOT, sd) > 0
    patch = f"""diff --git a/{sd} b/{sd}
--- a/{sd}
+++ b/{sd}
@@ -1,3 +1,4 @@
 WIDTH = 32
 FUSION = 32
+WIDTH = 64
"""
    validate_source_patch(patch, sd)  # does not raise
    foreign = patch.replace(f"b/{sd}", "b/src/research/frozen.py").replace(
        f"a/{sd}", "a/src/research/frozen.py"
    )
    with pytest.raises(PatchRejected, match="outside the editable file"):
        validate_source_patch(foreign, sd)
    with pytest.raises(PatchRejected, match="dynamic code execution"):
        validate_source_ast('exec("x = 1")\n')


def test_syndrome_model_exports_and_rebuilds_state_dict_only(tmp_path):
    import torch

    import research.candidate_sd as candidate_sd
    from research.model_export import load_eval_model, save_eval_model
    from research.task_sd import build_fold as build_sd_fold

    records = fake_sd_records()
    _base, path = _frozen(tmp_path, records)
    _records = _records_path(tmp_path, records)
    selection_records, selection_manifest = _selection_artifacts(path, records)
    config = {
        "train_provider": {"batch_size": 8},
        "trainer": {"seed": 0},
    }
    fold, _fold_manifest = build_sd_fold(
        config, selection_manifest.data, fold=0, folds=4, repeat=0,
        records=selection_records,
    )
    model = candidate_sd.build_model(fold)
    candidate_sd.train(model, fold, budget_seconds=0.05, seed=0)
    batch = next(iter(fold.train)).batch

    saved = save_eval_model(model, tmp_path, "m0")
    state = torch.load(saved["weights_path"], weights_only=True)
    assert isinstance(state, dict) and state
    loaded = load_eval_model(saved["weights_path"], saved["architecture_path"])
    with torch.no_grad():
        original = model(batch)
        copy = loaded(batch)
    assert torch.equal(
        original.predictions["syndrome_probs"], copy.predictions["syndrome_probs"]
    )
    assert loaded.input_spec.version == model.input_spec.version
    assert loaded.model_type == "tcm_syndrome_v1"
    # the rebuild is NOT the candidate's class: no candidate code ran
    assert type(loaded) is not candidate_sd.SyndromeMLP


# -- task-level unlock ledger ------------------------------------------------------


def _git_repo(repo: Path, editable_rel: str, content: str) -> Path:
    (repo / editable_rel).parent.mkdir(parents=True, exist_ok=True)
    (repo / editable_rel).write_text(content)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=repo, check=True)
    return repo


def _git_commit(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True, text=True, capture_output=True,
    ).stdout.strip()
def test_ledger_params_and_paths_are_per_task(tmp_path):
    from research.finalize import (
        MAX_UNLOCKS,
        MAX_UNLOCKS_SD,
        UNLOCK_LEDGER_FILE,
        UNLOCK_LEDGER_FILE_SD,
        global_ledger_path,
        ledger_params,
    )

    assert ledger_params("constitution") == (UNLOCK_LEDGER_FILE, MAX_UNLOCKS)
    assert ledger_params("syndrome") == (UNLOCK_LEDGER_FILE_SD, MAX_UNLOCKS_SD)
    assert (UNLOCK_LEDGER_FILE, MAX_UNLOCKS) == (
        "holdout-unlock-ledger.jsonl", 2
    )
    assert (UNLOCK_LEDGER_FILE_SD, MAX_UNLOCKS_SD) == (
        "holdout-unlock-ledger-sd.jsonl", 1
    )
    with pytest.raises(ValueError, match="unknown research task"):
        ledger_params("physio")
    repo = _git_repo(tmp_path / "wt", "src/research/candidate_sd.py", _MINIMAL_CANDIDATE)
    ledger = global_ledger_path(repo, "syndrome")
    assert ledger.name == UNLOCK_LEDGER_FILE_SD
    assert ledger.parent.name == "runs"
    # the default stays the constitution ledger, byte for byte
    assert global_ledger_path(repo).name == UNLOCK_LEDGER_FILE
    assert global_ledger_path(repo) == repo / "runs" / UNLOCK_LEDGER_FILE


def _patch_finalize_preflight(monkeypatch):
    from research import finalize as finalize_mod
    from research import spawn as spawn_mod
    image_id = "sha256:" + "d" * 64
    def sandbox_run(config, *, preflight=False, **kwargs):
        assert preflight is True
        from research.spawn import sandbox_worker_process_timeout
        assert config["time_budget_seconds"] > 0
        assert kwargs["timeout"] == sandbox_worker_process_timeout(
            config["time_budget_seconds"]
        )
        return {
            "candidate_sha256": config["candidate_sha256"],
            "task": config["task"],
            "output_key": "syndrome_probs",
            "output_shape": [8, N_CLASSES],
            "weights_sha256": "a" * 64,
            "architecture_sha256": "b" * 64,
            "sandbox_image_id": image_id,
        }
    def validate_preflight(result, task, candidate_sha, config):
        assert result["task"] == task
        assert result["candidate_sha256"] == candidate_sha
        return {
            "output_key": result["output_key"],
            "output_shape": result["output_shape"],
            "weights_sha256": result["weights_sha256"],
            "architecture_sha256": result["architecture_sha256"],
            "sandbox_image_id": image_id,
        }
    monkeypatch.setattr(spawn_mod, "run_sandboxed_candidate", sandbox_run)
    monkeypatch.setattr(spawn_mod, "sandbox_image_identity", lambda: image_id)
    monkeypatch.setattr(
        finalize_mod, "_validate_preflight_artifacts", validate_preflight,
    )

def _patch_fitted_artifacts(monkeypatch):
    from research import finalize as finalize_mod
    def validate(result, **kwargs):
        return {
            "weights_sha256": result["weights_sha256"],
            "architecture_sha256": result["architecture_sha256"],
            "sandbox_image_id": "sha256:" + "d" * 64,
        }
    monkeypatch.setattr(finalize_mod, "_validate_fitted_artifacts", validate)

def test_syndrome_fit_failure_does_not_spend_unlock(tmp_path, monkeypatch):
    from research import finalize as finalize_mod
    _patch_finalize_preflight(monkeypatch)
    repo = _git_repo(tmp_path / "wt", "src/research/candidate_sd.py", _MINIMAL_CANDIDATE)
    records = fake_sd_records()
    dataset_path = _records_path(tmp_path, records)
    dataset_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    _base, manifest_path = _frozen(
        tmp_path, records, dataset_sha256=dataset_sha256
    )
    spawned = []
    def fail_fit(config, config_path):
        spawned.append(config)
        raise RuntimeError("worker crashed mid-fit")
    kwargs = dict(
        worktree=repo, out_dir=tmp_path / "out",
        manifest_path=manifest_path, dataset_path=dataset_path,
        editable="src/research/candidate_sd.py", commit=_git_commit(repo),
        task="syndrome", spawn=fail_fit,
    )
    with pytest.raises(RuntimeError, match="crashed"):
        finalize_mod.refit_and_score(**kwargs)
    ledger = finalize_mod.global_ledger_path(repo, "syndrome")
    assert len(spawned) == 1
    assert not ledger.exists()
    assert not finalize_mod.global_ledger_path(repo).exists()

    retry_spawned = []
    def fail_retry(config, config_path):
        retry_spawned.append(config)
        raise RuntimeError("retry reached fit")
    with pytest.raises(RuntimeError, match="retry reached fit"):
        finalize_mod.refit_and_score(
            **{**kwargs, "out_dir": tmp_path / "retry-out", "spawn": fail_retry}
        )
    assert len(retry_spawned) == 1
    assert not ledger.exists()


def test_syndrome_probability_contract_failure_precedes_dataset_read_and_unlock(
    tmp_path, monkeypatch,
):
    from research import finalize as finalize_mod
    from research import spawn as spawn_mod

    repo = _git_repo(
        tmp_path / "wt", "src/research/candidate_sd.py", _MINIMAL_CANDIDATE,
    )
    records = fake_sd_records()
    _base, manifest_path = _frozen(tmp_path, records)
    missing_dataset = tmp_path / "not-mounted-selection-data.json"
    observed = []

    def reject_probability_contract(config, *, preflight=False, **_kwargs):
        assert preflight is True
        assert config["time_budget_seconds"] > 0
        observed.append("trusted synthetic preflight")
        raise RuntimeError(
            "candidate predictions['syndrome_probs'] probabilities must sum to one per row"
        )

    monkeypatch.setattr(
        spawn_mod, "run_sandboxed_candidate", reject_probability_contract,
    )
    with pytest.raises(RuntimeError, match="probabilities must sum to one"):
        finalize_mod.refit_and_score(
            worktree=repo,
            out_dir=tmp_path / "out",
            manifest_path=manifest_path,
            dataset_path=missing_dataset,
            editable="src/research/candidate_sd.py",
            commit=_git_commit(repo),
            task="syndrome",
        )

    ledger = finalize_mod.global_ledger_path(repo, "syndrome")
    assert observed == ["trusted synthetic preflight"]
    assert not missing_dataset.exists()
    assert not ledger.exists()

def test_charging_refuses_the_second_syndrome_event_atomically(tmp_path):
    from research import finalize as finalize_mod

    ledger = tmp_path / "runs" / "holdout-unlock-ledger-sd.jsonl"
    event = {"event_id": "a" * 32}
    finalize_mod._charge_unlock_event(
        ledger, event, max_unlocks=finalize_mod.MAX_UNLOCKS_SD
    )
    with pytest.raises(RuntimeError, match="unlock budget"):
        finalize_mod._charge_unlock_event(
            ledger, {"event_id": "b" * 32}, max_unlocks=finalize_mod.MAX_UNLOCKS_SD
        )
    assert finalize_mod.unlock_count(ledger) == 1
    # the constitution default is unchanged: two events fit, the third refuses
    old = tmp_path / "runs" / "holdout-unlock-ledger.jsonl"
    for i in range(2):
        finalize_mod._charge_unlock_event(old, {"event_id": str(i) * 32})
    with pytest.raises(RuntimeError, match="unlock budget"):
        finalize_mod._charge_unlock_event(old, {"event_id": "z" * 32})
    assert finalize_mod.unlock_count(old) == 2


def test_syndrome_finalize_driver_uses_the_syndrome_adapter(tmp_path, monkeypatch):
    _patch_finalize_preflight(monkeypatch)
    _patch_fitted_artifacts(monkeypatch)
    from research import finalize as finalize_mod

    repo = _git_repo(tmp_path / "wt", "src/research/candidate_sd.py", _MINIMAL_CANDIDATE)
    records = fake_sd_records()
    dataset_path = _records_path(tmp_path, records)
    dataset_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    base, manifest_path = _frozen(
        tmp_path, records, dataset_sha256=dataset_sha256
    )
    spawned: list[dict] = []

    def spawn(config, config_path):
        # the fit worker's config carries the syndrome task and NO dataset
        # path and NO frozen-manifest path
        assert config["task"] == "syndrome"
        assert "dataset_path" not in config["data"]
        assert "frozen_manifest_path" not in config["data"]
        spawned.append(config)
        return {
            "finalize": {"candidate_sha256": "x"},
            "weights_path": "unused.pt", "architecture_path": "unused.json",
            "weights_sha256": "0" * 64, "architecture_sha256": "0" * 64,
        }

    def spawn_holdout(config, request):
        assert config["task"] == "syndrome"
        assert config["data"]["frozen_manifest_path"]
        return {"holdout": {"partition": "holdout"}, "baselines": {}}

    out = finalize_mod.refit_and_score(
        worktree=repo, out_dir=tmp_path / "out",
        manifest_path=manifest_path, dataset_path=dataset_path,
        editable="src/research/candidate_sd.py", commit=_git_commit(repo),
        task="syndrome",
        spawn=spawn, spawn_holdout=spawn_holdout,
    )
    assert len(spawned) == 1
    assert out["holdout"]["partition"] == "holdout"
    pool = json.loads(
        Path(spawned[0]["data"]["pool_manifest_path"]).read_text()
    )
    expected = set(base["split"]["train"]) | set(base["split"]["dev"])
    assert set(pool["subject_row_by_id"]) == expected
    assert pool["split"]["holdout"] == []
    # the pool records are syndrome records the worker can parse back
    parsed = records_from_json(
        Path(spawned[0]["data"]["pool_records_path"]).read_text()
    )
    assert {r.subject_index for r in parsed} == set(pool["subject_row_by_id"].values())


# -- the full runner chain, on subprocesses ------------------------------------------


def _link_src(worktree: Path) -> None:
    """A fake worktree whose src is the real package."""
    src = worktree / "src"
    src.mkdir(parents=True)
    for item in SRC_ROOT.iterdir():
        (src / item.name).symlink_to(item)


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_syndrome_runner_full_chain_scores_synthetic(tmp_path, monkeypatch, docker_daemon):
    """The money smoke: runner (in-process orchestrator) -> one-shot
    train_worker subprocess on candidate_sd.py -> evaluator subprocess
    scoring the state_dict — all through the syndrome adapter, all files."""
    from research import runner
    from research.runner import run
    original_child_env = runner._child_env
    monkeypatch.setattr(
        runner,
        "_child_env",
        lambda: {**original_child_env(), "TMPDIR": str(tmp_path)},
    )

    records = fake_sd_records()
    _base, manifest_path = _frozen(tmp_path, records)
    # the runner consumes selection-only inputs (empty holdout, parent-bound
    # hashes); the full frozen manifest is never handed to a candidate chain
    loaded = load_real_manifest(manifest_path)
    selection_records, selection_manifest = build_syndrome_selection_artifacts(
        RealManifest(data=loaded, sha256=loaded["manifest_sha256"]), records
    )
    selection_manifest_path = tmp_path / "selection-manifest.json"
    selection_manifest_path.write_text(
        json.dumps(selection_manifest.data, ensure_ascii=False)
    )
    selection_records_path = tmp_path / "selection-records.json"
    selection_records_path.write_text(records_to_json(selection_records))
    worktree = tmp_path / "wt"
    _link_src(worktree)
    candidate_sha = _file_sha(worktree / "src" / "research" / "candidate_sd.py")
    config = {
        "task": "syndrome",
        "data": {"manifest_path": str(selection_manifest_path),
                 "dataset_path": str(selection_records_path)},
        "train_provider": {"batch_size": 8},
        "trainer": {"seed": 0},
        "time_budget_seconds": 8.0,
        "worktree_path": str(worktree),
        "editable": "src/research/candidate_sd.py",
        "candidate_sha256": candidate_sha,
    }
    result = run(config, folds=2, repeats=1)
    assert result["n_folds"] == 2
    assert isinstance(result["primary_value"], float)
    assert 0.0 <= result["primary_value"] <= 1.0
    assert not result["overran"]
    assert all(0.0 <= s <= 1.0 for s in result["folds"])


def test_syndrome_worker_and_evaluator_subprocesses_end_to_end(tmp_path, docker_daemon):
    """The evaluator subprocess path for the syndrome task: train in a
    one-shot worker subprocess on a trimmed fit manifest, then score the
    state_dict in an evaluator subprocess — no candidate import between."""
    records = fake_sd_records()
    _base, manifest_path = _frozen(tmp_path, records)
    # the evaluator subprocess also consumes selection-only inputs; the fit
    # manifest is derived below from the full frozen manifest's train side
    worktree = tmp_path / "wt"
    _link_src(worktree)
    candidate_sha = _file_sha(worktree / "src" / "research" / "candidate_sd.py")

    from research.frozen import selection_fold_ids
    from research.tasks import adapter_for

    manifest = load_real_manifest(manifest_path)
    train_ids, _dev = selection_fold_ids(manifest, 0, 4, 0)
    fit_manifest = adapter_for("syndrome").fold_manifest(
        manifest, records, train_ids, [], notes="fit"
    ).data
    assert "user_partition_by_id" not in fit_manifest
    assert set(fit_manifest["subject_user_by_id"]) == set(
        fit_manifest["split"]["train"]
    )
    fit_manifest_path = tmp_path / "fit-manifest.json"
    fit_manifest_path.write_text(json.dumps(fit_manifest, ensure_ascii=False))
    fit_rows = set(fit_manifest["subject_row_by_id"].values())
    fit_records_path = tmp_path / "fit-records.json"
    fit_records_path.write_text(
        records_to_json([r for r in records if r.subject_index in fit_rows])
    )
    worker_config = {
        "worktree_path": str(worktree),
        "editable": "src/research/candidate_sd.py",
        "task": "syndrome",
        "candidate_sha256": candidate_sha,
        "seed": 0,
        "time_budget_seconds": 8.0,
        "train_batch_size": 8,
        "fit_manifest_path": str(fit_manifest_path),
        "fit_records_path": str(fit_records_path),
        "output_dir": str(tmp_path / "out"),
        "output_stem": "m0",
    }
    payload = _run_child(worktree, "research.train_worker", worker_config)
    assert payload["ok"], payload
    assert payload["input_spec_version"].startswith("tcm-sd-text-v1")

    selection_records, selection_manifest = build_syndrome_selection_artifacts(
        RealManifest(data=manifest, sha256=manifest["manifest_sha256"]), records
    )
    selection_manifest_path = tmp_path / "selection-manifest.json"
    selection_manifest_path.write_text(
        json.dumps(selection_manifest.data, ensure_ascii=False)
    )
    selection_records_path = tmp_path / "selection-records.json"
    selection_records_path.write_text(records_to_json(selection_records))
    eval_config = {
        "task": "syndrome",
        "data": {"manifest_path": str(selection_manifest_path),
                 "dataset_path": str(selection_records_path)},
        "worktree_path": str(worktree),
        "editable": "src/research/candidate_sd.py",
    }
    request = {
        "fold": 0, "folds": 4, "repeat": 0,
        "weights_path": payload["weights_path"],
        "architecture_path": payload["architecture_path"],
        "weights_sha256": payload["weights_sha256"],
        "architecture_sha256": payload["architecture_sha256"],
        "integrity": {
            "manifest_sha256": _file_sha(selection_manifest_path),
            "dataset_sha256": _file_sha(selection_records_path),
            "candidate_sha256": candidate_sha,
        },
    }
    scored = _run_child(worktree, "research.evaluator", eval_config,
                        request=request, extra_args=["--mode", "fold"])
    assert scored["ok"], scored
    assert isinstance(scored["primary_value"], float)


def _run_child(worktree: Path, module: str, config: dict,
               extra_args: list[str] | None = None, request: dict | None = None):
    from research.spawn import bootstrap_command

    config_path = worktree.parent / f"{module.replace('.', '-')}-config.json"
    config_path.write_text(json.dumps(config))
    args = ["--config", str(config_path)]
    if request is not None:
        request_path = worktree.parent / f"{module.replace('.', '-')}-request.json"
        request_path.write_text(json.dumps(request))
        args += ["--request", str(request_path)]
    result = subprocess.run(
        bootstrap_command(str(worktree), module, *args, *(extra_args or [])),
        cwd=worktree, text=True, capture_output=True, timeout=300,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "HOME": str(worktree.parent),
            "TMPDIR": str(worktree.parent),
        },
    )
    lines = [line for line in result.stdout.strip().splitlines() if line.strip()]
    assert lines, result.stdout + result.stderr
    return json.loads(lines[-1])


def test_published_manifest_requires_complete_frozen_user_strata():
    records = fake_sd_records()
    partitions = subject_partitions(records)
    strata = user_majority_strata(records)
    kwargs = {
        "dataset_sha256": DATASET_SHA,
        "n_classes": N_CLASSES,
        "label_taxonomy_version": TAXONOMY,
        "label_tier": "PUBLISHED",
        "frozen_user_partitions": partitions,
    }

    with pytest.raises(ValueError, match="frozen_user_strata is required"):
        build_syndrome_manifest(records, **kwargs)

    incomplete = dict(strata)
    incomplete.pop(next(iter(incomplete)))
    with pytest.raises(ValueError, match="frozen user strata"):
        build_syndrome_manifest(
            records,
            frozen_user_strata=incomplete,
            **kwargs,
        )



def test_published_manifest_contract_rejects_incomplete_user_strata():
    from research.tasks import validate_manifest_contract

    records = fake_sd_records()
    manifest = build_syndrome_manifest(
        records,
        dataset_sha256=DATASET_SHA,
        n_classes=N_CLASSES,
        label_taxonomy_version=TAXONOMY,
        label_tier="PUBLISHED",
        frozen_user_partitions=subject_partitions(records),
        frozen_user_strata=user_majority_strata(records),
    ).data
    validate_manifest_contract("syndrome", manifest)

    manifest["user_stratum_by_id"].pop(next(iter(manifest["user_stratum_by_id"])))
    with pytest.raises(ValueError, match="do not cover frozen partitions"):
        validate_manifest_contract("syndrome", manifest)


def _selection_user_assignments(manifest, folds, repeat):
    assignments = {}
    fold_users = []
    for fold in range(folds):
        train_ids, dev_ids = selection_fold_ids(
            manifest,
            fold=fold,
            folds=folds,
            repeat=repeat,
        )
        train_users = {
            manifest["subject_user_by_id"][subject_id]
            for subject_id in train_ids
        }
        dev_users = {
            manifest["subject_user_by_id"][subject_id]
            for subject_id in dev_ids
        }
        assert train_users.isdisjoint(dev_users)
        for user_id in dev_users:
            assert user_id not in assignments
            assignments[user_id] = fold
        fold_users.append(dev_users)
    return assignments, fold_users


def _synthetic_selection_manifest(user_strata, recorded_users, seed=2026):
    subject_rows = {}
    subject_users = {}
    train_ids = []
    for user_id in sorted(recorded_users):
        for scenario in range(2):
            subject_id = f"{user_id}-case-{scenario}"
            subject_rows[subject_id] = len(subject_rows)
            subject_users[subject_id] = user_id
            train_ids.append(subject_id)
    return {
        "task": "syndrome",
        "split": {
            "train": train_ids,
            "dev": [],
            "holdout": [],
            "seed": seed,
        },
        "subject_row_by_id": subject_rows,
        "subject_user_by_id": subject_users,
        "user_partition_by_id": {
            user_id: "train" for user_id in user_strata
        },
        "user_stratum_by_id": dict(user_strata),
    }


def _force_selection_fold_offset(monkeypatch, offset):
    from research import frozen

    class FixedOffsetRandom:
        def __init__(self, seed):
            self.seed = seed

        def shuffle(self, values):
            return None

        def randrange(self, stop):
            assert offset < stop
            return offset

    monkeypatch.setattr(frozen.random, "Random", FixedOffsetRandom)


def test_selection_folds_keep_nonempty_mapping_unchanged(monkeypatch):
    users = [f"scored-{index}" for index in range(4)]
    manifest = _synthetic_selection_manifest(
        {user_id: "majority-class-0" for user_id in users},
        set(users),
    )
    _force_selection_fold_offset(monkeypatch, 1)

    assignments, _fold_users = _selection_user_assignments(manifest, 4, 0)

    assert assignments == {
        "scored-0": 1,
        "scored-1": 2,
        "scored-2": 3,
        "scored-3": 0,
    }


def test_selection_folds_fail_closed_when_folds_have_no_observed_users(monkeypatch):
    """Frozen fold membership never rebalances around scenario row filtering.

    Users without scenario rows keep their frozen fold assignment; folds left
    without any observed user fail closed instead of moving another user
    between folds (protocol invariant shared with the CV evaluator).
    """
    user_strata = {}
    recorded_users = set()
    for class_index in range(4):
        stratum = f"majority-class-{class_index}"
        recorded_user = f"recorded-{class_index}"
        recorded_users.add(recorded_user)
        user_strata[recorded_user] = stratum
        for no_scene_index in range(3):
            user_strata[f"no-scene-{class_index}-{no_scene_index}"] = stratum
    manifest = _synthetic_selection_manifest(user_strata, recorded_users)
    _force_selection_fold_offset(monkeypatch, 0)

    with pytest.raises(
        ValueError,
        match="no observed users in folds \\[0, 1, 2\\]",
    ):
        selection_fold_ids(manifest, fold=0, folds=4, repeat=0)


def test_selection_folds_reject_too_few_users_with_scenario_records(monkeypatch):
    user_strata = {
        "recorded-0": "majority-class-0",
        "recorded-1": "majority-class-1",
        "recorded-2": "majority-class-2",
        "no-scene": "majority-class-3",
    }
    manifest = _synthetic_selection_manifest(
        user_strata, {"recorded-0", "recorded-1", "recorded-2"}
    )
    _force_selection_fold_offset(monkeypatch, 0)

    with pytest.raises(
        ValueError,
        match="only 3 users with scenario records for 4 folds",
    ):
        selection_fold_ids(manifest, fold=0, folds=4, repeat=0)


def test_selection_folds_stratify_frozen_users_and_ignore_row_filtering():
    records = fake_sd_records(n=90)
    initial_partitions = subject_partitions(records)
    repeated_user = sorted(initial_partitions["train"] | initial_partitions["dev"])[0]
    repeated = next(record for record in records if record.user_id == repeated_user)
    records.append(
        SyndromeRecord(
            subject_index=91,
            user_id=repeated.user_id,
            norm_syndrome=repeated.norm_syndrome,
            features=dict(repeated.features),
            label=repeated.label,
            input_text=repeated.input_text,
            provenance={"source": "test-fixture", "repeat": True},
        )
    )
    partitions = subject_partitions(records)
    strata = user_majority_strata(records)
    manifest = build_syndrome_manifest(
        records,
        dataset_sha256=DATASET_SHA,
        n_classes=N_CLASSES,
        label_taxonomy_version=TAXONOMY,
        label_tier="SMOKE_ONLY",
        frozen_user_partitions=partitions,
        frozen_user_strata=strata,
    ).data

    assignments, fold_users = _selection_user_assignments(manifest, 4, 0)
    pool_ids = (
        set(manifest["split"]["train"]) | set(manifest["split"]["dev"])
    )
    pool_users = {
        manifest["subject_user_by_id"][subject_id] for subject_id in pool_ids
    }
    assert set(assignments) == pool_users
    repeated_ids = {
        subject_id
        for subject_id in pool_ids
        if manifest["subject_user_by_id"][subject_id] == repeated_user
    }
    for fold in range(4):
        train_ids, dev_ids = selection_fold_ids(manifest, fold, 4, 0)
        assert repeated_ids <= set(train_ids) or repeated_ids <= set(dev_ids)

    for stratum in set(strata.values()):
        counts = [
            sum(strata[user_id] == stratum for user_id in dev_users)
            for dev_users in fold_users
        ]
        assert max(counts) - min(counts) <= 1

    same_repeat, _ = _selection_user_assignments(manifest, 4, 0)
    next_repeat, _ = _selection_user_assignments(manifest, 4, 1)
    assert assignments == same_repeat
    assert assignments != next_repeat

    removed_user = next(
        user_id for user_id, stratum in strata.items()
        if stratum == sorted(set(strata.values()))[0]
        and user_id in (partitions["train"] | partitions["dev"])
    )
    filtered_records = [
        record for record in records if record.user_id != removed_user
    ]
    filtered = build_syndrome_manifest(
        filtered_records,
        dataset_sha256=DATASET_SHA,
        n_classes=N_CLASSES,
        label_taxonomy_version=TAXONOMY,
        label_tier="SMOKE_ONLY",
        frozen_user_partitions=partitions,
        frozen_user_strata=strata,
    ).data
    filtered_assignments, _ = _selection_user_assignments(filtered, 4, 0)
    assert {
        user_id: fold
        for user_id, fold in assignments.items()
        if user_id != removed_user
    } == filtered_assignments


def test_macro_f1_uses_fixed_taxonomy_and_reports_absent_classes(tmp_path):
    records = fake_sd_records(n=60, n_classes=N_CLASSES)
    for record in records:
        record.label = 1
    base, _path = _frozen(tmp_path, records)
    report = SyndromeTrustedEvaluator(base).evaluate(
        _ConstantModel(base, N_CLASSES, 1),
        SyndromeBatchProvider(base, records, "dev", batch_size=8),
        partition="dev",
    )

    assert _macro_f1([(0, 0)], n_classes=4) == pytest.approx(0.25)
    assert report.macro_f1 == pytest.approx(1 / N_CLASSES)
    assert set(report.per_class) == {str(index) for index in range(N_CLASSES)}
    assert report.per_class[str(N_CLASSES - 1)]["support"] == 0
    assert report.per_class[str(N_CLASSES - 1)]["f1"] == 0.0


def test_sd_v2_builder_published_manifest_is_one_based_and_validates(
    tmp_path, monkeypatch,
):
    from dataclasses import replace

    from data.tcm_sd import TcmSdRecord
    from research.tasks import validate_manifest_contract
    from scripts import build_sd_campaign_inputs as builder

    records = []
    for index in range(40):
        label = "甲" if index % 2 == 0 else "乙"
        records.append(
            TcmSdRecord(
                user_id=f"builder-user-{index:03d}",
                lcd_id=f"lcd-{index:03d}",
                lcd_name="合成证候",
                syndrome=label,
                norm_syndrome=label,
                chief_complaint=f"synthetic symptom {index}",
                description="",
                detection="",
                source_file="train.json",
            )
        )
    partitions = subject_partitions(records)
    train_record = next(
        record for record in records
        if record.user_id in partitions["train"]
    )
    dev_record = next(
        record for record in records
        if record.user_id in partitions["dev"]
    )
    dev_user = dev_record.user_id
    records[records.index(dev_record)] = replace(
        dev_record,
        chief_complaint=train_record.chief_complaint,
    )
    frozen_strata = user_majority_strata(records)
    monkeypatch.setattr(
        builder,
        "load_tcm_sd_development_records",
        lambda _raw_dir: records,
    )
    monkeypatch.setattr(
        builder,
        "load_syndrome_vocab",
        lambda _raw_dir: ["甲", "乙"],
    )
    out_dir = tmp_path / "sd-v2-short"
    assert builder.main([
        "--raw-dir",
        str(tmp_path / "unused-raw"),
        "--out",
        str(out_dir),
    ]) == 0

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert min(manifest["subject_row_by_id"].values()) == 1
    assert manifest["user_stratum_by_id"] == frozen_strata
    assert manifest["user_partition_by_id"][dev_user] == "dev"
    # the dev user keeps its frozen partition; its rows must not leak into the
    # train split (a complete subject→user mapping is required by the contract)
    assert dev_user not in {
        manifest["subject_user_by_id"][subject_id]
        for subject_id in manifest["split"]["train"]
    }
    validate_manifest_contract("syndrome", manifest)

    selection_records_path = out_dir / "selection_records.json"
    selection_manifest_path = out_dir / "selection_manifest.json"
    selection_manifest = load_real_manifest(selection_manifest_path)
    selection_records = records_from_json(
        selection_records_path.read_text(encoding="utf-8")
    )
    validate_manifest_contract(
        "syndrome", selection_manifest, selection_pool=True
    )
    assert selection_manifest["split"]["train"] == manifest["split"]["train"]
    assert selection_manifest["split"]["dev"] == manifest["split"]["dev"]
    assert selection_manifest["split"]["holdout"] == []
    assert selection_manifest["selection_parent_manifest_sha256"] == (
        manifest["manifest_sha256"]
    )
    assert selection_manifest["selection_parent_dataset_sha256"] == (
        manifest["dataset_sha256"]
    )
    selection_bytes = selection_records_path.read_bytes()
    assert hashlib.sha256(selection_bytes).hexdigest() == (
        selection_manifest["dataset_sha256"]
    )
    selection_subjects = set(selection_manifest["split"]["train"]) | set(
        selection_manifest["split"]["dev"]
    )
    assert set(selection_manifest["subject_row_by_id"]) == selection_subjects
    assert {record.subject_index for record in selection_records} == set(
        selection_manifest["subject_row_by_id"].values()
    )
    assert set(selection_manifest["user_partition_by_id"]) == {
        user for user, partition in manifest["user_partition_by_id"].items()
        if partition in {"train", "dev"}
    }
    assert selection_manifest["user_stratum_by_id"] == {
        user: frozen_strata[user]
        for user in selection_manifest["user_partition_by_id"]
    }

    snapshot = {
        path.name: path.read_bytes()
        for path in out_dir.iterdir()
        if path.is_file()
    }
    monkeypatch.setattr(
        builder,
        "load_tcm_sd_development_records",
        lambda _raw_dir: pytest.fail("existing output must be rejected first"),
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        builder.main([
            "--raw-dir",
            str(tmp_path / "unused-raw"),
            "--out",
            str(out_dir),
        ])
    assert {
        path.name: path.read_bytes()
        for path in out_dir.iterdir()
        if path.is_file()
    } == snapshot


def test_manifest_hash_covers_frozen_user_strata(tmp_path):
    records = fake_sd_records()
    manifest, path = _frozen(tmp_path, records)
    assert manifest["user_stratum_by_id"] == user_majority_strata(records)

    tampered = json.loads(path.read_text(encoding="utf-8"))
    user_id = next(iter(tampered["user_stratum_by_id"]))
    tampered["user_stratum_by_id"][user_id] = "changed-stratum"
    path.write_text(json.dumps(tampered, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        load_real_manifest(path)
