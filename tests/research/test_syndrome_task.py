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

from data.real_split import load_real_manifest
from research.task_sd import (
    FEATURE_GROUP,
    PRIMARY_METRIC_SD,
    SyndromeBatchProvider,
    SyndromeRecord,
    SyndromeTrustedEvaluator,
    _macro_f1,
    build_syndrome_manifest,
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
    manifest = build_syndrome_manifest(
        records, dataset_sha256=DATASET_SHA, n_classes=N_CLASSES,
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
            label_taxonomy_version=TAXONOMY,
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
            label_taxonomy_version=TAXONOMY,
        )
    with pytest.raises(ValueError, match="too few cases"):
        build_syndrome_manifest(
            good[:5], dataset_sha256=DATASET_SHA, n_classes=N_CLASSES,
            label_taxonomy_version=TAXONOMY,
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
            label_taxonomy_version=TAXONOMY, modality_kind="hashed_text",
        )
    with pytest.raises(ValueError, match="unknown modality_kind"):
        build_syndrome_manifest(
            fake_sd_text_records(12), dataset_sha256=DATASET_SHA,
            n_classes=N_CLASSES, label_taxonomy_version=TAXONOMY,
            modality_kind="tfidf_idf",
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
    config = {
        "data": {"manifest_path": str(path), "dataset_path": "unused"},
        "train_provider": {"batch_size": 8},
        "trainer": {"seed": 0},
    }
    fold_data, fold = build_sd_fold(
        config, str(path), "unused", fold=0, folds=4, repeat=0, records=records
    )
    assert fold_data.dims["modalities"]["text"] == 65536
    assert fold["scaler"] == {"text": {"kind": "identity"}}
    assert fold["split"]["holdout"] == []
    identity = {"text": {"kind": "identity"}}
    pool = adapter_for("syndrome").pool_manifest(fold, records, notes="pool")
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
    saved = save_eval_model(model, tmp_path, "ht0")
    loaded = load_eval_model(saved["weights_path"], saved["architecture_path"])
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
    assert _macro_f1([(0, 0), (0, 0), (0, 1), (1, 0)]) == pytest.approx(1 / 3)
    assert _macro_f1([(1, 1), (1, 1)]) == 1.0
    assert _macro_f1([]) is None


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
    classes = sorted({p for p, _ in pairs} | {lbl for _, lbl in pairs})
    f1s = []
    for c in classes:
        tp = sum(1 for p, lbl in pairs if p == c and lbl == c)
        fp = sum(1 for p, lbl in pairs if p == c and lbl != c)
        fn = sum(1 for p, lbl in pairs if p != c and lbl == c)
        f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    assert report.primary_metric == "syndrome/macro_f1"
    assert report.macro_f1 == pytest.approx(sum(f1s) / len(f1s))
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
    assert report.macro_f1 == 1.0

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
    config = {
        "data": {"manifest_path": str(path), "dataset_path": "unused"},
        "train_provider": {"batch_size": 8},
        "trainer": {"seed": 0},
    }
    fold_data, fold = build_sd_fold(
        config, str(path), "unused", fold=0, folds=4, repeat=0, records=records
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

    assert import_candidate({}).__name__ == "research.candidate"
    assert import_candidate({"editable": "src/research/candidate.py"}).__name__ == (
        "research.candidate"
    )
    assert import_candidate(
        {"editable": "src/research/candidate_sd.py"}
    ).__name__ == "research.candidate_sd"
    with pytest.raises(RuntimeError, match="src/<package>/<module>"):
        import_candidate({"editable": "research/evil.py"})


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
    config = {
        "data": {"manifest_path": str(path), "dataset_path": "unused"},
        "train_provider": {"batch_size": 8},
        "trainer": {"seed": 0},
    }
    fold, _fold_manifest = build_sd_fold(
        config, str(path), "unused", fold=0, folds=4, repeat=0, records=records
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
    repo = _git_repo(tmp_path / "wt", "src/research/candidate_sd.py", "WIDTH = 32\n")
    ledger = global_ledger_path(repo, "syndrome")
    assert ledger.name == UNLOCK_LEDGER_FILE_SD
    assert ledger.parent.name == "runs"
    # the default stays the constitution ledger, byte for byte
    assert global_ledger_path(repo).name == UNLOCK_LEDGER_FILE
    assert global_ledger_path(repo) == repo / "runs" / UNLOCK_LEDGER_FILE


def test_syndrome_ledger_charges_one_unlock_and_refuses_the_second(tmp_path):
    from research import finalize as finalize_mod

    repo = _git_repo(tmp_path / "wt", "src/research/candidate_sd.py", "WIDTH = 32\n")
    records = fake_sd_records()
    base, manifest_path = _frozen(tmp_path, records)
    spawned: list[dict] = []

    def spawn(config, config_path):
        spawned.append(config)
        raise RuntimeError("worker crashed mid-fit")

    kwargs = dict(
        worktree=repo, out_dir=tmp_path / "out",
        manifest_path=manifest_path, dataset_path="unused",
        editable="src/research/candidate_sd.py", commit="0123456789abcdef",
        records=records, task="syndrome", spawn=spawn,
    )
    with pytest.raises(RuntimeError, match="crashed"):
        finalize_mod.refit_and_score(**kwargs)
    assert len(spawned) == 1
    ledger = finalize_mod.global_ledger_path(repo, "syndrome")
    events = [json.loads(line) for line in ledger.read_text().splitlines() if line]
    assert len(events) == 1 and events[0]["task"] == "syndrome"
    # the budget is ONE for the syndrome task: the second attempt is refused
    # before any worker spawns, and the constitution ledger is untouched
    refused: list = []
    with pytest.raises(RuntimeError, match="unlock budget"):
        finalize_mod.refit_and_score(
            **{**kwargs, "spawn": lambda c, p: refused.append(c)}
        )
    assert refused == []
    assert finalize_mod.unlock_count(ledger) == 1
    assert not finalize_mod.global_ledger_path(repo).exists()

    # the worker-side check uses the same per-task budget: one event is this
    # run's own legal charge; a second event is an over-spend and is refused
    assert finalize_mod._assert_worker_budget(
        ledger, max_unlocks=finalize_mod.MAX_UNLOCKS_SD
    ) == 1
    over = ledger.read_text() + json.dumps({"event_id": "f" * 32}) + "\n"
    ledger.write_text(over)
    with pytest.raises(RuntimeError, match="unlock budget"):
        finalize_mod._assert_worker_budget(
            ledger, max_unlocks=finalize_mod.MAX_UNLOCKS_SD
        )


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


def test_syndrome_finalize_driver_uses_the_syndrome_adapter(tmp_path):
    from research import finalize as finalize_mod

    repo = _git_repo(tmp_path / "wt", "src/research/candidate_sd.py", "WIDTH = 32\n")
    records = fake_sd_records()
    base, manifest_path = _frozen(tmp_path, records)
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
        manifest_path=manifest_path, dataset_path="unused",
        editable="src/research/candidate_sd.py", commit="0123456789abcdef",
        records=records, task="syndrome",
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


def test_syndrome_runner_full_chain_scores_synthetic(tmp_path):
    """The money smoke: runner (in-process orchestrator) -> one-shot
    train_worker subprocess on candidate_sd.py -> evaluator subprocess
    scoring the state_dict — all through the syndrome adapter, all files."""
    from research.runner import run

    records = fake_sd_records()
    _base, manifest_path = _frozen(tmp_path, records)
    records_path = _records_path(tmp_path, records)
    worktree = tmp_path / "wt"
    _link_src(worktree)
    candidate_sha = _file_sha(worktree / "src" / "research" / "candidate_sd.py")
    config = {
        "task": "syndrome",
        "data": {"manifest_path": str(manifest_path),
                 "dataset_path": str(records_path)},
        "train_provider": {"batch_size": 8},
        "trainer": {"seed": 0},
        "time_budget_seconds": 0.05,
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


def test_syndrome_worker_and_evaluator_subprocesses_end_to_end(tmp_path):
    """The evaluator subprocess path for the syndrome task: train in a
    one-shot worker subprocess on a trimmed fit manifest, then score the
    state_dict in an evaluator subprocess — no candidate import between."""
    records = fake_sd_records()
    _base, manifest_path = _frozen(tmp_path, records)
    records_path = _records_path(tmp_path, records)
    worktree = tmp_path / "wt"
    _link_src(worktree)
    candidate_sha = _file_sha(worktree / "src" / "research" / "candidate_sd.py")

    from data.real_split import fold_manifest as _prune
    from research.frozen import selection_fold_ids

    manifest = load_real_manifest(manifest_path)
    train_ids, _dev = selection_fold_ids(manifest, 0, 4, 0)
    fit_manifest = _prune(manifest, records, train_ids, [], notes="fit").data
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
        "time_budget_seconds": 0.05,
        "train_batch_size": 8,
        "fit_manifest_path": str(fit_manifest_path),
        "fit_records_path": str(fit_records_path),
        "output_dir": str(tmp_path / "out"),
        "output_stem": "m0",
    }
    payload = _run_child(worktree, "research.train_worker", worker_config)
    assert payload["ok"], payload
    assert payload["input_spec_version"].startswith("tcm-sd-text-v1")

    eval_config = {
        "task": "syndrome",
        "data": {"manifest_path": str(manifest_path),
                 "dataset_path": str(records_path)},
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
            "manifest_sha256": _file_sha(manifest_path),
            "dataset_sha256": _file_sha(records_path),
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
        env={"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(worktree.parent)},
    )
    lines = [line for line in result.stdout.strip().splitlines() if line.strip()]
    assert lines, result.stdout + result.stderr
    return json.loads(lines[-1])
