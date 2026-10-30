"""Real-data batch pipeline tests (#50): providers, masking, model factory,
fail-closed evaluation and the checkpoint smoke_only gateways."""

import random

import pytest
import torch

from data.mendeley import SubjectRecord
from data.real_split import build_real_manifest
from runtime.checkpoint import load_training_checkpoint, save_training_checkpoint
from runtime.real_data import (
    RealBatchProvider,
    build_real_model,
    real_eval_manifest,
)
from runtime.tcm_eval import TCMTrustedEvaluator
from runtime.trainer import Trainer, TrainerConfig
from runtime.worker import _score_real

pytestmark = pytest.mark.synthetic

DATASET_SHA = "b" * 64


def fake_records(n=30, seed=7, all_unlabeled=False):
    rng = random.Random(seed)
    records = []
    for i in range(1, n + 1):
        if all_unlabeled:
            scores = [None] * 9
        else:
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


def manifest_file(tmp_path, task_type="derived_multilabel_classification", **kw):
    m = build_real_manifest(fake_records(**kw), dataset_sha256=DATASET_SHA,
                            task_type=task_type)
    path = tmp_path / "manifest.json"
    m.save(path)
    return m.data, path


def test_provider_is_deterministic(tmp_path):
    m, _ = manifest_file(tmp_path)
    records = fake_records()
    a = list(RealBatchProvider(m, records, "train", batch_size=8))
    b = list(RealBatchProvider(m, records, "train", batch_size=8))
    assert len(a) == len(b)
    for ia, ib in zip(a, b, strict=True):
        assert ia.subject_ids == ib.subject_ids
        for name in ia.batch.features:
            assert torch.equal(ia.batch.features[name], ib.batch.features[name])
            assert torch.equal(ia.batch.availability[name], ib.batch.availability[name])


def test_nan_row_is_availability_masked_and_zero_filled(tmp_path):
    m, _ = manifest_file(tmp_path)
    records = fake_records()
    train_id = m["split"]["train"][0]
    train_row = m["subject_row_by_id"][train_id]
    records[train_row - 1].features["meridian"][1] = float("nan")
    provider = RealBatchProvider(m, records, "train", batch_size=32)
    item = next(iter(provider))
    # train batches are epoch-shuffled: locate the row by subject id
    position = item.subject_ids.index(train_id)
    assert bool(item.batch.availability["exam_features"][position]) is False
    assert torch.count_nonzero(item.batch.features["exam_features"][position]) == 0
    other = next(i for i, s in enumerate(item.subject_ids) if s != train_id)
    assert bool(item.batch.availability["exam_features"][other]) is True


def test_classification_targets_carry_category_labels_with_balanced_masked(tmp_path):
    m, _ = manifest_file(tmp_path)
    provider = RealBatchProvider(m, fake_records(), "dev", batch_size=64)
    batch = next(iter(provider)).batch
    assert "category_labels" in batch.targets
    assert not bool(batch.target_mask["category_labels"][:, 0].any())
    assert bool(batch.target_mask["category_labels"][:, 1:].all())


def test_regression_targets_carry_constitution_scores(tmp_path):
    m, _ = manifest_file(tmp_path, task_type="score_regression")
    provider = RealBatchProvider(m, fake_records(), "dev", batch_size=64)
    batch = next(iter(provider)).batch
    assert "constitution_scores" in batch.targets
    assert "category_labels" not in batch.targets
    assert not bool(batch.target_mask["constitution_scores"][:, 0].any())


def test_real_model_trains_and_saves_non_smoke_checkpoint(tmp_path):
    m, _ = manifest_file(tmp_path, task_type="score_regression")
    model = build_real_model(m, {"include_category_head": False,
                                 "include_score_head": True})
    trainer = Trainer(model, TrainerConfig(seed=0, epochs=1, learning_rate=3e-3))
    trainer.fit(
        RealBatchProvider(m, fake_records(), "train", batch_size=16),
        RealBatchProvider(m, fake_records(), "dev", batch_size=16),
    )
    artifact = trainer.save_checkpoint(
        tmp_path / "artifact", smoke_only=False,
        extra_state={"manifest_sha256": m["manifest_sha256"]},
    )
    loaded, state, metadata = load_training_checkpoint(artifact)
    assert metadata["model_type"] == "tcm_constitution_v1"
    assert state["smoke_only"] is False
    out = loaded(next(iter(RealBatchProvider(m, fake_records(), "dev"))).batch)
    assert "constitution_scores" in out.predictions


def test_real_eval_manifest_mirrors_frozen_split(tmp_path):
    m, _ = manifest_file(tmp_path)
    em = real_eval_manifest(m)
    assert em.version.startswith("real-")
    assert em.input_spec_version == "tcm-input-real-v1"
    assert em.label_tier == "RULE"
    assert em.dev_subjects == tuple(m["split"]["dev"])


def test_evaluator_fails_closed_when_no_label_is_observable(tmp_path):
    m, _ = manifest_file(tmp_path, all_unlabeled=True)
    model = build_real_model(m, {})
    provider = RealBatchProvider(m, fake_records(all_unlabeled=True), "dev")
    report = TCMTrustedEvaluator(real_eval_manifest(m)).evaluate(
        model, provider, partition="dev"
    )
    assert report.n_labeled == 0
    assert report.subject_mae is None
    assert report.subject_rmse is None
    assert any("NOT_COMPUTABLE" in entry for entry in report.not_computable)


def test_score_real_refuses_smoke_only_checkpoint(tmp_path):
    from core.models import default_tcm_model

    model, _ = default_tcm_model(hidden_dim=8, fusion_dim=8)
    artifact = save_training_checkpoint(model, {}, tmp_path / "smoke", smoke_only=True)
    m, path = manifest_file(tmp_path)
    with pytest.raises(ValueError, match="SMOKE_ONLY"):
        _score_real(str(artifact), str(path), "unused.xls")


def test_score_real_refuses_foreign_manifest(tmp_path):
    m, path = manifest_file(tmp_path)
    model = build_real_model(m, {})
    state = {"epoch": 1, "global_step": 1, "best_epoch": None,
             "best_eval_loss": None, "optimizer": {}, "trainer_config": {},
             "torch_rng_state": torch.get_rng_state(),
             "manifest_sha256": "not-this-manifest"}
    artifact = save_training_checkpoint(model, state, tmp_path / "real",
                                        smoke_only=False)
    with pytest.raises(ValueError, match="different real-data manifest"):
        _score_real(str(artifact), str(path), "unused.xls")
