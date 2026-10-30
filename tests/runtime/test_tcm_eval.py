"""TCM trusted evaluator + inference API tests on synthetic tensors (#40)."""

import math

import pytest
import torch

from core.models import default_tcm_model
from runtime.batches import SyntheticBatchProvider
from runtime.checkpoint import load_training_checkpoint
from runtime.tcm_eval import (
    PRIMARY_METRIC,
    EvalManifest,
    TCMTrustedEvaluator,
    compute_per_dim_stats,
    compute_subject_level_metrics,
    make_synthetic_tcm_manifest,
)
from runtime.tcm_infer import TCMPredictor
from runtime.trainer import Trainer, TrainerConfig

pytestmark = pytest.mark.synthetic

NUM_SCORES = 9


def tcm_batch_like(batch_size=4):
    """Local synthetic batch factory (tests/runtime has no shared conftest)."""
    from core.batch import ModelBatch

    g = torch.Generator().manual_seed(11)
    labels = torch.randint(0, 2, (batch_size, NUM_SCORES), generator=g).float()
    return ModelBatch(
        features={
            "body_features": torch.rand(batch_size, 12, generator=g),
            "physio_features": torch.rand(batch_size, 7, generator=g),
        },
        targets={"category_labels": labels},
        target_mask={"category_labels": torch.ones(batch_size, NUM_SCORES, dtype=torch.bool)},
    )


@pytest.fixture()
def trained_artifact(tmp_path):
    torch.manual_seed(0)
    model, _ = default_tcm_model(hidden_dim=16, fusion_dim=16)
    trainer = Trainer(model, TrainerConfig(epochs=1, seed=0))
    trainer.fit(SyntheticBatchProvider("tcm", n_batches=2, batch_size=8, seed=0))
    return trainer.save_checkpoint(tmp_path / "artifact")


def manifest() -> EvalManifest:
    train = SyntheticBatchProvider("tcm", n_batches=2, seed=0)
    dev = SyntheticBatchProvider("tcm", n_batches=2, seed=1,
                                 subject_prefix="dev")
    holdout = SyntheticBatchProvider("tcm", n_batches=2, seed=2,
                                     subject_prefix="holdout")
    return make_synthetic_tcm_manifest(train, dev, holdout)


def test_subject_level_metrics_hand_computed():
    pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.tensor([[2.0, 2.0], [3.0, 6.0]])
    mask = torch.tensor([[True, True], [True, False]])
    # subject a: diffs (-1, 0) -> abs mean 0.5, sq mean 0.5
    # subject b: only dim 0 observed, diff 0 -> abs 0, sq 0
    stats = compute_subject_level_metrics(
        pred, target, mask, ["a", "b"]
    )
    assert stats["n_labeled"] == 2
    assert stats["subject_mae"] == pytest.approx((0.5 + 0.0) / 2)
    assert stats["subject_rmse"] == pytest.approx(math.sqrt((0.5 + 0.0) / 2))


def test_subject_metrics_exclude_unlabeled_subjects():
    pred = torch.tensor([[5.0]])
    target = torch.tensor([[1.0]])
    mask = torch.tensor([[False]])
    stats = compute_subject_level_metrics(pred, target, mask, ["a"])
    # fail-closed: nothing observable must never look like a perfect score
    assert stats["n_labeled"] == 0 and stats["subject_mae"] is None


def test_per_dim_stats_count_only_valid_labels():
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    mask = torch.tensor([[True, False], [True, True]])
    stats = compute_per_dim_stats(target, mask)
    assert stats[0]["n"] == 2 and stats[0]["missing"] == 0
    assert stats[0]["positive_rate"] == pytest.approx(0.5)
    assert stats[1]["n"] == 1 and stats[1]["missing"] == 1


def test_evaluator_reports_smoke_only_and_frozen_fields():
    evaluator = TCMTrustedEvaluator(manifest())
    model, _ = default_tcm_model(hidden_dim=16, fusion_dim=16)
    dev = SyntheticBatchProvider("tcm", n_batches=2, seed=1,
                                 subject_prefix="dev")
    report = evaluator.evaluate(model, dev, partition="dev")
    assert report.smoke_only is True
    assert report.primary_metric == PRIMARY_METRIC
    assert report.n_subjects == 16
    assert report.n_labeled == 16
    assert report.per_dim[0]["missing"] >= 0
    assert report.not_computable
    from dataclasses import FrozenInstanceError

    with pytest.raises(FrozenInstanceError):
        report.subject_mae = 0.0  # frozen dataclass: read-only gate


def test_evaluator_rejects_subject_crossing():
    evaluator = TCMTrustedEvaluator(manifest())
    model, _ = default_tcm_model(hidden_dim=16, fusion_dim=16)
    # train-namespace subjects offered as dev -> crossing must fail
    train = SyntheticBatchProvider("tcm", n_batches=2, seed=0)
    with pytest.raises(ValueError, match="sample crossing"):
        evaluator.evaluate(model, train, partition="dev")
    with pytest.raises(ValueError, match="unknown partition"):
        evaluator.evaluate(model, train, partition="test")


def test_evaluator_rejects_spec_version_mismatch():
    evaluator = TCMTrustedEvaluator(manifest())
    model, _ = default_tcm_model(hidden_dim=16, fusion_dim=16)
    model.input_spec = type(model.input_spec)(
        version="tcm-input-v999", modalities=model.input_spec.modalities
    )
    dev = SyntheticBatchProvider("tcm", n_batches=2, seed=1,
                                 subject_prefix="dev")
    with pytest.raises(ValueError, match="version mismatch"):
        evaluator.evaluate(model, dev, partition="dev")


def test_evaluator_rejects_unknown_prediction_keys():
    evaluator = TCMTrustedEvaluator(manifest())

    class StubModel(torch.nn.Module):
        model_type = "stub"

        def __init__(self):
            super().__init__()
            self.input_spec = default_tcm_model(
                hidden_dim=8, fusion_dim=8
            )[0].input_spec

        def forward(self, batch):
            from core.batch import ModelOutput

            return ModelOutput(
                predictions={
                    "category_probs": torch.zeros(batch.batch_size, NUM_SCORES),
                    "bogus_key": torch.zeros(batch.batch_size, 1),
                }
            )

    dev = SyntheticBatchProvider("tcm", n_batches=1, seed=1,
                                 subject_prefix="dev")
    with pytest.raises(ValueError, match="unknown prediction keys"):
        evaluator.evaluate(StubModel(), dev, partition="dev")


def test_manifest_rejects_leakage():
    with pytest.raises(ValueError, match="leakage"):
        EvalManifest(
            version="bad",
            input_spec_version="tcm-input-v0",
            standard_version="S",
            label_tier="SMOKE_ONLY",
            train_subjects=("a", "b"),
            dev_subjects=("b", "c"),
            holdout_subjects=("d",),
        )


def test_predictor_reload_consistency(trained_artifact):
    predictor = TCMPredictor(trained_artifact)
    model, _, _ = load_training_checkpoint(trained_artifact)
    batch = tcm_batch_like(batch_size=4)
    direct = model(batch).predictions["category_probs"]
    result = predictor.predict_batch(batch)
    assert result["status"] == "OK" and result["n_abstained"] == 0
    for i in range(4):
        assert result["category_probs"][i] == pytest.approx(
            direct[i].tolist(), rel=1e-6
        )


def test_predictor_single_and_abstain(trained_artifact):
    predictor = TCMPredictor(trained_artifact)
    feats = {"body_features": torch.rand(12).tolist()}
    result = predictor.predict_single(feats)
    assert result["status"] == "OK"
    assert len(result["category_probs"][0]) == NUM_SCORES
    assert result["schema_version"] == "tcm-inference-v0"

    blocked = predictor.predict_single(
        feats, availability={"body_features": [False]}
    )
    assert blocked["status"] == "ABSTAIN" and blocked["category_probs"] is None
    assert predictor.predict_single({})["status"] == "ABSTAIN"


def test_predictor_missing_and_unknown_fields(trained_artifact):
    predictor = TCMPredictor(trained_artifact)
    with pytest.raises(KeyError, match="unknown feature keys"):
        predictor.predict_single({"eeg_during": [0.1] * 9})


def test_predictor_rejects_non_tcm_or_bad_format(tmp_path, trained_artifact):
    import json

    metadata_path = trained_artifact / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["format_version"] = "999"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="format_version"):
        TCMPredictor(trained_artifact)


def test_predictor_reports_real_marker_from_training_state(tmp_path):
    from runtime.checkpoint import save_training_checkpoint

    torch.manual_seed(0)
    model, _ = default_tcm_model(hidden_dim=8, fusion_dim=8)
    artifact = save_training_checkpoint(
        model,
        {"epoch": 1, "global_step": 1, "best_epoch": None,
         "best_eval_loss": None, "optimizer": {}, "trainer_config": {},
         "torch_rng_state": torch.get_rng_state(),
         "manifest_sha256": "c" * 64},
        tmp_path / "real", smoke_only=False,
    )
    predictor = TCMPredictor(artifact)
    assert predictor.smoke_only is False
    feats = {"body_features": torch.rand(12).tolist()}
    assert predictor.predict_single(feats)["smoke_only"] is False

    state = torch.load(artifact / "training_state.pt", weights_only=False)
    state.pop("smoke_only")
    torch.save(state, artifact / "training_state.pt")
    with pytest.raises(ValueError, match="smoke_only marker"):
        TCMPredictor(artifact)
