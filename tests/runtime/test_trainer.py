"""Trainer/Evaluator/Checkpoint contract tests on synthetic fixtures (#49)."""

import pytest
import torch

from core.models import default_physio_model, default_tcm_model
from runtime.batches import SyntheticBatchProvider, assert_subject_disjoint
from runtime.checkpoint import load_training_checkpoint
from runtime.evaluation import METRIC_VERSION
from runtime.trainer import Trainer, TrainerConfig

pytestmark = pytest.mark.synthetic

HIDDEN = 32


def tcm_trainer(**cfg) -> Trainer:
    torch.manual_seed(0)
    model, _ = default_tcm_model(hidden_dim=HIDDEN, fusion_dim=HIDDEN)
    return Trainer(model, TrainerConfig(**cfg))


def physio_trainer(**cfg) -> Trainer:
    torch.manual_seed(1)
    model, _ = default_physio_model(hidden_dim=HIDDEN, fusion_dim=HIDDEN)
    return Trainer(model, TrainerConfig(**cfg))


def test_one_batch_fit_produces_finite_history_and_eval_metrics(tmp_path):
    trainer = tcm_trainer(epochs=2, seed=0)
    train = SyntheticBatchProvider("tcm", n_batches=2, batch_size=8, seed=0)
    eval_ = SyntheticBatchProvider(
        "tcm", n_batches=1, batch_size=8, seed=5, subject_prefix="eval"
    )
    report = trainer.fit(train, eval_)
    assert report.epochs_run == 2
    assert all(e["train_loss"] > 0 for e in report.history)
    metrics = report.history[-1]["eval"]
    assert metrics["metric_version"] == METRIC_VERSION
    assert metrics["n_batches"] == 1
    assert metrics["eval_loss"] > 0
    assert metrics["category_probs/n"] > 0


def test_same_seed_is_reproducible_on_cpu():
    def run():
        trainer = tcm_trainer(epochs=2, seed=3)
        provider = SyntheticBatchProvider("tcm", n_batches=2, batch_size=8, seed=3)
        report = trainer.fit(provider)
        return [e["train_loss"] for e in report.history]

    assert run() == run()


def test_subject_disjoint_split_violation_is_rejected():
    a = SyntheticBatchProvider("tcm", seed=0)
    b = SyntheticBatchProvider("tcm", seed=0)  # same seed -> same subject prefix
    with pytest.raises(ValueError, match="subject-level split violation"):
        assert_subject_disjoint(a, b)
    c = SyntheticBatchProvider("tcm", seed=0, subject_prefix="other")
    assert_subject_disjoint(a, c)  # must not raise


def test_early_stop_requires_eval_provider():
    trainer = tcm_trainer(early_stop_patience=1)
    with pytest.raises(ValueError, match="early_stop_patience requires"):
        trainer.fit(SyntheticBatchProvider("tcm"))


def test_physio_smoke_fit():
    trainer = physio_trainer(epochs=1, seed=2)
    report = trainer.fit(SyntheticBatchProvider("physio", n_batches=2, seed=2))
    assert report.epochs_run == 1


def test_fit_fails_loudly_on_missing_targets():
    trainer = tcm_trainer(epochs=1)
    provider = SyntheticBatchProvider("tcm", with_targets=False)
    with pytest.raises(ValueError, match="targets"):
        trainer.fit(provider)


class _DivergingStub(torch.nn.Module):
    """compute_loss scales a parameter-dependent term by a per-call counter:
    the counter grows linearly with steps while Adam's step size is
    scale-invariant, so eval loss provably rises after epoch 0 and
    best_epoch == 0 by construction — restore is actually exercised."""

    model_type = "stub_diverging"

    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(4, 2)
        self._calls = 0

    @property
    def model_config(self) -> dict:
        return {"stub": "diverging"}

    def forward(self, batch):
        from core.batch import ModelOutput

        return ModelOutput(predictions={})

    def compute_loss(self, batch):
        self._calls += 1
        probe = self.lin(torch.ones(4))
        return float(self._calls) * (probe**2).mean(), {}


def test_fit_restore_semantics():
    def run(epochs: int, **cfg) -> Trainer:
        torch.manual_seed(0)  # seed BEFORE stub construction: identical init
        trainer = Trainer(_DivergingStub(), TrainerConfig(seed=0, epochs=epochs, **cfg))
        trainer.fit(
            SyntheticBatchProvider("tcm", n_batches=2, batch_size=8, seed=0),
            SyntheticBatchProvider(
                "tcm", n_batches=1, batch_size=8, seed=5, subject_prefix="eval"
            ),
        )
        return trainer

    one = run(1)
    # default: LAST-epoch weights, declared in provenance (measured on real
    # data: best-eval-loss restore picks underfit base-rate epochs)
    three = run(3)
    assert three.best_epoch == 0
    assert any(
        not torch.equal(pa, pb)
        for pa, pb in zip(one.model.parameters(), three.model.parameters(), strict=True)
    ), "default weights are not last-epoch"
    # opt-in: restore_best selects the best epoch's weights
    three_best = run(3, restore_best=True)
    assert three_best.best_epoch == 0
    for pa, pb in zip(one.model.parameters(), three_best.model.parameters(), strict=True):
        assert torch.equal(pa, pb), "restore_best did not restore best-epoch weights"


def test_checkpoint_resume_consistency(tmp_path):
    def provider():
        return SyntheticBatchProvider("tcm", n_batches=3, batch_size=8, seed=6)

    trainer_a = tcm_trainer(epochs=4, seed=6)
    report_a = trainer_a.fit(provider())
    final_loss_a = report_a.history[-1]["train_loss"]

    # split run: 2 epochs -> checkpoint -> resume to 4 must reproduce the loss
    first_half = tcm_trainer(epochs=2, seed=6)
    first_half.fit(provider())
    ckpt = first_half.save_checkpoint(tmp_path / "half")
    state = torch.load(ckpt / "training_state.pt", weights_only=False)
    assert state["smoke_only"] is True
    assert state["epoch"] == 2

    loaded, resume = Trainer.load(ckpt, epochs=4)
    assert resume["epoch"] == 2
    report_b = loaded.fit(provider(), resume_state=resume)
    assert report_b.epochs_run == 2  # epochs 2 and 3 only
    assert report_b.history[-1]["train_loss"] == pytest.approx(final_loss_a, rel=1e-5)


def test_tampered_smoke_marker_is_refused(tmp_path):
    trainer = tcm_trainer(epochs=1)
    trainer.fit(SyntheticBatchProvider("tcm", n_batches=1))
    ckpt = trainer.save_checkpoint(tmp_path / "ckpt")
    state_file = ckpt / "training_state.pt"
    state = torch.load(state_file, weights_only=False)
    state["smoke_only"] = False
    torch.save(state, state_file)
    with pytest.raises(ValueError, match="SMOKE_ONLY"):
        load_training_checkpoint(ckpt)


def test_checkpoint_summary_fields(tmp_path):
    trainer = tcm_trainer(epochs=1, seed=4)
    trainer.fit(SyntheticBatchProvider("tcm", n_batches=1))
    ckpt = trainer.save_checkpoint(tmp_path / "ckpt", source_commit="test-sha")
    from runtime.checkpoint import checkpoint_summary

    summary = checkpoint_summary(ckpt)
    assert summary["model_type"] == "tcm_constitution_v1"
    assert summary["smoke_only"] is True
    assert summary["seed"] == 4
    assert len(summary["weights_sha256"]) == 64
