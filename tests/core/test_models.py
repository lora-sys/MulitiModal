"""Model forward/backward/mask behavior on synthetic tensors (#48/#39)."""

import pytest
import torch
from conftest import HIDDEN, NUM_SCORES, physio_batch, tcm_batch

from core.losses import (
    masked_mae_loss,
    masked_mse_loss,
    masked_multilabel_bce_loss,
    masked_softmax_ce_loss,
)
from core.models import default_physio_model, default_tcm_model

pytestmark = pytest.mark.synthetic


def build_tcm(**overrides):
    torch.manual_seed(7)
    model, config = default_tcm_model(hidden_dim=HIDDEN, fusion_dim=HIDDEN, **overrides)
    return model, config


def build_physio(**overrides):
    torch.manual_seed(8)
    model, config = default_physio_model(hidden_dim=HIDDEN, fusion_dim=HIDDEN, **overrides)
    return model, config


def test_tcm_forward_probs_and_logits_shapes():
    model, config = build_tcm()
    assert config["include_category_head"] is True
    for n in (1, 8):
        out = model(tcm_batch(n))
        probs = out.predictions["category_probs"]
        assert probs.shape == (n, NUM_SCORES)
        assert bool(((probs >= 0) & (probs <= 1)).all())
        assert out.auxiliary["category_logits"].shape == (n, NUM_SCORES)


def test_tcm_score_head_disabled_by_default():
    model, config = build_tcm()
    assert config["include_score_head"] is False
    assert model.score_head is None
    out = model(tcm_batch())
    assert "constitution_scores" not in out.predictions


def test_tcm_score_head_opt_in_range_and_loss():
    model, _ = build_tcm(include_score_head=True)
    assert [t.name for t in model.target_specs] == ["category_labels", "constitution_scores"]
    out = model(tcm_batch(with_score_targets=True))
    scores = out.predictions["constitution_scores"]
    assert scores.shape == (8, NUM_SCORES)
    assert bool(((scores >= 0) & (scores <= 100)).all())
    out.check_ranges({"constitution_scores": (0.0, 100.0)})
    _, parts = model.compute_loss(tcm_batch(with_score_targets=True))
    assert set(parts) == {"category_bce", "score_mse"}


def test_tcm_exclusive_mode_uses_softmax_and_ce():
    model, config = build_tcm(category_exclusive=True)
    assert config["category_exclusive"] is True
    probs = model(tcm_batch()).predictions["category_probs"]
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(8), rtol=1e-5, atol=1e-5)
    _, parts = model.compute_loss(tcm_batch())
    assert set(parts) == {"category_ce"}


def test_tcm_missing_modality_and_zero_availability_stay_finite():
    model, _ = build_tcm()
    dropped = model(tcm_batch(drop_modality="physio_features"))
    assert torch.isfinite(dropped.predictions["category_probs"]).all()
    masked = model(tcm_batch(zero_out_availability="body_features"))
    assert torch.isfinite(masked.predictions["category_probs"]).all()


def test_tcm_backward_produces_grads():
    model, _ = build_tcm()
    loss, parts = model.compute_loss(tcm_batch())
    assert torch.isfinite(loss)
    assert set(parts) == {"category_bce"}
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_tcm_rejects_nan_input():
    model, _ = build_tcm()
    with pytest.raises(ValueError, match="NaN"):
        model(tcm_batch(inject_nan=True))


def test_tcm_loss_requires_targets():
    model, _ = build_tcm()
    with pytest.raises(ValueError, match="targets"):
        model.compute_loss(tcm_batch(with_targets=False))


def test_tcm_requires_at_least_one_head():
    with pytest.raises(ValueError, match="at least one output head"):
        build_tcm(include_category_head=False, include_score_head=False)


def test_tcm_exclusive_requires_category_head():
    with pytest.raises(ValueError, match="category_exclusive requires"):
        build_tcm(include_category_head=False, category_exclusive=True)


def test_tcm_synthetic_train_save_load_predict_smoke(tmp_path):
    """#39 acceptance: train/save/load/predict round trip on contract data."""
    from core.registry import batch_predict, load_model, save_model

    model, config = build_tcm()
    batch = tcm_batch(batch_size=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    first_loss = None
    for _ in range(5):
        optimizer.zero_grad()
        loss, _ = model.compute_loss(batch)
        if first_loss is None:
            first_loss = loss.item()
        loss.backward()
        optimizer.step()
    assert first_loss is not None and first_loss > 0
    assert torch.isfinite(loss)

    out_dir = save_model(model, config, tmp_path / "tcm_smoke")
    loaded, metadata = load_model(out_dir)
    before = batch_predict(model, batch)
    after = batch_predict(loaded, batch)
    torch.testing.assert_close(
        before.predictions["category_probs"], after.predictions["category_probs"]
    )
    assert metadata["model_type"] == "tcm_constitution_v1"


def test_physio_forward_and_aux_range():
    model, config = build_physio()
    assert config["include_bpsi_aux"] is True
    out = model(physio_batch())
    assert out.predictions["efri"].shape == (8, 1)
    assert out.auxiliary["bpsi"].shape == (8, 1)
    assert bool(((out.auxiliary["bpsi"] >= 0) & (out.auxiliary["bpsi"] <= 100)).all())


def test_physio_rejects_during_post_features():
    model, _ = build_physio()
    with pytest.raises(ValueError, match="not an approved input"):
        model(physio_batch(extra_during_post=True))


def test_physio_backward_with_masked_aux():
    model, _ = build_physio()
    loss, parts = model.compute_loss(physio_batch())
    assert torch.isfinite(loss)
    assert set(parts) == {"efri_mse", "bpsi_aux_mae"}
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_physio_bpsi_aux_can_be_disabled():
    model, _ = build_physio(include_bpsi_aux=False)
    out = model(physio_batch(with_targets=False))
    assert "bpsi" not in out.auxiliary


def test_masked_losses_count_only_observed_entries():
    pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    mask = torch.tensor([[True, False], [False, False]])
    # only the (0,0) entry is observed: (1-10)^2 = 81
    assert masked_mse_loss(pred, target, mask).item() == pytest.approx(81.0)
    assert masked_mae_loss(pred, target, mask).item() == pytest.approx(9.0)

    logits = torch.tensor([[2.0], [-3.0]])
    labels = torch.tensor([[1.0], [0.0]])
    full = torch.tensor([[True], [True]])
    partial = torch.tensor([[True], [False]])
    loss_full = masked_multilabel_bce_loss(logits, labels, full)
    loss_partial = masked_multilabel_bce_loss(logits, labels, partial)
    assert loss_full.item() > 0 and loss_partial.item() > 0
    assert loss_full.item() != loss_partial.item()  # unobserved row not counted


def test_all_masked_loss_is_connected_zero():
    pred = torch.tensor([[1.0, 2.0]], requires_grad=True)
    target = torch.zeros(1, 2)
    mask = torch.zeros(1, 2, dtype=torch.bool)
    loss = masked_mse_loss(pred, target, mask)
    assert loss.item() == 0.0
    loss.backward()  # graph stays connected: no NaN, no crash


def test_masked_softmax_ce_counts_only_observed_rows():
    logits = torch.tensor([[2.0, 1.0], [0.5, -0.5]], requires_grad=True)
    onehot = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    mask = torch.tensor([[True, True], [False, False]])
    loss = masked_softmax_ce_loss(logits, onehot, mask)
    expected = -(2.0 - torch.logsumexp(logits[0], dim=0))
    assert loss.item() == pytest.approx(expected.item(), rel=1e-5)

    empty = torch.zeros(2, 2, dtype=torch.bool)
    zero_loss = masked_softmax_ce_loss(logits, onehot, empty)
    assert zero_loss.item() == 0.0
    zero_loss.backward()
