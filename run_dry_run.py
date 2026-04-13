#!/usr/bin/env python3
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.encoders import get_dynamic_encoder
from src.models.fusion import BaselineSignalRegressor, DualGatingModel


ENCODERS = ["inceptiontime", "os-cnn", "xcm", "1d-resnet", "tcn"]


def _assert_no_nan_grad(model: nn.Module) -> None:
    for name, p in model.named_parameters():
        if p.grad is None:
            raise AssertionError(f"Gradient is None for parameter: {name}")
        if torch.isnan(p.grad).any():
            raise AssertionError(f"NaN gradient detected: {name}")


def _single_step(model: nn.Module, pred: torch.Tensor, target: torch.Tensor) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    loss = loss_fn(pred, target)
    loss.backward()
    _assert_no_nan_grad(model)
    optimizer.step()


def run_dry_run() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dynamic = torch.randn(4, 2, 1000, device=device)
    static = torch.randn(4, 8, device=device)
    target = torch.randn(4, 1, device=device)

    for encoder_name in ENCODERS:
        # 1) Encoder instantiation + forward shape check
        enc = get_dynamic_encoder(encoder_name).to(device)
        z = enc(dynamic)
        assert z.shape == (4, 128), f"{encoder_name} encoder output shape mismatch: {z.shape}"

        # 2) Baseline model checks
        baseline = BaselineSignalRegressor(encoder_name=encoder_name).to(device)
        baseline.train()
        pred_base = baseline(dynamic, static)
        assert pred_base.shape == (4, 1), f"Baseline pred shape mismatch: {pred_base.shape}"
        _single_step(baseline, pred_base, target)

        # 3) Full DualGating model checks
        model = DualGatingModel(
            encoder_name=encoder_name,
            freeze_tcm=False,
            use_gate_a=True,
            use_gate_b=True,
            use_tcm_encoder=True,
        ).to(device)
        model.train()
        pred = model(dynamic, static)
        assert pred.shape == (4, 1), f"DualGating pred shape mismatch: {pred.shape}"

        static_embed, probs = model.tcm_encoder(static)
        assert static_embed.shape == (4, 128), f"TCM static embedding shape mismatch: {static_embed.shape}"
        assert probs.shape == (4, 9), f"TCM probs shape mismatch: {probs.shape}"
        assert torch.allclose(
            probs.sum(dim=1),
            torch.ones(4, device=device),
            atol=1e-4,
        ), "TCM probs do not sum to 1.0"

        _single_step(model, pred, target)

    print("Dry Run Passed: all models/encoders forward-backward-update are healthy.")


if __name__ == "__main__":
    try:
        run_dry_run()
    except Exception as exc:
        raise RuntimeError("Dry Run Failed! 请勿启动正式训练！") from exc

