#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import ROOT, Paths, TCM_CHECKPOINT_PATH, TCM_SCALER_PATH, ensure_dirs, override_from_env, resolve_device
from src.data_loader import WESADDataset, make_train_val_loaders
from src.models.fusion import DualGatingModel
from src.utils import timestamp


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grad-CAM visualization for WESAD samples")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--wesad-dir", type=str, default=None)
    p.add_argument("--tcm-checkpoint", type=str, default=str(TCM_CHECKPOINT_PATH))
    p.add_argument("--tcm-scaler", type=str, default=str(TCM_SCALER_PATH))
    return p.parse_args()


def _find_last_conv1d(module: torch.nn.Module) -> torch.nn.Module:
    candidates = [m for m in module.modules() if isinstance(m, torch.nn.Conv1d)]
    if not candidates:
        raise RuntimeError("No Conv1d layer found for Grad-CAM.")
    return candidates[-1]


def _compute_1d_gradcam(model: DualGatingModel, dynamic: torch.Tensor, static_8d: torch.Tensor, target_layer: torch.nn.Module):
    activations = {}
    gradients = {}

    def fwd_hook(_, __, output):
        activations["value"] = output

    def bwd_hook(_, grad_in, grad_out):
        _ = grad_in
        gradients["value"] = grad_out[0]

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)
    try:
        model.zero_grad(set_to_none=True)
        pred = model(dynamic, static_8d)
        pred.mean().backward()
    finally:
        h1.remove()
        h2.remove()

    if "value" not in activations or "value" not in gradients:
        raise RuntimeError("Failed to capture activations/gradients for Grad-CAM.")

    acts = activations["value"]  # [B, C, L]
    grads = gradients["value"]  # [B, C, L]
    weights = grads.mean(dim=2, keepdim=True)  # [B, C, 1]
    cam = torch.relu((weights * acts).sum(dim=1, keepdim=True))  # [B, 1, L]
    cam = torch.nn.functional.interpolate(cam, size=dynamic.shape[-1], mode="linear", align_corners=False)
    cam = cam.squeeze(0).squeeze(0)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam_2ch = cam.unsqueeze(0).repeat(2, 1)  # [2, 1000]
    return cam_2ch.detach().cpu().numpy()


def _pick_sample_indices(dataset, val_indices):
    stress_idx = None
    baseline_idx = None
    best_stress_dist = 1e9
    best_base_dist = 1e9

    for idx in val_indices:
        _, _, target = dataset[idx]
        v = float(target.item())
        d_stress = abs(v - 0.0)
        d_base = abs(v - 1.0)
        if d_stress < best_stress_dist:
            best_stress_dist = d_stress
            stress_idx = idx
        if d_base < best_base_dist:
            best_base_dist = d_base
            baseline_idx = idx

    if stress_idx is None or baseline_idx is None:
        raise RuntimeError("Cannot find stress/baseline samples in test split.")
    return stress_idx, baseline_idx


def _plot_wave_and_cam(dynamic_np: np.ndarray, cam_np: np.ndarray, title: str, out_path: Path):
    x = np.arange(dynamic_np.shape[-1])
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(x, dynamic_np[0], label="ECG", color="#1f77b4", linewidth=1.0)
    axes[0].plot(x, dynamic_np[1], label="EDA", color="#2ca02c", linewidth=1.0, alpha=0.8)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(f"{title} - Raw Signals")
    axes[0].legend(loc="upper right")

    heat = cam_np.mean(axis=0, keepdims=True)
    y_min = float(np.min(dynamic_np))
    y_max = float(np.max(dynamic_np))
    axes[1].plot(x, dynamic_np[0], color="black", linewidth=0.8, alpha=0.55, label="PPG-like focus trace")
    axes[1].imshow(
        heat,
        aspect="auto",
        cmap="coolwarm",
        extent=[0, dynamic_np.shape[-1] - 1, y_min, y_max],
        alpha=0.6,
        origin="lower",
    )
    axes[1].set_ylabel("Amplitude")
    axes[1].set_xlabel("Time Index")
    axes[1].set_title(f"{title} - Grad-CAM Heatmap Overlay")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    paths = override_from_env(Paths())
    if args.wesad_dir:
        paths.wesad_dir = Path(args.wesad_dir)
    ensure_dirs(paths)
    device = resolve_device(args.device)

    dataset = WESADDataset(paths.wesad_dir, Path(args.tcm_scaler))
    _, val_loader = make_train_val_loaders(dataset, batch_size=64, seed=42)
    val_indices = val_loader.dataset.indices
    stress_idx, baseline_idx = _pick_sample_indices(dataset, val_indices)

    ckpt = torch.load(paths.checkpoints / "best_model.pth", map_location="cpu", weights_only=True)
    best_encoder = ckpt.get("best_encoder", "tcn")
    model = DualGatingModel(
        encoder_name=best_encoder,
        tcm_checkpoint_path=Path(args.tcm_checkpoint),
        freeze_tcm=True,
        use_tcm=ckpt.get("use_tcm", True),
        use_gate_a=ckpt.get("use_gate_a", True),
        use_gate_b=ckpt.get("use_gate_b", True),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    target_layer = _find_last_conv1d(model.dynamic_encoder)

    for idx, fig_name, title in [
        (stress_idx, "fig4_gradcam_stress.png", "Stress Sample"),
        (baseline_idx, "fig5_gradcam_baseline.png", "Baseline Sample"),
    ]:
        dynamic, static_8d, target = dataset[idx]
        dynamic = dynamic.unsqueeze(0).to(device)
        static_8d = static_8d.unsqueeze(0).to(device)
        cam = _compute_1d_gradcam(model, dynamic, static_8d, target_layer)
        dyn_np = dynamic.squeeze(0).detach().cpu().numpy()
        out_path = ROOT / "figures" / fig_name
        _plot_wave_and_cam(dyn_np, cam, f"{title} (target={float(target.item()):.2f})", out_path)
        print(f"[{timestamp()}] saved: {out_path}")


if __name__ == "__main__":
    main()

