#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.config import (
    EARLY_STOPPING_PATIENCE,
    MAX_EPOCHS,
    Paths,
    TCM_CHECKPOINT_PATH,
    TCM_SCALER_PATH,
    TrainConfig,
    ensure_dirs,
    override_from_env,
    resolve_device,
)
from src.data_loader import WESADDataset, make_train_val_loaders
from src.models.fusion import DualGatingModel
from src.training import fit_regression_model_with_history, save_checkpoint
from src.utils import load_json, save_json, set_seed, timestamp
from src.utils.plotting import plot_ablation, plot_comparison, plot_selection


ENCODERS = ["inceptiontime", "os-cnn", "xcm", "1d-resnet", "tcn"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run 9-step WESAD experiment matrix")
    p.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--wesad-dir", type=str, default=None)
    p.add_argument("--tcm-checkpoint", type=str, default=str(TCM_CHECKPOINT_PATH))
    p.add_argument("--tcm-scaler", type=str, default=str(TCM_SCALER_PATH))
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def run_single_experiment(name: str, model, train_loader, val_loader, cfg: TrainConfig, result_dir: Path):
    print(f"[{timestamp()}] >>> run_single_experiment: {name} (epochs={cfg.epochs})")
    model, train_metrics, val_metrics, val_mse_history, best_epoch = fit_regression_model_with_history(
        model,
        train_loader,
        val_loader,
        epochs=cfg.epochs,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        device=cfg.device,
        patience=EARLY_STOPPING_PATIENCE,
    )
    print(f"[{timestamp()}] >>> run_single_experiment completed: {name}")
    return model, val_metrics, val_mse_history, best_epoch
    payload = {
        "name": name,
        "train": train_metrics,
        "val": val_metrics,
        "val_mse_history": val_mse_history,
        "recommended_best_epoch": best_epoch,
    }
    save_json(payload, result_dir / "metrics.json")
    save_checkpoint(
        model,
        result_dir / "model_best.pth",
        extra={
            "name": name,
            "val_metrics": val_metrics,
            "recommended_best_epoch": best_epoch,
            "val_mse_history": val_mse_history,
        },
    )
    print(
        f"[{timestamp()}] {name} | "
        f"MSE={val_metrics['mse']:.6f} RMSE={val_metrics['rmse']:.6f} "
        f"MAE={val_metrics['mae']:.6f} Pearson={val_metrics['pearson']:.6f} "
        f"[推荐最佳 Epoch: {best_epoch}]",
        flush=True,
    )
    return model, val_metrics, val_mse_history, best_epoch


def instantiate_model(encoder: str, args: argparse.Namespace, *, use_tcm: bool, use_gate_a: bool, use_gate_b: bool):
    print(f"[{timestamp()}] >>> instantiate_model: encoder={encoder}, use_tcm={use_tcm}, gate_a={use_gate_a}, gate_b={use_gate_b}")
    print(f"[{timestamp()}] >>> Creating DualGatingModel...")
    model = DualGatingModel(
        encoder_name=encoder,
        tcm_checkpoint_path=Path(args.tcm_checkpoint),
        tcm_scaler_path=Path(args.tcm_scaler),
        freeze_tcm=True,
        use_tcm=use_tcm,
        use_gate_a=use_gate_a,
        use_gate_b=use_gate_b,
    )
    print(f"[{timestamp()} >>> DualGatingModel created successfully")
    return model


def run_step(
    *,
    step_name: str,
    result_dir: Path,
    encoder: str,
    use_tcm: bool,
    use_gate_a: bool,
    use_gate_b: bool,
    args: argparse.Namespace,
    cfg: TrainConfig,
    train_loader,
    val_loader,
):
    print(
        f"[{timestamp()}] >>> {step_name}: re-instantiate model "
        f"(encoder={encoder}, use_tcm={use_tcm}, gate_a={use_gate_a}, gate_b={use_gate_b})",
        flush=True,
    )
    print(f"[{timestamp()}] >>> Instantiating model...")
    model = instantiate_model(
        encoder=encoder,
        args=args,
        use_tcm=use_tcm,
        use_gate_a=use_gate_a,
        use_gate_b=use_gate_b,
    )
    print(f"[{timestamp()}] >>> Model instantiated, starting training...")
    model, metrics, val_mse_history, best_epoch = run_single_experiment(
        step_name, model, train_loader, val_loader, cfg, result_dir
    )
    return model, metrics, val_mse_history, best_epoch


def main() -> None:
    args = parse_args()
    paths = override_from_env(Paths())
    if args.wesad_dir:
        paths.wesad_dir = Path(args.wesad_dir)
    ensure_dirs(paths)

    print(f"[{timestamp()}] >>> Loading best parameters from Optuna...")
    best_params = load_json(paths.checkpoints / "optuna_best_params.json", default={})
    print(f"[{timestamp()}] >>> Best params: {best_params}")
    
    print(f"[{timestamp()}] >>> Creating TrainConfig with best parameters...")
    cfg = TrainConfig(
        batch_size=int(best_params.get("batch_size", 64)),
        lr=float(best_params.get("lr", 1e-3)),
        weight_decay=float(best_params.get("weight_decay", 1e-4)),
        epochs=3 if args.dry_run else args.epochs,
        device=resolve_device(args.device),
    )
    print(f"[{timestamp()}] >>> TrainConfig: batch_size={cfg.batch_size}, lr={cfg.lr}, epochs={cfg.epochs}, device={cfg.device}")
    set_seed(cfg.seed)

    print(f"[{timestamp()}] >>> Loading WESAD dataset from {paths.wesad_dir} ...", flush=True)
    dataset = WESADDataset(paths.wesad_dir, Path(args.tcm_scaler), window_size=cfg.window_size, overlap=cfg.window_overlap)
    print(f"[{timestamp()}] >>> WESAD loaded. total_windows={len(dataset)}", flush=True)
    train_loader, val_loader = make_train_val_loaders(dataset, batch_size=cfg.batch_size, seed=cfg.seed)
    print(
        f"[{timestamp()}] >>> DataLoader ready. train_batches={len(train_loader)} val_batches={len(val_loader)} "
        f"batch_size={cfg.batch_size} epochs={cfg.epochs}",
        flush=True,
    )

    detailed_logs = []
    experiment_logs = []  # Must end with exactly 9 step-level logs.

    # Step 1
    print(f"[{timestamp()}] >>> Starting Step 1: Baseline A (Weak)")
    model, metrics, hist, best_epoch = run_step(
        step_name="Step1-BaselineA-Weak",
        result_dir=paths.results / "step1_baseline_a_weak",
        encoder="tcn",
        use_tcm=False,
        use_gate_a=False,
        use_gate_b=False,
        args=args,
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    detailed_logs.append({"step": 1, "name": "Baseline A", "encoder": "tcn", "metrics": metrics, "val_mse_history": hist, "best_epoch": best_epoch})
    experiment_logs.append({"step": 1, "name": "Baseline A", "mse": metrics["mse"]})
    del model
    clear_cuda_cache()

    # Step 2
    model, metrics, hist, best_epoch = run_step(
        step_name="Step2-BaselineB-Strong",
        result_dir=paths.results / "step2_baseline_b_strong",
        encoder="tcn",
        use_tcm=False,
        use_gate_a=False,
        use_gate_b=True,
        args=args,
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    detailed_logs.append({"step": 2, "name": "Baseline B", "encoder": "tcn", "metrics": metrics, "val_mse_history": hist, "best_epoch": best_epoch})
    experiment_logs.append({"step": 2, "name": "Baseline B", "mse": metrics["mse"]})
    del model
    clear_cuda_cache()

    # Step 3
    model, metrics, hist, best_epoch = run_step(
        step_name="Step3-Ours-TCN",
        result_dir=paths.results / "step3_ours_tcn",
        encoder="tcn",
        use_tcm=True,
        use_gate_a=True,
        use_gate_b=True,
        args=args,
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    detailed_logs.append({"step": 3, "name": "Ours", "encoder": "tcn", "metrics": metrics, "val_mse_history": hist, "best_epoch": best_epoch})
    experiment_logs.append({"step": 3, "name": "Ours", "mse": metrics["mse"]})
    del model
    clear_cuda_cache()

    # Step 4 (encoder selection loop)
    encoder_selection_results = []
    best_encoder = None
    best_encoder_mse = float("inf")
    for enc in ENCODERS:
        model, metrics, hist, best_epoch = run_step(
            step_name=f"Step4-EncoderSelection-{enc}",
            result_dir=paths.results / f"step4_encoder_{enc}",
            encoder=enc,
            use_tcm=True,
            use_gate_a=True,
            use_gate_b=True,
            args=args,
            cfg=cfg,
            train_loader=train_loader,
            val_loader=val_loader,
        )
        encoder_selection_results.append({"name": enc, "encoder": enc, "metrics": metrics, "val_mse_history": hist, "best_epoch": best_epoch})
        if metrics["mse"] < best_encoder_mse:
            best_encoder_mse = metrics["mse"]
            best_encoder = enc
        del model
        clear_cuda_cache()

    assert best_encoder is not None
    print(f"[{timestamp()}] Step4 best encoder: {best_encoder} (MSE={best_encoder_mse:.6f})", flush=True)
    detailed_logs.append(
        {
            "step": 4,
            "name": "Encoder Selection",
            "best_encoder": best_encoder,
            "best_mse": best_encoder_mse,
            "encoders": encoder_selection_results,
        }
    )
    experiment_logs.append({"step": 4, "name": "Encoder Selection", "best_encoder": best_encoder, "best_mse": best_encoder_mse})

    # Step 5
    model, metrics, hist, best_epoch = run_step(
        step_name="Step5-Ablation1-woDualGating",
        result_dir=paths.results / "step5_ablation_wo_dual_gating",
        encoder=best_encoder,
        use_tcm=True,
        use_gate_a=False,
        use_gate_b=False,
        args=args,
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    detailed_logs.append({"step": 5, "name": "w/o Dual Gating", "encoder": best_encoder, "metrics": metrics, "val_mse_history": hist, "best_epoch": best_epoch})
    experiment_logs.append({"step": 5, "name": "w/o Dual Gating", "mse": metrics["mse"]})
    del model
    clear_cuda_cache()

    # Step 6
    model, metrics, hist, best_epoch = run_step(
        step_name="Step6-Ablation2-woTCMPrior",
        result_dir=paths.results / "step6_ablation_wo_tcm_prior",
        encoder=best_encoder,
        use_tcm=False,
        use_gate_a=False,
        use_gate_b=True,
        args=args,
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    detailed_logs.append({"step": 6, "name": "w/o TCM Prior", "encoder": best_encoder, "metrics": metrics, "val_mse_history": hist, "best_epoch": best_epoch})
    experiment_logs.append({"step": 6, "name": "w/o TCM Prior", "mse": metrics["mse"]})
    del model
    clear_cuda_cache()

    # Step 7
    model, metrics, hist, best_epoch = run_step(
        step_name="Step7-Ablation3-woTCMGate",
        result_dir=paths.results / "step7_ablation_wo_tcm_gate",
        encoder=best_encoder,
        use_tcm=True,
        use_gate_a=False,
        use_gate_b=True,
        args=args,
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    detailed_logs.append({"step": 7, "name": "w/o TCM_Gate", "encoder": best_encoder, "metrics": metrics, "val_mse_history": hist, "best_epoch": best_epoch})
    experiment_logs.append({"step": 7, "name": "w/o TCM_Gate", "mse": metrics["mse"]})
    del model
    clear_cuda_cache()

    # Step 8
    model, metrics, hist, best_epoch = run_step(
        step_name="Step8-FinalOurs",
        result_dir=paths.results / "step8_final_ours",
        encoder=best_encoder,
        use_tcm=True,
        use_gate_a=True,
        use_gate_b=True,
        args=args,
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    detailed_logs.append({"step": 8, "name": "Full Model", "encoder": best_encoder, "metrics": metrics, "val_mse_history": hist, "best_epoch": best_epoch})
    experiment_logs.append({"step": 8, "name": "Final Ours", "mse": metrics["mse"]})
    save_checkpoint(
        model,
        paths.checkpoints / "best_model.pth",
        extra={
            "best_encoder": best_encoder,
            "metrics": metrics,
            "use_tcm": True,
            "use_gate_a": True,
            "use_gate_b": True,
        },
    )
    del model
    clear_cuda_cache()

    # Step 9 plotting call
    comparison_data = [
        {"name": "Baseline A", "metrics": detailed_logs[0]["metrics"]},
        {"name": "Baseline B", "metrics": detailed_logs[1]["metrics"]},
        {"name": "Ours", "metrics": detailed_logs[2]["metrics"]},
    ]
    selection_data = [
        {"name": e["name"], "metrics": e["metrics"]}
        for e in detailed_logs[3]["encoders"]
    ]
    ablation_data = [
        {"name": "Full Model", "metrics": detailed_logs[7]["metrics"]},
        {"name": "w/o Dual Gating", "metrics": detailed_logs[4]["metrics"]},
        {"name": "w/o TCM Prior", "metrics": detailed_logs[5]["metrics"]},
        {"name": "w/o TCM_Gate", "metrics": detailed_logs[6]["metrics"]},
    ]
    fig1 = plot_comparison(comparison_data)
    fig2 = plot_selection(selection_data)
    fig3 = plot_ablation(ablation_data)
    print(f"[{timestamp()}] >>> Figures saved: {fig1}, {fig2}, {fig3}", flush=True)
    experiment_logs.append({"step": 9, "name": "Plotting", "figures": [str(fig1), str(fig2), str(fig3)]})

    assert len(experiment_logs) == 9, f"Expected 9 experiment steps, got {len(experiment_logs)}"

    save_json(
        {
            "best_encoder": best_encoder,
            "experiment_logs": experiment_logs,
            "detailed_logs": detailed_logs,
        },
        paths.results / "experiments_summary.json",
    )
    print(f"[{timestamp()}] >>> 9-step experiment matrix completed.", flush=True)


if __name__ == "__main__":
    main()
