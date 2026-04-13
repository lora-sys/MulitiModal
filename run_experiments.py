#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.config import Paths, TCM_CHECKPOINT_PATH, TCM_SCALER_PATH, TrainConfig, ensure_dirs, override_from_env, resolve_device
from src.data_loader import WESADDataset, make_train_val_loaders
from src.models.fusion import BaselineSignalRegressor, DualGatingModel
from src.training import fit_regression_model, save_checkpoint
from src.utils import load_json, save_json, set_seed, timestamp


ENCODERS = ["inceptiontime", "os-cnn", "xcm", "1d-resnet", "tcn"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Exp1..Exp5 on WESAD")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--wesad-dir", type=str, default=None)
    p.add_argument("--tcm-checkpoint", type=str, default=str(TCM_CHECKPOINT_PATH))
    p.add_argument("--tcm-scaler", type=str, default=str(TCM_SCALER_PATH))
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def run_single_experiment(name: str, model, train_loader, val_loader, cfg: TrainConfig, result_dir: Path):
    model, train_metrics, val_metrics = fit_regression_model(
        model,
        train_loader,
        val_loader,
        epochs=cfg.epochs,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        device=cfg.device,
    )
    payload = {"name": name, "train": train_metrics, "val": val_metrics}
    save_json(payload, result_dir / "metrics.json")
    print(
        f"[{timestamp()}] {name} | "
        f"MSE={val_metrics['mse']:.6f} RMSE={val_metrics['rmse']:.6f} "
        f"MAE={val_metrics['mae']:.6f} Pearson={val_metrics['pearson']:.6f}"
    )
    return model, val_metrics


def clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    paths = override_from_env(Paths())
    if args.wesad_dir:
        paths.wesad_dir = Path(args.wesad_dir)
    ensure_dirs(paths)

    best_params = load_json(paths.checkpoints / "optuna_best_params.json", default={})
    cfg = TrainConfig(
        batch_size=int(best_params.get("batch_size", 64)),
        lr=float(best_params.get("lr", 1e-3)),
        weight_decay=float(best_params.get("weight_decay", 1e-4)),
        epochs=3 if args.dry_run else args.epochs,
        device=resolve_device(args.device),
    )
    set_seed(cfg.seed)

    dataset = WESADDataset(paths.wesad_dir, Path(args.tcm_scaler), window_size=cfg.window_size, overlap=cfg.window_overlap)
    train_loader, val_loader = make_train_val_loaders(dataset, batch_size=cfg.batch_size, seed=cfg.seed)

    # Exp1
    exp1_dir = paths.results / "exp1_baseline"
    model = BaselineSignalRegressor("inceptiontime")
    model, exp1_metrics = run_single_experiment("Exp1-Baseline", model, train_loader, val_loader, cfg, exp1_dir)
    del model
    clear_cuda_cache()

    # Exp2: encoder selection
    best_encoder = None
    best_encoder_mse = float("inf")
    for enc in ENCODERS:
        print(f"[{timestamp()}] >>> Exp2 instantiate model encoder={enc}")
        model = DualGatingModel(
            encoder_name=enc,
            tcm_checkpoint_path=Path(args.tcm_checkpoint),
            freeze_tcm=True,
            use_gate_a=True,
            use_gate_b=True,
            use_tcm_encoder=True,
        )
        _, metrics = run_single_experiment(
            f"Exp2-Encoder-{enc}",
            model,
            train_loader,
            val_loader,
            cfg,
            paths.results / f"exp2_encoder_{enc}",
        )
        if metrics["mse"] < best_encoder_mse:
            best_encoder_mse = metrics["mse"]
            best_encoder = enc
        del model
        clear_cuda_cache()

    assert best_encoder is not None
    save_json({"best_encoder": best_encoder, "best_mse": best_encoder_mse}, paths.results / "exp2_best_encoder.json")

    # Exp3: remove gate A
    print(f"[{timestamp()}] >>> Exp3 instantiate model (no Gate A)")
    model = DualGatingModel(
        encoder_name=best_encoder,
        tcm_checkpoint_path=Path(args.tcm_checkpoint),
        freeze_tcm=True,
        use_gate_a=False,
        use_gate_b=True,
        use_tcm_encoder=True,
    )
    model, exp3_metrics = run_single_experiment("Exp3-Ablation-NoGateA", model, train_loader, val_loader, cfg, paths.results / "exp3_no_gate_a")
    del model
    clear_cuda_cache()

    # Exp4: remove gate B
    print(f"[{timestamp()}] >>> Exp4 instantiate model (no Gate B)")
    model = DualGatingModel(
        encoder_name=best_encoder,
        tcm_checkpoint_path=Path(args.tcm_checkpoint),
        freeze_tcm=True,
        use_gate_a=True,
        use_gate_b=False,
        use_tcm_encoder=True,
    )
    model, exp4_metrics = run_single_experiment("Exp4-Ablation-NoGateB", model, train_loader, val_loader, cfg, paths.results / "exp4_no_gate_b")
    del model
    clear_cuda_cache()

    # Exp5: full model
    print(f"[{timestamp()}] >>> Exp5 instantiate full model")
    model = DualGatingModel(
        encoder_name=best_encoder,
        tcm_checkpoint_path=Path(args.tcm_checkpoint),
        freeze_tcm=True,
        use_gate_a=True,
        use_gate_b=True,
        use_tcm_encoder=True,
    )
    model, exp5_metrics = run_single_experiment("Exp5-FullDualGating", model, train_loader, val_loader, cfg, paths.results / "exp5_full")
    save_checkpoint(
        model,
        paths.checkpoints / "best_model.pth",
        extra={"best_encoder": best_encoder, "metrics": exp5_metrics},
    )

    summary = {
        "exp1": exp1_metrics,
        "exp3": exp3_metrics,
        "exp4": exp4_metrics,
        "exp5": exp5_metrics,
        "best_encoder": best_encoder,
    }
    save_json(summary, paths.results / "experiments_summary.json")
    print(f"[{timestamp()}] >>> Experiment pipeline completed.")


if __name__ == "__main__":
    main()
