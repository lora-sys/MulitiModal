#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.config import Paths, TCM_CHECKPOINT_PATH, TCM_SCALER_PATH, TrainConfig, ensure_dirs, override_from_env, resolve_device
from src.data_loader import WESADDataset, make_train_val_loaders
from src.models.fusion import DualGatingModel
from src.training import fit_regression_model
from src.utils import save_json, set_seed, timestamp

try:
    import optuna
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Optuna is required for run_optuna.py") from exc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Constrained Optuna pre-search on WESAD (Ours/TCN only)")
    p.add_argument("--n-trials", type=int, default=50)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--wesad-dir", type=str, default=None)
    p.add_argument("--tcm-checkpoint", type=str, default=str(TCM_CHECKPOINT_PATH))
    p.add_argument("--tcm-scaler", type=str, default=str(TCM_SCALER_PATH))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = override_from_env(Paths())
    if args.wesad_dir:
        paths.wesad_dir = Path(args.wesad_dir)
    ensure_dirs(paths)

    cfg = TrainConfig(epochs=args.epochs, device=resolve_device(args.device))
    set_seed(cfg.seed)

    dataset = WESADDataset(paths.wesad_dir, Path(args.tcm_scaler), window_size=cfg.window_size, overlap=cfg.window_overlap)

    def objective(trial: "optuna.Trial") -> float:
        # Constrained search space for stable optimization.
        lr = trial.suggest_float("lr", 5e-4, 2e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64])

        tr_loader, va_loader = make_train_val_loaders(dataset, batch_size=batch_size, seed=cfg.seed)

        # Locked evaluation target: Ours architecture only.
        model = DualGatingModel(
            encoder_name="tcn",
            tcm_checkpoint_path=Path(args.tcm_checkpoint),
            freeze_tcm=True,
            use_tcm=True,
            use_gate_a=True,
            use_gate_b=True,
        )
        _, _, val_metrics = fit_regression_model(
            model,
            tr_loader,
            va_loader,
            epochs=cfg.epochs,
            lr=lr,
            weight_decay=weight_decay,
            device=cfg.device,
        )
        return float(val_metrics["mse"])

    print(f"[{timestamp()}] >>> Starting constrained Optuna ({args.n_trials} trials)")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.n_trials)

    best = {
        "lr": float(study.best_trial.params["lr"]),
        "weight_decay": float(study.best_trial.params["weight_decay"]),
        "batch_size": int(study.best_trial.params["batch_size"]),
        "best_mse": float(study.best_value),
    }

    print(f"[{timestamp()}] >>> Best lr: {best['lr']}")
    print(f"[{timestamp()}] >>> Best weight_decay: {best['weight_decay']}")
    print(f"[{timestamp()}] >>> Best batch_size: {best['batch_size']}")

    output = paths.checkpoints / "optuna_best_params.json"
    save_json(best, output)
    print(f"[{timestamp()}] >>> Optuna done. Best params saved to {output}")


if __name__ == "__main__":
    main()

