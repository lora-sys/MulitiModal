#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import OPLRIRegressor
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
from src.data_loader import (
    WESADDataset,
    make_loaders_from_indices,
    make_subject_level_split_indices,
)
from src.utils import regression_metrics, save_json, set_seed, timestamp, to_numpy
from src.utils.plotting import plot_ablation, plot_comparison, plot_selection


ENCODERS = ["tcn", "inceptiontime", "os-cnn", "xcm", "1d-resnet"]
RUN_EXPERIMENTS_SIGNATURE = "RUN_EXPERIMENTS_V2_LOSO_GUARDED"


@dataclass
class HyperParams:
    lr: float
    weight_decay: float
    batch_size: int


class FrozenTCMPrior:
    """Script-side TCM inference: 4D -> scaler -> frozen FT-Transformer -> 9D probs."""

    def __init__(self, checkpoint_path: Path, scaler_path: Path, device: str):
        self.device = device
        resolved_ckpt = self._resolve_checkpoint_path(checkpoint_path)
        resolved_scaler = self._resolve_scaler_path(scaler_path, resolved_ckpt)
        print(f"[{timestamp()}] >>> TCM checkpoint resolved to: {resolved_ckpt}", flush=True)
        print(f"[{timestamp()}] >>> TCM scaler resolved to: {resolved_scaler}", flush=True)

        self.model = self._load_tcm_model(resolved_ckpt).to(device)
        self.scaler = self._load_scaler(resolved_scaler)

        # 铁律: 永久冻结
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def _resolve_checkpoint_path(self, checkpoint_path: Path) -> Path:
        candidates = [
            checkpoint_path,
            checkpoint_path.parent / "best_model.pth",
            checkpoint_path.parent / "best_tcm_model.pth",
            Path("/root/work/MulitiModal/checkpoints/best_model.pth"),
            Path("/root/work/MulitiModal/checkpoints/best_tcm_model.pth"),
            Path("/root/work/MulitiModal/tcm_ft_transformer/checkpoints/best_model.pth"),
            Path("/root/work/MulitiModal/tcm_ft_transformer/checkpoints/best_tcm_model.pth"),
        ]
        seen = set()
        ordered = []
        for c in candidates:
            s = str(c)
            if s not in seen:
                seen.add(s)
                ordered.append(c)
        for c in ordered:
            if c.exists():
                return c
        raise FileNotFoundError(
            "TCM checkpoint not found. Searched:\n" + "\n".join(str(p) for p in ordered)
        )

    def _resolve_scaler_path(self, scaler_path: Path, resolved_ckpt: Path) -> Path:
        candidates = [
            scaler_path,
            resolved_ckpt.parent / "tcm_scaler.pkl",
            resolved_ckpt.parent / "scaler_params.npz",
            Path("/root/work/MulitiModal/checkpoints/tcm_scaler.pkl"),
            Path("/root/work/MulitiModal/checkpoints/scaler_params.npz"),
            Path("/root/work/MulitiModal/tcm_ft_transformer/scaler_params.npz"),
            Path("/root/work/MulitiModal/tcm_ft_transformer/checkpoints/scaler_params.npz"),
        ]
        seen = set()
        ordered = []
        for c in candidates:
            s = str(c)
            if s not in seen:
                seen.add(s)
                ordered.append(c)
        for c in ordered:
            if c.exists():
                return c
        raise FileNotFoundError(
            "TCM scaler not found. Searched:\n" + "\n".join(str(p) for p in ordered)
        )

    def _load_tcm_model(self, checkpoint_path: Path) -> nn.Module:
        repo_root = Path(__file__).resolve().parent
        sys.path.insert(0, str(repo_root / "tcm_ft_transformer"))
        from ft_transformer import get_model  # pylint: disable=import-error

        model = get_model(n_features=4, n_classes=9)
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = loaded["model_state_dict"] if isinstance(loaded, dict) and "model_state_dict" in loaded else loaded
        model.load_state_dict(state_dict, strict=False)
        return model

    def _load_scaler(self, scaler_path: Path):
        try:
            import joblib  # type: ignore

            scaler = joblib.load(scaler_path)
            if hasattr(scaler, "transform"):
                return scaler
        except Exception:
            pass
        try:
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            if hasattr(scaler, "transform"):
                return scaler
        except Exception:
            pass
        try:
            arr = np.load(scaler_path)
            if "mean" in arr and "std" in arr:
                return {"mean": arr["mean"].astype(np.float32), "std": arr["std"].astype(np.float32)}
        except Exception:
            pass
        raise RuntimeError(f"Failed to load scaler content from: {scaler_path}")

    @torch.no_grad()
    def infer_probs(self, static_4d: torch.Tensor) -> torch.Tensor:
        x = static_4d[:, :4].detach().cpu().numpy().astype(np.float32)
        if isinstance(self.scaler, dict):
            mean = np.asarray(self.scaler["mean"], dtype=np.float32)
            std = np.asarray(self.scaler["std"], dtype=np.float32)
            if mean.shape[0] >= 4:
                mean = mean[:4]
            if std.shape[0] >= 4:
                std = std[:4]
            std = np.where(std == 0, 1.0, std)
            x_scaled = (x - mean) / std
        else:
            x_scaled = self.scaler.transform(x).astype(np.float32)
            if x_scaled.shape[1] >= 4:
                x_scaled = x_scaled[:, :4]

        x_scaled_t = torch.from_numpy(x_scaled.astype(np.float32)).to(self.device)
        self.model.eval()
        probs = self.model(x_scaled_t)  # [B, 9]
        return probs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="3-stage experiment flow with late reinjection (WESAD only)")
    p.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    p.add_argument("--selection-epochs", type=int, default=15)
    p.add_argument("--search-epochs", type=int, default=15)
    p.add_argument("--search-trials", type=int, default=15)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--wesad-dir", type=str, default=None)
    p.add_argument("--tcm-checkpoint", type=str, default=str(TCM_CHECKPOINT_PATH))
    p.add_argument("--tcm-scaler", type=str, default=str(TCM_SCALER_PATH))
    p.add_argument("--override-params", type=str, default="")
    p.add_argument("--skip-fast-search", action="store_true")
    p.add_argument("--gate-a-scale", type=float, default=0.6)
    p.add_argument("--gate-b-scale", type=float, default=0.35)
    p.add_argument("--final-lr-mult", type=float, default=0.7)
    p.add_argument("--gate-b-sweep", type=str, default="", help="Comma-separated gate_b_scale values, e.g. 0.1,0.2,0.3")
    p.add_argument("--final-lr-mult-sweep", type=str, default="", help="Comma-separated final_lr_mult values, e.g. 0.5,0.7")
    p.add_argument("--sweep-epochs", type=int, default=-1, help="Epochs used by Step8 sweep; <=0 means use --epochs")
    p.add_argument("--protocol", type=str, default="loso", choices=["loso", "subject_split"])
    p.add_argument("--loso-subject", type=str, default="", help="Internal: run only one LOSO fold with this held-out subject")
    p.add_argument("--output-json", type=str, default="", help="Optional override path for fold summary json")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _unpack_batch(batch, device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dynamic, second, third = batch
    dynamic = dynamic.to(device)
    # WESAD now returns (dynamic, target, static), keep compatibility fallback.
    if second.dim() >= 2 and second.shape[-1] in (4, 8):
        static = second.to(device)
        target = third.to(device)
    else:
        target = second.to(device)
        static = third.to(device)
    return dynamic, static, target


def _run_forward(
    model: OPLRIRegressor,
    tcm_prior: FrozenTCMPrior,
    dynamic_x: torch.Tensor,
    static_x: torch.Tensor,
    *,
    use_tcm: bool,
    use_gate_a: bool,
    use_gate_b: bool,
    gate_a_scale: float,
    gate_b_scale: float,
) -> torch.Tensor:
    if use_tcm:
        tcm_probs = tcm_prior.infer_probs(static_x)  # [B, 9]
    else:
        tcm_probs = torch.zeros(dynamic_x.size(0), 9, device=dynamic_x.device, dtype=torch.float32)

    z_raw = model.dynamic_encoder(dynamic_x)  # [B,128]
    if use_gate_a and use_tcm:
        gate_a = torch.sigmoid(model.gate_a_linear(tcm_probs))
        z_modulated = z_raw * ((1.0 - gate_a_scale) + gate_a_scale * gate_a)
    else:
        z_modulated = z_raw

    if use_gate_b:
        gate_b = torch.sigmoid(model.gate_b_linear(z_modulated))
        z_pure_dynamic = z_modulated * (1.0 - gate_b_scale * gate_b)
    else:
        z_pure_dynamic = z_modulated

    # OP-LRI anti-pollution: dynamic branch detached before late reinjection.
    z_pure_dynamic = z_pure_dynamic.detach()
    final_input = torch.cat([z_pure_dynamic, tcm_probs], dim=-1)  # [B,137]
    pred = model.forward_from_final_input(final_input)
    return pred


def _freeze_non_head(model: OPLRIRegressor) -> None:
    for p in model.dynamic_encoder.parameters():
        p.requires_grad = False
    for p in model.gate_a_linear.parameters():
        p.requires_grad = False
    for p in model.gate_b_linear.parameters():
        p.requires_grad = False
    for p in model.reg_head.parameters():
        p.requires_grad = True


def train_eval_step(
    *,
    step_name: str,
    encoder: str,
    use_tcm: bool,
    use_gate_a: bool,
    use_gate_b: bool,
    hparams: HyperParams,
    epochs: int,
    patience: int,
    dataset: WESADDataset,
    train_indices: List[int],
    val_indices: List[int],
    seed: int,
    device: str,
    tcm_prior: FrozenTCMPrior,
    gate_a_scale: float,
    gate_b_scale: float,
) -> Dict:
    train_loader, val_loader = make_loaders_from_indices(
        dataset,
        train_indices,
        val_indices,
        batch_size=hparams.batch_size,
        seed=seed,
    )
    model = OPLRIRegressor(encoder_name=encoder, use_gate_a=True, use_gate_b=True).to(device)
    _freeze_non_head(model)
    optimizer = torch.optim.AdamW(model.reg_head.parameters(), lr=hparams.lr, weight_decay=hparams.weight_decay)
    loss_fn = nn.MSELoss()

    best_val_mse = float("inf")
    best_epoch = 1
    best_state = None
    counter = 0
    val_mse_history: List[float] = []

    for epoch in range(1, max(1, epochs) + 1):
        model.train()
        total_loss = 0.0
        n_items = 0
        for batch in train_loader:
            dynamic_x, static_x, target = _unpack_batch(batch, device)
            pred = _run_forward(
                model,
                tcm_prior,
                dynamic_x,
                static_x,
                use_tcm=use_tcm,
                use_gate_a=use_gate_a,
                use_gate_b=use_gate_b,
                gate_a_scale=gate_a_scale,
                gate_b_scale=gate_b_scale,
            )
            loss = loss_fn(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(dynamic_x)
            n_items += len(dynamic_x)
        train_loss = total_loss / max(n_items, 1)

        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for batch in val_loader:
                dynamic_x, static_x, target = _unpack_batch(batch, device)
                pred = _run_forward(
                    model,
                    tcm_prior,
                    dynamic_x,
                    static_x,
                    use_tcm=use_tcm,
                    use_gate_a=use_gate_a,
                    use_gate_b=use_gate_b,
                    gate_a_scale=gate_a_scale,
                    gate_b_scale=gate_b_scale,
                )
                ys.append(to_numpy(target))
                ps.append(to_numpy(pred))

        val_metrics = regression_metrics(np.concatenate(ys, axis=0), np.concatenate(ps, axis=0))
        val_mse = float(val_metrics["mse"])
        val_mse_history.append(val_mse)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1

        print(
            f"[{timestamp()}] {step_name} | Epoch {epoch}/{epochs} "
            f"train_loss={train_loss:.6f} val_mse={val_mse:.6f} "
            f"best_val={best_val_mse:.6f} patience={counter}/{patience}",
            flush=True,
        )

        if counter >= max(1, patience):
            print(f"[{timestamp()}] {step_name} | Early stop at epoch {epoch}, best={best_epoch}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final metrics on val using best state.
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in val_loader:
            dynamic_x, static_x, target = _unpack_batch(batch, device)
            pred = _run_forward(
                model,
                tcm_prior,
                dynamic_x,
                static_x,
                use_tcm=use_tcm,
                use_gate_a=use_gate_a,
                use_gate_b=use_gate_b,
                gate_a_scale=gate_a_scale,
                gate_b_scale=gate_b_scale,
            )
            ys.append(to_numpy(target))
            ps.append(to_numpy(pred))
    final_metrics = regression_metrics(np.concatenate(ys, axis=0), np.concatenate(ps, axis=0))

    print(
        f"[{timestamp()}] {step_name} DONE | MSE={final_metrics['mse']:.6f} "
        f"RMSE={final_metrics['rmse']:.6f} MAE={final_metrics['mae']:.6f} "
        f"Pearson={final_metrics['pearson']:.6f} [best_epoch={best_epoch}]",
        flush=True,
    )

    out = {
        "step_name": step_name,
        "encoder": encoder,
        "use_tcm": use_tcm,
        "use_gate_a": use_gate_a,
        "use_gate_b": use_gate_b,
        "hparams": {"lr": hparams.lr, "weight_decay": hparams.weight_decay, "batch_size": hparams.batch_size},
        "best_epoch": best_epoch,
        "val_mse_history": val_mse_history,
        "metrics": final_metrics,
        "model_state_dict": model.state_dict(),
    }
    del model
    clear_cuda_cache()
    return out


def _strip_for_json(step: Dict) -> Dict:
    """Remove non-JSON fields such as tensors/state_dict before summary dump."""
    safe = dict(step)
    safe.pop("model_state_dict", None)
    return safe


def _parse_override_params(text: str) -> HyperParams | None:
    s = (text or "").strip()
    if not s:
        return None
    if s.startswith("{"):
        obj = json.loads(s)
        return HyperParams(
            lr=float(obj.get("lr", 5e-4)),
            weight_decay=float(obj.get("weight_decay", 1e-5)),
            batch_size=int(obj.get("batch_size", 32)),
        )
    kv = {}
    for part in s.split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        kv[k.strip()] = v.strip()
    if not kv:
        return None
    return HyperParams(
        lr=float(kv.get("lr", 5e-4)),
        weight_decay=float(kv.get("weight_decay", 1e-5)),
        batch_size=int(kv.get("batch_size", 32)),
    )


def _parse_float_list(text: str) -> List[float]:
    s = (text or "").strip()
    if not s:
        return []
    vals = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    return vals


def _summarize_loso(fold_payloads: List[Dict]) -> Dict:
    per_step: Dict[int, List[float]] = {}
    name_map: Dict[int, str] = {}
    for payload in fold_payloads:
        for row in payload.get("stage3_matrix_best_val_mse", []):
            step = int(row["step"])
            val = row["best_val_mse"]
            name_map[step] = row["name"]
            if val is None:
                continue
            per_step.setdefault(step, []).append(float(val))

    summary_rows = []
    for step in sorted(name_map):
        vals = per_step.get(step, [])
        if not vals:
            summary_rows.append(
                {"step": step, "name": name_map[step], "mean_best_val_mse": None, "std_best_val_mse": None, "n_folds": 0}
            )
        else:
            arr = np.asarray(vals, dtype=np.float64)
            summary_rows.append(
                {
                    "step": step,
                    "name": name_map[step],
                    "mean_best_val_mse": float(arr.mean()),
                    "std_best_val_mse": float(arr.std(ddof=0)),
                    "n_folds": int(arr.size),
                }
            )
    return {"loso_stage3_summary": summary_rows}


def _sample_hparams(rng: random.Random) -> HyperParams:
    # Conservative quick-search region for late reinjection.
    lr = 10 ** rng.uniform(np.log10(2.5e-4), np.log10(8e-4))
    weight_decay = 10 ** rng.uniform(np.log10(1e-6), np.log10(1e-4))
    batch_size = 32
    return HyperParams(lr=float(lr), weight_decay=float(weight_decay), batch_size=int(batch_size))


def _print_selection_table(selection_rows: List[Dict]) -> None:
    print("\n===== Stage1 Backbone Selection (late reinjection, full gates) =====")
    print(f"{'Encoder':<15} {'BestValMSE':>12} {'RMSE':>10} {'MAE':>10} {'Pearson':>10}")
    for r in selection_rows:
        m = r["metrics"]
        print(f"{r['encoder']:<15} {m['mse']:>12.6f} {m['rmse']:>10.6f} {m['mae']:>10.6f} {m['pearson']:>10.6f}")
    print("=====================================================================\n")


def main() -> None:
    args = parse_args()
    runner_file = Path(__file__).resolve()
    runner_mtime = datetime.fromtimestamp(runner_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"[{timestamp()}] >>> Runner signature: {RUN_EXPERIMENTS_SIGNATURE} | "
        f"file={runner_file} | mtime={runner_mtime}",
        flush=True,
    )
    paths = override_from_env(Paths())
    if args.wesad_dir:
        paths.wesad_dir = Path(args.wesad_dir)
    ensure_dirs(paths)

    device = resolve_device(args.device)
    cfg = TrainConfig(device=device, epochs=(3 if args.dry_run else args.epochs))
    set_seed(cfg.seed)

    print(f"[{timestamp()}] >>> Loading WESAD dataset from {paths.wesad_dir}", flush=True)
    dataset = WESADDataset(
        paths.wesad_dir,
        Path(args.tcm_scaler),
        window_size=cfg.window_size,
        overlap=cfg.window_overlap,
    )
    print(f"[{timestamp()}] >>> WESAD loaded. total_windows={len(dataset)}", flush=True)
    print(
        f"[{timestamp()}] >>> Gate scales: gate_a_scale={args.gate_a_scale:.3f}, gate_b_scale={args.gate_b_scale:.3f}",
        flush=True,
    )
    if args.protocol == "loso" and not args.loso_subject:
        unique_subjects = sorted(set(dataset.sample_subject_ids))
        print(f"[{timestamp()}] >>> LOSO driver mode: {len(unique_subjects)} folds", flush=True)
        fold_dir = paths.results / "loso_folds"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_payloads: List[Dict] = []
        for sid in unique_subjects:
            fold_out = fold_dir / f"experiments_summary_{sid}.json"
            cmd = [
                "python3",
                "-u",
                "run_experiments.py",
                "--protocol",
                "loso",
                "--loso-subject",
                sid,
                "--output-json",
                str(fold_out),
                "--epochs",
                str(args.epochs),
                "--selection-epochs",
                str(args.selection_epochs),
                "--search-epochs",
                str(args.search_epochs),
                "--search-trials",
                str(args.search_trials),
                "--device",
                str(args.device),
                "--tcm-checkpoint",
                str(args.tcm_checkpoint),
                "--tcm-scaler",
                str(args.tcm_scaler),
                "--override-params",
                str(args.override_params),
                "--gate-a-scale",
                str(args.gate_a_scale),
                "--gate-b-scale",
                str(args.gate_b_scale),
                "--final-lr-mult",
                str(args.final_lr_mult),
                "--gate-b-sweep",
                str(args.gate_b_sweep),
                "--final-lr-mult-sweep",
                str(args.final_lr_mult_sweep),
                "--sweep-epochs",
                str(args.sweep_epochs),
            ]
            if args.wesad_dir:
                cmd.extend(["--wesad-dir", str(args.wesad_dir)])
            if args.skip_fast_search:
                cmd.append("--skip-fast-search")
            if args.dry_run:
                cmd.append("--dry-run")
            print(f"[{timestamp()}] >>> LOSO fold holdout={sid}", flush=True)
            subprocess.run(cmd, check=True)
            with open(fold_out, "r", encoding="utf-8") as f:
                fold_payloads.append(json.load(f))

        loso_summary = _summarize_loso(fold_payloads)
        loso_summary["protocol"] = "loso"
        loso_summary["n_folds"] = len(fold_payloads)
        loso_summary["fold_subjects"] = unique_subjects
        loso_summary["runner"] = {
            "signature": RUN_EXPERIMENTS_SIGNATURE,
            "file": str(runner_file),
            "mtime": runner_mtime,
        }
        out_path = Path(args.output_json) if args.output_json else (paths.results / "experiments_summary_loso.json")
        save_json(loso_summary, out_path)
        print(f"[{timestamp()}] >>> LOSO summary saved to {out_path}", flush=True)
        return

    if args.protocol == "loso":
        holdout = args.loso_subject.strip()
        if not holdout:
            raise RuntimeError("LOSO single-fold mode requires --loso-subject")
        train_indices = [i for i, sid in enumerate(dataset.sample_subject_ids) if sid != holdout]
        val_indices = [i for i, sid in enumerate(dataset.sample_subject_ids) if sid == holdout]
        if not train_indices or not val_indices:
            raise RuntimeError(f"Invalid LOSO holdout subject: {holdout}")
    else:
        train_indices, val_indices = make_subject_level_split_indices(dataset, val_ratio=0.2, seed=cfg.seed)

    train_subjects = {dataset.sample_subject_ids[i] for i in train_indices}
    val_subjects = {dataset.sample_subject_ids[i] for i in val_indices}
    overlap_subjects = train_subjects.intersection(val_subjects)
    if overlap_subjects:
        raise RuntimeError(f"Subject-level split leakage detected: {sorted(overlap_subjects)}")
    print(
        f"[{timestamp()}] >>> Split ready (protocol={args.protocol}): "
        f"train_windows={len(train_indices)} val_windows={len(val_indices)} "
        f"train_subjects={len(train_subjects)} val_subjects={len(val_subjects)}",
        flush=True,
    )

    tcm_prior = FrozenTCMPrior(Path(args.tcm_checkpoint), Path(args.tcm_scaler), device)

    # ---------------- Stage 1: force re-selection ----------------
    stage1_epochs = 3 if args.dry_run else args.selection_epochs
    base_hparams = HyperParams(lr=5e-4, weight_decay=1e-5, batch_size=32)
    selection_rows: List[Dict] = []
    best_encoder = None
    best_encoder_mse = float("inf")

    print(f"[{timestamp()}] >>> Stage 1: backbone reselection on new architecture", flush=True)
    for encoder in ENCODERS:
        row = train_eval_step(
            step_name=f"Stage1-Select-{encoder}",
            encoder=encoder,
            use_tcm=True,
            use_gate_a=True,
            use_gate_b=True,
            hparams=base_hparams,
            epochs=stage1_epochs,
            patience=EARLY_STOPPING_PATIENCE,
            dataset=dataset,
            train_indices=train_indices,
            val_indices=val_indices,
            seed=cfg.seed,
            device=device,
            tcm_prior=tcm_prior,
            gate_a_scale=args.gate_a_scale,
            gate_b_scale=args.gate_b_scale,
        )
        selection_rows.append(row)
        mse = row["metrics"]["mse"]
        if mse < best_encoder_mse:
            best_encoder_mse = mse
            best_encoder = encoder

    assert best_encoder is not None
    _print_selection_table(selection_rows)
    print(f"[{timestamp()}] >>> Stage 1 winner: {best_encoder} (best_val_mse={best_encoder_mse:.6f})", flush=True)

    # ---------------- Stage 2: reset/tune params ----------------
    print(f"[{timestamp()}] >>> Stage 2: hyper-parameter reset/tuning for encoder={best_encoder}", flush=True)
    override_hp = _parse_override_params(args.override_params)
    if override_hp is not None:
        best_hp = override_hp
        tune_logs = [{"trial": 0, "hparams": best_hp.__dict__, "mse": None, "note": "override_params"}]
        print(f"[{timestamp()}] >>> Using override params: {best_hp}", flush=True)
    elif args.skip_fast_search:
        best_hp = base_hparams
        tune_logs = [{"trial": 0, "hparams": best_hp.__dict__, "mse": None, "note": "conservative_default"}]
        print(f"[{timestamp()}] >>> Using conservative default params: {best_hp}", flush=True)
    else:
        rng = random.Random(cfg.seed + 7)
        trials = max(1, args.search_trials if not args.dry_run else 3)
        search_epochs = 3 if args.dry_run else args.search_epochs
        best_tune_mse = float("inf")
        best_hp = base_hparams
        tune_logs = []
        for t in range(1, trials + 1):
            hp = _sample_hparams(rng)
            res = train_eval_step(
                step_name=f"Stage2-Tune-T{t}",
                encoder=best_encoder,
                use_tcm=True,
                use_gate_a=True,
                use_gate_b=True,
                hparams=hp,
                epochs=search_epochs,
                patience=EARLY_STOPPING_PATIENCE,
                dataset=dataset,
                train_indices=train_indices,
                val_indices=val_indices,
                seed=cfg.seed + t,
                device=device,
                tcm_prior=tcm_prior,
                gate_a_scale=args.gate_a_scale,
                gate_b_scale=args.gate_b_scale,
            )
            mse = float(res["metrics"]["mse"])
            tune_logs.append({"trial": t, "hparams": hp.__dict__, "mse": mse})
            if mse < best_tune_mse:
                best_tune_mse = mse
                best_hp = hp
        print(f"[{timestamp()}] >>> Stage 2 best params: {best_hp} (best_val_mse={best_tune_mse:.6f})", flush=True)

    save_json(
        {
            "best_encoder": best_encoder,
            "best_params": best_hp.__dict__,
            "selection_results": [
                {"encoder": r["encoder"], "metrics": r["metrics"], "best_epoch": r["best_epoch"]}
                for r in selection_rows
            ],
            "tuning_logs": tune_logs,
        },
        paths.checkpoints / "reinjection_best_setup.json",
    )

    # Optional focused sweep for Step8 stability:
    # - gate_b_scale in a small range
    # - final_lr_mult for head-only late stage
    sweep_gate_b = _parse_float_list(args.gate_b_sweep)
    sweep_lr_mult = _parse_float_list(args.final_lr_mult_sweep)
    selected_gate_b_scale = float(args.gate_b_scale)
    selected_final_lr_mult = float(args.final_lr_mult)
    sweep_results: List[Dict] = []
    if sweep_gate_b or sweep_lr_mult:
        if not sweep_gate_b:
            sweep_gate_b = [selected_gate_b_scale]
        if not sweep_lr_mult:
            sweep_lr_mult = [selected_final_lr_mult]
        if args.dry_run:
            sweep_epochs = 3
        else:
            sweep_epochs = cfg.epochs if int(args.sweep_epochs) <= 0 else max(1, int(args.sweep_epochs))
        best_sweep_mse = float("inf")
        print(f"[{timestamp()}] >>> Running Step8-focused sweep (epochs={sweep_epochs})", flush=True)
        for gb in sweep_gate_b:
            for lm in sweep_lr_mult:
                hp = HyperParams(
                    lr=float(best_hp.lr * max(0.1, lm)),
                    weight_decay=float(best_hp.weight_decay),
                    batch_size=int(best_hp.batch_size),
                )
                res = train_eval_step(
                    step_name=f"Sweep-Step8-gb{gb:.3f}-lm{lm:.3f}",
                    encoder=best_encoder,
                    use_tcm=True,
                    use_gate_a=True,
                    use_gate_b=True,
                    hparams=hp,
                    epochs=sweep_epochs,
                    patience=EARLY_STOPPING_PATIENCE,
                    dataset=dataset,
                    train_indices=train_indices,
                    val_indices=val_indices,
                    seed=cfg.seed + 100 + int(gb * 1000) + int(lm * 1000),
                    device=device,
                    tcm_prior=tcm_prior,
                    gate_a_scale=args.gate_a_scale,
                    gate_b_scale=float(gb),
                )
                mse = float(res["metrics"]["mse"])
                sweep_results.append(
                    {"gate_b_scale": float(gb), "final_lr_mult": float(lm), "mse": mse}
                )
                if mse < best_sweep_mse:
                    best_sweep_mse = mse
                    selected_gate_b_scale = float(gb)
                    selected_final_lr_mult = float(lm)
        print(
            f"[{timestamp()}] >>> Sweep best: gate_b_scale={selected_gate_b_scale:.3f}, "
            f"final_lr_mult={selected_final_lr_mult:.3f}, mse={best_sweep_mse:.6f}",
            flush=True,
        )

    # ---------------- Stage 3: 9-step matrix ----------------
    print(f"[{timestamp()}] >>> Stage 3: running 9-step matrix with fixed best params", flush=True)
    matrix_logs: List[Dict] = []

    step1 = train_eval_step(
        step_name="Step1-BaselineA-Weak",
        encoder="tcn",
        use_tcm=False,
        use_gate_a=False,
        use_gate_b=False,
        hparams=best_hp,
        epochs=cfg.epochs,
        patience=EARLY_STOPPING_PATIENCE,
        dataset=dataset,
        train_indices=train_indices,
        val_indices=val_indices,
        seed=cfg.seed,
        device=device,
        tcm_prior=tcm_prior,
        gate_a_scale=args.gate_a_scale,
        gate_b_scale=args.gate_b_scale,
    )
    matrix_logs.append({"step": 1, "name": "Baseline A", "mse": step1["metrics"]["mse"], "full": _strip_for_json(step1)})

    step2 = train_eval_step(
        step_name="Step2-BaselineB-Strong",
        encoder="tcn",
        use_tcm=False,
        use_gate_a=False,
        use_gate_b=True,
        hparams=best_hp,
        epochs=cfg.epochs,
        patience=EARLY_STOPPING_PATIENCE,
        dataset=dataset,
        train_indices=train_indices,
        val_indices=val_indices,
        seed=cfg.seed + 1,
        device=device,
        tcm_prior=tcm_prior,
        gate_a_scale=args.gate_a_scale,
        gate_b_scale=args.gate_b_scale,
    )
    matrix_logs.append({"step": 2, "name": "Baseline B", "mse": step2["metrics"]["mse"], "full": _strip_for_json(step2)})

    step3 = train_eval_step(
        step_name="Step3-Ours-TCN",
        encoder="tcn",
        use_tcm=True,
        use_gate_a=True,
        use_gate_b=True,
        hparams=best_hp,
        epochs=cfg.epochs,
        patience=EARLY_STOPPING_PATIENCE,
        dataset=dataset,
        train_indices=train_indices,
        val_indices=val_indices,
        seed=cfg.seed + 2,
        device=device,
        tcm_prior=tcm_prior,
        gate_a_scale=args.gate_a_scale,
        gate_b_scale=args.gate_b_scale,
    )
    matrix_logs.append({"step": 3, "name": "Ours-TCN", "mse": step3["metrics"]["mse"], "full": _strip_for_json(step3)})

    # Step4 uses Stage1 selection outcome.
    matrix_logs.append(
        {
            "step": 4,
            "name": "Encoder Selection",
            "mse": best_encoder_mse,
            "best_encoder": best_encoder,
            "selection_rows": [{"encoder": r["encoder"], "mse": r["metrics"]["mse"]} for r in selection_rows],
        }
    )

    step5 = train_eval_step(
        step_name="Step5-Ablation1-woDualGating",
        encoder=best_encoder,
        use_tcm=True,
        use_gate_a=False,
        use_gate_b=False,
        hparams=best_hp,
        epochs=cfg.epochs,
        patience=EARLY_STOPPING_PATIENCE,
        dataset=dataset,
        train_indices=train_indices,
        val_indices=val_indices,
        seed=cfg.seed + 3,
        device=device,
        tcm_prior=tcm_prior,
        gate_a_scale=args.gate_a_scale,
        gate_b_scale=args.gate_b_scale,
    )
    matrix_logs.append({"step": 5, "name": "w/o Dual Gating", "mse": step5["metrics"]["mse"], "full": _strip_for_json(step5)})

    step6 = train_eval_step(
        step_name="Step6-Ablation2-woTCMPrior",
        encoder=best_encoder,
        use_tcm=False,
        use_gate_a=False,
        use_gate_b=True,
        hparams=best_hp,
        epochs=cfg.epochs,
        patience=EARLY_STOPPING_PATIENCE,
        dataset=dataset,
        train_indices=train_indices,
        val_indices=val_indices,
        seed=cfg.seed + 4,
        device=device,
        tcm_prior=tcm_prior,
        gate_a_scale=args.gate_a_scale,
        gate_b_scale=args.gate_b_scale,
    )
    matrix_logs.append({"step": 6, "name": "w/o TCM Prior", "mse": step6["metrics"]["mse"], "full": _strip_for_json(step6)})

    step7 = train_eval_step(
        step_name="Step7-Ablation3-woTCMGate",
        encoder=best_encoder,
        use_tcm=True,
        use_gate_a=False,
        use_gate_b=True,
        hparams=best_hp,
        epochs=cfg.epochs,
        patience=EARLY_STOPPING_PATIENCE,
        dataset=dataset,
        train_indices=train_indices,
        val_indices=val_indices,
        seed=cfg.seed + 5,
        device=device,
        tcm_prior=tcm_prior,
        gate_a_scale=args.gate_a_scale,
        gate_b_scale=args.gate_b_scale,
    )
    matrix_logs.append({"step": 7, "name": "w/o TCM_Gate", "mse": step7["metrics"]["mse"], "full": _strip_for_json(step7)})

    final_hp = HyperParams(
        lr=float(best_hp.lr * max(0.1, selected_final_lr_mult)),
        weight_decay=float(best_hp.weight_decay),
        batch_size=int(best_hp.batch_size),
    )
    print(f"[{timestamp()}] >>> Final Ours stabilized hparams: {final_hp}", flush=True)

    step8 = train_eval_step(
        step_name="Step8-FinalOurs",
        encoder=best_encoder,
        use_tcm=True,
        use_gate_a=True,
        use_gate_b=True,
        hparams=final_hp,
        epochs=cfg.epochs,
        patience=EARLY_STOPPING_PATIENCE,
        dataset=dataset,
        train_indices=train_indices,
        val_indices=val_indices,
        seed=cfg.seed + 6,
        device=device,
        tcm_prior=tcm_prior,
        gate_a_scale=args.gate_a_scale,
        gate_b_scale=selected_gate_b_scale,
    )
    matrix_logs.append({"step": 8, "name": "Final Ours", "mse": step8["metrics"]["mse"], "full": _strip_for_json(step8)})

    # Save final best model for downstream cross-domain.
    best_model_path = paths.checkpoints / "best_model.pth"
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": step8["model_state_dict"],
            "best_encoder": best_encoder,
            "metrics": step8["metrics"],
            "hparams": best_hp.__dict__,
            "architecture": "late_reinjection",
        },
        best_model_path,
    )
    print(f"[{timestamp()}] >>> Saved final model to {best_model_path}", flush=True)

    # Step 9 plotting
    comparison_data = [
        {"name": "Baseline A", "metrics": step1["metrics"]},
        {"name": "Baseline B", "metrics": step2["metrics"]},
        {"name": "Ours", "metrics": step3["metrics"]},
    ]
    selection_data = [{"name": r["encoder"], "metrics": r["metrics"]} for r in selection_rows]
    ablation_data = [
        {"name": "Full Model", "metrics": step8["metrics"]},
        {"name": "w/o Dual Gating", "metrics": step5["metrics"]},
        {"name": "w/o TCM Prior", "metrics": step6["metrics"]},
        {"name": "w/o TCM_Gate", "metrics": step7["metrics"]},
    ]
    fig1 = plot_comparison(comparison_data)
    fig2 = plot_selection(selection_data)
    fig3 = plot_ablation(ablation_data)
    matrix_logs.append({"step": 9, "name": "Plotting", "mse": None, "figures": [str(fig1), str(fig2), str(fig3)]})
    assert len(matrix_logs) == 9, f"Expected 9 matrix steps, got {len(matrix_logs)}"

    # Required outputs
    print("\n===== Stage 2 Chosen Hyper-Parameters =====")
    print(best_hp)
    print("===========================================\n")
    print(
        f"Step8 tuned controls: gate_b_scale={selected_gate_b_scale:.3f}, "
        f"final_lr_mult={selected_final_lr_mult:.3f}"
    )

    print("===== Final 9-Step best_val_mse List =====")
    for row in matrix_logs:
        print(f"Step {row['step']}: {row['name']} -> best_val_mse={row['mse']}")
    print("===========================================")

    summary_payload = {
            "stage1_selection_table": [
                {"encoder": r["encoder"], "metrics": r["metrics"], "best_epoch": r["best_epoch"]}
                for r in selection_rows
            ],
            "runner": {
                "signature": RUN_EXPERIMENTS_SIGNATURE,
                "file": str(runner_file),
                "mtime": runner_mtime,
            },
            "protocol": args.protocol,
            "holdout_subject": args.loso_subject or None,
            "stage2_best_params": best_hp.__dict__,
            "step8_controls": {
                "gate_a_scale": args.gate_a_scale,
                "gate_b_scale": selected_gate_b_scale,
                "final_lr_mult": selected_final_lr_mult,
            },
            "step8_sweep_results": sweep_results,
            "stage3_matrix_best_val_mse": [
                {"step": r["step"], "name": r["name"], "best_val_mse": r["mse"]}
                for r in matrix_logs
            ],
            "matrix_logs": matrix_logs,
        }
    summary_path = Path(args.output_json) if args.output_json else (paths.results / "experiments_summary.json")
    save_json(summary_payload, summary_path)
    print(f"[{timestamp()}] >>> Experiment flow completed. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
