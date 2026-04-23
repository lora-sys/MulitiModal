#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import OPLRIRegressor
from src.config import TCM_CHECKPOINT_PATH, TCM_SCALER_PATH, resolve_device
from src.data_loader import WESADDataset, make_train_val_loaders
from src.utils import regression_metrics, set_seed, timestamp, to_numpy


class FrozenTCMPrior:
    """
    Static path executor in script layer (NOT in model.py):
      4D static -> scaler.transform -> frozen FT-Transformer -> 9D probs
    """

    def __init__(
        self,
        checkpoint_path: Path,
        scaler_path: Path,
        device: str,
        *,
        prob_eps: float = 0.0,
        temperature: float = 1.0,
    ) -> None:
        self.device = device
        self.prob_eps = float(prob_eps)
        self.temperature = float(temperature)
        self.model = self._load_tcm_model(checkpoint_path).to(device)
        self.scaler = self._load_scaler(scaler_path)

        # 铁律 1: TCM必须绝对冻结
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self._probs_cache: Dict[bytes, torch.Tensor] = {}

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
        if not scaler_path.exists():
            raise FileNotFoundError(f"TCM scaler not found: {scaler_path}")
        try:
            import joblib  # type: ignore

            return joblib.load(scaler_path)
        except Exception:
            pass
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        if not hasattr(scaler, "transform"):
            raise TypeError(f"Scaler has no transform(): {scaler_path}")
        return scaler

    @torch.no_grad()
    def infer_probs(self, static_4d: torch.Tensor) -> torch.Tensor:
        # 铁律 2: 特征顺序必须严格 [Age, Gender, BMI, HeartRate]
        x = static_4d[:, :4].detach().cpu().numpy().astype(np.float32)
        x_key = np.round(x, 4).astype(np.float32)
        cached: list[torch.Tensor | None] = [None] * x_key.shape[0]
        miss_rows = []
        miss_idx = []
        for i in range(x_key.shape[0]):
            k = x_key[i].tobytes()
            v = self._probs_cache.get(k)
            if v is None:
                miss_rows.append(x[i])
                miss_idx.append(i)
            else:
                cached[i] = v

        if miss_rows:
            x_miss = np.stack(miss_rows, axis=0).astype(np.float32)
            x_scaled = self.scaler.transform(x_miss).astype(np.float32)
            x_scaled_t = torch.from_numpy(x_scaled).to(self.device)
            self.model.eval()
            probs_miss = self.model(x_scaled_t)  # [M, 9]
            for j, i in enumerate(miss_idx):
                k = x_key[i].tobytes()
                v = probs_miss[j].detach()
                self._probs_cache[k] = v
                cached[i] = v

        if any(t is None for t in cached):
            raise RuntimeError("TCM probs cache internal error: missing entries after inference.")
        probs = torch.stack([t.to(self.device) for t in cached], dim=0)
        if self.temperature != 1.0:
            probs = torch.clamp(probs, 1e-8, 1.0)
            logits = torch.log(probs)
            probs = torch.softmax(logits / max(self.temperature, 1e-6), dim=1)
        if self.prob_eps > 0.0:
            k = probs.shape[1]
            probs = (1.0 - self.prob_eps) * probs + (self.prob_eps / float(k))
        return probs


def _unpack_batch(batch, device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dynamic, second, third = batch
    dynamic = dynamic.to(device)
    if second.dim() >= 2 and second.shape[-1] == 4:
        static = second.to(device)
        target = third.to(device)
    else:
        target = second.to(device)
        static = third.to(device)
    return dynamic, static, target


def train_one_epoch(
    model: OPLRIRegressor,
    tcm_prior: FrozenTCMPrior,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    model.train()
    loss_fn = nn.MSELoss()
    total_loss = 0.0
    total_n = 0

    for batch in loader:
        dynamic_x, static_x, target = _unpack_batch(batch, device)

        # Script-layer static stream (strict TCM chain)
        tcm_probs_9d = tcm_prior.infer_probs(static_x)  # [B, 9], no_grad
        z_pure_dynamic, _ = model.extract_pure_dynamic(dynamic_x, tcm_probs_9d)  # [B, 128], detached

        # Late reinjection in script (not in model body)
        final_input = torch.cat([z_pure_dynamic, tcm_probs_9d], dim=-1)  # [B, 137]
        pred = model.forward_from_final_input(final_input)

        loss = loss_fn(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * len(dynamic_x)
        total_n += len(dynamic_x)

    return total_loss / max(total_n, 1)


@torch.no_grad()
def evaluate(model: OPLRIRegressor, tcm_prior: FrozenTCMPrior, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    ys, ps = [], []
    for batch in loader:
        dynamic_x, static_x, target = _unpack_batch(batch, device)
        tcm_probs_9d = tcm_prior.infer_probs(static_x)  # [B, 9]
        z_pure_dynamic, _ = model.extract_pure_dynamic(dynamic_x, tcm_probs_9d)  # detached
        final_input = torch.cat([z_pure_dynamic, tcm_probs_9d], dim=-1)
        pred = model.forward_from_final_input(final_input)

        ys.append(to_numpy(target))
        ps.append(to_numpy(pred))

    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(ps, axis=0)
    return regression_metrics(y_true, y_pred)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single WESAD experiment with script-side frozen TCM late reinjection.")
    p.add_argument("--wesad-dir", type=str, default="data/wesad")
    p.add_argument("--tcm-checkpoint", type=str, default=str(TCM_CHECKPOINT_PATH))
    p.add_argument("--tcm-scaler", type=str, default=str(TCM_SCALER_PATH))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save", type=str, default="checkpoints/single_experiment_best.pth")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    set_seed(args.seed)

    print(f"[{timestamp()}] >>> Loading WESAD from {args.wesad_dir}")
    dataset = WESADDataset(Path(args.wesad_dir), Path(args.tcm_scaler))
    train_loader, val_loader = make_train_val_loaders(dataset, batch_size=args.batch_size, seed=args.seed)

    print(f"[{timestamp()}] >>> Building model and frozen TCM prior")
    model = OPLRIRegressor(encoder_name="tcn", use_gate_a=True, use_gate_b=True).to(device)
    tcm_prior = FrozenTCMPrior(Path(args.tcm_checkpoint), Path(args.tcm_scaler), device)

    # 铁律 4: 只更新回归头，动态主干/门控完全不更新
    for p in model.dynamic_encoder.parameters():
        p.requires_grad = False
    for p in model.gate_a_linear.parameters():
        p.requires_grad = False
    for p in model.gate_b_linear.parameters():
        p.requires_grad = False
    for p in model.cross_attention.parameters():
        p.requires_grad = False
    model.constitution_tokens.requires_grad = False
    optimizer = torch.optim.AdamW(model.reg_head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_mse = float("inf")
    best_state = None
    best_epoch = 1

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, tcm_prior, train_loader, optimizer, device)
        val_metrics = evaluate(model, tcm_prior, val_loader, device)
        val_mse = float(val_metrics["mse"])

        if val_mse < best_mse:
            best_mse = val_mse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"[Epoch {epoch}/{args.epochs}] train_loss={train_loss:.6f} "
            f"val_mse={val_metrics['mse']:.6f} val_rmse={val_metrics['rmse']:.6f} "
            f"val_mae={val_metrics['mae']:.6f} val_pearson={val_metrics['pearson']:.6f}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    final_val = evaluate(model, tcm_prior, val_loader, device)
    print(
        f"[{timestamp()}] >>> Best epoch={best_epoch} | "
        f"MSE={final_val['mse']:.6f} RMSE={final_val['rmse']:.6f} "
        f"MAE={final_val['mae']:.6f} Pearson={final_val['pearson']:.6f}",
        flush=True,
    )

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_epoch": best_epoch,
            "val_metrics": final_val,
            "config": vars(args),
        },
        save_path,
    )
    print(f"[{timestamp()}] >>> Saved best checkpoint to {save_path}")


if __name__ == "__main__":
    main()
