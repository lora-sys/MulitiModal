"""Training runtime (#49): config, device, masked loss, optimizer,
epochs/checkpoint/resume, early stop, determinism.

Engineering smoke on synthetic fixtures only (SMOKE_ONLY); a run here is
never scientific training evidence (#22/#47).
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from core.registry import resolve_device

from .batches import BatchProvider, assert_subject_disjoint
from .checkpoint import load_training_checkpoint, save_training_checkpoint


@dataclass
class TrainerConfig:
    seed: int = 0
    device: str = "cpu"
    epochs: int = 3
    learning_rate: float = 1e-3
    grad_clip: float | None = None
    early_stop_patience: int | None = None
    amp: bool = False


@dataclass
class FitReport:
    """Per-epoch history plus best-epoch bookkeeping; SMOKE_ONLY numbers."""

    history: list[dict] = field(default_factory=list)
    epochs_run: int = 0
    best_epoch: int | None = None
    best_eval_loss: float | None = None

    def to_jsonable(self) -> dict:
        return asdict(self)


class Trainer:
    """Fits a registry-built model over a BatchProvider.

    The model must expose `compute_loss(batch)` and `input_spec` (i.e. one of
    the core models). Eval providers must be subject-disjoint from training
    (#27 assertion). Everything runs on synthetic fixtures in this phase.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: TrainerConfig | None = None,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        self.config = config or TrainerConfig()
        self.device = resolve_device(self.config.device)
        self.model = model.to(self.device)
        model_config = getattr(model, "model_config", None)
        if model_config is None:
            raise ValueError(
                "model was not built by a default_*_model factory; cannot checkpoint"
            )
        self.model_config = model_config
        self.optimizer = optimizer or torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )
        self._seed_everything(self.config.seed)
        self.global_step = 0
        self._epochs_run = 0
        self.best_epoch: int | None = None
        self.best_eval_loss: float | None = None

    def _seed_everything(self, seed: int) -> None:
        random.seed(seed)
        torch.manual_seed(seed)

    def fit(
        self,
        train_provider: BatchProvider,
        eval_provider: BatchProvider | None = None,
        resume_state: dict | None = None,
    ) -> FitReport:
        cfg = self.config
        if eval_provider is not None:
            assert_subject_disjoint(train_provider, eval_provider)
        if cfg.early_stop_patience is not None and eval_provider is None:
            raise ValueError("early_stop_patience requires an eval_provider")

        start_epoch = 0
        patience_left = cfg.early_stop_patience
        if resume_state is not None:
            start_epoch = resume_state["epoch"]
            self.global_step = resume_state["global_step"]
            self.best_epoch = resume_state["best_epoch"]
            self.best_eval_loss = resume_state["best_eval_loss"]
            self.optimizer.load_state_dict(resume_state["optimizer"])
            torch.set_rng_state(resume_state["torch_rng_state"])
            if self.device.type == "cuda" and resume_state.get("cuda_rng_state"):
                torch.cuda.set_rng_state(resume_state["cuda_rng_state"])

        use_amp = cfg.amp and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        report = FitReport()

        for epoch in range(start_epoch, cfg.epochs):
            self.model.train()
            losses: list[float] = []
            for item in train_provider:
                loss, _ = self.model.compute_loss(item.batch.to(self.device))
                self.optimizer.zero_grad()
                if use_amp:
                    scaler.scale(loss).backward()
                    if cfg.grad_clip:
                        scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), cfg.grad_clip
                        )
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if cfg.grad_clip:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), cfg.grad_clip
                        )
                    self.optimizer.step()
                self.global_step += 1
                losses.append(float(loss.detach()))
            entry: dict = {
                "epoch": epoch,
                "train_loss": sum(losses) / len(losses) if losses else 0.0,
                "eval": None,
            }

            if eval_provider is not None:
                from .evaluation import Evaluator

                metrics = Evaluator().evaluate(self.model, eval_provider, self.device)
                entry["eval"] = metrics
                eval_loss = metrics["eval_loss"]
                improved = self.best_eval_loss is None or eval_loss < self.best_eval_loss
                if improved:
                    self.best_eval_loss = eval_loss
                    self.best_epoch = epoch
                    patience_left = cfg.early_stop_patience
                elif patience_left is not None:
                    patience_left -= 1
                    if patience_left <= 0:
                        report.history.append(entry)
                        break

            report.history.append(entry)

        report.epochs_run = len(report.history)
        report.best_epoch = self.best_epoch
        report.best_eval_loss = self.best_eval_loss
        self._epochs_run = start_epoch + len(report.history)
        return report

    def save_checkpoint(
        self,
        out_dir: str | Path,
        source_commit: str | None = None,
        smoke_only: bool = True,
        extra_state: dict | None = None,
    ) -> Path:
        state = {
            "epoch": self._epochs_run,
            "global_step": self.global_step,
            "best_epoch": self.best_epoch,
            "best_eval_loss": self.best_eval_loss,
            "optimizer": self.optimizer.state_dict(),
            "trainer_config": asdict(self.config),
            "torch_rng_state": torch.get_rng_state(),
        }
        if self.device.type == "cuda":
            state["cuda_rng_state"] = torch.cuda.get_rng_state()
        return save_training_checkpoint(
            self.model, state, out_dir, source_commit=source_commit,
            smoke_only=smoke_only, extra_state=extra_state,
        )

    @classmethod
    def load(
        cls, model_dir: str | Path, device: str = "cpu", epochs: int | None = None
    ) -> tuple[Trainer, dict]:
        """Rebuild a Trainer (model + optimizer + config) from a checkpoint.

        `epochs` overrides the stored total (e.g. resume a 2-epoch checkpoint
        into a 4-epoch schedule); resume_state["epoch"] is where fit continues.
        """
        model, state, metadata = load_training_checkpoint(model_dir, device=device)
        cfg = TrainerConfig(**state["trainer_config"])
        cfg.device = device
        if epochs is not None:
            cfg.epochs = epochs
        trainer = cls(model, cfg)
        trainer.global_step = state["global_step"]
        trainer.best_epoch = state["best_epoch"]
        trainer.best_eval_loss = state["best_eval_loss"]
        trainer.optimizer.load_state_dict(state["optimizer"])
        resume = {
            "epoch": state["epoch"],
            "global_step": state["global_step"],
            "best_epoch": state["best_epoch"],
            "best_eval_loss": state["best_eval_loss"],
            "optimizer": state["optimizer"],
            "torch_rng_state": state["torch_rng_state"],
            "cuda_rng_state": state.get("cuda_rng_state"),
        }
        assert metadata["model_type"]  # metadata returned for provenance logging
        return trainer, resume
