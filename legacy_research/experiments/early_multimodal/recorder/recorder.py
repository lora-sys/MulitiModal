import os
import json
import csv
import time
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import numpy as np
import matplotlib.pyplot as plt
import torch

from metrics import MetricsResult, compute_metrics


@dataclass
class ExperimentConfig:
    experiment_id: str
    run_id: str
    seed: int
    model: str
    fusion_type: str
    batch_size: int
    lr: float
    optimizer: str
    weight_decay: float
    num_epochs: int
    num_workers: int
    device: str
    scheduler: str = ""
    warmup_epochs: int = 0
    encoder_lr_ratio: float = 1.0
    dropout: float = 0.3
    augmentation: str = "none"
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExperimentResult:
    experiment_id: str
    run_id: str
    seed: int
    model: str
    fusion_type: str
    batch_size: int
    lr: float
    optimizer: str
    weight_decay: float
    num_epochs: int
    num_workers: int
    device: str
    train_time_min: float
    train_loss_final: float
    val_loss_best: float
    val_epoch_of_best: int
    val_accuracy_best: float
    val_macro_f1_best: float
    test_accuracy: float
    test_macro_f1: float
    notes: str
    per_class_precision: str = ""
    per_class_recall: str = ""
    confusion_matrix_path: str = ""
    attention_stats_path: str = ""
    checkpoint_path: str = ""
    gpu_mem_peak_gb: float = 0.0
    train_steps_per_epoch: int = 0
    augmentation: str = "none"
    timestamp: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class ExperimentRecorder:
    CSV_REQUIRED_FIELDS = (
        'experiment_id', 'run_id', 'seed', 'model', 'fusion_type',
        'batch_size', 'lr', 'optimizer', 'weight_decay', 'num_epochs',
        'num_workers', 'device', 'train_time_min', 'train_loss_final',
        'val_loss_best', 'val_epoch_of_best', 'val_accuracy_best',
        'val_macro_f1_best', 'test_accuracy', 'test_macro_f1', 'notes'
    )

    CSV_OPTIONAL_FIELDS = (
        'per_class_precision', 'per_class_recall', 'confusion_matrix_path',
        'attention_stats_path', 'checkpoint_path', 'gpu_mem_peak_gb',
        'train_steps_per_epoch', 'augmentation', 'timestamp'
    )

    def __init__(
        self,
        output_dir: str,
        experiment_id: str,
        run_id: str,
        seed: int,
    ):
        self.output_dir = output_dir
        self.experiment_id = experiment_id
        self.run_id = run_id
        self.seed = seed
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.run_dir = os.path.join(output_dir, experiment_id, run_id)
        self.csv_path = os.path.join(output_dir, "experiments_log.csv")
        self.config_path = os.path.join(self.run_dir, "run_config.json")
        self.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
        self.figures_dir = os.path.join(self.run_dir, "figures")

        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

        self.history: Dict[str, List] = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_macro_f1': [],
            'lr': [],
        }

        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.best_epoch = 0

        self.config: Optional[ExperimentConfig] = None
        self.result: Optional[ExperimentResult] = None
        
    def save_config(
        self,
        model: str,
        fusion_type: str,
        batch_size: int,
        lr: float,
        optimizer: str = "AdamW",
        weight_decay: float = 1e-4,
        num_epochs: int = 50,
        num_workers: int = 0,
        device: str = "cpu",
        scheduler: str = "",
        warmup_epochs: int = 0,
        encoder_lr_ratio: float = 1.0,
        dropout: float = 0.3,
        augmentation: str = "none",
        notes: str = "",
        **kwargs
    ) -> None:
        self.config = ExperimentConfig(
            experiment_id=self.experiment_id,
            run_id=self.run_id,
            seed=self.seed,
            model=model,
            fusion_type=fusion_type,
            batch_size=batch_size,
            lr=lr,
            optimizer=optimizer,
            weight_decay=weight_decay,
            num_epochs=num_epochs,
            num_workers=num_workers,
            device=device,
            scheduler=scheduler,
            warmup_epochs=warmup_epochs,
            encoder_lr_ratio=encoder_lr_ratio,
            dropout=dropout,
            augmentation=augmentation,
            notes=notes,
        )

        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config.to_dict(), f, indent=2, ensure_ascii=False)

        print(f"[Recorder] 配置已保存: {self.config_path}")
    
    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_accuracy: float,
        val_macro_f1: float,
        lr: float = None,
        print_log: bool = True
    ) -> bool:
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['val_accuracy'].append(val_accuracy)
        self.history['val_macro_f1'].append(val_macro_f1)
        if lr is not None:
            self.history['lr'].append(lr)

        is_best = val_macro_f1 > self.best_val_f1
        if is_best:
            self.best_val_f1 = val_macro_f1
            self.best_val_acc = val_accuracy
            self.best_val_loss = val_loss
            self.best_epoch = epoch

        if print_log:
            best_marker = " *" if is_best else ""
            print(
                f"Epoch [{epoch+1:2d}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_accuracy:.2f}% F1: {val_macro_f1:.4f}"
                f"{best_marker}"
            )

        return is_best
    
    def save_result(
        self,
        test_metrics: MetricsResult,
        train_time_min: float,
        train_loss_final: float = None,
        notes: str = "",
    ) -> None:
        if train_loss_final is None:
            train_loss_final = self.history['train_loss'][-1] if self.history['train_loss'] else 0.0

        per_class_p = json.dumps(test_metrics.per_class_precision)
        per_class_r = json.dumps(test_metrics.per_class_recall)

        self.result = ExperimentResult(
            experiment_id=self.experiment_id,
            run_id=self.run_id,
            seed=self.seed,
            model=self.config.model if self.config else "unknown",
            fusion_type=self.config.fusion_type if self.config else "unknown",
            batch_size=self.config.batch_size if self.config else 0,
            lr=self.config.lr if self.config else 0.0,
            optimizer=self.config.optimizer if self.config else "unknown",
            weight_decay=self.config.weight_decay if self.config else 0.0,
            num_epochs=len(self.history['train_loss']),
            num_workers=self.config.num_workers if self.config else 0,
            device=self.config.device if self.config else "unknown",
            train_time_min=train_time_min,
            train_loss_final=train_loss_final,
            val_loss_best=self.best_val_loss,
            val_epoch_of_best=self.best_epoch + 1,
            val_accuracy_best=self.best_val_acc,
            val_macro_f1_best=self.best_val_f1,
            test_accuracy=test_metrics.accuracy * 100,
            test_macro_f1=test_metrics.macro_f1,
            notes=notes,
            per_class_precision=per_class_p,
            per_class_recall=per_class_r,
            timestamp=self.timestamp,
        )

        self._append_to_csv()
        print(f"[Recorder] 结果已保存到: {self.csv_path}")
    
    def _append_to_csv(self) -> None:
        file_exists = os.path.exists(self.csv_path)
        all_fields = self.CSV_REQUIRED_FIELDS + self.CSV_OPTIONAL_FIELDS

        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_fields)
            if not file_exists:
                writer.writeheader()
            row = self.result.to_dict()
            writer.writerow(row)

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        is_best: bool = False,
        extra_info: dict = None
    ) -> str:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_f1': self.best_val_f1,
            'best_val_acc': self.best_val_acc,
            'history': self.history,
        }

        if extra_info:
            checkpoint.update(extra_info)

        latest_path = os.path.join(self.checkpoint_dir, "latest.pth")
        torch.save(checkpoint, latest_path)

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best.pth")
            torch.save(checkpoint, best_path)
            if self.result:
                self.result.checkpoint_path = best_path
            return best_path

        return latest_path
    
    def save_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str] = None,
        title: str = None
    ) -> str:
        if class_names is None:
            class_names = ['一般', '正常', '良好']

        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)

        _fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names,
               yticklabels=class_names,
               ylabel='True label',
               xlabel='Predicted label')

        if title:
            ax.set_title(title)

        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()

        path = os.path.join(self.figures_dir, "confusion_matrix.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()

        if self.result:
            self.result.confusion_matrix_path = path

        print(f"[Recorder] 混淆矩阵已保存: {path}")
        return path

    def save_attention_stats(
        self,
        attention_weights: np.ndarray,
        gate_values: np.ndarray = None,
        labels: np.ndarray = None,
    ) -> str:
        stats = {
            'attention_weights': attention_weights,
            'gate_values': gate_values,
            'labels': labels,
        }

        path = os.path.join(self.run_dir, "attention_stats.npz")
        np.savez(path, **stats)

        if self.result:
            self.result.attention_stats_path = path

        print(f"[Recorder] Attention 统计已保存: {path}")
        return path
    
    def save_training_curves(self) -> str:
        if not self.history['train_loss']:
            return ""

        _fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        epochs = range(1, len(self.history['train_loss']) + 1)

        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss')
        axes[0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss Curves')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, self.history['val_accuracy'], 'g-', label='Val Accuracy')
        axes[1].axhline(y=self.best_val_acc, color='r', linestyle='--', label=f'Best: {self.best_val_acc:.2f}%')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(epochs, self.history['val_macro_f1'], 'm-', label='Val Macro F1')
        axes[2].axhline(y=self.best_val_f1, color='r', linestyle='--', label=f'Best: {self.best_val_f1:.4f}')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Macro F1')
        axes[2].set_title('Validation Macro F1')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.figures_dir, "training_curves.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"[Recorder] 训练曲线已保存: {path}")
        return path

    def get_summary(self) -> str:
        lines = [
            f"{'='*50}",
            f"Experiment: {self.experiment_id} / {self.run_id}",
            f"Seed: {self.seed}",
            f"{'='*50}",
        ]

        if self.config:
            lines.extend([
                f"Model: {self.config.model}",
                f"Fusion: {self.config.fusion_type}",
                f"LR: {self.config.lr}, Batch: {self.config.batch_size}",
            ])

        if self.history['train_loss']:
            lines.extend([
                f"\nBest Val: Epoch {self.best_epoch + 1}",
                f"  Loss: {self.best_val_loss:.4f}",
                f"  Acc:  {self.best_val_acc:.2f}%",
                f"  F1:   {self.best_val_f1:.4f}",
            ])

        if self.result:
            lines.extend([
                f"\nTest Results:",
                f"  Acc: {self.result.test_accuracy:.2f}%",
                f"  F1:  {self.result.test_macro_f1:.4f}",
                f"  Time: {self.result.train_time_min:.1f} min",
            ])

        return "\n".join(lines)
