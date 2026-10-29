"""Phase-1 training runtime (#49): synthetic smoke Trainer/Evaluator/
Checkpoint/CLI plus the reserved #45 train_candidate / trusted_score surface.

Everything here runs on synthetic fixtures (SMOKE_ONLY); no real dataset
file is ever read. See issues #49 and #45 for the acceptance contracts.
"""

from .batches import (
    BatchItem,
    BatchProvider,
    SyntheticBatchProvider,
    assert_subject_disjoint,
)
from .checkpoint import (
    checkpoint_summary,
    load_training_checkpoint,
    save_training_checkpoint,
)
from .evaluation import METRIC_VERSION, Evaluator
from .protocols import TASK_BUILDERS, train_candidate, trusted_score
from .trainer import FitReport, Trainer, TrainerConfig

__all__ = [
    "BatchItem",
    "BatchProvider",
    "SyntheticBatchProvider",
    "assert_subject_disjoint",
    "checkpoint_summary",
    "load_training_checkpoint",
    "save_training_checkpoint",
    "METRIC_VERSION",
    "Evaluator",
    "TASK_BUILDERS",
    "trusted_score",
    "train_candidate",
    "FitReport",
    "Trainer",
    "TrainerConfig",
]
