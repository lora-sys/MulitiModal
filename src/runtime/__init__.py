"""Phase-1 training runtime (#49): synthetic smoke Trainer/Evaluator/
Checkpoint/CLI plus the reserved #45 train_candidate / trusted_score surface.

Everything here runs on synthetic fixtures (SMOKE_ONLY); no real dataset
file is ever read. See issues #49 and #45 for the acceptance contracts.
"""

from .agent import (
    AgentProvider,
    DeterministicStubProvider,
    ExternalAgentProvider,
    Proposal,
    apply_config_patch,
    validate_config_patch,
    validate_source_patch,
)
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
from .harness import AutoResearchHarness, default_tcm_runspec
from .manifest import RunManifestStore
from .protocols import TASK_BUILDERS, train_candidate, trusted_score
from .tcm_eval import (
    EVAL_METRIC_VERSION,
    PRIMARY_METRIC,
    EvalManifest,
    TCMReport,
    TCMTrustedEvaluator,
    compute_per_dim_stats,
    compute_subject_level_metrics,
    make_synthetic_tcm_manifest,
)
from .tcm_infer import TCM_OUTPUT_SCHEMA_VERSION, TCMPredictor
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
    "EVAL_METRIC_VERSION",
    "PRIMARY_METRIC",
    "EvalManifest",
    "TCMReport",
    "TCMTrustedEvaluator",
    "compute_per_dim_stats",
    "compute_subject_level_metrics",
    "make_synthetic_tcm_manifest",
    "TCM_OUTPUT_SCHEMA_VERSION",
    "TCMPredictor",
    "AgentProvider",
    "DeterministicStubProvider",
    "ExternalAgentProvider",
    "Proposal",
    "apply_config_patch",
    "validate_config_patch",
    "validate_source_patch",
    "AutoResearchHarness",
    "default_tcm_runspec",
    "RunManifestStore",
]
