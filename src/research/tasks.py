"""Task dispatch for the source-level AutoResearch engine (#53).

The engine was welded to the nine-dimension constitution table task: records
came from `data.mendeley`, folds from `data.real_split` + `runtime.real_data`,
scoring from `runtime.tcm_eval`. The engine is now task-parameterized: config
carries `task` — `"constitution"` (the frozen default, byte-identical
behavior) or `"syndrome"` (issue #53's TCM-SD text classification, adapter in
`research/task_sd.py`) — and every engine entrypoint (runner, train_worker,
evaluator, finalize, loop) resolves its data and scoring seams through the
adapter returned by `adapter_for`.

The adapter surface is exactly the seams the two tasks must NOT drift on:
which records a fold hands the candidate, how manifests are pruned, which
trusted evaluator scores the saved model, and which metric is primary.
Everything task-agnostic (guard, spawn, model_export, the finalize ledger
mechanics) stays shared code. The registry is a closed set, not a
plugin point: a new task is a code change, reviewed as one.

Nothing here imports the heavy task packages at module load; every adapter
method imports its own dependencies lazily, like the rest of src/research/.
"""

from __future__ import annotations

from pathlib import Path


class TaskAdapter:
    """The per-task seams of the research engine. Subclasses implement all."""

    #: registry name, and the value config["task"] must carry
    name: str = ""
    #: the editable file a fresh run of this task starts from
    default_editable: str = ""

    # -- doctor hooks (#61) --------------------------------------------------
    #
    # The Doctor is task-agnostic; every task-specific rule reaches it through
    # these hooks. A hook that returns None means the check CANNOT BE PERFORMED
    # and the Doctor reports UNVERIFIED — never that it passed.

    #: does this task's input include raw text (do text-leak checks apply)?
    has_text: bool = False
    #: does a record carry exactly one label (do single-label checks apply)?
    single_label: bool = False

    def record_user_key(self, record):
        """Identity of the real subject behind a record, for the #1 leak check."""
        return None

    def record_text_key(self, record):
        """The input text, for exact-duplicate and label-conflict checks."""
        return None

    def record_label_key(self, record):
        """The single label of a record, or None."""
        return None

    def record_has_label(self, record) -> bool | None:
        """Is any label observable for this record?"""
        return None

    def label_space(self, manifest: dict):
        """The frozen class taxonomy, in the order the evaluator indexes it."""
        return None

    def metric_names(self):
        """Metric names the trusted evaluator can actually report."""
        return None

    @property
    def primary_metric(self) -> str:
        raise NotImplementedError

    # -- records -------------------------------------------------------------

    def load_manifest(self, path: str | Path) -> dict:
        raise NotImplementedError

    def load_records(self, dataset_path: str | Path) -> list:
        """The full-set records, from the task's dataset file."""
        raise NotImplementedError

    def load_eval_records(self, config: dict) -> list:
        """Trusted-side records for the evaluator: the JSON the driver wrote,
        or the dataset file."""
        raise NotImplementedError

    def records_from_json(self, text: str) -> list:
        raise NotImplementedError

    def records_to_json(self, records: list) -> str:
        raise NotImplementedError

    # -- manifests -----------------------------------------------------------

    def fold_manifest(self, manifest: dict, records: list,
                      train_ids: list[str], dev_ids: list[str],
                      notes: str = ""):
        """Prune a frozen manifest to one selection fold (holdout emptied,
        row mapping pruned, fold-own statistics)."""
        raise NotImplementedError

    def pool_manifest(self, frozen: dict, records: list, notes: str = ""):
        """The selection-pool manifest for the finalize refit."""
        raise NotImplementedError

    # -- folds ---------------------------------------------------------------

    def build_fold(self, config: dict, manifest_path, dataset_path,
                   fold: int, folds: int, repeat: int, records=None):
        """(candidate-facing FoldData, fold manifest) for one selection fold."""
        raise NotImplementedError

    def training_fold_data(self, fit_manifest: dict, records, config: dict):
        """The FoldData a candidate trains against: train rows only, no dev."""
        raise NotImplementedError

    # -- scoring ---------------------------------------------------------------

    def evaluate_model(self, model, fold_manifest: dict, provider,
                       partition: str):
        """The trusted evaluation of one saved model against one partition."""
        raise NotImplementedError

    def score_holdout(self, config: dict, request: dict, records,
                      model) -> dict:
        """The single holdout read of a finalize run, plus baselines."""
        raise NotImplementedError


class ConstitutionAdapter(TaskAdapter):
    """The nine-dimension constitution table task, exactly as before #53.

    Every method is the code the engine used to inline, unchanged — this
    adapter exists so the syndrome task can share the engine without the
    constitution path moving at all."""

    name = "constitution"
    default_editable = "src/research/candidate.py"

    # -- doctor hooks ---------------------------------------------------------
    # A tabular task with no text and eight independent labels: text-leak and
    # label-conflict checks do not apply, and saying so is more honest than
    # returning a value that pretends they were performed.
    has_text = False
    single_label = False

    def record_user_key(self, record):
        # this dataset has no subject column; identity is the row, which is
        # exactly the assumption the reports record as UNVERIFIED
        provenance = getattr(record, "provenance", None) or {}
        subject = provenance.get("subject_id") if isinstance(provenance, dict) else None
        if subject:
            return str(subject)
        index = getattr(record, "subject_index", None)
        return None if index is None else f"row-{index}"

    def record_has_label(self, record) -> bool | None:
        mask = getattr(record, "score_mask", None)
        return None if mask is None else bool(any(mask))

    def metric_names(self):
        from runtime.tcm_eval import PRIMARY_METRIC

        return (PRIMARY_METRIC,)

    @property
    def primary_metric(self) -> str:
        from runtime.tcm_eval import PRIMARY_METRIC

        return PRIMARY_METRIC

    def load_manifest(self, path):
        from data.real_split import load_real_manifest

        return load_real_manifest(path)

    def load_records(self, dataset_path):
        from data.mendeley import load_mendeley_scores

        return load_mendeley_scores(dataset_path)

    def load_eval_records(self, config):
        from data.mendeley import load_mendeley_scores, records_from_json

        data = config["data"]
        if data.get("records_path"):
            return records_from_json(Path(data["records_path"]).read_text())
        return load_mendeley_scores(data["dataset_path"])

    def records_from_json(self, text):
        from data.mendeley import records_from_json

        return records_from_json(text)

    def records_to_json(self, records):
        from data.mendeley import records_to_json

        return records_to_json(records)

    def fold_manifest(self, manifest, records, train_ids, dev_ids, notes=""):
        from data.real_split import fold_manifest

        return fold_manifest(manifest, records, train_ids, dev_ids, notes=notes)

    def pool_manifest(self, frozen, records, notes=""):
        from data.real_split import pool_manifest

        return pool_manifest(frozen, records, notes=notes)

    def build_fold(self, config, manifest_path, dataset_path, fold, folds,
                   repeat, records=None):
        from research.frozen import build_fold

        return build_fold(
            config, manifest_path, dataset_path, fold, folds, repeat,
            records=records,
        )

    def training_fold_data(self, fit_manifest, records, config):
        from research.frozen import training_fold_data

        return training_fold_data(fit_manifest, records, config)

    def evaluate_model(self, model, fold_manifest, provider, partition):
        from runtime.real_data import real_eval_manifest
        from runtime.tcm_eval import TCMTrustedEvaluator

        return TCMTrustedEvaluator(real_eval_manifest(fold_manifest)).evaluate(
            model, provider, partition=partition,
        )

    def score_holdout(self, config, request, records, model) -> dict:
        from data.real_split import holdout_view, load_real_manifest
        from runtime.baselines import evaluate_baselines
        from runtime.real_data import RealBatchProvider

        pool = load_real_manifest(config["data"]["pool_manifest_path"])
        frozen = load_real_manifest(config["data"]["frozen_manifest_path"])
        view = holdout_view(pool, frozen)
        holdout = RealBatchProvider(view, records, "holdout", batch_size=64)
        report = self.evaluate_model(model, view, holdout, partition="holdout")
        baselines = evaluate_baselines(view, records)
        return {
            "holdout": report.to_dict(),
            "baselines": {
                name: {
                    "macro_ap": vals["holdout"]["macro_ap"],
                    "macro_f1": vals["holdout"]["macro_f1"],
                    "subject_mae": vals["holdout"]["subject_mae"],
                }
                for name, vals in baselines.items()
                if isinstance(vals, dict) and "holdout" in vals
            },
        }


def adapter_for(task: str) -> TaskAdapter:
    """The adapter for config["task"]; anything unknown is refused loudly."""
    if task == "constitution":
        return ConstitutionAdapter()
    if task == "syndrome":
        from research.task_sd import SyndromeAdapter

        return SyndromeAdapter()
    if task == "toy":
        # the TCM-free example task (#61): proves the engine can run with no
        # project data at all, on a synthetic dataset it generates itself
        from research.task_toy import ToyAdapter

        return ToyAdapter()
    raise ValueError(
        f"unknown research task {task!r}; expected 'constitution', 'syndrome' or 'toy'"
    )
