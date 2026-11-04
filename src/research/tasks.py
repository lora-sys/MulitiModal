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

import json
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

    def resolve_editable(self, editable: str | None = None) -> str:
        """Return the run's anchor candidate file; reject frozen paths.

        Editable surface v2: ``editable`` is the ANCHOR for complexity pricing
        and provenance, not a one-file allowlist. The loop's patch guard lets
        the Agent edit any non-frozen ``src/**/*.py`` file.
        """
        from research.guard import is_frozen

        if editable is None:
            return self.default_editable
        if is_frozen(editable):
            raise ValueError(
                f"candidate {editable!r} is a frozen path; only model/train/"
                "feature sources are writable"
            )
        if not (str(editable).startswith("src/") and str(editable).endswith(".py")):
            raise ValueError(
                f"candidate {editable!r} must be a src/**/*.py path"
            )
        return str(editable)

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

    def build_fold(
        self, config: dict, manifest: dict, fold: int, folds: int,
        repeat: int, records: list,
    ):
        """Build a fold from an already-validated selection manifest and rows."""
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

    def build_fold(self, config, manifest, fold, folds, repeat, records):
        from research.frozen import build_fold

        return build_fold(
            config, manifest, fold, folds, repeat, records=records,
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


def validate_manifest_contract(
    task: str, manifest: dict, *, selection_pool: bool = False,
) -> None:
    """Validate the task-specific, hash-checked frozen manifest shape.

    The adapter loader verifies the canonical digest. This second check makes
    the task routing contract explicit before any unlock event can be charged.
    """
    import re

    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    for name in ("manifest_sha256", "dataset_sha256"):
        value = manifest.get(name)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"manifest {name} must be a lowercase SHA-256")
    modalities = manifest.get("modalities")
    if not isinstance(modalities, dict) or not modalities:
        raise ValueError("manifest modalities must be a non-empty object")
    split = manifest.get("split")
    row_map = manifest.get("subject_row_by_id")
    if not isinstance(split, dict) or not isinstance(row_map, dict) or not row_map:
        raise ValueError("manifest must contain split and subject_row_by_id objects")
    if selection_pool:
        split_seed = split.get("seed")
        if (
            isinstance(split_seed, bool)
            or not isinstance(split_seed, int)
            or split_seed < 0
        ):
            raise ValueError(
                "selection manifest split seed must be a non-negative integer"
            )
    if any(
        not isinstance(subject_id, str) or not subject_id
        or isinstance(row, bool) or not isinstance(row, int) or row < 1
        for subject_id, row in row_map.items()
    ):
        raise ValueError("manifest subject_row_by_id has an invalid subject or row")
    if len(set(row_map.values())) != len(row_map):
        raise ValueError("manifest subject_row_by_id rows must be unique")
    partitions = []
    for name in ("train", "dev", "holdout"):
        values = split.get(name)
        empty_pool_partition = selection_pool and name in ("dev", "holdout")
        if (
            not isinstance(values, list)
            or (not values and not empty_pool_partition)
        ):
            qualifier = "a list" if empty_pool_partition else "a non-empty list"
            raise ValueError(f"manifest split {name} must be {qualifier}")
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError(f"manifest split {name} contains an invalid subject id")
        if len(values) != len(set(values)):
            raise ValueError(f"manifest split {name} contains duplicate subjects")
        partitions.append(set(values))
    if any(partitions[i] & partitions[j]
           for i in range(len(partitions)) for j in range(i + 1, len(partitions))):
        raise ValueError("manifest subject partitions overlap")
    if set.union(*partitions) != set(row_map):
        raise ValueError("manifest partitions do not match subject_row_by_id")
    for name, modality in modalities.items():
        if not isinstance(name, str) or not name or not isinstance(modality, dict):
            raise ValueError("manifest modality entries must be named objects")
        dim = modality.get("dim")
        if isinstance(dim, bool) or not isinstance(dim, int) or dim < 1:
            raise ValueError(f"manifest modality {name!r} has an invalid dim")
    if task == "constitution":
        if manifest.get("task_type") != "derived_multilabel_classification":
            raise ValueError("constitution task requires a derived multilabel manifest")
    elif task == "syndrome":
        if (
            manifest.get("task") != "syndrome"
            or manifest.get("task_type") != "syndrome_classification"
        ):
            raise ValueError("syndrome task requires a syndrome classification manifest")
        n_classes = manifest.get("n_classes")
        if isinstance(n_classes, bool) or not isinstance(n_classes, int) or n_classes < 2:
            raise ValueError("syndrome manifest n_classes must be an integer >= 2")
        taxonomy_version = manifest.get("label_taxonomy_version")
        if not isinstance(taxonomy_version, str) or not taxonomy_version:
            raise ValueError("syndrome manifest requires label_taxonomy_version")
        label_tier = manifest.get("label_tier")
        if "text" not in modalities or not isinstance(label_tier, str):
            raise ValueError("syndrome manifest requires the text modality and label_tier")
        if label_tier == "PUBLISHED" and not selection_pool:
            subject_users = manifest.get("subject_user_by_id")
            user_partitions = manifest.get("user_partition_by_id")
            user_strata = manifest.get("user_stratum_by_id")
            if not all(
                isinstance(value, dict)
                for value in (subject_users, user_partitions, user_strata)
            ):
                raise ValueError(
                    "PUBLISHED syndrome manifest requires frozen user strata "
                    "and partition metadata"
                )
            if set(subject_users) != set(row_map):
                raise ValueError(
                    "PUBLISHED syndrome manifest requires complete subject-to-user mapping"
                )
            if any(
                not isinstance(user_id, str) or not user_id
                for user_id in subject_users.values()
            ):
                raise ValueError("PUBLISHED syndrome manifest has an invalid user mapping")
            if set(user_strata) != set(user_partitions):
                raise ValueError(
                    "PUBLISHED syndrome manifest frozen user strata do not "
                    "cover frozen partitions"
                )
            if any(
                not isinstance(user_id, str) or not user_id
                for user_id in user_partitions
            ) or any(
                not isinstance(stratum, str) or not stratum
                for stratum in user_strata.values()
            ):
                raise ValueError(
                    "PUBLISHED syndrome manifest has invalid user strata metadata"
                )
            if not set(subject_users.values()) <= set(user_partitions):
                raise ValueError(
                    "PUBLISHED syndrome subject mapping references an unfrozen user"
                )
            expected_parts = {"train": "train", "dev": "dev", "holdout": "holdout"}
            for part, user_part in expected_parts.items():
                if any(
                    user_partitions[subject_users[subject_id]] != user_part
                    for subject_id in split[part]
                ):
                    raise ValueError(
                        f"PUBLISHED syndrome user partitions disagree with split {part!r}"
                    )
    elif task == "toy":
        # SMOKE_ONLY example task (#61). It is generated, so there is no DOI,
        # no PUBLISHED tier and no user-partition metadata to check; validate
        # what the engine actually relies on: task routing, class count and
        # the declared synthetic label tier.
        if manifest.get("task") != "toy" or (
            manifest.get("task_type") != "toy_classification"
        ):
            raise ValueError("toy task requires a toy classification manifest")
        n_classes = manifest.get("n_classes")
        if isinstance(n_classes, bool) or not isinstance(n_classes, int) or n_classes < 2:
            raise ValueError("toy manifest n_classes must be an integer >= 2")
        if manifest.get("label_tier") != "SMOKE_ONLY":
            raise ValueError("toy manifest must be explicitly SMOKE_ONLY")
    else:
        raise ValueError(f"unknown research task {task!r}")


def write_synthetic_preflight_inputs(
    task: str,
    reference_manifest: dict,
    out_dir: str | Path,
) -> tuple[Path, Path]:
    """Write a tiny synthetic training contract for isolated candidate checks.

    Only the task schema and modality dimensions are copied from the frozen
    manifest. No dataset records or labels are read.
    """
    import random

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter = adapter_for(task)
    records = []
    if task == "constitution":
        from data.mendeley import FEATURE_GROUP_SIZES, SubjectRecord
        from data.real_split import build_real_manifest

        rng = random.Random(0)
        for index in range(1, 31):
            scores = [None] + [float(rng.randint(10, 90)) for _ in range(8)]
            records.append(SubjectRecord(
                subject_index=index,
                features={
                    name: [rng.random() for _ in range(size)]
                    for name, size in FEATURE_GROUP_SIZES.items()
                },
                scores=scores,
                score_mask=[value is not None for value in scores],
                provenance={"source": "synthetic-candidate-preflight"},
            ))
        frozen = build_real_manifest(
            records, dataset_sha256="0" * 64,
        ).data
    elif task == "syndrome":
        from research.task_sd import (
            FEATURE_GROUP,
            MODALITY_KIND_FLOAT_GROUPS,
            MODALITY_KIND_HASHED_TEXT,
            SyndromeRecord,
            build_syndrome_manifest,
        )

        modalities = reference_manifest["modalities"]
        source_modality = modalities["text"]
        kind = source_modality.get("kind", MODALITY_KIND_FLOAT_GROUPS)
        if kind not in (MODALITY_KIND_FLOAT_GROUPS, MODALITY_KIND_HASHED_TEXT):
            raise ValueError(f"unsupported syndrome preflight modality {kind!r}")
        n_classes = int(reference_manifest["n_classes"])
        taxonomy = reference_manifest["label_taxonomy_version"]
        dim = int(source_modality["dim"])
        for index in range(1, 31):
            rng = random.Random(index)
            features = (
                {}
                if kind == MODALITY_KIND_HASHED_TEXT
                else {FEATURE_GROUP: [rng.random() for _ in range(dim)]}
            )
            records.append(SyndromeRecord(
                subject_index=index,
                user_id=f"preflight-user-{index:04d}",
                norm_syndrome=f"synthetic-{index % 3}",
                features=features,
                input_text=f"synthetic symptom pattern {index}" if not features else "",
                label=index % n_classes,
                provenance={"source": "synthetic-candidate-preflight"},
            ))
        frozen = build_syndrome_manifest(
            records,
            dataset_sha256="0" * 64,
            n_classes=n_classes,
            label_taxonomy_version=taxonomy,
            label_tier="SMOKE_ONLY",
            modality_kind=kind,
            notes="synthetic candidate output-contract preflight",
        ).data
    elif task == "toy":
        # SMOKE_ONLY: build the same synthetic records the toy task uses, then
        # a frozen manifest through the toy adapter's own builder.
        import random

        from research.task_toy import ToyRecord, build_manifest

        rng = random.Random(0)
        for index in range(1, 31):
            records.append(ToyRecord(
                subject_index=index, user_id=f"preflight-user-{index:04d}",
                text=f"synthetic case {index}",
                numeric=[rng.random() for _ in range(4)],
                label=index % 3, label_mask=True,
            ))
        frozen = build_manifest(records, seed=0)
    else:
        raise ValueError(f"unknown research task {task!r}")
    fit_manifest = adapter.pool_manifest(
        frozen, records, notes="synthetic candidate output-contract preflight",
    ).data
    fit_rows = set(fit_manifest["subject_row_by_id"].values())
    fit_records = [record for record in records if record.subject_index in fit_rows]
    manifest_path = out_dir / "fit-manifest.json"
    manifest_path.write_text(json.dumps(fit_manifest, indent=2, ensure_ascii=False))
    records_path = out_dir / "fit-records.json"
    records_path.write_text(adapter.records_to_json(fit_records))
    return manifest_path, records_path
