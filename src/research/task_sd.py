"""The syndrome-task adapter (issue #53, config task="syndrome").

TCM-SD symptom/sign text -> syndrome-tendency classification, through the
same source-level AutoResearch engine as the constitution task. This module
is the task's seam definitions; the engine calls them via
`research.tasks.adapter_for("syndrome")`.

What is shared with the constitution task, deliberately: the manifest
CONTRACT of `data.real_split` (subject-level split #27, fold-internal scaler
#29, holdout emptied + row mapping pruned in every derived manifest) and the
frozen derivation `research.frozen.selection_fold_ids`. The manifest
machinery in data.real_split is record-shape agnostic (it only reads
`.subject_index` and `.features`), so the fold/pool/holdout/load functions
are reused verbatim — the leak-protection properties are inherited, not
re-promised.

What is task-owned here:

* `SyndromeRecord` — one case: a real subject `user_id` (R1 F3: the manifest
  split is subject-level, hard rule 2), a frozen feature vector per feature
  group (v0 contract: the featurizer runs OUTSIDE the loop; the candidate
  never sees raw text) plus one integer syndrome label or None (unlabeled
  cases are ABSTAIN, never zero-filled — AGENTS.md rule 5);
* `build_syndrome_manifest` — the frozen manifest builder (task_type
  "syndrome_classification", label taxonomy version, n_classes); the split
  reuses `data.tcm_sd.subject_partitions` as the single source of truth;
* `SyndromeBatchProvider` — batches in the core ModelBatch contract,
  targets `syndrome_labels` [B, 1] class indices with an observation mask;
* `SyndromeTrustedEvaluator` — the trusted scorer: macro-F1 over the frozen
  taxonomy (issue #53's primary metric), partition-leakage refusal,
  fail-closed when nothing is observable;
* the majority-class baseline for the holdout read.

Evidence state: this adapter and `research/candidate_sd.py` are validated on
synthetic fixtures only (SMOKE_ONLY — engineering validity, no scientific
claim; AGENTS.md rule 7). The real TCM-SD data path (download, audit,
featurization) lands via #53 scope 1 in src/data/ + src/syndrome/ and is
expected to produce these records/manifests; the JSON contracts here are the
interface that work must satisfy.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import torch

from core.batch import ModelBatch
from core.specs import InputSpecSet, ModalitySpec
from data.tcm_sd import SPLIT_SEED
from research.frozen import FoldData, selection_fold_ids
from research.tasks import TaskAdapter

MANIFEST_VERSION = "tcm-sd-manifest-v2"
TASK_TYPE = "syndrome_classification"
DATASET_ID = "tcm-sd-v1"
#: v0 record contract: one feature group holding the frozen text features
FEATURE_GROUP = "frozen_text_features"
MODALITY_NAME = "text"

INPUT_SPEC_PREFIX = "tcm-sd-text-v1"
PRIMARY_METRIC_SD = "syndrome/macro_f1"
EVAL_METRIC_VERSION = "tcm-sd-eval-v1"
PREDICTION_KEY = "syndrome_probs"
TARGET_KEY = "syndrome_labels"

SUBJECT_ID_RE = re.compile(r"^sd-r\d{5,}$")


# -- records ------------------------------------------------------------------


@dataclass
class SyndromeRecord:
    """One case: frozen features per group, one syndrome label or None.

    `label` is an index into the frozen taxonomy (0 .. n_classes-1). None
    means the case's label is unavailable — it stays unlabeled everywhere
    (provider mask, evaluator denominator); it is never guessed.

    `user_id` identifies the real subject behind the case (R1 F3): the
    manifest split is subject-level (AGENTS.md hard rule 2), and all cases
    of one user land in one partition. `norm_syndrome` feeds the same
    majority-label stratification the data layer uses; when unknown, the
    split degrades to a single stratum and stays deterministic.
    """

    subject_index: int
    user_id: str
    features: dict[str, list[float]]
    label: int | None
    norm_syndrome: str = ""
    provenance: dict = field(default_factory=dict)


def records_to_json(records: list[SyndromeRecord]) -> str:
    return json.dumps(
        [
            {
                "subject_index": r.subject_index,
                "user_id": r.user_id,
                "features": r.features,
                "label": r.label,
                "norm_syndrome": r.norm_syndrome,
                "provenance": r.provenance,
            }
            for r in records
        ],
        ensure_ascii=False,
    )


def records_from_json(text: str) -> list[SyndromeRecord]:
    return [
        SyndromeRecord(
            subject_index=int(item["subject_index"]),
            user_id=str(item["user_id"]),
            features={
                group: [float(v) for v in values]
                for group, values in item["features"].items()
            },
            label=None if item.get("label") is None else int(item["label"]),
            norm_syndrome=str(item.get("norm_syndrome", "")),
            provenance=dict(item.get("provenance", {})),
        )
        for item in json.loads(text)
    ]


# -- manifests ----------------------------------------------------------------


def _manifest_digest(data: dict) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(data, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def subject_id(row: int) -> str:
    return f"sd-r{row:05d}"


def build_syndrome_manifest(
    records: list[SyndromeRecord],
    *,
    dataset_sha256: str,
    n_classes: int,
    label_taxonomy_version: str,
    label_tier: str = "PUBLISHED",
    split_seed: int = SPLIT_SEED,
    notes: str = "",
):
    """Freeze the syndrome split into a hash-identified manifest.

    Mirrors data.real_split.build_real_manifest's structure and guarantees:
    the scaler is fit on TRAIN rows only (#29), and no raw text enters the
    manifest — only row indices, split assignment, scaler statistics and
    contract versions. `label_tier` records where labels came from
    ("PUBLISHED" for author-provided TCM-SD answers; "SMOKE_ONLY" for
    synthetic fixtures).

    The split itself is NOT drawn here (R1 F3): the internal 0.70/0.15/0.15
    case-level shuffle is gone, and the partition of the records' deduplicated
    user set is delegated to the single source of truth
    `data.tcm_sd.subject_partitions(records, seed=split_seed)`. That is the
    AGENTS.md hard rule 2 semantics: every case of one user_id lands in
    exactly one partition, partitions are pairwise disjoint by user, and the
    assignment is a deterministic function of (records, seed). The
    data-layer partitions name the third split "test"; the manifest exposes
    it as "holdout" (the engine's naming) with identical membership.
    """
    from data.real_split import RealManifest, _fit_scaler
    from data.tcm_sd import DEV_RATIO, TEST_RATIO, TRAIN_RATIO, subject_partitions

    if n_classes < 2:
        raise ValueError(f"n_classes must be >= 2, got {n_classes}")
    indices = [r.subject_index for r in records]
    if len(set(indices)) != len(indices):
        raise ValueError(
            "duplicate subject_index in records; refusing to build a manifest"
        )
    if len(indices) < 10:
        raise ValueError(f"too few cases for a split: {len(indices)}")
    if any(not r.user_id for r in records):
        raise ValueError(
            "record without user_id; the subject-level split (hard rule 2) "
            "cannot be built from anonymous cases"
        )
    for r in records:
        if r.label is not None and not 0 <= r.label < n_classes:
            raise ValueError(
                f"record {r.subject_index} label {r.label} outside the frozen "
                f"taxonomy of {n_classes} classes"
            )
        if FEATURE_GROUP not in r.features:
            raise ValueError(
                f"record {r.subject_index} carries no {FEATURE_GROUP!r} group; "
                "the v0 contract is one frozen feature vector per case"
            )

    user_parts = subject_partitions(records, seed=split_seed)
    for a in ("train", "dev", "test"):
        for b in ("train", "dev", "test"):
            if a < b and user_parts[a] & user_parts[b]:
                raise ValueError(
                    f"subject_partitions returned overlapping users for "
                    f"{a!r}/{b!r}; refusing a leaky manifest"
                )
    by_row = {r.subject_index: r for r in records}
    partitions = {
        part: sorted(
            r.subject_index
            for r in records
            if r.user_id in user_parts[source]
        )
        for part, source in (("train", "train"), ("dev", "dev"), ("holdout", "test"))
    }
    for name, members in partitions.items():
        if not members:
            raise ValueError(
                f"empty {name!r} partition for {len(records)} cases over "
                f"{len(user_parts['train'] | user_parts['dev'] | user_parts['test'])} "
                "users; the subject-level ratios leave nothing to score"
            )
    # fold-only scaler: statistics from TRAIN rows exclusively (#29). The
    # features are frozen (featurized outside the loop), so this standardizes
    # them; any future featurization that fits statistics must go through
    # this same train-fold-only path.
    scaler = _fit_scaler(
        partitions["train"], by_row, {MODALITY_NAME: (FEATURE_GROUP,)},
    )
    data = {
        "version": MANIFEST_VERSION,
        "dataset_id": DATASET_ID,
        "dataset_sha256": dataset_sha256,
        "task": "syndrome",
        "task_type": TASK_TYPE,
        "standard_version": label_taxonomy_version,
        "label_tier": label_tier,
        "label_version": f"tcm-sd-labels-v1:{label_taxonomy_version}",
        "label_taxonomy_version": label_taxonomy_version,
        "n_classes": n_classes,
        "modalities": {
            MODALITY_NAME: {"feature_groups": [FEATURE_GROUP],
                            "dim": len(scaler[MODALITY_NAME]["mean"])},
        },
        "split": {
            "seed": split_seed,
            "fractions": {"train": TRAIN_RATIO, "dev": DEV_RATIO, "holdout": TEST_RATIO},
            **{p: [subject_id(r) for r in sorted(rows)]
               for p, rows in partitions.items()},
        },
        "subject_row_by_id": {subject_id(r): r for r in sorted(indices)},
        "scaler": scaler,
        "scaler_policy": (
            "per-column mean/std over finite values of TRAIN-fold rows only "
            "(#29); degenerate columns transform to 0.0; features are frozen "
            "vectors produced outside the loop"
        ),
        "scaler_fit_rows": len(partitions["train"]),
        "notes": notes,
    }
    digest = _manifest_digest(data)
    data["manifest_sha256"] = digest
    return RealManifest(data=data, sha256=digest)


def _input_spec_version(manifest: dict) -> str:
    dims = ",".join(
        f"{name}={params['dim']}" for name, params in manifest["modalities"].items()
    )
    return f"{INPUT_SPEC_PREFIX}:{manifest['label_taxonomy_version']}:{dims}"


def syndrome_input_spec(manifest: dict) -> InputSpecSet:
    return InputSpecSet(
        version=_input_spec_version(manifest),
        modalities=tuple(
            ModalitySpec(
                name=name,
                kind="tabular",
                dim=params["dim"],
                units="frozen_features",
                description="train-fold standardized frozen text features",
            )
            for name, params in manifest["modalities"].items()
        ),
    )


# -- batches ------------------------------------------------------------------


class SyndromeBatchProvider:
    """BatchProvider over one frozen syndrome manifest partition.

    Deterministic like runtime.real_data.RealBatchProvider (identical
    manifest + partition + batch_size -> identical batches; train epochs
    shuffle under a manifest-derived seed). Unlabeled cases travel with
    mask False and are skipped by the evaluator — never zero-filled.
    """

    def __init__(
        self,
        manifest: dict,
        records: list[SyndromeRecord],
        partition: str,
        batch_size: int = 32,
        subject_ids: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        if partition not in ("train", "dev", "holdout"):
            raise ValueError(f"unknown partition {partition!r}")
        self.manifest = manifest
        self.partition = partition
        self.batch_size = max(1, int(batch_size))
        all_ids = tuple(manifest["split"][partition])
        if subject_ids is None:
            self.subject_ids = all_ids
        else:
            allowed = set(all_ids)
            foreign = [s for s in subject_ids if s not in allowed]
            if foreign:
                raise ValueError(
                    f"subject_ids outside partition {partition!r}: {foreign[:3]}"
                )
            self.subject_ids = tuple(subject_ids)
        self.records_by_row = {r.subject_index: r for r in records}
        if not self.subject_ids:
            raise ValueError(
                f"partition {partition!r} has no cases; refusing to build a provider"
            )
        missing = [s for s in self.subject_ids
                   if manifest["subject_row_by_id"][s] not in self.records_by_row]
        if missing:
            raise ValueError(f"manifest cases missing from records: {missing[:3]}")
        (self._features, self._availability,
         self._targets, self._target_mask) = self._materialize()
        self._epoch = 0

    def _materialize(self):
        modalities = self.manifest["modalities"]
        features = {name: [] for name in modalities}
        availability = {name: [] for name in modalities}
        labels: list[list[float]] = []
        mask: list[list[bool]] = []
        for sid in self.subject_ids:
            record = self.records_by_row[self.manifest["subject_row_by_id"][sid]]
            for name, params in modalities.items():
                values = [
                    v for g in params["feature_groups"] for v in record.features[g]
                ]
                stats = self.manifest["scaler"][name]
                ok = not any(math.isnan(v) for v in values)
                scaled = [
                    0.0 if d or not ok else (v - m) / s
                    for v, m, s, d in zip(
                        values, stats["mean"], stats["std"],
                        stats["degenerate"], strict=True,
                    )
                ]
                features[name].append(scaled)
                availability[name].append(ok)
            label = record.label
            if label is not None and not 0 <= label < self.manifest["n_classes"]:
                raise ValueError(
                    f"record {record.subject_index} label {label} outside the "
                    f"frozen taxonomy of {self.manifest['n_classes']} classes"
                )
            labels.append([0.0 if label is None else float(label)])
            mask.append([label is not None])
        return (
            {name: torch.tensor(features[name], dtype=torch.float32)
             for name in modalities},
            {name: torch.tensor(availability[name], dtype=torch.bool)
             for name in modalities},
            {TARGET_KEY: torch.tensor(labels, dtype=torch.float32)},
            {TARGET_KEY: torch.tensor(mask, dtype=torch.bool)},
        )

    def input_spec(self) -> InputSpecSet:
        return syndrome_input_spec(self.manifest)

    def __len__(self) -> int:
        return (len(self.subject_ids) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        from runtime.batches import BatchItem

        n = len(self.subject_ids)
        order = list(range(n))
        if self.partition == "train":
            rng = random.Random(
                f"{self.manifest['manifest_sha256']}:{self._epoch}"
            )
            rng.shuffle(order)
            self._epoch += 1
        for lo in range(0, n, self.batch_size):
            hi = min(lo + self.batch_size, n)
            idx = order[lo:hi]
            yield BatchItem(
                batch=ModelBatch(
                    features={k: v[idx] for k, v in self._features.items()},
                    availability={k: v[idx] for k, v in self._availability.items()},
                    targets={k: v[idx] for k, v in self._targets.items()},
                    target_mask={k: v[idx] for k, v in self._target_mask.items()},
                ),
                subject_ids=[self.subject_ids[i] for i in idx],
            )


# -- trusted evaluation -------------------------------------------------------


def _macro_f1(pairs: list[tuple[int, int]]) -> float | None:
    """Macro-F1 over observed (predicted, true) pairs, classes = true ∪ pred.

    A class with no true and no predicted occurrence cannot appear; a class
    with an empty F1 denominator scores 0.0 (the sklearn convention), so the
    macro is always defined when at least one labeled pair exists.

    NOTE (R1 F4): this is an OBSERVED-CLASS average — its denominator is
    only the classes that actually occur in the pairs. It is deliberately
    different from `syndrome.metrics.macro_f1`, which averages over ALL 148
    frozen classes with zero F1 for absent ones. The two numbers must never
    be compared side by side (this one is >= the full-taxonomy number)."""
    if not pairs:
        return None
    classes = sorted({p for p, _ in pairs} | {lbl for _, lbl in pairs})
    f1s = []
    for c in classes:
        tp = sum(1 for p, lbl in pairs if p == c and lbl == c)
        fp = sum(1 for p, lbl in pairs if p == c and lbl != c)
        fn = sum(1 for p, lbl in pairs if p != c and lbl == c)
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom else 0.0)
    return sum(f1s) / len(f1s)


@dataclass(frozen=True)
class SyndromeReport:
    """Immutable evaluation report for the syndrome task (issue #53).

    The primary value is macro-F1 over the frozen label taxonomy. It is None
    when no label is observable (fail-closed: an empty observation must never
    look like a perfect score). With synthetic inputs the report is
    SMOKE_ONLY and carries no scientific claim."""

    metric_version: str
    primary_metric: str
    partition: str
    label_taxonomy_version: str
    input_spec_version: str
    smoke_only: bool
    n_subjects: int
    n_labeled: int
    macro_f1: float | None = None
    per_class: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    not_computable: tuple = field(default_factory=tuple)

    @property
    def primary_value(self) -> float | None:
        if self.primary_metric != PRIMARY_METRIC_SD:
            raise KeyError(f"unknown primary metric {self.primary_metric!r}")
        return self.macro_f1

    def to_dict(self) -> dict:
        return {
            "metric_version": self.metric_version,
            "primary_metric": self.primary_metric,
            "partition": self.partition,
            "label_taxonomy_version": self.label_taxonomy_version,
            "input_spec_version": self.input_spec_version,
            "smoke_only": self.smoke_only,
            "n_subjects": self.n_subjects,
            "n_labeled": self.n_labeled,
            "macro_f1": self.macro_f1,
            "per_class": dict(self.per_class),
            "provenance": dict(self.provenance),
            "not_computable": list(self.not_computable),
        }


class SyndromeTrustedEvaluator:
    """Trusted scorer for the syndrome task: frozen manifest, macro-F1,
    leakage refusal, SMOKE_ONLY reporting.

    Mirrors runtime.tcm_eval's contract: it consumes any model-like object
    (forward -> ModelOutput) plus a BatchProvider, refuses inconsistent
    input-spec versions, unknown prediction keys and cases crossing between
    partitions, and never trusts a number the model reports about itself.
    """

    PARTITIONS = ("train", "dev", "holdout")

    def __init__(self, manifest: dict) -> None:
        self.manifest = manifest
        sets = {p: set(manifest["split"][p]) for p in self.PARTITIONS}
        for a in self.PARTITIONS:
            for b in self.PARTITIONS:
                if a < b and sets[a] & sets[b]:
                    raise ValueError(
                        f"manifest leakage: case present in both {a!r} and {b!r}"
                    )
        self.input_spec_version = _input_spec_version(manifest)

    def evaluate(self, model, provider, partition: str) -> SyndromeReport:
        if partition not in self.PARTITIONS:
            raise ValueError(
                f"unknown partition {partition!r}; expected {self.PARTITIONS}"
            )
        if provider.partition != partition:
            raise ValueError(
                f"provider partition {provider.partition!r} does not match "
                f"the evaluated partition {partition!r}"
            )
        spec = provider.input_spec()
        if spec.version != self.input_spec_version:
            raise ValueError(
                f"input spec mismatch: provider spec {spec.version!r} "
                f"against manifest {self.input_spec_version!r} — refusing to "
                "score against a contract the manifest does not declare"
            )
        model_spec = getattr(model, "input_spec", None)
        if model_spec is not None and model_spec.version != self.input_spec_version:
            raise ValueError(
                f"input spec mismatch: model spec {model_spec.version!r} "
                f"against manifest {self.input_spec_version!r} — refusing to "
                "score a model against a contract it was not built for"
            )
        allowed = set(self.manifest["split"][partition])
        foreign = [s for s in provider.subject_ids if s not in allowed]
        if foreign:
            raise ValueError(
                f"leakage refusal: {foreign[:3]} are not in partition "
                f"{partition!r} of the frozen manifest"
            )
        pairs: list[tuple[int, int]] = []
        n_subjects = len(provider.subject_ids)
        for item in provider:
            out = model(item.batch)
            if PREDICTION_KEY not in out.predictions:
                raise ValueError(
                    f"model did not predict {PREDICTION_KEY!r}; unknown "
                    "prediction keys are refused, never ignored"
                )
            probs = out.predictions[PREDICTION_KEY]
            labels = item.batch.targets[TARGET_KEY].squeeze(-1)
            mask = item.batch.target_mask[TARGET_KEY].squeeze(-1)
            if probs.shape != (len(labels), self.manifest["n_classes"]):
                raise ValueError(
                    f"{PREDICTION_KEY!r} must be [B, n_classes="
                    f"{self.manifest['n_classes']}], got {tuple(probs.shape)}"
                )
            pred = probs.argmax(dim=-1)
            for p, lbl, observed in zip(pred.tolist(), labels.tolist(),
                                        mask.tolist(), strict=True):
                if observed:
                    pairs.append((int(p), int(lbl)))
        smoke_only = self.manifest["label_tier"] == "SMOKE_ONLY"
        macro = _macro_f1(pairs)
        not_computable: tuple = ()
        if macro is None:
            not_computable = ("no labeled cases in this partition",)
        per_class = self._per_class(pairs)
        return SyndromeReport(
            metric_version=EVAL_METRIC_VERSION,
            primary_metric=PRIMARY_METRIC_SD,
            partition=partition,
            label_taxonomy_version=self.manifest["label_taxonomy_version"],
            input_spec_version=self.input_spec_version,
            smoke_only=smoke_only,
            n_subjects=n_subjects,
            n_labeled=len(pairs),
            macro_f1=macro,
            per_class=per_class,
            provenance={
                "manifest_sha256": self.manifest["manifest_sha256"],
                "label_tier": self.manifest["label_tier"],
            },
            not_computable=not_computable,
        )

    @staticmethod
    def _per_class(pairs: list[tuple[int, int]]) -> dict:
        if not pairs:
            return {}
        classes = sorted({p for p, _ in pairs} | {lbl for _, lbl in pairs})
        report: dict[str, dict] = {}
        for c in classes:
            tp = sum(1 for p, lbl in pairs if p == c and lbl == c)
            fp = sum(1 for p, lbl in pairs if p == c and lbl != c)
            fn = sum(1 for p, lbl in pairs if p != c and lbl == c)
            support = tp + fn
            precision = tp / (tp + fp) if tp + fp else None
            recall = tp / support if support else None
            denom = 2 * tp + fp + fn
            report[str(c)] = {
                "precision": precision,
                "recall": recall,
                "f1": (2 * tp / denom) if denom else 0.0,
                "support": support,
            }
        return report


# -- folds ----------------------------------------------------------------------


def build_fold(config: dict, manifest_path, dataset_path, fold: int,
               folds: int, repeat: int, records=None):
    """One selection fold of the syndrome task: same derivation and the same
    manifest pruning as the constitution path (holdout emptied, row mapping
    pruned to the fold's own train∪dev ids, fold-own scaler)."""
    from data.real_split import fold_manifest as _prune_fold
    from data.real_split import load_real_manifest

    manifest = load_real_manifest(manifest_path)
    if records is None:
        records = records_from_json(Path(dataset_path).read_text())
    train_ids, dev_ids = selection_fold_ids(manifest, fold, folds, repeat)
    data = _prune_fold(
        manifest, records, train_ids, dev_ids,
        notes=f"syndrome selection fold {fold} repeat {repeat}",
    ).data
    fold_rows = set(data["subject_row_by_id"].values())
    fold_records = [r for r in records if r.subject_index in fold_rows]
    batch_size = int(config.get("train_provider", {}).get("batch_size", 32))
    fold_data = FoldData(
        train=SyndromeBatchProvider(data, fold_records, "train",
                                    batch_size=batch_size),
        dev=SyndromeBatchProvider(data, fold_records, "dev", batch_size=64),
        input_spec=syndrome_input_spec(data),
        dims={
            "modalities": {name: p["dim"] for name, p in data["modalities"].items()},
            "targets": int(data["n_classes"]),
        },
        n_train_subjects=len(train_ids),
        n_dev_subjects=len(dev_ids),
        seed=int(config.get("trainer", {}).get("seed", 0)),
    )
    return fold_data, data


def training_fold_data(fit_manifest: dict, records, config: dict) -> FoldData:
    """The FoldData a syndrome candidate trains against: train rows only,
    no dev, no holdout rows anywhere in the fit manifest (F-06/F-08)."""
    return FoldData(
        train=SyndromeBatchProvider(
            fit_manifest, records, "train",
            batch_size=int(config.get("train_batch_size", 32)),
        ),
        dev=None,
        input_spec=syndrome_input_spec(fit_manifest),
        dims={
            "modalities": {
                name: params["dim"]
                for name, params in fit_manifest["modalities"].items()
            },
            "targets": int(fit_manifest["n_classes"]),
        },
        n_train_subjects=len(fit_manifest["split"]["train"]),
        n_dev_subjects=0,
        seed=int(config.get("seed", 0)),
    )


# -- holdout baselines ----------------------------------------------------------


def syndrome_baselines(view: dict, records: list[SyndromeRecord]) -> dict:
    """The majority-class baseline for the single holdout read.

    The class prior comes from the POOL (train) rows only — the holdout is
    never used to fit anything, and the holdout labels are read exactly once
    here, in the trusted evaluator process."""
    by_row = {r.subject_index: r for r in records}
    train_labels = [
        by_row[view["subject_row_by_id"][s]].label for s in view["split"]["train"]
    ]
    labeled = [lbl for lbl in train_labels if lbl is not None]
    if not labeled:
        return {"majority": {
            "holdout": {"macro_f1": None, "n_labeled": 0},
            "not_computable": "no labeled pool rows",
        }}
    counts = Counter(labeled)
    majority = min(counts, key=lambda c: (-counts[c], c))
    pairs = [
        (majority, by_row[view["subject_row_by_id"][s]].label)
        for s in view["split"]["holdout"]
        if by_row[view["subject_row_by_id"][s]].label is not None
    ]
    return {"majority": {
        "holdout": {"macro_f1": _macro_f1(pairs), "n_labeled": len(pairs)},
    }}


# -- adapter --------------------------------------------------------------------


class SyndromeAdapter(TaskAdapter):
    """The TCM-SD syndrome classification task (issue #53)."""

    name = "syndrome"
    default_editable = "src/research/candidate_sd.py"

    @property
    def primary_metric(self) -> str:
        return PRIMARY_METRIC_SD

    def load_manifest(self, path):
        from data.real_split import load_real_manifest

        return load_real_manifest(path)

    def load_records(self, dataset_path):
        return records_from_json(Path(dataset_path).read_text())

    def load_eval_records(self, config):
        data = config["data"]
        if data.get("records_path"):
            return records_from_json(Path(data["records_path"]).read_text())
        return records_from_json(Path(data["dataset_path"]).read_text())

    def records_from_json(self, text):
        return records_from_json(text)

    def records_to_json(self, records):
        return records_to_json(records)

    def fold_manifest(self, manifest, records, train_ids, dev_ids, notes=""):
        from data.real_split import fold_manifest

        return fold_manifest(manifest, records, train_ids, dev_ids, notes=notes)

    def pool_manifest(self, frozen, records, notes=""):
        from data.real_split import pool_manifest

        return pool_manifest(frozen, records, notes=notes)

    def build_fold(self, config, manifest_path, dataset_path, fold, folds,
                   repeat, records=None):
        return build_fold(
            config, manifest_path, dataset_path, fold, folds, repeat,
            records=records,
        )

    def training_fold_data(self, fit_manifest, records, config):
        return training_fold_data(fit_manifest, records, config)

    def evaluate_model(self, model, fold_manifest, provider, partition):
        return SyndromeTrustedEvaluator(fold_manifest).evaluate(
            model, provider, partition=partition,
        )

    def score_holdout(self, config, request, records, model) -> dict:
        from data.real_split import holdout_view, load_real_manifest

        pool = load_real_manifest(config["data"]["pool_manifest_path"])
        frozen = load_real_manifest(config["data"]["frozen_manifest_path"])
        view = holdout_view(pool, frozen)
        holdout = SyndromeBatchProvider(view, records, "holdout", batch_size=64)
        report = self.evaluate_model(model, view, holdout, partition="holdout")
        return {
            "holdout": report.to_dict(),
            "baselines": syndrome_baselines(view, records),
        }
