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
  group (float_groups modality, v0: the featurizer runs OUTSIDE the loop;
  the candidate never sees raw text) OR the raw `input_text` (hashed_text
  modality, below), plus one integer syndrome label or None (unlabeled
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

Modality "hashed_text" (the real-scale path, ~44k cases): the manifest
freezes a STATELESS sklearn HashingVectorizer parameter set (char_wb 1-3
grams, 2^16 buckets, alternate_sign off, L2 norm) and the provider rebuilds
the vectorizer from those parameters and vectorizes each batch on the fly —
inlining 65k-dim vectors in records JSON is not storable, and materializing
them dense is not tractable. Because the vectorizer has no fitted state,
hard rule 3 (in-fold fitting) is vacuously satisfied: no statistics are
estimated from ANY data, train fold included, so the manifest records an
identity scaler and derived (fold/pool) manifests carry it verbatim. This
is the same shape as the constitution task's manifest-frozen scaler: the
evaluator subprocess never executes candidate code, and features are
rebuilt deterministically from manifest parameters alone. Raw case text
lives in the records JSON (upstream, outside Git — AGENTS.md rule 4), never
in the manifest.

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

MANIFEST_VERSION = "tcm-sd-manifest-v3"
TASK_TYPE = "syndrome_classification"
DATASET_ID = "tcm-sd-v1"
SELECTION_FOLD_VERSION = "tcm-sd-stratified-user-fold-v2"
#: v0 record contract: one feature group holding the frozen text features
FEATURE_GROUP = "frozen_text_features"
MODALITY_NAME = "text"

#: modality kinds: "float_groups" carries frozen inline vectors (v0, the
#: default); "hashed_text" carries raw text the provider vectorizes lazily
MODALITY_KIND_FLOAT_GROUPS = "float_groups"
MODALITY_KIND_HASHED_TEXT = "hashed_text"
#: frozen stateless vectorization (mainline arbitration): exact sklearn
#: HashingVectorizer semantics, recorded verbatim in the manifest modality
#: params and rebuilt from them by the provider. Seed-free by construction:
#: hashing has no fitted state and no randomness.
HASHED_TEXT_PARAMS = {
    "analyzer": "char_wb",
    "ngram_range": [1, 3],
    "n_features": 65536,
    "alternate_sign": False,
    "norm": "l2",
}

INPUT_SPEC_PREFIX = "tcm-sd-text-v1"
INPUT_SPEC_VERSION_HASHED_PREFIX = "tcm-sd-text-v2"
PRIMARY_METRIC_SD = "syndrome/macro_f1"
EVAL_METRIC_VERSION = "tcm-sd-eval-v3"
PREDICTION_KEY = "syndrome_probs"
TARGET_KEY = "syndrome_labels"

SUBJECT_ID_RE = re.compile(r"^sd-r\d{5,}$")


# -- records ------------------------------------------------------------------


@dataclass
class SyndromeRecord:
    """One case: frozen features OR raw text, one syndrome label or None.

    `label` is an index into the frozen taxonomy (0 .. n_classes-1). None
    means the case's label is unavailable — it stays unlabeled everywhere
    (provider mask, evaluator denominator); it is never guessed.

    `user_id` identifies the real subject behind the case (R1 F3): the
    manifest split is subject-level (AGENTS.md hard rule 2), and all cases
    of one user land in one partition. `norm_syndrome` feeds the same
    majority-label stratification the data layer uses; when unknown, the
    split degrades to a single stratum and stays deterministic.

    `input_text` is the raw case text for the hashed_text modality (short
    text or full clinical narrative, decided upstream; it may carry real
    clinical content and must never be committed to Git — AGENTS.md rule 4).
    Contract: `features` may be an empty dict if and only if `input_text` is
    non-empty and the manifest modality is hashed_text; a record with
    neither is refused everywhere (fail-closed).
    """

    subject_index: int
    user_id: str
    features: dict[str, list[float]]
    label: int | None
    norm_syndrome: str = ""
    input_text: str = ""
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
                "input_text": r.input_text,
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
            input_text=str(item.get("input_text", "")),
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


def _modality_kind(manifest: dict) -> str:
    """The manifest's text-modality kind ("float_groups" for v0 manifests)."""
    return manifest["modalities"][MODALITY_NAME].get(
        "kind", MODALITY_KIND_FLOAT_GROUPS
    )


def _hashing_vectorizer(params: dict):
    """Rebuild the frozen vectorizer from manifest parameters.

    Deterministic by construction: HashingVectorizer is stateless (nothing
    fitted, nothing random), so identical parameters yield identical
    features on every process, every run.
    """
    from sklearn.feature_extraction.text import HashingVectorizer

    return HashingVectorizer(
        analyzer=params["analyzer"],
        ngram_range=tuple(params["ngram_range"]),
        n_features=int(params["n_features"]),
        alternate_sign=bool(params["alternate_sign"]),
        norm=params["norm"],
    )


def _frozen_scaler(manifest: dict) -> dict | None:
    """The verbatim scaler for derived manifests, or None to refit.

    hashed_text features are stateless: derived (fold/pool) manifests carry
    the parent's identity scaler instead of a refit, which would crash on
    feature-less records and would estimate nothing anyway (hard rule 3 is
    vacuous when no statistics are ever fitted)."""
    if _modality_kind(manifest) != MODALITY_KIND_HASHED_TEXT:
        return None
    return json.loads(json.dumps(manifest["scaler"]))


def build_syndrome_manifest(
    records: list[SyndromeRecord],
    *,
    dataset_sha256: str,
    n_classes: int,
    label_taxonomy_version: str,
    label_tier: str = "PUBLISHED",
    split_seed: int = SPLIT_SEED,
    modality_kind: str = MODALITY_KIND_FLOAT_GROUPS,
    notes: str = "",
    frozen_user_partitions: dict[str, frozenset[str]] | None = None,
    frozen_user_strata: dict[str, str] | None = None,
):
    """Freeze the syndrome split into a hash-identified manifest.

    Mirrors data.real_split.build_real_manifest's structure and guarantees:
    the scaler is fit on TRAIN rows only (#29), and no raw text enters the
    manifest — only row indices, split assignment, scaler statistics and
    contract versions. `label_tier` records where labels came from
    ("PUBLISHED" for author-provided TCM-SD answers; "SMOKE_ONLY" for
    synthetic fixtures).

    PUBLISHED manifests require the caller to freeze subject_partitions over
    the complete input user set before evaluation filtering. Only explicit
    SMOKE_ONLY fixtures may derive partitions from their synthetic records.
    Passing frozen_user_partitions preserves that assignment after dev rows
    are removed. Every case for one user therefore
    remains in one partition, with the original user mapping included in the
    manifest hash. The data-layer third split is named holdout here.

    `modality_kind` selects the input contract: "float_groups" keeps the v0
    frozen inline vectors with a train-fold-fitted scaler (#29); "hashed_text"
    freezes the stateless HashingVectorizer parameter set into the modality
    params, requires every record to carry non-empty `input_text`, and
    records an identity scaler — the vectorizer has no fitted state, so no
    statistics are estimated from any data and the in-fold fitting rule is
    vacuously satisfied. Raw text never enters the manifest under either
    kind.
    """
    from data.real_split import RealManifest, _fit_scaler
    from data.tcm_sd import (
        DEV_RATIO,
        TEST_RATIO,
        TRAIN_RATIO,
        subject_partition_sha256,
        subject_partitions,
        user_majority_strata,
    )

    if n_classes < 2:
        raise ValueError(f"n_classes must be >= 2, got {n_classes}")
    if modality_kind not in (MODALITY_KIND_FLOAT_GROUPS, MODALITY_KIND_HASHED_TEXT):
        raise ValueError(
            f"unknown modality_kind {modality_kind!r}; expected "
            f"{(MODALITY_KIND_FLOAT_GROUPS, MODALITY_KIND_HASHED_TEXT)}"
        )
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
        if modality_kind == MODALITY_KIND_HASHED_TEXT:
            if not r.input_text.strip():
                raise ValueError(
                    f"record {r.subject_index} carries neither features nor "
                    "input_text; the hashed_text contract is fail-closed "
                    "(empty features are admitted only against non-empty text)"
                )
        elif FEATURE_GROUP not in r.features:
            raise ValueError(
                f"record {r.subject_index} carries no {FEATURE_GROUP!r} group; "
                "the v0 contract is one frozen feature vector per case"
            )

    if frozen_user_partitions is None:
        if label_tier != "SMOKE_ONLY":
            raise ValueError(
                "frozen_user_partitions is required for non-SMOKE_ONLY "
                "syndrome manifests"
            )
        user_parts = subject_partitions(records, seed=split_seed)
    else:
        user_parts = {
            name: frozenset(users)
            for name, users in frozen_user_partitions.items()
        }
    if frozen_user_strata is None:
        if label_tier != "SMOKE_ONLY":
            raise ValueError(
                "frozen_user_strata is required for non-SMOKE_ONLY "
                "syndrome manifests"
            )
        user_strata = user_majority_strata(records)
    else:
        user_strata = dict(frozen_user_strata)
    expected_user_parts = {"train", "dev", "test"}
    if set(user_parts) != expected_user_parts:
        raise ValueError(
            "frozen_user_partitions must contain exactly train, dev, and test"
        )
    all_partitioned_users: set[str] = set()
    for name in ("train", "dev", "test"):
        overlap = all_partitioned_users & user_parts[name]
        if overlap:
            raise ValueError(
                f"frozen user partitions overlap in {name!r}: "
                f"{sorted(overlap)[:3]}"
            )
        all_partitioned_users.update(user_parts[name])
    if set(user_strata) != all_partitioned_users or any(
        not isinstance(stratum, str) or not stratum
        for stratum in user_strata.values()
    ):
        raise ValueError(
            "frozen user strata must exactly cover every frozen user with "
            "a non-empty syndrome"
        )
    missing_users = {r.user_id for r in records} - all_partitioned_users
    if missing_users:
        raise ValueError(
            "records reference users outside frozen_user_partitions: "
            f"{sorted(missing_users)[:3]}"
        )
    user_partition_by_id = {
        user_id: partition
        for partition, source in (
            ("train", "train"), ("dev", "dev"), ("holdout", "test")
        )
        for user_id in sorted(user_parts[source])
    }
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
                f"{len(all_partitioned_users)} "
                "users; the subject-level ratios leave nothing to score"
            )
    # fold-only scaler: statistics from TRAIN rows exclusively (#29) — for
    # float_groups, whose features are frozen (featurized outside the loop),
    # this standardizes them; any future featurization that fits statistics
    # must go through this same train-fold-only path. hashed_text features
    # are stateless, so nothing is fit anywhere and the scaler is identity.
    if modality_kind == MODALITY_KIND_HASHED_TEXT:
        scaler = {MODALITY_NAME: {"kind": "identity"}}
        scaler_policy = (
            "identity: hashed_text features come from the stateless "
            "HashingVectorizer whose parameters are frozen in this manifest; "
            "no statistics are estimated from any data (in-fold rule #29 "
            "vacuous) and L2 normalization is built into the vectorizer"
        )
        scaler_fit_rows = 0
        modality_params = {
            "feature_groups": [FEATURE_GROUP],
            "dim": HASHED_TEXT_PARAMS["n_features"],
            "kind": MODALITY_KIND_HASHED_TEXT,
            **HASHED_TEXT_PARAMS,
        }
    else:
        scaler = _fit_scaler(
            partitions["train"], by_row, {MODALITY_NAME: (FEATURE_GROUP,)},
        )
        scaler_policy = (
            "per-column mean/std over finite values of TRAIN-fold rows only "
            "(#29); degenerate columns transform to 0.0; features are frozen "
            "vectors produced outside the loop"
        )
        scaler_fit_rows = len(partitions["train"])
        modality_params = {
            "feature_groups": [FEATURE_GROUP],
            "dim": len(scaler[MODALITY_NAME]["mean"]),
        }
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
        "modalities": {MODALITY_NAME: modality_params},
        "split": {
            "seed": split_seed,
            "fractions": {"train": TRAIN_RATIO, "dev": DEV_RATIO, "holdout": TEST_RATIO},
            **{p: [subject_id(r) for r in sorted(rows)]
               for p, rows in partitions.items()},
        },
        "subject_row_by_id": {subject_id(r): r for r in sorted(indices)},
        "subject_user_by_id": {
            subject_id(r.subject_index): r.user_id
            for r in sorted(records, key=lambda record: record.subject_index)
        },
        "user_partition_by_id": user_partition_by_id,
        "user_stratum_by_id": dict(sorted(user_strata.items())),
        # canonical hash over the complete frozen user→partition assignment;
        # derived manifests inherit it unchanged so the full assignment stays
        # auditable even after holdout users are pruned from a selection pool
        "user_partition_sha256": subject_partition_sha256(user_parts),
        "scaler": scaler,
        "scaler_policy": scaler_policy,
        "scaler_fit_rows": scaler_fit_rows,
        "notes": notes,
    }
    digest = _manifest_digest(data)
    data["manifest_sha256"] = digest
    return RealManifest(data=data, sha256=digest)


def build_syndrome_selection_artifacts(manifest, records):
    """Derive train+dev-only CV inputs from a frozen syndrome manifest.

    The user partition and majority-stratum maps are retained for every
    original train/dev user, including users whose rows were removed by
    scenario text filtering. This keeps CV assignment stable under row
    filtering. All holdout rows and users are removed from the derived
    artifact, and both dataset and manifest hashes are recomputed.
    """
    import hashlib

    from data.real_split import RealManifest

    data = json.loads(json.dumps(manifest.data))
    parent_sha = data.get("manifest_sha256")
    if not isinstance(parent_sha, str) or parent_sha != manifest.sha256:
        raise ValueError(
            "parent syndrome manifest SHA-256 is missing or inconsistent"
        )

    split = data.get("split")
    row_map = data.get("subject_row_by_id")
    subject_users = data.get("subject_user_by_id")
    user_partitions = data.get("user_partition_by_id")
    user_strata = data.get("user_stratum_by_id")
    if not all(
        isinstance(value, dict)
        for value in (split, row_map, subject_users, user_partitions, user_strata)
    ):
        raise ValueError("parent syndrome manifest lacks selection metadata")
    train_ids = split.get("train")
    dev_ids = split.get("dev")
    holdout_ids = split.get("holdout")
    if not all(
        isinstance(value, list)
        for value in (train_ids, dev_ids, holdout_ids)
    ):
        raise ValueError("parent syndrome manifest has invalid split lists")
    pool_ids = set(train_ids) | set(dev_ids)
    if (
        set(row_map) != pool_ids | set(holdout_ids)
        or set(subject_users) != set(row_map)
        or set(user_partitions) != set(user_strata)
    ):
        raise ValueError("parent syndrome manifest mappings are incomplete")
    pool_users = {
        user_id
        for user_id, partition in user_partitions.items()
        if partition in {"train", "dev"}
    }
    if any(
        user_partitions[user_id] not in {"train", "dev", "holdout"}
        for user_id in user_partitions
    ):
        raise ValueError("parent syndrome manifest has an invalid user partition")
    if any(
        user_partitions[subject_users[subject_id]] != expected
        for partition, expected in (
            (train_ids, "train"),
            (dev_ids, "dev"),
            (holdout_ids, "holdout"),
        )
        for subject_id in partition
    ):
        raise ValueError("parent syndrome user partitions disagree with split rows")
    if not pool_users <= set(user_strata):
        raise ValueError("parent syndrome manifest lacks frozen pool user strata")

    row_by_subject = {int(row): subject_id for subject_id, row in row_map.items()}
    expected_rows = {int(row_map[subject_id]) for subject_id in pool_ids}
    record_rows = [record.subject_index for record in records]
    if len(record_rows) != len(set(record_rows)):
        raise ValueError("parent syndrome records contain duplicate rows")
    if set(record_rows) != set(row_map.values()):
        raise ValueError("parent syndrome records do not match the manifest rows")
    selected_records = sorted(
        (record for record in records if record.subject_index in expected_rows),
        key=lambda record: record.subject_index,
    )
    selected_subject_ids = {
        row_by_subject[record.subject_index] for record in selected_records
    }
    if selected_subject_ids != pool_ids:
        raise ValueError("selection records do not cover parent train+dev rows")

    payload = records_to_json(selected_records)
    data["dataset_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    data["selection_parent_manifest_sha256"] = parent_sha
    data["selection_parent_dataset_sha256"] = manifest.data["dataset_sha256"]
    data["split"]["holdout"] = []
    data["subject_row_by_id"] = {
        subject_id: row_map[subject_id] for subject_id in sorted(pool_ids)
    }
    data["subject_user_by_id"] = {
        subject_id: subject_users[subject_id] for subject_id in sorted(pool_ids)
    }
    data["user_partition_by_id"] = {
        user_id: user_partitions[user_id] for user_id in sorted(pool_users)
    }
    data["user_stratum_by_id"] = {
        user_id: user_strata[user_id] for user_id in sorted(pool_users)
    }
    data["notes"] = (
        f"{data.get('notes', '')}; selection_pool=train+dev; holdout omitted"
    )
    data.pop("manifest_sha256", None)
    digest = _manifest_digest(data)
    data["manifest_sha256"] = digest
    return selected_records, RealManifest(data=data, sha256=digest)


def _rehash_manifest_data(data: dict) -> str:
    data.pop("manifest_sha256", None)
    digest = _manifest_digest(data)
    data["manifest_sha256"] = digest
    return digest


def _prune_syndrome_user_metadata(manifest):
    """Keep only in-scope subject identities in a derived manifest."""
    from data.real_split import RealManifest

    data = json.loads(json.dumps(manifest.data))
    row_ids = set(data["subject_row_by_id"])
    subject_users = data.get("subject_user_by_id")
    if not isinstance(subject_users, dict) or not row_ids <= set(subject_users):
        raise ValueError(
            "syndrome manifest lacks a complete subject-to-user mapping"
        )
    data["subject_user_by_id"] = {
        subject_id: subject_users[subject_id] for subject_id in sorted(row_ids)
    }
    user_strata = data.get("user_stratum_by_id")
    current_users = set(data["subject_user_by_id"].values())
    if not isinstance(user_strata, dict) or not current_users <= set(user_strata):
        raise ValueError(
            "syndrome manifest lacks complete frozen user strata"
        )
    data["user_stratum_by_id"] = {
        user_id: user_strata[user_id] for user_id in sorted(current_users)
    }
    data.pop("user_partition_by_id", None)
    digest = _rehash_manifest_data(data)
    return RealManifest(data=data, sha256=digest)


def _build_syndrome_fold_manifest(
    manifest: dict, records, train_ids, dev_ids, notes: str = ""
):
    from data.real_split import fold_manifest

    derived = fold_manifest(
        manifest, records, train_ids, dev_ids, notes=notes,
        scaler=_frozen_scaler(manifest),
    )
    return _prune_syndrome_user_metadata(derived)


def _build_syndrome_pool_manifest(manifest: dict, records, notes: str = ""):
    from data.real_split import pool_manifest

    derived = pool_manifest(
        manifest, records, notes=notes, scaler=_frozen_scaler(manifest)
    )
    return _prune_syndrome_user_metadata(derived)


def _input_spec_version(manifest: dict) -> str:
    taxonomy = manifest["label_taxonomy_version"]
    params = manifest["modalities"][MODALITY_NAME]
    if params.get("kind") == MODALITY_KIND_HASHED_TEXT:
        # v2 encodes the vectorization contract itself: kind + bucket count
        # (the full parameter set lives frozen in the manifest)
        return (
            f"{INPUT_SPEC_VERSION_HASHED_PREFIX}:{taxonomy}:"
            f"{params['kind']}:{params['n_features']}"
        )
    dims = ",".join(
        f"{name}={m['dim']}" for name, m in manifest["modalities"].items()
    )
    return f"{INPUT_SPEC_PREFIX}:{taxonomy}:{dims}"


def syndrome_input_spec(manifest: dict) -> InputSpecSet:
    hashed = _modality_kind(manifest) == MODALITY_KIND_HASHED_TEXT
    return InputSpecSet(
        version=_input_spec_version(manifest),
        modalities=tuple(
            ModalitySpec(
                name=name,
                kind="tabular",
                dim=params["dim"],
                units="hashed_text_l2" if hashed else "frozen_features",
                description=(
                    "L2-normalized hashed char 1-3 gram text features "
                    "(stateless, manifest-frozen; no scaler)"
                ) if hashed else "train-fold standardized frozen text features",
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

    Under the hashed_text modality the partition is held as raw text plus
    the manifest-frozen HashingVectorizer and each batch is vectorized on
    the fly; determinism is unchanged because the vectorizer is stateless.
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
        self._hashed = _modality_kind(manifest) == MODALITY_KIND_HASHED_TEXT
        if self._hashed:
            stats = manifest["scaler"][MODALITY_NAME]
            if stats.get("kind") != "identity":
                raise ValueError(
                    "hashed_text manifest carries a fitted scaler; the "
                    "stateless contract admits only the identity scaler"
                )
            self._vectorizer = _hashing_vectorizer(
                manifest["modalities"][MODALITY_NAME]
            )
        (self._features, self._availability,
         self._targets, self._target_mask) = self._materialize()
        self._epoch = 0

    def _materialize(self):
        if self._hashed:
            return self._materialize_hashed()
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

    def _materialize_hashed(self):
        """hashed_text rows stay as text; vectorization happens per batch.

        Nothing dense is materialized here (2^16 features x ~44k rows would
        not fit memory): `__iter__` vectorizes each batch on the fly from
        the manifest-frozen parameters. No z-score: the identity-scaler
        manifest was checked at build time, and L2 normalization is built
        into the vectorizer. Availability is input_text presence — records
        without text are refused here (fail-closed), so it is all True.
        """
        texts: list[str] = []
        availability: list[bool] = []
        labels: list[list[float]] = []
        mask: list[list[bool]] = []
        for sid in self.subject_ids:
            record = self.records_by_row[self.manifest["subject_row_by_id"][sid]]
            if not record.input_text.strip():
                raise ValueError(
                    f"record {record.subject_index} carries neither features "
                    "nor input_text; refusing a fail-closed contract breach"
                )
            label = record.label
            if label is not None and not 0 <= label < self.manifest["n_classes"]:
                raise ValueError(
                    f"record {record.subject_index} label {label} outside the "
                    f"frozen taxonomy of {self.manifest['n_classes']} classes"
                )
            texts.append(record.input_text)
            availability.append(True)
            labels.append([0.0 if label is None else float(label)])
            mask.append([label is not None])
        self._texts = texts
        return (
            {},
            {MODALITY_NAME: torch.tensor(availability, dtype=torch.bool)},
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
            if self._hashed:
                dense = self._vectorizer.transform(
                    [self._texts[i] for i in idx]
                ).toarray()
                features = {
                    MODALITY_NAME: torch.from_numpy(dense).to(torch.float32),
                }
            else:
                features = {k: v[idx] for k, v in self._features.items()}
            yield BatchItem(
                batch=ModelBatch(
                    features=features,
                    availability={k: v[idx] for k, v in self._availability.items()},
                    targets={k: v[idx] for k, v in self._targets.items()},
                    target_mask={k: v[idx] for k, v in self._target_mask.items()},
                ),
                subject_ids=[self.subject_ids[i] for i in idx],
            )


# -- trusted evaluation -------------------------------------------------------


def _per_class_metrics(
    pairs: list[tuple[int, int]], n_classes: int
) -> dict[str, dict]:
    if isinstance(n_classes, bool) or not isinstance(n_classes, int) or n_classes < 2:
        raise ValueError(f"n_classes must be an integer >= 2, got {n_classes!r}")
    if any(
        prediction < 0 or prediction >= n_classes
        or label < 0 or label >= n_classes
        for prediction, label in pairs
    ):
        raise ValueError("observed syndrome class is outside the frozen taxonomy")
    report = {}
    for class_id in range(n_classes):
        tp = sum(
            1 for prediction, label in pairs
            if prediction == class_id and label == class_id
        )
        fp = sum(
            1 for prediction, label in pairs
            if prediction == class_id and label != class_id
        )
        fn = sum(
            1 for prediction, label in pairs
            if prediction != class_id and label == class_id
        )
        support = tp + fn
        precision_denom = tp + fp
        f1_denom = 2 * tp + fp + fn
        report[str(class_id)] = {
            "precision": tp / precision_denom if precision_denom else 0.0,
            "recall": tp / support if support else 0.0,
            "f1": 2 * tp / f1_denom if f1_denom else 0.0,
            "support": support,
        }
    return report


def _macro_f1(pairs: list[tuple[int, int]], n_classes: int) -> float | None:
    """Macro-F1 over the complete frozen taxonomy; absent classes score 0."""
    if not pairs:
        return None
    per_class = _per_class_metrics(pairs, n_classes)
    return sum(metrics["f1"] for metrics in per_class.values()) / n_classes


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
        provider_ids = tuple(provider.subject_ids)
        duplicates = len(provider_ids) != len(set(provider_ids))
        missing = sorted(allowed - set(provider_ids))
        foreign = sorted(set(provider_ids) - allowed)
        if duplicates or missing or foreign:
            raise ValueError(
                f"provider sample set mismatch for partition {partition!r}: "
                f"missing={missing[:3]}, foreign={foreign[:3]}, "
                f"duplicates={duplicates}"
            )
        from core.batch import ModelOutput

        pairs: list[tuple[int, int]] = []
        n_subjects = len(provider_ids)
        for item in provider:
            with torch.no_grad():
                out = model(item.batch)
            if not isinstance(out, ModelOutput) or not isinstance(
                out.predictions, dict
            ):
                raise ValueError("model must return ModelOutput with a prediction mapping")
            if PREDICTION_KEY not in out.predictions:
                raise ValueError(
                    f"model did not predict {PREDICTION_KEY!r}; unknown "
                    "prediction keys are refused, never ignored"
                )
            probs = out.predictions[PREDICTION_KEY]
            labels = item.batch.targets[TARGET_KEY].squeeze(-1)
            mask = item.batch.target_mask[TARGET_KEY].squeeze(-1)
            if not isinstance(probs, torch.Tensor):
                raise ValueError(f"{PREDICTION_KEY!r} probability output must be a tensor")
            if probs.shape != (len(labels), self.manifest["n_classes"]):
                raise ValueError(
                    f"{PREDICTION_KEY!r} must be [B, n_classes="
                    f"{self.manifest['n_classes']}], got {tuple(probs.shape)}"
                )
            if not torch.isfinite(probs).all():
                raise ValueError(f"{PREDICTION_KEY!r} probabilities must be finite")
            if torch.any(probs < 0.0) or torch.any(probs > 1.0):
                raise ValueError(
                    f"{PREDICTION_KEY!r} probabilities must be within [0, 1]"
                )
            row_sums = probs.sum(dim=-1)
            if not torch.allclose(
                row_sums, torch.ones_like(row_sums), rtol=1e-4, atol=1e-4
            ):
                raise ValueError(
                    f"{PREDICTION_KEY!r} probabilities must sum to one per row"
                )
            pred = probs.argmax(dim=-1)
            for p, lbl, observed in zip(pred.tolist(), labels.tolist(),
                                        mask.tolist(), strict=True):
                if observed:
                    pairs.append((int(p), int(lbl)))
        smoke_only = self.manifest["label_tier"] == "SMOKE_ONLY"
        n_classes = int(self.manifest["n_classes"])
        macro = _macro_f1(pairs, n_classes)
        not_computable: tuple = ()
        if macro is None:
            not_computable = ("no labeled cases in this partition",)
        per_class = self._per_class(pairs, n_classes)
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
    def _per_class(pairs: list[tuple[int, int]], n_classes: int) -> dict:
        if not pairs:
            return {}
        return _per_class_metrics(pairs, n_classes)



# -- folds ----------------------------------------------------------------------


def build_fold(
    config: dict,
    manifest: dict,
    fold: int,
    folds: int,
    repeat: int,
    records: list[SyndromeRecord],
):
    """Build a fold from a validated selection-only manifest and records."""
    from research.tasks import validate_manifest_contract

    validate_manifest_contract("syndrome", manifest, selection_pool=True)
    if manifest["split"]["holdout"] != []:
        raise ValueError("selection manifest must have an empty holdout partition")
    if records is None:
        raise ValueError("fold records must come from validated selection inputs")

    train_ids, dev_ids = selection_fold_ids(manifest, fold, folds, repeat)
    row_by_id = manifest["subject_row_by_id"]
    pool_subject_ids = set(manifest["split"]["train"]) | set(manifest["split"]["dev"])
    pool_rows = {row_by_id[subject_id] for subject_id in pool_subject_ids}
    record_rows = [record.subject_index for record in records]
    if len(record_rows) != len(set(record_rows)):
        raise ValueError("duplicate subject_index in syndrome fold pool records")
    if set(record_rows) != pool_rows:
        raise ValueError("syndrome fold records do not match selection manifest rows")
    records_by_row = {record.subject_index: record for record in records}

    dev_before = len(dev_ids)
    text_dedup = {
        "policy": "one-way exact input_text dedup from dev against train",
        "text_basis": None,
        "n_before": dev_before,
        "n_dropped": 0,
        "n_after": dev_before,
    }
    if _modality_kind(manifest) == MODALITY_KIND_HASHED_TEXT:
        train_texts = {
            records_by_row[row_by_id[subject_id]].input_text
            for subject_id in train_ids
        }
        dev_ids = [
            subject_id
            for subject_id in dev_ids
            if records_by_row[row_by_id[subject_id]].input_text not in train_texts
        ]
        text_dedup.update(
            {
                "text_basis": "input_text",
                "n_dropped": dev_before - len(dev_ids),
                "n_after": len(dev_ids),
            }
        )

    derived = _build_syndrome_fold_manifest(
        manifest,
        records,
        train_ids,
        dev_ids,
        notes=(
            f"syndrome selection fold {fold} repeat {repeat}; "
            f"{SELECTION_FOLD_VERSION}"
        ),
    )
    data = derived.data
    data["selection_fold"] = {
        "version": SELECTION_FOLD_VERSION,
        "grouping": "user_id",
        "fold": fold,
        "folds": folds,
        "repeat": repeat,
        "dev_text_dedup": text_dedup,
    }
    _rehash_manifest_data(data)
    fold_rows = set(data["subject_row_by_id"].values())
    fold_records = [record for record in records if record.subject_index in fold_rows]
    batch_size = int(config.get("train_provider", {}).get("batch_size", 32))
    fold_data = FoldData(
        train=SyndromeBatchProvider(data, fold_records, "train", batch_size=batch_size),
        dev=SyndromeBatchProvider(data, fold_records, "dev", batch_size=64),
        input_spec=syndrome_input_spec(data),
        dims={
            "modalities": {
                name: params["dim"] for name, params in data["modalities"].items()
            },
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
        "holdout": {
            "macro_f1": _macro_f1(pairs, int(view["n_classes"])),
            "n_labeled": len(pairs),
        },
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

    def load_records_verified(self, dataset_path, expected_sha256):
        from research.evaluator import read_verified_dataset_bytes

        payload = read_verified_dataset_bytes(
            dataset_path, expected_sha256, "syndrome dataset"
        )
        return records_from_json(payload.decode("utf-8"))

    def load_eval_records(self, config):
        data = config["data"]
        if data.get("records_path"):
            raise ValueError(
                "syndrome evaluation records must come from the dataset bytes "
                "bound to the frozen manifest"
            )
        frozen = self.load_manifest(data["frozen_manifest_path"])
        return self.load_records_verified(
            data["dataset_path"], frozen["dataset_sha256"]
        )

    def records_from_json(self, text):
        return records_from_json(text)

    def records_to_json(self, records):
        return records_to_json(records)

    def fold_manifest(self, manifest, records, train_ids, dev_ids, notes=""):
        return _build_syndrome_fold_manifest(
            manifest, records, train_ids, dev_ids, notes=notes,
        )

    def pool_manifest(self, frozen, records, notes=""):
        return _build_syndrome_pool_manifest(frozen, records, notes=notes)

    def build_fold(self, config, manifest, fold, folds, repeat, records):
        return build_fold(
            config, manifest, fold, folds, repeat, records=records,
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
