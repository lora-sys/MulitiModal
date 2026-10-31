"""Frozen real-data manifest for TCM training (#50).

Turns audited Mendeley subject records into a frozen, hash-identified
manifest: subject-level split (#27), train-fold-fitted scaler (#29),
rule-derived multilabel labels or PUBLISHED scores as targets (#38), and
the deployable/full modality sets (#50 deployability requirement). The
manifest contains no raw questionnaire rows — only subject row indices,
split assignment, scaler statistics and contract versions — so it can be
committed-adjacent metadata; the underlying workbook stays git-ignored.

Contract decisions recorded here (NOT silent):
- derived_multilabel_classification: label = 1 iff published score >= 40
  (rule YES); TENDENCY [30,40) and NO count as 0; the balanced dimension
  (index 0) is always unobserved -> mask False, never negative-filled.
- score_regression: targets are the 8 published 0-100 scores; balanced
  unobserved.
- a modality row with any NaN feature is availability-masked (zero-filled
  embedding, never imputed values).
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from data.mendeley import SubjectRecord

MANIFEST_VERSION = "tcm-real-manifest-v1"
LABEL_VERSION_CLASSIFICATION = (
    "derived-multilabel-v1:RULE(ZYYXH/T157-2009):YES>=40 -> 1, "
    "TENDENCY/NO -> 0, balanced abstain"
)
LABEL_VERSION_REGRESSION = "published-scores-v1:PUBLISHED(0-100, 8/9 dims)"

TASK_TYPES = ("derived_multilabel_classification", "score_regression")
#: deployable Body/Vitals modalities (available without clinic examination
#: hardware, #50 deployability); "exam" adds meridian/pulse/tongue features
MODALITY_SETS = {
    "deployable": {
        "body_features": ("demographics", "anthropometrics"),
    },
    "full": {
        "body_features": ("demographics", "anthropometrics"),
        "exam_features": ("meridian", "pulse_wave", "tongue"),
    },
}


def _fit_scaler(
    train_rows: list[int],
    by_row: dict[int, SubjectRecord],
    groups_by_modality: dict[str, tuple[str, ...]],
) -> dict[str, dict]:
    """Column statistics from TRAIN rows exclusively (#29).

    A column with no finite train-fold value or zero variance is marked
    degenerate: transforms must emit 0.0 for it (neutral) instead of
    passing raw-magnitude values through at inference time.
    """
    scaler: dict[str, dict] = {}
    for modality, feature_groups in groups_by_modality.items():
        cols = [[v for g in feature_groups for v in by_row[r].features[g]] for r in train_rows]
        n_cols = len(cols[0])
        means: list[float] = []
        stds: list[float] = []
        degenerate: list[bool] = []
        for j in range(n_cols):
            finite = [c[j] for c in cols if not math.isnan(c[j])]
            if not finite:
                means.append(0.0)
                stds.append(1.0)
                degenerate.append(True)
                continue
            mean = sum(finite) / len(finite)
            std = math.sqrt(sum((v - mean) ** 2 for v in finite) / len(finite))
            if std == 0.0:
                means.append(mean)
                stds.append(1.0)
                degenerate.append(True)
                continue
            means.append(mean)
            stds.append(std)
            degenerate.append(False)
        scaler[modality] = {"mean": means, "std": stds, "degenerate": degenerate}
    return scaler


def _manifest_digest(data: dict) -> str:
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


@dataclass
class RealManifest:
    data: dict
    sha256: str

    def save(self, path: str | Path) -> str:
        Path(path).write_text(json.dumps(self.data, indent=2, ensure_ascii=False))
        return self.sha256


def build_real_manifest(
    records: list[SubjectRecord],
    *,
    dataset_sha256: str,
    task_type: str = "derived_multilabel_classification",
    modality_set: str = "full",
    split_seed: int = 20261030,
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    exclude_subject_rows: tuple[int, ...] = (),
    notes: str = "",
) -> RealManifest:
    """Build the frozen manifest; the scaler is fit on TRAIN subjects only."""
    if task_type not in TASK_TYPES:
        raise ValueError(f"unknown task_type {task_type!r}; expected {TASK_TYPES}")
    if modality_set not in MODALITY_SETS:
        raise ValueError(
            f"unknown modality_set {modality_set!r}; expected {sorted(MODALITY_SETS)}"
        )
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1")
    indices = [r.subject_index for r in records]
    if len(set(indices)) != len(indices):
        raise ValueError(
            "duplicate subject_index in records; refusing to build a manifest"
        )

    rows = [r.subject_index for r in records if r.subject_index not in exclude_subject_rows]
    if len(rows) < 10:
        raise ValueError(f"too few subjects left for a split: {len(rows)}")
    # shuffle the FULL index list once, fix the partition CUT POINTS on the
    # full count, and only then drop exclusions: partition membership of the
    # remaining subjects is stable no matter which rows are excluded, so
    # derived manifests (regression / deployable / QC) rebuild to the SAME
    # split as the frozen classification manifest.
    rng = random.Random(split_seed)
    order = [r.subject_index for r in records]
    rng.shuffle(order)
    n = len(order)
    n_train = int(n * fractions[0])
    n_dev = int(n * fractions[1])
    full_partitions = {
        "train": order[:n_train],
        "dev": order[n_train : n_train + n_dev],
        "holdout": order[n_train + n_dev :],
    }
    partitions = {
        name: [r for r in members if r not in exclude_subject_rows]
        for name, members in full_partitions.items()
    }
    for name, members in partitions.items():
        if not members:
            raise ValueError(
                f"empty {name!r} partition for {n} subjects with fractions "
                f"{list(fractions)}; adjust the fractions"
            )
    by_row = {r.subject_index: r for r in records}

    # --- fold-only scaler: statistics from TRAIN rows exclusively (#29) ----
    scaler = _fit_scaler(partitions["train"], by_row, MODALITY_SETS[modality_set])
    scaler_policy = (
        "per-column mean/std over finite values of TRAIN-fold rows only (#29); "
        "degenerate (no finite value or zero variance) columns transform to 0.0"
    )

    label_version = (
        LABEL_VERSION_CLASSIFICATION
        if task_type == "derived_multilabel_classification"
        else LABEL_VERSION_REGRESSION
    )
    data = {
        "version": MANIFEST_VERSION,
        "dataset_id": "mendeley-cxtph3tjsg-v1",
        "dataset_sha256": dataset_sha256,
        "standard_version": "ZYYXH/T157-2009",
        "label_tier": "RULE" if task_type == "derived_multilabel_classification" else "PUBLISHED",
        "label_version": label_version,
        "task_type": task_type,
        "modality_set": modality_set,
        "modalities": {
            modality: {
                "feature_groups": list(feature_groups),
                "dim": len(scaler[modality]["mean"]),
            }
            for modality, feature_groups in MODALITY_SETS[modality_set].items()
        },
        "split": {
            "seed": split_seed,
            "fractions": list(fractions),
            **{p: [f"mendeley-r{r:03d}" for r in sorted(rows)] for p, rows in partitions.items()},
        },
        "subject_row_by_id": {f"mendeley-r{r:03d}": r for r in sorted(rows)},
        "scaler": scaler,
        "scaler_policy": scaler_policy,
        "scaler_fit_rows": len(partitions["train"]),
        "excluded_subject_rows": list(exclude_subject_rows),
        "notes": notes,
    }
    digest = _manifest_digest(data)
    data["manifest_sha256"] = digest
    return RealManifest(data=data, sha256=digest)


def exclude_subjects_from_manifest(
    manifest: dict,
    records: list[SubjectRecord],
    exclude_rows: tuple[int, ...],
    notes: str = "",
) -> RealManifest:
    """Derive a QC-variant manifest WITHOUT reshuffling the frozen split.

    A fresh build_real_manifest with the same seed would reshuffle every
    remaining subject across partitions once the row set changes; this
    keeps the original partition assignment minus the excluded subjects
    and refits the scaler on the reduced train fold only (#eval E01).
    """
    data = json.loads(json.dumps(manifest))
    data.pop("manifest_sha256", None)
    dropped = {f"mendeley-r{r:03d}" for r in exclude_rows}
    for partition in ("train", "dev", "holdout"):
        data["split"][partition] = [
            s for s in data["split"][partition] if s not in dropped
        ]
        # a partition that was already empty (the selection-pool manifest has
        # no dev rows) is not "emptied" by the exclusion; one that loses its
        # last subject is
        if not data["split"][partition] and manifest["split"][partition]:
            raise ValueError(f"excluding {sorted(dropped)} empties {partition!r}")
    data["subject_row_by_id"] = {
        k: v for k, v in data["subject_row_by_id"].items() if k not in dropped
    }
    by_row = {r.subject_index: r for r in records}
    groups = {
        modality: tuple(params["feature_groups"])
        for modality, params in data["modalities"].items()
    }
    data["scaler"] = _fit_scaler(
        [
            data["subject_row_by_id"][s]
            for s in data["split"]["train"]
        ],
        by_row,
        groups,
    )
    data["excluded_subject_rows"] = sorted(
        set(data.get("excluded_subject_rows", [])) | set(exclude_rows)
    )
    # the scaler was refit on the REDUCED train fold: provenance must say so
    data["scaler_fit_rows"] = len(data["split"]["train"])
    if notes:
        data["notes"] = notes
    digest = _manifest_digest(data)
    data["manifest_sha256"] = digest
    return RealManifest(data=data, sha256=digest)


def pool_manifest(manifest: dict, records: list[SubjectRecord], notes: str = "") -> RealManifest:
    """Derive the SELECTION-POOL manifest (train + dev) for the final refit.

    AutoResearch selects a configuration by subject-level cross-validation
    over train+dev; once selected, the final model is fitted on every row
    that took part in selection. Fitting on train alone would discard 41 of
    232 selection subjects for no gain in honesty — dev is no longer a
    validation set once it has steered selection. The holdout partition is
    untouched and stays unread until the report scores it once.

    Like exclude_subjects_from_manifest, this keeps the frozen row-to-
    partition assignment and only moves dev rows into train; nothing is
    reshuffled. The scaler is refit on the pool, which is exactly the data
    the final model sees.
    """
    data = json.loads(json.dumps(manifest))
    data.pop("manifest_sha256", None)
    data["split"] = {
        "train": list(data["split"]["train"]) + list(data["split"]["dev"]),
        "dev": [],
        "holdout": list(data["split"]["holdout"]),
    }
    by_row = {r.subject_index: r for r in records}
    groups = {
        modality: tuple(params["feature_groups"])
        for modality, params in data["modalities"].items()
    }
    data["scaler"] = _fit_scaler(
        [data["subject_row_by_id"][s] for s in data["split"]["train"]],
        by_row,
        groups,
    )
    data["scaler_fit_rows"] = len(data["split"]["train"])
    data["selection_parent_manifest_sha256"] = manifest["manifest_sha256"]
    if notes:
        data["notes"] = notes
    digest = _manifest_digest(data)
    data["manifest_sha256"] = digest
    return RealManifest(data=data, sha256=digest)


def fold_manifest(
    manifest: dict,
    records: list[SubjectRecord],
    train_ids: Sequence[str],
    dev_ids: Sequence[str],
    notes: str = "",
) -> RealManifest:
    """Derive the manifest for ONE cross-validation fold.

    Two things must not survive from the parent manifest:

    1. **The scaler.** Keeping the parent's statistics means every validation
       subject that also sits in the parent's train partition contributes to
       the mean/std used to train the fold model — a leak into the very
       number the fold exists to produce. The scaler is refit on the fold's
       train rows only.
    2. **The holdout partition.** Emptied here, so "selection never reads the
       holdout" becomes a structural property of the fold manifest instead of
       a convention the calling code is trusted to follow.
    """
    data = json.loads(json.dumps(manifest))
    data.pop("manifest_sha256", None)
    data["split"] = {
        "train": list(train_ids),
        "dev": list(dev_ids),
        "holdout": [],
    }
    by_row = {r.subject_index: r for r in records}
    groups = {
        modality: tuple(params["feature_groups"])
        for modality, params in data["modalities"].items()
    }
    data["scaler"] = _fit_scaler(
        [data["subject_row_by_id"][s] for s in train_ids], by_row, groups
    )
    data["scaler_fit_rows"] = len(train_ids)
    data["selection_parent_manifest_sha256"] = manifest["manifest_sha256"]
    if notes:
        data["notes"] = notes
    digest = _manifest_digest(data)
    data["manifest_sha256"] = digest
    return RealManifest(data=data, sha256=digest)


def load_real_manifest(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text())
    expected = data.pop("manifest_sha256", None)
    actual = hashlib.sha256(
        json.dumps(data, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    if expected != actual:
        raise ValueError(
            f"manifest hash mismatch: expected {expected}, computed {actual} — refusing"
        )
    data["manifest_sha256"] = expected
    return data


def derive_classification_targets(
    scores: list[float | None], score_mask: list[bool] | None = None
) -> tuple[list[float], list[bool]]:
    """RULE-derived 9D binary labels from canonical scores (see module doc).

    Missing (balanced) dims are unobserved — never zero-filled as negative.
    When score_mask is given it must agree with the score values; a
    disagreement means the adapter's audit trail and the label derivation
    have drifted apart, which is refused rather than silently resolved.
    """
    if score_mask is not None:
        for score, observed in zip(scores, score_mask, strict=True):
            if observed != (score is not None):
                raise ValueError(
                    "score_mask disagrees with score values; refusing to "
                    "derive labels from an inconsistent audit trail"
                )
    labels: list[float] = []
    mask: list[bool] = []
    for s in scores:
        if s is None:
            labels.append(0.0)
            mask.append(False)
        else:
            labels.append(1.0 if s >= 40.0 else 0.0)
            mask.append(True)
    return labels, mask
