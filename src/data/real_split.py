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


def _nan_aware(values: list[float]) -> tuple[float, float]:
    """Mean/std over finite values only; (0, 1) when nothing is finite."""
    finite = [v for v in values if not math.isnan(v)]
    if not finite:
        return 0.0, 1.0
    mean = sum(finite) / len(finite)
    var = sum((v - mean) ** 2 for v in finite) / len(finite)
    return mean, math.sqrt(var) or 1.0


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

    rows = [r.subject_index for r in records if r.subject_index not in exclude_subject_rows]
    if len(rows) < 10:
        raise ValueError(f"too few subjects left for a split: {len(rows)}")
    rng = random.Random(split_seed)
    order = rows[:]
    rng.shuffle(order)
    n = len(order)
    n_train = int(n * fractions[0])
    n_dev = int(n * fractions[1])
    partitions = {
        "train": order[:n_train],
        "dev": order[n_train : n_train + n_dev],
        "holdout": order[n_train + n_dev :],
    }
    by_row = {r.subject_index: r for r in records}

    # --- fold-only scaler: statistics from TRAIN rows exclusively (#29) ----
    groups = MODALITY_SETS[modality_set]
    scaler: dict[str, dict] = {}
    for modality, feature_groups in groups.items():
        cols: list[list[float]] = []
        for r in partitions["train"]:
            values = [v for g in feature_groups for v in by_row[r].features[g]]
            cols.append(values)
        n_cols = len(cols[0])
        scaler[modality] = {
            "mean": [_nan_aware([c[j] for c in cols])[0] for j in range(n_cols)],
            "std": [_nan_aware([c[j] for c in cols])[1] for j in range(n_cols)],
        }

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
            for modality, feature_groups in groups.items()
        },
        "split": {
            "seed": split_seed,
            "fractions": list(fractions),
            **{p: [f"mendeley-r{r:03d}" for r in sorted(rows)] for p, rows in partitions.items()},
        },
        "subject_row_by_id": {f"mendeley-r{r:03d}": r for r in sorted(rows)},
        "scaler": scaler,
        "excluded_subject_rows": list(exclude_subject_rows),
        "notes": notes,
    }
    digest = hashlib.sha256(
        json.dumps(data, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
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


def derive_classification_targets(scores: list[float | None]) -> tuple[list[float], list[bool]]:
    """RULE-derived 9D binary labels from canonical scores (see module doc).

    Missing (balanced) dims are unobserved — never zero-filled as negative.
    """
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
