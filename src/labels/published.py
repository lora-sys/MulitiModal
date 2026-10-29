"""PUBLISHED label export (#38): author-published scores as named 9D labels.

PUBLISHED means the dataset authors measured and published per-person
constitution scores (here: the Mendeley Su et al. 2019 workbook, edition
fingerprint-verified ZYYXH/T157-2009). This module re-keys those scores
onto the canonical nine-dimension order through the single audited
SOURCE_COLUMN_MAPPING and never re-scores, re-editions, or imputes them.
Rule-derived categories are a separate, clearly labeled derivation —
they are never passed off as published labels.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from labels.ccmq60 import CCMQ60Scorer, LabelRecord

if TYPE_CHECKING:  # import inversion: data depends on labels, never conversely
    from data.mendeley import SubjectRecord

PUBLISHED_INSTRUMENT = "CCMQ_60_HISTORICAL"


def _label_from_subject(sr: SubjectRecord) -> LabelRecord:
    anomalies = sr.provenance.get("fingerprint_anomalies", {})
    qc_flags = sorted(dim for dim, rows in anomalies.items() if sr.subject_index in rows)
    return LabelRecord(
        scores=list(sr.scores),
        mask=list(sr.score_mask),
        label_tier="PUBLISHED",
        standard_version=sr.provenance["standard_version"],
        instrument_id=PUBLISHED_INSTRUMENT,
        missing_policy="abstain",
        source_id=sr.provenance["dataset_id"],
        source_row=sr.subject_index,
        canonical_mapping_verified=False,  # paper definition table not yet checked
        extra_provenance={
            "doi": sr.provenance["doi"],
            "file_sha256": sr.provenance["file_sha256"],
            "canonical_mapping_state": sr.provenance["canonical_mapping_state"],
            "score_dims": sr.provenance["score_dims"],
            **({"qc_fingerprint_anomaly": qc_flags} if qc_flags else {}),
        },
    )


def published_label_records(
    subjects: Iterable[SubjectRecord],
) -> list[LabelRecord]:
    """Re-key audited subject records (e.g. data.mendeley.load_mendeley_scores
    output) into named 9D PUBLISHED labels.

    The balanced dimension stays unobserved (the source has no such column);
    subject rows flagged by the item-count fingerprint QC carry a
    qc_fingerprint_anomaly provenance entry (#38: never silently mixed)."""
    return [_label_from_subject(sr) for sr in subjects]


def derive_rule_categories(record: LabelRecord) -> dict[str, str]:
    """Apply the verified researcher decision rules (30/40 biased, 60/30/40
    balanced) to a PUBLISHED score record. The result is RULE-derived
    categorical data with its own provenance — suitable for the
    classification head only when recorded as derived, never as the
    authors' published output."""
    if record.label_tier != "PUBLISHED":
        raise ValueError(
            f"derive_rule_categories expects PUBLISHED scores, got {record.label_tier!r}"
        )
    return CCMQ60Scorer().classify(record)
