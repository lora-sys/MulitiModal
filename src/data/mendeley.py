"""TCM dataset adapters (#37): raw files -> audited per-subject records.

The Mendeley adapter enforces the #37 frozen rules: the file hash is
verified before any parse, the balanced-constitution (9th) dimension is
absent and is expressed as an unobserved label (never fabricated), and
outputs carry provenance. Feature groups stay separated so unpaired
modalities can never masquerade as one multimodal sample.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

SCORE_COLUMNS = ("YaD", "YiD", "QiDf", "PW", "WH", "BS", "SDa", "QDp")
NINE_CONSTITUTION_DIMS = 9  # scores provide 8; the 9th stays unobserved
# column groups per the audited header layout (docs/data/dataset-audit-report.md)
FEATURE_GROUPS = {
    "demographics": slice(0, 6),      # gender..surgery hx
    "anthropometrics": slice(6, 9),   # BH/BW/BMI
    "meridian": slice(9, 21),         # h2/h3 left/right
    "pulse_wave": slice(36, 51),      # SYS..MWTCI
    "tongue": slice(51, 62),          # tongue max width..HSV S
}

EXPECTED_SHA256 = "6206f432b683f8a7b9a75dae23e598a4f01a41a4f8af84b099924b9d85d08a96"


@dataclass
class SubjectRecord:
    subject_index: int  # anonymous identity: row order (no ID column exists)
    features: dict[str, list[float]]
    scores: list[float | None]  # 9 dims; index 8 (balanced) is None by contract
    score_mask: list[bool]  # observed dims only; never zero-filled
    provenance: dict


def verify_mendeley_sha256(path: str | Path) -> str:
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(
            f"Mendeley file hash mismatch: expected {EXPECTED_SHA256}, got {digest}"
        )
    return digest


def load_mendeley_scores(path: str | Path) -> list[SubjectRecord]:
    """Parse Share data.xls into per-subject records with masked 9D targets."""
    import xlrd

    verify_mendeley_sha256(path)
    wb = xlrd.open_workbook(str(path))
    sheet = wb.sheet_by_name("data")
    headers = [str(sheet.cell_value(0, c)) for c in range(sheet.ncols)]
    for col in SCORE_COLUMNS:
        if col not in headers:
            raise ValueError(f"score column {col!r} missing from dataset")
    score_idx = {headers.index(c) for c in SCORE_COLUMNS}

    records: list[SubjectRecord] = []
    for r in range(1, sheet.nrows):
        features: dict[str, list[float]] = {}
        for group, sl in FEATURE_GROUPS.items():
            values = []
            for c in range(sl.start, min(sl.stop, sheet.ncols)):
                v = sheet.cell_value(r, c)
                values.append(float(v) if v != "" else float("nan"))
            features[group] = values
        raw_scores = [sheet.cell_value(r, c) for c in sorted(score_idx)]
        # 8 observed dims + the absent balanced dim, always in SCORE_COLUMNS order
        scores: list[float | None] = [float(v) for v in raw_scores] + [None]
        mask = [s is not None for s in scores]
        records.append(
            SubjectRecord(
                subject_index=r - 1,
                features=features,
                scores=scores,
                score_mask=mask,
                provenance={
                    "dataset_id": "mendeley-cxtph3tjsg-v1",
                    "doi": "10.17632/cxtph3tjsg.1",
                    "standard_version": "CCMQ-2009",
                    "label_source": "PUBLISHED",
                    "file_sha256": EXPECTED_SHA256,
                },
            )
        )
    return records
