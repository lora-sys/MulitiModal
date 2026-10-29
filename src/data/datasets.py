"""Public dataset registry for TCM data (#37).

Machine-checkable dataset records with provenance; eligibility values map
to the #37 frozen vocabulary (train_candidate / rules_test_only /
pending_verification / rejected). Raw files stay read-only under
data/local/ (git-ignored); only hashes and audit results are tracked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ELIGIBILITY_TRAIN_CANDIDATE = "train_candidate"
ELIGIBILITY_RULES_TEST_ONLY = "rules_test_only"
ELIGIBILITY_PENDING = "pending_verification"
ELIGIBILITY_REJECTED = "rejected"
ELIGIBILITIES = (
    ELIGIBILITY_TRAIN_CANDIDATE,
    ELIGIBILITY_RULES_TEST_ONLY,
    ELIGIBILITY_PENDING,
    ELIGIBILITY_REJECTED,
)


@dataclass(frozen=True)
class DatasetFile:
    path: str
    sha256: str = ""
    md5: str = ""
    size_bytes: int = 0


@dataclass(frozen=True)
class DatasetRecord:
    dataset_id: str
    primary_url: str
    doi: str
    version: str
    license: str
    source_type: str  # public_dataset / scoring_reference
    label_source: str  # PUBLISHED / RULE / NONE
    standard_version: str
    n_subjects: int
    modality_summary: str
    label_availability: str
    eligibility: str
    retrieved_at: str
    files: tuple[DatasetFile, ...] = field(default_factory=tuple)
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.dataset_id or not self.primary_url or not self.doi:
            raise ValueError("dataset_id, primary_url and doi are required")
        if self.eligibility not in ELIGIBILITIES:
            raise ValueError(
                f"unknown eligibility {self.eligibility!r}; expected {ELIGIBILITIES}"
            )
        if self.eligibility == ELIGIBILITY_TRAIN_CANDIDATE and self.label_source == "NONE":
            raise ValueError("train_candidate requires available labels")


#: Mendeley Su et al. 2019 — audited on dev host 2026-10-30 (see
#: docs/data/dataset-audit-report.md). 274 subjects, eight 0-100
#: constitution scores fully observed (NO ninth/balanced column — never
#: fabricate it), tabular meridian/pulse/tongue features. Scores follow the
#: 2009 CCMQ edition; conversion to GB/T 46939-2025 is forbidden without a
#: certified mapping (#38).
DATASET_CXTPH3TJSG = DatasetRecord(
    dataset_id="mendeley-cxtph3tjsg-v1",
    primary_url="https://data.mendeley.com/datasets/cxtph3tjsg/1",
    doi="10.17632/cxtph3tjsg.1",
    version="1",
    license="CC BY 4.0",
    source_type="public_dataset",
    label_source="PUBLISHED",
    standard_version="CCMQ-2009",
    n_subjects=274,
    modality_summary="tabular: demographics+meridian+pulse wave+tongue features (62 cols) + 8 constitution score columns",
    label_availability="8/9 dimension scores 0-100, fully observed, no missing; balanced-constitution dimension absent",
    eligibility=ELIGIBILITY_TRAIN_CANDIDATE,
    retrieved_at="2026-10-30",
    files=(
        DatasetFile(
            path="data/local/mendeley-cxtph3tjsg-v1/Share data.xls",
            sha256="6206f432b683f8a7b9a75dae23e598a4f01a41a4f8af84b099924b9d85d08a96",
            size_bytes=289792,
        ),
    ),
    notes="No explicit subject ID column; rows are one subject each (row order = "
          "anonymous identity). 1 missing row in h2/h3 meridian features. Authors' "
          "score-column abbreviations (YaD/YiD/QiDf/PW/WH/BS/SDa/QDp) need certified "
          "mapping to CCMQ dimensions in #38 before semantic claims.",
)

#: Zenodo CCMQ researcher scoring code + test data — reference material to
#: verify a #38 rule implementation against, NOT a supervised training set.
DATASET_ZENODO_4431679 = DatasetRecord(
    dataset_id="zenodo-ccmq-scoring-4431679",
    primary_url="https://zenodo.org/records/4431679",
    doi="10.5281/zenodo.4431679",
    version="1",
    license="CC BY 4.0",
    source_type="scoring_reference",
    label_source="RULE",
    standard_version="CCMQ-2009",
    n_subjects=14,
    modality_summary="SAS/SPSS scoring code + TEST_DATA.xlsx (14 subjects x 63 CCMQ item responses)",
    label_availability="researcher-computed constitution classifications for 14 test rows",
    eligibility=ELIGIBILITY_RULES_TEST_ONLY,
    retrieved_at="2026-10-30",
    files=(
        DatasetFile(
            path="data/local/zenodo-4431679/TEST_DATA.xlsx",
            md5="363d3e388df625425242fb2e45ec3d97",
            size_bytes=14606,
        ),
        DatasetFile(
            path="data/local/zenodo-4431679/CCMQ SAS code.sas",
            md5="3b622e25070847c767c31806d07afcf2",
            size_bytes=31320,
        ),
        DatasetFile(
            path="data/local/zenodo-4431679/CCMQ SPSS syntax.sps",
            md5="d843895a1fa7f626a7aa62c116757704",
            size_bytes=23657,
        ),
    ),
    notes="Researcher code is not an official scorer (#47); use only to test rule "
          "engines, never as supervised training data.",
)

DATASET_REGISTRY: dict[str, DatasetRecord] = {
    DATASET_CXTPH3TJSG.dataset_id: DATASET_CXTPH3TJSG,
    DATASET_ZENODO_4431679.dataset_id: DATASET_ZENODO_4431679,
}


def verify_files(record: DatasetRecord, data_root: str | Path = ".") -> dict[str, bool]:
    """Recompute file checksums from disk; read-only, no copying into Git."""
    import hashlib

    results = {}
    for f in record.files:
        path = Path(data_root) / f.path
        if not path.exists():
            results[f.path] = False
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if f.sha256:
            results[f.path] = digest == f.sha256
        else:
            md5 = hashlib.md5(path.read_bytes()).hexdigest()
            results[f.path] = md5 == f.md5
    return results
