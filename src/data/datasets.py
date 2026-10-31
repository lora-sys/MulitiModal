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
#: fabricate it), tabular meridian/pulse/tongue features. Edition evidence:
#: all 8 score columns' item-count fingerprints match ZYYXH/T157-2009
#: subscale sizes (incl. damp-heat=6, PMC10164641); item texts not compared.
#: Conversion to GB/T 46939-2025 is forbidden without a certified
#: mapping (#38).
DATASET_CXTPH3TJSG = DatasetRecord(
    dataset_id="mendeley-cxtph3tjsg-v1",
    primary_url="https://data.mendeley.com/datasets/cxtph3tjsg/1",
    doi="10.17632/cxtph3tjsg.1",
    version="1",
    license="CC BY 4.0",
    source_type="public_dataset",
    label_source="PUBLISHED",
    standard_version="ZYYXH/T157-2009",
    n_subjects=274,
    modality_summary=(
        "tabular: demographics+meridian+pulse wave+tongue features (62 cols)"
        " + 8 constitution score columns"
    ),
    label_availability=(
        "8/9 dimension scores 0-100, fully observed, no missing; "
        "balanced-constitution dimension absent"
    ),
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
          "anonymous identity). 1 missing row in h2/h3 meridian features. Score "
          "columns mapped to canonical dimensions via fingerprint-cross-checked "
          "SOURCE_COLUMN_MAPPING (src/data/mendeley.py, #38); subject row 273 "
          "violates the per-column item-count fingerprint on YaD/PW/WH/QDp — "
          "flag for QC, never silently mixed into rule cross-checks.",
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
    standard_version="ZYYXH/T157-2009",
    n_subjects=14,
    modality_summary=(
        "SAS/SPSS scoring code + TEST_DATA.xlsx (14 subjects x 63 CCMQ item responses)"
    ),
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

#: 2026-10-31 expansion search. The model pairs physiological/lifestyle
#: features with CCMQ-derived categories, so a usable dataset must carry
#: BOTH sides per subject. The records below are the candidates that search
#: surfaced; none could be downloaded in this session, and the record exists
#: so the search is checkable instead of being re-run from memory.
DATASET_NCMI_PULSE = DatasetRecord(
    dataset_id="ncmi-phda-pulse-constitution-2015-2019",
    primary_url=(
        "https://www.ncmi.cn/phda/dataDetails.html?type=project_data"
        "&id=CSTR:A0006.11.A001I.202006.001095-V1.0"
    ),
    doi="10.12041/geodata.1588738452788.ver1.db",
    version="V1.0",
    license="PHDA service terms; registration + per-dataset application",
    source_type="public_dataset",
    label_source="PUBLISHED",
    standard_version="ZYYXH/T157-2009",
    n_subjects=26154,
    modality_summary=(
        "pulse-diagnosis physiological parameters, 58,863 records, "
        "~75.8-96.9 GB, four regions; constitution type per subject"
    ),
    label_availability=(
        "constitution type per subject (project reports 26,154 person-times); "
        "row-level label column not inspected"
    ),
    eligibility=ELIGIBILITY_PENDING,
    retrieved_at="2026-10-31",
    notes=(
        "CSTR:A0006.11.A001I.202006.001095 (PHDA) and geodata DOI above are "
        "two records of the same CACMS project. Metadata labels it "
        "完全共享/线上共享, but the detail page returns HTTP 403 without an "
        "account and the download needs an order plus manual review. This is "
        "the only candidate that would genuinely extend the training set "
        "(pulse features + constitution labels). Application is a human step."
    ),
)

DATASET_TCMECQ_ELDERLY = DatasetRecord(
    dataset_id="frontiers-tcmecq-elderly-2026",
    primary_url="https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2026.1698576/full",
    doi="10.3389/fmed.2026.1698576",
    version="published article",
    license="not stated for row-level data",
    source_type="publication",
    label_source="PUBLISHED",
    standard_version="ZYYXH/T157-2009",
    n_subjects=25586,
    modality_summary="TCMECQ questionnaire (33 items) + 13 biochemical indicators",
    label_availability="nine-class constitution + transformation trend, aggregate in paper",
    eligibility=ELIGIBILITY_REJECTED,
    retrieved_at="2026-10-31",
    notes=(
        "25,586 cleaned records with paired biochemical features would have "
        "been the best extension found, but the data availability statement "
        "puts row-level data with the corresponding author and no repository. "
        "Rejected as NOT_PUBLICLY_DOWNLOADABLE; no adapter written."
    ),
)

DATASET_CCMQ_NATIONAL = DatasetRecord(
    dataset_id="ccmq-national-91145",
    primary_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC11403957/",
    doi="10.1186/s13020-024-00992-0",
    version="published article",
    license="not stated",
    source_type="publication",
    label_source="RULE",
    standard_version="ZYYXH/T157-2009",
    n_subjects=91145,
    modality_summary="CCMQ-60 item responses, web survey (no physiological features)",
    label_availability="item-level responses; data on request from the corresponding author",
    eligibility=ELIGIBILITY_REJECTED,
    retrieved_at="2026-10-31",
    notes=(
        "Rejected on two independent grounds: no downloadable file, and no "
        "physiological features — it could only validate the #38 scoring "
        "rule, which the 14-row Zenodo fixture already covers."
    ),
)

DATASET_REGISTRY: dict[str, DatasetRecord] = {
    DATASET_CXTPH3TJSG.dataset_id: DATASET_CXTPH3TJSG,
    DATASET_ZENODO_4431679.dataset_id: DATASET_ZENODO_4431679,
    DATASET_NCMI_PULSE.dataset_id: DATASET_NCMI_PULSE,
    DATASET_TCMECQ_ELDERLY.dataset_id: DATASET_TCMECQ_ELDERLY,
    DATASET_CCMQ_NATIONAL.dataset_id: DATASET_CCMQ_NATIONAL,
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
