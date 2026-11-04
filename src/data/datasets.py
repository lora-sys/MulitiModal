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

#: Borororo/ZY-BERT TCM-SD — full audit 2026-11-01 on this machine, see
#: docs/data/tcm-sd-syndrome-audit.md. 54,152 Chinese clinical cases,
#: 148-syndrome closed vocabulary, single norm_syndrome label per record,
#: zero out-of-vocab labels. The official train/dev/test user_id sets
#: overlap (19/18/13), so training must use the subject-level repartition
#: in src/data/tcm_sd.py (subject_partitions), never the official split.
#: NOTE: n_subjects is the audited union 52,295; the 52,345 figure that
#: circulated in the #53 probe sums per-file unique users and double-counts
#: the 50 users that appear in more than one official file.
DATASET_TCM_SD = DatasetRecord(
    dataset_id="tcm-sd-syndrome-v1",
    primary_url="https://github.com/Borororo/ZY-BERT",
    doi="NOT_AVAILABLE (GitHub-only release, no DOI)",
    version="1",
    license="CC BY-NC-SA 4.0",
    source_type="public_dataset",
    label_source="PUBLISHED",
    standard_version="tcm-sd-syndrome-vocab-v1",
    n_subjects=52295,
    modality_summary=(
        "text: chief_complaint + description + detection per case, "
        "single norm_syndrome label from a frozen 148-syndrome vocabulary"
    ),
    label_availability=(
        "norm_syndrome present and 100% in-vocab for all 54,152 records; "
        "raw `syndrome` field retained but differs from norm in 6,725 records"
    ),
    eligibility=ELIGIBILITY_TRAIN_CANDIDATE,
    retrieved_at="2026-11-01",
    files=(
        DatasetFile(
            path="data/local/tcm-sd/train.json",
            sha256="d29d3be5eccc82a1a2654f06f3aa09f5aff8dbecbb29515980a272b1c39663eb",
            size_bytes=57187960,
        ),
        DatasetFile(
            path="data/local/tcm-sd/dev.json",
            sha256="34e0e85f03d2d3e4fab93856c32dc99c71e8f76b74dc5f9b92123127669b1540",
            size_bytes=7577225,
        ),
        DatasetFile(
            path="data/local/tcm-sd/test.json",
            sha256="cfcd6f20e74c6658e4677cbaed93622a90eceb35a62bad792ba443549af31ea9",
            size_bytes=7398155,
        ),
        DatasetFile(
            path="data/local/tcm-sd/syndrome_vocab.txt",
            sha256="63ee108754d90914a3833328abda7e04130f200617ca79033a66d22e47c45c83",
            size_bytes=2462,
        ),
        DatasetFile(
            path="data/local/tcm-sd/syndrome_knowledge.json",
            sha256="c98b830f4d13c44b106f3bc2d40571c6070f9da7228e8c264cb41e8f8ec3bd9e",
            size_bytes=1230148,
        ),
    ),
    notes=(
        "License is CC BY-NC-SA 4.0: research use only; the dataset, its "
        "derivatives and any model trained on it must not be used for "
        "commercial deployment without separate licensing. Label leakage "
        "into input text measured at 1/54,152 records; explicit "
        "differentiation-conclusion phrasing (证属/辨证为/诊为/诊断为) present "
        "in 17.6% of inputs and prescription phrasing in 0.6% — see the "
        "audit report for the two evaluation text scenarios. 2,635 "
        "exact-duplicate text groups (8,612 records; 334 with conflicting "
        "norm labels; 1,382 spanning different users)."
    ),
)

#: 2020MEAI/PresRecST TCM-Lung — audited 2026-11-01 from a fresh clone.
#: Rejected for #53 cross-corpus scoring: the release contains ONLY
#: anonymized integer IDs. The Syndrome/Symptom/Treat dictionary sheets
#: hold bare integers 1..N with no name decoding, so the 50-syndrome label
#: space cannot be aligned with the TCM-SD 148-term vocabulary — mapping
#: coverage is NOT_COMPUTABLE (0 alignable labels), and there is no
#: subject ID either (4,484 rows of symptom-set -> syndrome-ID combos).
#: No LICENSE file and no license statement in the repo.
DATASET_TCM_LUNG = DatasetRecord(
    dataset_id="presrecst-tcm-lung-v1",
    primary_url="https://github.com/2020MEAI/PresRecST",
    doi="NOT_AVAILABLE (repository release; companion paper doi:10.1093/jamia/ocae066)",
    version="1",
    license="not stated (no LICENSE file in repository)",
    source_type="public_dataset",
    label_source="PUBLISHED",
    standard_version="NOT_AVAILABLE (syndrome labels are anonymous integer IDs)",
    n_subjects=0,
    modality_summary=(
        "anonymized integer IDs: 4,484 rows of SymptomID-set -> SyndromeID "
        "-> TreatmentID -> HerbID; dictionaries are bare ID lists without names"
    ),
    label_availability=(
        "SyndromeID per row (50 anonymous classes, multi-valued); no textual syndrome names"
    ),
    eligibility=ELIGIBILITY_REJECTED,
    retrieved_at="2026-11-01",
    files=(
        DatasetFile(
            path="data/local/tcm-lung-presrecst/TCM_Lung.xlsx",
            sha256="68e911dd0fe0c35a2fae46706f682cff9e6530858fbeb8cb3d4d97b7354f325d",
            size_bytes=274951,
        ),
    ),
    notes=(
        "Rejected as NOT_ALIGNABLE + LICENSE_NOT_STATED: without a name "
        "decoding of the ID space, one-to-one mapping verification against "
        "tcm-sd-syndrome-vocab-v1 is impossible (coverage 0/148), and the "
        "missing license forbids assuming reuse. prescript_1195.csv in the "
        "same repo was not audited (prescriptions, out of scope for "
        "syndrome-label alignment)."
    ),
)

#: TCMEval-SDT (figshare 27184596) — audited 2026-11-01 from the current
#: figshare version (v5; issue #53 references v4). 300 classical-style
#: cases (200/50/50). Train carries gold `TCM Syndrome` text; Validation/
#: Test gold labels are decodable only via the per-record letter answers
#: in the later-uploaded *_data_result.txt files joined to each record's
#: own `Options of TCM Syndrome`. Label space is free-form classical
#: phrasing, mostly multi-label (117/200 train records carry >=2 labels).
#: Alignment to the 148-term TCM-SD vocabulary: exact match 0/407 distinct
#: labels; dropping the trailing 证 gives 42/407 distinct (18.9% of label
#: instances) — cross-corpus scoring only on that aligned subset, else
#: NOT_COMPUTABLE. Verbatim text overlap with TCM-SD: 0 fragments (no
#: cross-corpus duplicate cases found). Case IDs (病例N) are case-level;
#: no subject identity beyond that.
DATASET_TCMEVAL_SDT = DatasetRecord(
    dataset_id="tcmeval-sdt-v1",
    primary_url="https://doi.org/10.6084/m9.figshare.27184596.v4",
    doi="10.6084/m9.figshare.27184596",
    version="5",
    license="CC BY 4.0",
    source_type="public_dataset",
    label_source="PUBLISHED",
    standard_version="NOT_AVAILABLE (free-form classical syndrome phrasing, multi-label)",
    n_subjects=300,
    modality_summary=(
        "text: classical TCM case records (Clinical Data / Clinical "
        "Information) with syndrome-differentiation multiple-choice options "
        "and letter answers"
    ),
    label_availability=(
        "train: gold TCM Syndrome text for 200/200; validation/test: gold "
        "letter answers via *_data_result.txt for 50/50 + 50/50 records"
    ),
    eligibility=ELIGIBILITY_PENDING,
    retrieved_at="2026-11-01",
    files=(
        DatasetFile(
            path="data/local/tcmeval-sdt/Train_TCM_Data_v1.json",
            sha256="cddcf9c7187d43462b8c80b1756527e506b2e8d9d0329171d50477f49bd679cd",
            size_bytes=403773,
        ),
        DatasetFile(
            path="data/local/tcmeval-sdt/Validation_TCM_Data_v1.json",
            sha256="80b5b836349da9c0c3f9be31255c8baad5669a4b33270778ce67f71187fe1e6c",
            size_bytes=56324,
        ),
        DatasetFile(
            path="data/local/tcmeval-sdt/Test_TCM_Data_v1.json",
            sha256="4b9f69308b46125ea6d979613d4b2717b1d930d7acefd2e3c58d274975420bff",
            size_bytes=56432,
        ),
        DatasetFile(
            path="data/local/tcmeval-sdt/Validation_data_result.txt",
            sha256="427de6e639addde75a05be08e99063df2bdfd8fbac9cb53b0d3bc339e642113a",
            size_bytes=20691,
        ),
        DatasetFile(
            path="data/local/tcmeval-sdt/Test_data_result.txt",
            sha256="68b521be0650796af8fcc96672b680368278b561c07203b03d10d1ef78530ee5",
            size_bytes=23113,
        ),
    ),
    notes=(
        "Pending verification as a cross-corpus evaluation set: usable only "
        "on the 证-suffix-aligned label subset (42 distinct labels, 18.9% of "
        "instances), single-label records only, and never as training data "
        "for the TCM-SD task. Multi-label nature of the source must be "
        "reported, not silently collapsed."
    ),
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
        "(pulse features + constitution labels). Application is a human step. "
        "label_source=PUBLISHED is a metadata-level claim from the project "
        "description; the row-level label column awaits download and audit."
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
    DATASET_TCM_SD.dataset_id: DATASET_TCM_SD,
    DATASET_TCM_LUNG.dataset_id: DATASET_TCM_LUNG,
    DATASET_TCMEVAL_SDT.dataset_id: DATASET_TCMEVAL_SDT,
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
