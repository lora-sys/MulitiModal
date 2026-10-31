"""TCM dataset adapters (#37/#38): raw files -> audited per-subject records.

The Mendeley adapter enforces the #37 frozen rules: the file hash is
verified before any parse, the balanced-constitution (9th) dimension is
absent and is expressed as an unobserved label (never fabricated), and
outputs carry provenance. Feature groups stay separated so unpaired
modalities can never masquerade as one multimodal sample.

Per the #38 frozen correction (2026-10-30), scores are emitted in the
project canonical order with named dimensions — never in source worksheet
order — through the single audited SOURCE_COLUMN_MAPPING below.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from labels.standards import CANONICAL_ORDER

# Source score columns exactly as they appear in the audited worksheet
# header (columns 63-70). Output order is CANONICAL_ORDER, never this tuple.
SCORE_COLUMNS = ("YaD", "YiD", "QiDf", "PW", "WH", "BS", "SDa", "QDp")

#: Single audited source_column -> canonical_dimension mapping (#38 frozen
#: correction). Evidence (audited 2026-10-30, file sha256 6206f432…):
#: 1. `QiDf` = qi-deficiency is stated verbatim in the Su et al. 2019
#:    indexed article text ("Qi-deficiency (QDf) constitution").
#: 2. The remaining abbreviations are the standard TCM-English renderings
#:    (YaD=Yang-Deficiency, YiD=Yin-Deficiency, PW=Phlegm-Wetness,
#:    WH=Wet-Heat, BS=Blood-Stasis, SDa=Special Diathesis, QDp=Qi-Depression),
#:    bijective with the eight biased constitutions the paper reports
#:    modelling on the Nine-Constitution Scale.
#: 3. Item-count fingerprint: under the CCMQ formula (sum-n)/(4n)*100 an
#:    n-item subscale only yields multiples of 100/(4n). Measured feasible
#:    counts over 273/274 rows match the ZYYXH/T157-2009 subscale sizes
#:    (PMC10164641 Table 1: qi 8, yang 7, yin 8, phlegm 8, damp-heat 6,
#:    blood 7, qi-stagnation 7, special 7) for every column under this
#:    mapping — including the uniquely distinctive damp-heat=6.
#: Limitation: the paper's own abbreviation definition table is paywalled;
#: evidence state is FINGERPRINT_CROSS_CHECKED, not paper-table VERIFIED.
SOURCE_COLUMN_MAPPING: dict[str, str] = {
    "YaD": "yang_deficiency",
    "YiD": "yin_deficiency",
    "QiDf": "qi_deficiency",
    "PW": "phlegm_damp",
    "WH": "damp_heat",
    "BS": "blood_stasis",
    "SDa": "special_diathesis",
    "QDp": "qi_depression",
}
CANONICAL_MAPPING_STATE = "FINGERPRINT_CROSS_CHECKED"

#: CCMQ-60 (ZYYXH/T157-2009) items per biased subscale (PMC10164641 Table 1)
EXPECTED_ITEMS = {
    "qi_deficiency": 8,
    "yang_deficiency": 7,
    "yin_deficiency": 8,
    "phlegm_damp": 8,
    "damp_heat": 6,
    "blood_stasis": 7,
    "qi_depression": 7,
    "special_diathesis": 7,
}

# column groups per the audited header layout (docs/data/dataset-audit-report.md)
#
# The meridian device block does not stop at column 20. Columns 21-35
# (RET L / RET_R resistance, HR bpm, six sSA_* and six tSA_* meridian-energy
# channels) were never read by any model in v1-v8: the first audit wrote
# "meridian 9-20" and the 15 columns between the meridian and pulse-wave
# blocks went unmentioned, while `b` (column 62, tongue-image blue channel,
# 274/274 observed) was dismissed as undefined. An independent review
# measured the gap: 16 of 63 input columns, i.e. 25%, never reached the model.
FEATURE_GROUPS = {
    "demographics": slice(0, 6),      # gender..surgery hx
    "anthropometrics": slice(6, 9),   # BH/BW/BMI
    "meridian": slice(9, 21),         # h2/h3 left/right conductance
    "meridian_dynamics": slice(21, 36),  # RET, HR, sSA_* and tSA_* channels
    "pulse_wave": slice(36, 51),      # SYS..MWTCI
    "tongue": slice(51, 63),          # tongue max width..HSV S and colour b
}

#: every feature column index the adapter is expected to read. Anything in the
#: workbook outside this set and outside SCORE_COLUMNS is an unread column and
#: must be justified in the audit report — the v1-v8 omission was not.
AUDITED_FEATURE_COLUMNS = tuple(
    i for s in FEATURE_GROUPS.values() for i in range(s.start, s.stop)
)

#: how many values each group must yield. Fixtures and the adapter both read
#: this instead of restating lengths: the v1-v8 feature omission survived
#: because three separate test fixtures each hardcoded their own copy of the
#: group layout and none of them noticed 15 columns were unread.
FEATURE_GROUP_SIZES = {
    name: group.stop - group.start for name, group in FEATURE_GROUPS.items()
}

EXPECTED_SHA256 = "6206f432b683f8a7b9a75dae23e598a4f01a41a4f8af84b099924b9d85d08a96"


def assert_mapping_is_audited() -> None:
    """Reject any mapping that is not a bijective cover of the eight biased
    canonical dimensions (#38: never emit an unaudited mapping)."""
    biased = [d for d in CANONICAL_ORDER if d != "balanced"]
    if sorted(SOURCE_COLUMN_MAPPING.values()) != sorted(biased):
        raise ValueError(
            "SOURCE_COLUMN_MAPPING must be bijective over the eight biased "
            f"canonical dims {biased}, got {sorted(SOURCE_COLUMN_MAPPING.values())}"
        )
    if set(SOURCE_COLUMN_MAPPING) != set(SCORE_COLUMNS):
        raise ValueError("SOURCE_COLUMN_MAPPING keys must equal SCORE_COLUMNS")
    if sorted(EXPECTED_ITEMS) != sorted(biased):
        raise ValueError(f"EXPECTED_ITEMS must cover the biased dims {biased}")


def feasible_item_counts(value: float) -> tuple[int, ...]:
    """Item counts n whose CCMQ granularity 100/(4n) the score fits."""
    return tuple(
        n for n in range(4, 13)
        if abs(value * (4 * n) / 100.0 - round(value * (4 * n) / 100.0)) < 1e-6
    )


def column_fingerprint(values: list[float]) -> int | None:
    """Smallest subscale size n whose CCMQ granularity 100/(4n) fits every
    value except <=1% anomalous rows, or None if no n reaches that coverage.
    (Many scores fit several granularities — e.g. 8.333 fits n=6, 9 and 12 —
    so a majority-of-unique-values vote is wrong; coverage is the criterion.)"""
    total = len(values)
    if not total:
        return None
    for n in range(4, 13):
        covered = sum(1 for v in values if n in feasible_item_counts(v))
        if covered >= total * 0.99:
            return n
    return None


def fingerprint_anomalies(values: list[float], expected_n: int) -> list[int]:
    """Row indices whose value is incompatible with the expected item count."""
    return [
        i for i, v in enumerate(values)
        if expected_n not in feasible_item_counts(v)
    ]


def resolve_score_columns(headers: list[str]) -> dict[str, int]:
    """Locate score columns by name; reject missing or duplicated ones."""
    idx: dict[str, int] = {}
    for col in SCORE_COLUMNS:
        hits = [i for i, h in enumerate(headers) if h == col]
        if not hits:
            raise ValueError(f"score column {col!r} missing from dataset")
        if len(hits) > 1:
            raise ValueError(f"score column {col!r} appears {len(hits)} times")
        idx[col] = hits[0]
    return idx


@dataclass
class SubjectRecord:
    subject_index: int  # anonymous identity: row order (no ID column exists)
    features: dict[str, list[float]]
    scores: list[float | None]  # 9 dims in CANONICAL_ORDER; balanced (index 0) is None
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
    """Parse Share data.xls into per-subject records with canonical 9D targets.

    Scores are emitted in CANONICAL_ORDER with balanced (index 0) unobserved;
    every dimension is addressable by name via provenance["score_dims"] and
    SOURCE_COLUMN_MAPPING, never by worksheet position (#38 frozen fix).
    """
    import xlrd

    assert_mapping_is_audited()
    verify_mendeley_sha256(path)
    wb = xlrd.open_workbook(str(path))
    sheet = wb.sheet_by_name("data")
    headers = [str(sheet.cell_value(0, c)) for c in range(sheet.ncols)]
    score_idx = resolve_score_columns(headers)

    anomalies: dict[str, list[int]] = {}
    for col, dim in SOURCE_COLUMN_MAPPING.items():
        values = [sheet.cell_value(r, score_idx[col]) for r in range(1, sheet.nrows)]
        fp = column_fingerprint(values)
        if fp != EXPECTED_ITEMS[dim]:
            raise ValueError(
                f"column {col!r} item-count fingerprint {fp} does not match "
                f"expected {EXPECTED_ITEMS[dim]} for {dim} — refusing to map"
            )
        bad = fingerprint_anomalies(values, EXPECTED_ITEMS[dim])
        if bad:
            anomalies[dim] = bad

    records: list[SubjectRecord] = []
    for r in range(1, sheet.nrows):
        features: dict[str, list[float]] = {}
        for group, sl in FEATURE_GROUPS.items():
            values = []
            for c in range(sl.start, min(sl.stop, sheet.ncols)):
                v = sheet.cell_value(r, c)
                values.append(float(v) if v != "" else float("nan"))
            features[group] = values
            # the contract, checked against the real workbook: a group whose
            # column range silently shrank (or was never widened to cover the
            # meridian-dynamics block) fails here instead of producing a model
            # that quietly reads fewer inputs than the audit describes
            if len(values) != FEATURE_GROUP_SIZES[group]:
                raise ValueError(
                    f"feature group {group!r} yielded {len(values)} values but "
                    f"the audited contract expects {FEATURE_GROUP_SIZES[group]} "
                    f"(columns {FEATURE_GROUPS[group].start}.."
                    f"{FEATURE_GROUPS[group].stop - 1}); refusing to load"
                )
        # canonical order: balanced unobserved at index 0, then the biased dims
        by_dim = {
            dim: float(sheet.cell_value(r, score_idx[col]))
            for col, dim in SOURCE_COLUMN_MAPPING.items()
        }
        scores: list[float | None] = [None]
        mask = [False]
        for dim in CANONICAL_ORDER[1:]:
            scores.append(by_dim[dim])
            mask.append(True)
        records.append(
            SubjectRecord(
                subject_index=r - 1,
                features=features,
                scores=scores,
                score_mask=mask,
                provenance={
                    "dataset_id": "mendeley-cxtph3tjsg-v1",
                    "doi": "10.17632/cxtph3tjsg.1",
                    "standard_version": "ZYYXH/T157-2009",
                    "standard_edition_evidence": (
                        "item-count fingerprint of all 8 score columns matches "
                        "ZYYXH/T157-2009 subscale sizes (PMC10164641); paper "
                        "definition table paywalled, item texts not compared"
                    ),
                    "label_source": "PUBLISHED",
                    "file_sha256": EXPECTED_SHA256,
                    "score_dims": list(CANONICAL_ORDER),
                    "source_column_mapping": dict(SOURCE_COLUMN_MAPPING),
                    "canonical_mapping_state": CANONICAL_MAPPING_STATE,
                    "fingerprint_anomalies": {
                        dim: list(rows) for dim, rows in anomalies.items()
                    },
                },
            )
        )
    return records
