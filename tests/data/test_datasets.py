"""Dataset registry + Mendeley adapter tests (#37/#38).

Tests touching the real downloaded file run only where data/local/ exists
(dev host); the file is git-ignored and never committed. All other tests
are synthetic.
"""

from pathlib import Path

import pytest

from data.datasets import (
    DATASET_CXTPH3TJSG,
    DATASET_REGISTRY,
    DATASET_ZENODO_4431679,
    ELIGIBILITY_RULES_TEST_ONLY,
    ELIGIBILITY_TRAIN_CANDIDATE,
    DatasetRecord,
    verify_files,
)
from data.mendeley import (
    EXPECTED_ITEMS,
    SCORE_COLUMNS,
    SOURCE_COLUMN_MAPPING,
    assert_mapping_is_audited,
    column_fingerprint,
    feasible_item_counts,
    fingerprint_anomalies,
    load_mendeley_scores,
    resolve_score_columns,
    verify_mendeley_sha256,
)
from labels.standards import CANONICAL_ORDER

pytestmark = pytest.mark.synthetic

DATA_ROOT = Path(__file__).resolve().parents[2]
MENDELEY_FILE = DATA_ROOT / DATASET_CXTPH3TJSG.files[0].path
has_mendeley_file = MENDELEY_FILE.exists()


def test_registry_records_are_valid():
    assert set(DATASET_REGISTRY) == {"mendeley-cxtph3tjsg-v1", "zenodo-ccmq-scoring-4431679"}
    for record in DATASET_REGISTRY.values():
        assert record.files, record.dataset_id
        assert record.eligibility in (
            ELIGIBILITY_TRAIN_CANDIDATE, ELIGIBILITY_RULES_TEST_ONLY,
            "pending_verification", "rejected",
        )


def test_train_candidate_requires_labels():
    with pytest.raises(ValueError, match="train_candidate requires"):
        DatasetRecord(
            dataset_id="x", primary_url="u", doi="d", version="1",
            license="CC BY 4.0", source_type="public_dataset",
            label_source="NONE", standard_version="s", n_subjects=1,
            modality_summary="m", label_availability="none",
            eligibility=ELIGIBILITY_TRAIN_CANDIDATE, retrieved_at="2026-10-30",
        )


def test_unknown_eligibility_rejected():
    with pytest.raises(ValueError, match="unknown eligibility"):
        DatasetRecord(
            dataset_id="x", primary_url="u", doi="d", version="1",
            license="l", source_type="s", label_source="NONE",
            standard_version="s", n_subjects=1, modality_summary="m",
            label_availability="a", eligibility="sure_why_not",
            retrieved_at="2026-10-30",
        )


def test_checksum_verification_reports_missing_files(tmp_path):
    results = verify_files(DATASET_ZENODO_4431679, data_root=tmp_path)
    assert all(v is False for v in results.values())  # no data/local in CI


def test_mapping_is_bijective_over_canonical_dims():
    """#38 frozen fix: exactly one source column per biased canonical dim;
    balanced is never mapped (it does not exist in the source)."""
    assert_mapping_is_audited()
    biased = [d for d in CANONICAL_ORDER if d != "balanced"]
    assert sorted(SOURCE_COLUMN_MAPPING.values()) == sorted(biased)
    assert "balanced" not in SOURCE_COLUMN_MAPPING.values()
    for col, dim in SOURCE_COLUMN_MAPPING.items():
        assert col in SCORE_COLUMNS and dim in EXPECTED_ITEMS


def test_resolve_score_columns_rejects_missing_and_duplicate():
    ok = resolve_score_columns(["a", *SCORE_COLUMNS, "z"])
    assert ok == {col: i + 1 for i, col in enumerate(SCORE_COLUMNS)}
    with pytest.raises(ValueError, match="missing"):
        resolve_score_columns(list(SCORE_COLUMNS)[:-1])
    with pytest.raises(ValueError, match="2 times"):
        resolve_score_columns([*SCORE_COLUMNS, SCORE_COLUMNS[0]])


def test_fingerprint_helpers():
    # an 8-item subscale only yields multiples of 100/32
    assert feasible_item_counts(21.875) == (8,)
    # a 7-item subscale only yields multiples of 100/28
    assert feasible_item_counts(3.571428571428571) == (7,)
    assert column_fingerprint([21.875, 46.875, 0.0, 87.5]) == 8
    assert column_fingerprint([3.571428571428571, 89.28571428571429, 0.0]) == 7
    # an all-ambiguous column (0/100-style values) fits every granularity:
    # the smallest consistent n is returned, real disambiguation needs the hash guard
    assert column_fingerprint([0.0] * 5) == 4
    # row 273 of the real file: value fits a different subscale size
    assert fingerprint_anomalies([28.571428571428573, 46.875], 7) == [1]


@pytest.mark.skipif(not has_mendeley_file, reason="real dataset file only on dev host")
def test_mendeley_sha256_guard(tmp_path):
    assert verify_mendeley_sha256(MENDELEY_FILE).startswith("6206f432")
    tampered = tmp_path / "tampered.xls"
    tampered.write_bytes(MENDELEY_FILE.read_bytes() + b"\x00")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_mendeley_scores(tampered)


@pytest.mark.skipif(not has_mendeley_file, reason="real dataset file only on dev host")
def test_mendeley_adapter_contract():
    records = load_mendeley_scores(MENDELEY_FILE)
    assert len(records) == 274  # paper cohort size confirmed at row level
    first = records[0]
    dims = first.provenance["score_dims"]
    assert dims == list(CANONICAL_ORDER)  # every dimension addressable BY NAME
    # balanced is canonical index 0 and unobserved — never appended at the end
    assert dims[0] == "balanced"
    assert first.scores[0] is None and first.score_mask[0] is False
    # all eight biased dims observed at their canonical positions, 0-100
    observed = {
        dim: first.scores[i] for i, dim in enumerate(dims) if first.score_mask[i]
    }
    assert set(observed) == set(SOURCE_COLUMN_MAPPING.values())
    assert all(0.0 <= s <= 100.0 for s in observed.values())
    assert set(first.features) == {
        "demographics", "anthropometrics", "meridian", "pulse_wave", "tongue",
    }
    assert len(first.features["pulse_wave"]) == 15  # SYS..MWTCI
    assert len(first.features["tongue"]) == 11
    provenance = first.provenance
    assert provenance["standard_version"] == "ZYYXH/T157-2009"
    assert provenance["label_source"] == "PUBLISHED"
    assert provenance["canonical_mapping_state"] == "FINGERPRINT_CROSS_CHECKED"
    assert provenance["source_column_mapping"] == SOURCE_COLUMN_MAPPING


@pytest.mark.skipif(not has_mendeley_file, reason="real dataset file only on dev host")
def test_mendeley_score_column_fingerprints():
    """Every score column must fingerprint to its mapped subscale's ZYYXH/
    T157-2009 item count; only the known row-273 anomaly may deviate."""
    import xlrd

    records = load_mendeley_scores(MENDELEY_FILE)  # raises on bad fingerprints
    wb = xlrd.open_workbook(str(MENDELEY_FILE))
    sheet = wb.sheet_by_name("data")
    headers = [str(sheet.cell_value(0, c)) for c in range(sheet.ncols)]
    for col, dim in SOURCE_COLUMN_MAPPING.items():
        values = [sheet.cell_value(r, headers.index(col)) for r in range(1, sheet.nrows)]
        assert column_fingerprint(values) == EXPECTED_ITEMS[dim], col
        assert set(fingerprint_anomalies(values, EXPECTED_ITEMS[dim])) <= {273}, col
    assert set(records[0].provenance["fingerprint_anomalies"]) <= {
        "yang_deficiency", "phlegm_damp", "damp_heat", "qi_depression",
    }


def test_adapter_refuses_unaudited_mapping(monkeypatch):
    """Negative (#38): a mapping that is not a bijective cover of the biased
    canonical dims aborts before any file is read."""
    monkeypatch.setattr(
        "data.mendeley.SOURCE_COLUMN_MAPPING",
        {**SOURCE_COLUMN_MAPPING, "BS": "qi_depression"},  # blood_stasis lost
    )
    with pytest.raises(ValueError, match="bijective"):
        assert_mapping_is_audited()
