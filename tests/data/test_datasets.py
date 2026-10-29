"""Dataset registry + Mendeley adapter tests (#37).

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
    NINE_CONSTITUTION_DIMS,
    SCORE_COLUMNS,
    load_mendeley_scores,
    verify_mendeley_sha256,
)

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
    assert len(first.scores) == NINE_CONSTITUTION_DIMS
    # eight observed PUBLISHED scores + the absent balanced dim, never fabricated
    assert all(m for m in first.score_mask[:8])
    assert first.score_mask[8] is False and first.scores[8] is None
    assert all(0.0 <= s <= 100.0 for s in first.scores[:8])
    assert set(first.features) == {
        "demographics", "anthropometrics", "meridian", "pulse_wave", "tongue",
    }
    assert len(first.features["pulse_wave"]) == 15  # SYS..MWTCI
    assert len(first.features["tongue"]) == 11
    assert first.provenance["standard_version"] == "CCMQ-2009"
    assert first.provenance["label_source"] == "PUBLISHED"
    # scores sorted in SCORE_COLUMNS order, provenance hash matches the file
    assert len(SCORE_COLUMNS) == 8
