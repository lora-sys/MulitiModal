"""PUBLISHED export + LLM_SILVER interface tests (#38).

Real-file PUBLISHED tests run only on the dev host (git-ignored data);
everything else is synthetic.
"""

from pathlib import Path

import pytest

from data.datasets import DATASET_CXTPH3TJSG
from data.mendeley import load_mendeley_scores
from labels.ccmq60 import LabelRecord
from labels.published import (
    PUBLISHED_INSTRUMENT,
    derive_rule_categories,
    published_label_records,
)
from labels.silver import (
    DeterministicAbstainProvider,
    SilverRequest,
    ensure_reference_tier,
)
from labels.standards import CANONICAL_ORDER

pytestmark = pytest.mark.synthetic

MENDELEY_FILE = Path(__file__).resolve().parents[2] / DATASET_CXTPH3TJSG.files[0].path
has_mendeley_file = MENDELEY_FILE.exists()


def make_record(tier: str = "PUBLISHED", **overrides) -> LabelRecord:
    kwargs = dict(
        scores=[75.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        mask=[True] * 9,
        label_tier=tier,
        standard_version="ZYYXH/T157-2009",
        instrument_id=PUBLISHED_INSTRUMENT,
        source_id="test",
        source_row=0,
    )
    kwargs.update(overrides)
    return LabelRecord(**kwargs)


def test_published_record_roundtrip_and_tier_discipline():
    record = make_record()
    assert record.to_dict()["dims"] == list(CANONICAL_ORDER)
    assert record.to_dict()["label_tier"] == "PUBLISHED"
    with pytest.raises(ValueError, match="expects PUBLISHED"):
        derive_rule_categories(make_record(tier="RULE"))


def test_derived_categories_are_rule_decisions_not_published_labels():
    """Balanced 75 with all biased 0 -> researcher criteria give balanced
    YES; the derivation is labeled as RULE-derived in its contract."""
    categories = derive_rule_categories(make_record())
    assert categories["balanced"] == "YES"
    for dim in CANONICAL_ORDER[1:]:
        assert categories[dim] == "NO"


@pytest.mark.skipif(not has_mendeley_file, reason="real dataset file only on dev host")
def test_published_export_names_every_dimension():
    records = published_label_records(load_mendeley_scores(MENDELEY_FILE))
    assert len(records) == 274
    first = records[0]
    assert first.label_tier == "PUBLISHED"
    assert first.instrument_id == PUBLISHED_INSTRUMENT
    assert first.standard_version == "ZYYXH/T157-2009"
    assert first.canonical_mapping_verified is False  # paper table unverified
    # balanced unobserved at canonical index 0, eight biased dims observed
    assert first.scores[0] is None and first.mask[0] is False
    assert sum(first.mask) == 8
    assert first.extra_provenance["canonical_mapping_state"] == "FINGERPRINT_CROSS_CHECKED"
    # subject 273 is the known fingerprint-anomaly row and must be flagged
    flagged = [r for r in records if "qc_fingerprint_anomaly" in r.extra_provenance]
    assert [r.source_row for r in flagged] == [273]
    assert set(flagged[0].extra_provenance["qc_fingerprint_anomaly"]) == {
        "yang_deficiency", "phlegm_damp", "damp_heat", "qi_depression",
    }


def test_silver_default_provider_always_abstains():
    provider = DeterministicAbstainProvider()
    request = SilverRequest(
        source_id="doc-1", source_row=3, prompt_id="p0", model_id=provider.model_id,
        text_evidence="患者形体肥胖，腹部肥满松软。",
    )
    record = provider.label(request)
    assert record.label_tier == "LLM_SILVER"
    assert record.scores == [None] * 9 and record.mask == [False] * 9
    assert "no validated LLM labeling method" in record.extra_provenance["abstain_reason"]
    empty = provider.label(SilverRequest("doc-1", None, "p0", "m", "   "))
    assert "no quoted text evidence" in empty.extra_provenance["abstain_reason"]


def test_silver_never_serves_as_reference_truth():
    provider = DeterministicAbstainProvider()
    silver = provider.label(SilverRequest("doc", None, "p", "m", "x"))
    with pytest.raises(ValueError, match="never be used as reference truth"):
        ensure_reference_tier(silver)
    assert ensure_reference_tier(make_record(tier="RULE")) is not None
    assert ensure_reference_tier(make_record(tier="PUBLISHED")) is not None
