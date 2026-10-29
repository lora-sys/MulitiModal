"""CCMQ-60 rule engine tests (#38).

Synthetic cases are hand-computed from the frozen formula
score = (sum(corrected items) - n) / (4n) * 100 with reverse items as
6-response. Cross-check tests run only where the Zenodo reference files
exist (dev host): the engine's item groups / reverse keys / scale sizes
are re-parsed from the researcher SAS source bytes, and the engine must
score the 14-row TEST_DATA sample under both missing policies.
"""

import re
from pathlib import Path

import pytest

from labels.ccmq60 import (
    RESEARCHER_MISSING_FIVE,
    SUBSCALES,
    CCMQ60Scorer,
    LabelRecord,
    transformed_value,
)
from labels.standards import CANONICAL_ORDER

pytestmark = pytest.mark.synthetic

DATA_ROOT = Path(__file__).resolve().parents[2]
ZENODO_DIR = DATA_ROOT / "data/local/zenodo-4431679"
SAS_FILE = ZENODO_DIR / "CCMQ SAS code.sas"
TEST_DATA = ZENODO_DIR / "TEST_DATA.xlsx"
has_zenodo = SAS_FILE.exists() and TEST_DATA.exists()

ALL_ONE = {i: 1 for i in range(1, 60)}
ALL_FIVE = {i: 5 for i in range(1, 60)}


def score_all(responses: dict[int, int], sex=None, **policy) -> dict[str, float | None]:
    record = CCMQ60Scorer(**policy).score(responses, sex=sex)
    return dict(zip(CANONICAL_ORDER, record.scores, strict=True))


def test_all_five_responses():
    """Forward items -> 100; balanced keeps 2 forward 5s + 6 reversed to 1:
    (2*5 + 6*1 - 8) / 32 * 100 = 25."""
    scores = score_all({**ALL_FIVE, 60: 5}, sex="male")
    assert scores["balanced"] == 25.0
    for dim in CANONICAL_ORDER[1:]:
        assert scores[dim] == 100.0, dim


def test_all_one_responses():
    """Forward items -> 0; balanced: 6 reversed to 5 + 2 forward 1s:
    (6*5 + 2*1 - 8) / 32 * 100 = 75."""
    scores = score_all({**ALL_ONE, 60: 1}, sex="female")
    assert scores["balanced"] == 75.0
    for dim in CANONICAL_ORDER[1:]:
        assert scores[dim] == 0.0, dim


def test_classification_thresholds():
    """Researcher criteria: biased NO <30, TENDENCY [30,40), YES >=40;
    balanced YES >=60 with all biased <30, BASICALLY_YES >=60 with all
    biased <40 and some >=30 (SAS lines 306-355)."""
    scorer = CCMQ60Scorer()

    yes = scorer.classify(scorer.score({**ALL_ONE, 60: 1}, sex="female"))
    assert set(yes.values()) == {"NO"} | {"YES"}  # biased NO, balanced YES

    # qi_deficiency sum 18 -> (18-8)/32*100 = 31.25 TENDENCY; balanced rises to
    # 71.875 (Q2/Q7 shared, reversed only in balanced) -> BASICALLY_YES
    tend = scorer.classify(scorer.score(
        {**ALL_ONE, 2: 2, 3: 2, 4: 2, 5: 2, 22: 2, 6: 2, 7: 3, 26: 3, 60: 1},
        sex="female",
    ))
    assert tend["qi_deficiency"] == "TENDENCY"
    assert tend["balanced"] == "BASICALLY_YES"

    # yang_deficiency sum 19 -> (19-7)/28*100 = 42.86 YES -> balanced NO
    hard = scorer.classify(scorer.score(
        {**ALL_ONE, 17: 3, 18: 3, 19: 3, 21: 3, 22: 3, 52: 2, 55: 2, 60: 1},
        sex="female",
    ))
    assert hard["yang_deficiency"] == "YES"
    assert hard["balanced"] == "NO"


def test_reverse_transformation_is_per_subscale():
    """Q2/Q7 are reverse-keyed inside balanced but RAW inside qi_deficiency
    (SAS: scoreA uses _transformed, scoreB uses _Trmissing for the same
    items) — reversal must never be applied globally."""
    assert transformed_value(2, 1, reverse=True) == 5
    assert transformed_value(2, 5, reverse=True) == 1
    assert transformed_value(2, 1, reverse=False) == 1
    assert transformed_value(1, 3) == 3  # forward
    assert transformed_value(53, 4) == 4  # forward despite imputation policy


def test_missing_item_abstains_never_zero_filled():
    record = CCMQ60Scorer().score({k: v for k, v in ALL_ONE.items() if k != 17}, sex="female")
    dims = dict(zip(CANONICAL_ORDER, record.scores, strict=True))
    assert dims["yang_deficiency"] is None
    assert record.mask[CANONICAL_ORDER.index("yang_deficiency")] is False
    assert dims["qi_deficiency"] == 0.0  # unaffected dim still scored
    status = CCMQ60Scorer().classify(record)
    assert status["yang_deficiency"] == "ABSTAIN"
    # balanced cannot establish "all others < 30" -> ABSTAIN, never guessed
    assert status["balanced"] == "ABSTAIN"


def test_researcher_imputation_mode():
    """Q1/Q53 missing -> 5, others -> 1 (SAS line 15 comment). Cross-check
    replication only; RULE exports use abstain mode."""
    base = {k: v for k, v in ALL_ONE.items() if k != 17}
    imp = CCMQ60Scorer(missing_policy="researcher_imputation")
    record = imp.score(base, sex="female")
    dims = dict(zip(CANONICAL_ORDER, record.scores, strict=True))
    assert dims["yang_deficiency"] == 0.0  # imputed 1, all corrected 1s
    record_q1 = imp.score({k: v for k, v in ALL_ONE.items() if k != 1}, sex="female")
    dims_q1 = dict(zip(CANONICAL_ORDER, record_q1.scores, strict=True))
    # balanced: Q1 imputed 5 (forward), other 7 items corrected as all-1 case
    assert dims_q1["balanced"] == (5 + 6 * 5 + 1 - 8) / 32 * 100
    assert set(RESEARCHER_MISSING_FIVE) == {1, 53}


def test_item60_sex_specific_damp_heat():
    scorer = CCMQ60Scorer()
    # answered without sex -> rejected (variant unknown)
    with pytest.raises(ValueError, match="requires sex"):
        scorer.score({**ALL_ONE, 60: 4})
    # abstain mode: unanswered -> damp_heat unobserved
    record = scorer.score(ALL_ONE, sex="male")
    assert record.scores[CANONICAL_ORDER.index("damp_heat")] is None
    # researcher mode: SAS imputes 1 for the applicable variant
    imp = CCMQ60Scorer(missing_policy="researcher_imputation")
    record_imp = imp.score(ALL_ONE, sex="male")
    assert record_imp.scores[CANONICAL_ORDER.index("damp_heat")] == 0.0


def test_invalid_inputs_rejected():
    scorer = CCMQ60Scorer()
    for bad_item in (0, 61, -1, True, 1.5, "3"):
        with pytest.raises(ValueError, match="unknown CCMQ item"):
            scorer.score({bad_item: 3})
    for bad_value in (0, 6, -2, True, 2.5, "3"):
        with pytest.raises(ValueError, match="1-5 scale"):
            scorer.score({1: bad_value})


def test_unknown_missing_policy_rejected():
    with pytest.raises(ValueError, match="unknown missing policy"):
        CCMQ60Scorer(missing_policy="fill_with_zero")


def test_label_record_invariants():
    with pytest.raises(ValueError, match="9-dimensional"):
        LabelRecord(scores=[0.0] * 8, mask=[True] * 8, label_tier="RULE",
                    standard_version="s", instrument_id="i")
    with pytest.raises(ValueError, match="never None"):
        LabelRecord(scores=[None] * 9, mask=[True] * 9, label_tier="RULE",
                    standard_version="s", instrument_id="i")
    with pytest.raises(ValueError, match="never zero-filled"):
        LabelRecord(scores=[0.0] * 9, mask=[False] * 9, label_tier="RULE",
                    standard_version="s", instrument_id="i")
    with pytest.raises(ValueError, match="0-100"):
        LabelRecord(scores=[150.0] * 9, mask=[True] * 9, label_tier="RULE",
                    standard_version="s", instrument_id="i")
    with pytest.raises(ValueError, match="unknown label tier"):
        LabelRecord(scores=[0.0] * 9, mask=[True] * 9, label_tier="GOLD",
                    standard_version="s", instrument_id="i")


def test_engine_subscales_cover_canonical_order():
    biased = [d for d in CANONICAL_ORDER if d != "balanced"]
    assert sorted(SUBSCALES) == sorted(CANONICAL_ORDER)
    assert SUBSCALES["balanced"]["items"] == [1, 2, 7, 8, 21, 53, 54, 27]
    assert SUBSCALES["balanced"]["reverse"] == {2, 7, 8, 21, 54, 27}
    for dim in biased:
        assert SUBSCALES[dim]["reverse"] == set()
        assert len(SUBSCALES[dim]["items"]) in (6, 7, 8)


# --- Zenodo cross-check (dev host only: real reference files) --------------

def parse_sas_subscales(text: str) -> dict[str, dict]:
    """Re-derive item groups / reverse keys / scale sizes from the researcher
    SAS bytes — an artifact-grounded check of the hand transcription."""
    parsed: dict[str, dict] = {}
    for letter in "ABCDEFGHI":
        m = re.search(rf"\bscore{letter}\s*=(.*?);", text, re.DOTALL)
        assert m, f"score{letter} definition not found in SAS"
        body = m.group(1)
        items = {
            int(n) for n in re.findall(r"TCM_consti(?:tution)?(\d+)_Trmissing", body)
        }
        reverse = {
            int(n) for n in re.findall(r"TCM_constitution(\d+)_transformed", body)
        }
        nm = re.search(
            rf"T_score{letter}\s*=\s*\(?\s*score{letter}\s*-\s*(\d+)\s*\)?\s*/\s*\(\s*4\s*\*\s*(\d+)\s*\)",
            text,
        )
        assert nm, f"T_score{letter} formula not found in SAS"
        n = int(nm.group(2))
        assert int(nm.group(1)) == n, f"score{letter}: formula min {nm.group(1)} != n {n}"
        assert len(items | reverse) == n, f"score{letter}: {len(items | reverse)} items != n {n}"
        parsed[letter] = {"items": items | reverse, "reverse": reverse, "n": n}
    return parsed


@pytest.mark.skipif(not has_zenodo, reason="Zenodo reference files only on dev host")
def test_engine_matches_sas_source_bytes():
    parsed = parse_sas_subscales(SAS_FILE.read_text(errors="replace"))
    # SAS letters A..I correspond to the canonical order (A=gentleness, then
    # qi/yang/yin/phlegm-wetness/wetness-heat/blood/qi-depression/special)
    for letter, dim in zip("ABCDEFGHI", CANONICAL_ORDER, strict=True):
        spec = SUBSCALES[dim]
        assert set(spec["items"]) == parsed[letter]["items"], dim
        assert set(spec["reverse"]) == parsed[letter]["reverse"], dim
        assert len(spec["items"]) == parsed[letter]["n"], dim


def load_test_data_rows() -> list[dict]:
    import openpyxl

    wb = openpyxl.load_workbook(TEST_DATA, read_only=True)
    ws = wb["Sheet1"]
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    out = []
    for row in rows:
        sex_code = int(row[1])
        responses = {i: int(row[1 + i]) for i in range(1, 60)}
        # SAS: sex=0 uses the Female variant column (col 61), sex=1 the Male
        # column (col 62); columns 2..60 hold items 1..59
        responses[60] = int(row[61 if sex_code == 0 else 62])
        out.append({"sex": "female" if sex_code == 0 else "male", "responses": responses})
    return out


@pytest.mark.skipif(not has_zenodo, reason="Zenodo reference files only on dev host")
def test_engine_scores_zenodo_reference_sample():
    rows = load_test_data_rows()
    assert len(rows) == 14  # reference sample size
    scorer = CCMQ60Scorer(missing_policy="researcher_imputation")
    for row in rows:
        record = scorer.score(row["responses"], sex=row["sex"])
        assert all(record.mask), "researcher mode must score all nine dims"
        assert all(0.0 <= s <= 100.0 for s in record.scores)
        status = scorer.classify(record)
        assert set(status.values()) <= {"NO", "TENDENCY", "YES", "BASICALLY_YES"}
    # abstain mode on the same rows: complete responses -> same scores
    rule = CCMQ60Scorer()
    for row in rows:
        record = rule.score(row["responses"], sex=row["sex"])
        assert all(record.mask), "TEST_DATA rows are complete; no dim may abstain"
