"""CCMQ-60 rule scoring engine (#38).

Deterministic, auditable RULE scoring on the historical 60-item CCMQ:
item groups and reverse keys come from the transcribed researcher SAS
implementation, the transformed-score formula is (sum - n)/(4n)*100 with
reverse items as 6-response, and missing answers ABSTAIN per dimension
(never filled with 0, never guessed). The researcher's own missing-value
imputation is available only in `researcher_imputation` mode for the
Zenodo cross-check and is never used for RULE exports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .standards import CANONICAL_ORDER, N_DIMS

MissingPolicy = Literal["abstain", "researcher_imputation"]


def _git_sha() -> str:
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


SUBSCALES: dict[str, dict] = {
    "balanced": {"items": [1, 2, 7, 8, 21, 53, 54, 27], "reverse": {2, 7, 8, 21, 54, 27}},
    "qi_deficiency": {"items": [2, 3, 4, 5, 22, 6, 7, 26], "reverse": set()},
    "yang_deficiency": {"items": [17, 18, 19, 21, 22, 52, 55], "reverse": set()},
    "yin_deficiency": {"items": [16, 20, 35, 29, 57, 38, 44, 46], "reverse": set()},
    "phlegm_damp": {"items": [13, 15, 50, 28, 42, 49, 51, 58], "reverse": set()},
    # item 60 is the sex-specific damp-heat alternative: responses carry it as
    # key 60 under the applicable sex; the inapplicable variant is never 0-filled
    "damp_heat": {"items": [39, 41, 48, 56, 59, 60], "reverse": set()},
    "blood_stasis": {"items": [33, 36, 37, 40, 43, 27, 45], "reverse": set()},
    "qi_depression": {"items": [8, 9, 10, 11, 12, 14, 47], "reverse": set()},
    "special_diathesis": {"items": [23, 24, 25, 30, 31, 32, 34], "reverse": set()},
}

#: positive-keyed items the researcher imputes as 5 when missing; all others
#: impute 1 (researcher_imputation mode only — see module docstring)
RESEARCHER_MISSING_FIVE = {1, 53}


def transformed_value(item: int, response: int, reverse: bool = False) -> int:
    """Reverse-keyed items score 6 - response — per subscale: the SAS applies
    the reversal only within the balanced scale (e.g. Q2/Q7 are raw inside
    qi_deficiency but reversed inside balanced), so `reverse` must be passed
    from the scoring subscale's own reverse set, never derived globally."""
    return 6 - response if reverse else response


class CCMQ60Scorer:
    """Scores canonical 9D constitution scores from 1-5 item responses."""

    standard_id = "CCMQ-60-2009"
    instrument_id = "CCMQ_60_HISTORICAL"
    label_tier = "RULE"

    def __init__(self, missing_policy: MissingPolicy = "abstain") -> None:
        if missing_policy not in ("abstain", "researcher_imputation"):
            raise ValueError(f"unknown missing policy {missing_policy!r}")
        self.missing_policy = missing_policy

    def score(
        self,
        responses: dict[int, int],
        sex: Literal["female", "male"] | None = None,
        source_id: str = "",
    ) -> LabelRecord:
        # strict input validation: any malformed response aborts (no guessing);
        # item 60 is the sex-specific damp-heat alternative (keys 1..60)
        clean: dict[int, int] = {}
        for item, value in responses.items():
            if not isinstance(item, int) or isinstance(item, bool) or not (1 <= item <= 60):
                raise ValueError(f"unknown CCMQ item {item!r}; expected int 1..60")
            if not isinstance(value, int) or isinstance(value, bool) or not (1 <= value <= 5):
                raise ValueError(
                    f"item {item}: response {value!r} outside the 1-5 scale"
                )
            clean[item] = value
        item60 = self._item60(clean, sex)

        scores: list[float | None] = []
        mask: list[bool] = []
        for dim in CANONICAL_ORDER:
            spec = SUBSCALES[dim]
            values: list[int] = []
            missing = False
            for item in spec["items"]:
                v = item60 if item == 60 else clean.get(item)
                if v is None:
                    if self.missing_policy == "researcher_imputation":
                        v = 5 if (item in RESEARCHER_MISSING_FIVE) else 1
                    else:
                        missing = True
                        break
                values.append(transformed_value(item, v, item in spec["reverse"]))
            if missing:
                scores.append(None)
                mask.append(False)
            else:
                n = len(spec["items"])
                scores.append((sum(values) - n) / (4 * n) * 100)
                mask.append(True)
        return LabelRecord(
            scores=scores,
            mask=mask,
            label_tier=self.label_tier,
            standard_version=self.standard_id,
            instrument_id=self.instrument_id,
            missing_policy=self.missing_policy,
            source_id=source_id,
        )

    def _item60(
        self, clean: dict[int, int], sex: Literal["female", "male"] | None
    ) -> int | None:
        if 60 in clean:
            if sex not in ("female", "male"):
                raise ValueError(
                    "item 60 (sex-specific damp-heat question) requires sex to "
                    "select the applicable variant"
                )
            return clean[60]
        # absent answer: the researcher imputes 1 for the applicable variant
        # (SAS: constitution60_Male=. and sex=1 -> 1); RULE mode abstains
        if self.missing_policy == "researcher_imputation" and sex in ("female", "male"):
            return 1
        return None

    def classify(self, record: LabelRecord) -> dict[str, str]:
        """NO / TENDENCY / YES per biased dim; balanced YES / BASICALLY_YES
        per the researcher SAS criteria (cross-check only, not an official
        oracle #38). Any unobserved dim ABSTAINs — including balanced, whose
        "all others below threshold" conditions cannot be established without
        every biased score (no guessing)."""
        status: dict[str, str] = {}
        biased = [d for d in CANONICAL_ORDER if d != "balanced"]
        all_biased_lt30 = True
        all_biased_lt40 = True
        for dim in biased:
            s = record.scores[CANONICAL_ORDER.index(dim)]
            if s is None:
                status[dim] = "ABSTAIN"
                all_biased_lt30 = all_biased_lt40 = False
                continue
            if s < 30:
                status[dim] = "NO"
            elif s < 40:
                status[dim] = "TENDENCY"
                all_biased_lt30 = False
            else:
                status[dim] = "YES"
                all_biased_lt30 = all_biased_lt40 = False
        a = record.scores[0]
        if a is None or any(v == "ABSTAIN" for v in status.values()):
            status["balanced"] = "ABSTAIN"
        elif a >= 60 and all_biased_lt30:
            status["balanced"] = "YES"
        elif a >= 60 and all_biased_lt40:
            status["balanced"] = "BASICALLY_YES"
        else:
            status["balanced"] = "NO"
        return status


@dataclass
class LabelRecord:
    """Versioned, provenance-carrying label output (#38 Scope)."""

    scores: list[float | None]
    mask: list[bool]
    label_tier: Literal["RULE", "PUBLISHED", "LLM_SILVER"]
    standard_version: str
    instrument_id: str
    missing_policy: str = "abstain"
    source_id: str = ""
    source_row: int | None = None
    scorer_commit: str = _git_sha()
    canonical_mapping_verified: bool = True
    extra_provenance: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.scores) != N_DIMS or len(self.mask) != N_DIMS:
            raise ValueError(f"labels must be {N_DIMS}-dimensional")
        if self.label_tier not in ("RULE", "PUBLISHED", "LLM_SILVER"):
            raise ValueError(f"unknown label tier {self.label_tier!r}")
        for s, m in zip(self.scores, self.mask, strict=True):
            if m and s is None:
                raise ValueError("observed dims must carry a score, never None")
            if not m and s is not None:
                raise ValueError("unobserved dims must be None (never zero-filled)")
            if m and not (0.0 <= s <= 100.0):
                raise ValueError(f"score {s} outside the 0-100 range")

    def to_dict(self) -> dict:
        return {
            "label_tier": self.label_tier,
            "standard_version": self.standard_version,
            "instrument_id": self.instrument_id,
            "dims": list(CANONICAL_ORDER),
            "scores": self.scores,
            "mask": self.mask,
            "missing_policy": self.missing_policy,
            "source_id": self.source_id,
            "source_row": self.source_row,
            "scorer_commit": self.scorer_commit,
            "canonical_mapping_verified": self.canonical_mapping_verified,
            **self.extra_provenance,
        }

