"""Scoring standard registry (#38).

Standards are versioned and never mixed: the historical 60-item CCMQ
(CCMQ_60_HISTORICAL) is cross-checked against the Zenodo researcher code
(independent reference, NOT an official oracle); GB/T 46939-2025 stays
NOT_IMPLEMENTED until the official full text has been checked item by item
(#38/#47). Canonical nine-dimension order is fixed project-wide.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

#: canonical 9D label order (#38 frozen spec), scores 0-100 + boolean mask
CANONICAL_ORDER = (
    "balanced",          # 平和
    "qi_deficiency",     # 气虚
    "yang_deficiency",   # 阳虚
    "yin_deficiency",    # 阴虚
    "phlegm_damp",       # 痰湿
    "damp_heat",         # 湿热
    "blood_stasis",      # 血瘀
    "qi_depression",     # 气郁
    "special_diathesis",  # 特禀
)
N_DIMS = len(CANONICAL_ORDER)

StandardState = Literal[
    "CROSS_CHECKED_RESEARCHER", "NOT_IMPLEMENTED_RULES_UNVERIFIED", "VERIFIED_OFFICIAL"
]


@dataclass(frozen=True)
class ScoringStandard:
    standard_id: str
    instrument_id: str
    version: str
    official_source: str
    state: StandardState
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.standard_id or not self.instrument_id:
            raise ValueError("standard_id and instrument_id are required")


#: Historical 60-item CCMQ. Item groups/reverse keys transcribed from the
#: Zenodo researcher SAS implementation (md5 3b622e25…, dev host
#: data/local/zenodo-4431679/); used as an independent cross-check source,
#: explicitly not an official standard oracle (#38). Formula per subscale:
#: (sum(corrected items) - n) / (4*n) * 100; reverse item -> 6 - response.
CCMQ_60_HISTORICAL = ScoringStandard(
    standard_id="CCMQ-60-2009",
    instrument_id="CCMQ_60_HISTORICAL",
    version="2009",
    official_source="https://pmc.ncbi.nlm.nih.gov/articles/PMC10164641/ (paper); "
                    "cross-check https://zenodo.org/records/4431679",
    state="CROSS_CHECKED_RESEARCHER",
    notes="Researcher missing-value policy (Q1/Q53 missing -> 5, others -> 1) "
          "is replicated ONLY in researcher_imputation mode for cross-checking; "
          "RULE exports use abstain semantics (missing -> unobserved dim).",
)

#: GB/T 46939-2025 — engineering interpretation from a third-party
#: transcription exists, but the official full text has not been obtained
#: and checked item by item, so no rule implementation is published here.
GBT_46939_2025 = ScoringStandard(
    standard_id="GBT-46939-2025",
    instrument_id="GBT_46939_2025",
    version="2025",
    official_source="https://openstd.samr.gov.cn/bzgk/std/newGbInfo?hcno=827D6D266BA52A2983F93638DA871028",
    state="NOT_IMPLEMENTED_RULES_UNVERIFIED",
    notes="Do not guess item texts, thresholds or reverse keys from the "
          "third-party transcription; implement only after official-PDF audit.",
)

SCORING_STANDARDS = {s.standard_id: s for s in (CCMQ_60_HISTORICAL, GBT_46939_2025)}
