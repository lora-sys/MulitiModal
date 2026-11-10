"""#95 unified engineering acceptance over synthetic fixtures."""

from __future__ import annotations

import pytest

from physio.acceptance import (
    EVIDENCE_LEVEL,
    ISOLATION_LEVEL,
    SMOKE_ONLY,
    run_acceptance,
)

pytestmark = pytest.mark.synthetic


def test_acceptance_all_checks_pass():
    report = run_acceptance()
    assert report["status"] == "PASS"
    assert report["passed"] == report["total"] == 16
    assert {check["issue"] for check in report["checks"]} == {
        "#95",
        "#96",
        "#97",
        "#98",
        "#99",
        "#100",
        "#102",
    }
    assert all(check["passed"] for check in report["checks"])


def test_acceptance_keeps_evidence_boundaries():
    report = run_acceptance()
    assert report["evidence_level"] == EVIDENCE_LEVEL == "LOCAL_EXPERIMENTAL"
    assert report["isolation"] == ISOLATION_LEVEL == "AGENT_ISOLATION_UNVERIFIED"
    assert report["smoke_only"] == SMOKE_ONLY == "SMOKE_ONLY"
    assert report["device_control"] == "NOT_CONNECTED"
    assert any("no model performance" in note for note in report["limitations"])


def test_acceptance_exposes_an_independent_real_data_mapping_verdict():
    """Engineering PASS must never be readable as mapping sign-off.

    Removing ``real_data_mapping_status`` from ``run_acceptance()`` would otherwise
    leave this suite green.
    """
    report = run_acceptance()
    assert "real_data_mapping_status" in report
    status = report["real_data_mapping_status"]
    assert status.startswith("MAPPING_NOT_VERIFIED"), status
    assert "blockers" in status
    # The two conclusions stay independent even though every check passes.
    assert report["status"] == "PASS"
    assert report["passed"] == report["total"]
    assert status.startswith("MAPPING_NOT_VERIFIED")
    assert any(
        "never that the 94-field mapping is verified" in item for item in report["limitations"]
    )
