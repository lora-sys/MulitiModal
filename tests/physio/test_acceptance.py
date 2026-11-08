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
    assert report["passed"] == report["total"] == 13
    assert {check["issue"] for check in report["checks"]} == {
        "#95",
        "#96",
        "#97",
        "#98",
        "#99",
        "#100",
    }
    assert all(check["passed"] for check in report["checks"])


def test_acceptance_keeps_evidence_boundaries():
    report = run_acceptance()
    assert report["evidence_level"] == EVIDENCE_LEVEL == "LOCAL_EXPERIMENTAL"
    assert report["isolation"] == ISOLATION_LEVEL == "AGENT_ISOLATION_UNVERIFIED"
    assert report["smoke_only"] == SMOKE_ONLY == "SMOKE_ONLY"
    assert report["device_control"] == "NOT_CONNECTED"
    assert any("no model performance" in note for note in report["limitations"])
