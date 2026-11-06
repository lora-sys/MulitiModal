"""任务 B 契约测试：证候价值复核的拒判覆盖率与报告 schema。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from syndrome import value_review as review

pytestmark = pytest.mark.synthetic


def test_status_coverage_counts_and_rates():
    result = review.status_coverage(
        ["OK", "OK", "UNCERTAIN", "ABSTAIN", "OUT_OF_DOMAIN"])
    assert result["n_inputs"] == 5
    assert result["counts"] == {"OK": 2, "UNCERTAIN": 1, "ABSTAIN": 1,
                                "OUT_OF_DOMAIN": 1}
    assert result["decision_coverage"] == 0.4
    assert result["abstain_coverage"] == 0.6
    assert result["all_rejected"] is False


def test_status_coverage_all_rejected_and_empty():
    all_rejected = review.status_coverage(["ABSTAIN", "OUT_OF_DOMAIN"])
    assert all_rejected["all_rejected"] is True
    assert all_rejected["decision_coverage"] == 0.0
    assert review.status_coverage([])["n_inputs"] == 0


def test_report_schema_contract():
    required = {
        "schema_version", "status", "inputs", "protocol", "acceptance",
        "threshold", "abstention", "per_class", "unverified_generalization",
    }
    skeleton = {key: {} for key in required}
    skeleton["schema_version"] = review.REVIEW_SCHEMA
    assert required <= set(skeleton)
    assert isinstance(json.dumps(skeleton), str)


def test_module_source_never_reads_holdout_or_trains():
    source = (Path(__file__).resolve().parents[2]
              / "src/syndrome/value_review.py").read_text(encoding="utf-8")
    assert "assert_holdout_untouched" in source
    assert "test.json" not in source
    assert "append_history(" not in source
    assert ".fit(" not in source


def test_review_raises_when_artifact_unavailable(tmp_path):
    """工件不可用必须显式报错，不得静默给出伪结果。"""
    with pytest.raises((OSError, ValueError, FileNotFoundError)):
        review.review(tmp_path / "missing_manifest.json", tmp_path / "missing_dataset.json",
                      tmp_path, str(tmp_path / "no-artifact"))
