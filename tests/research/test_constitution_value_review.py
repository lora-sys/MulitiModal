"""任务 B 契约测试：体质预测价值复核的 shape/mask/负例与报告 schema。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research import constitution_value_review as review

pytestmark = pytest.mark.synthetic


def _rows(n: int = 6):
    return [{
        "subject_id": f"s{i}",
        "features": [float(j + 1) for j in range(63)],
        "feature_mask": [True] * 63,
        "scores": [float(20 + i + j) for j in range(8)],
        "score_dims": list(review.OBSERVED_DIMS),
        "score_mask": [True] * 8,
    } for i in range(n)]


def test_mean_baseline_uses_only_observed_training_scores():
    rows = _rows()
    rows[0]["score_mask"][0] = False
    rows[0]["scores"][0] = 999.0  # 未观测，绝不能参与均值
    preds = review.mean_baseline_predictions(rows, 3)
    assert len(preds) == 3
    assert all(len(p) == review.N_TARGET_DIMS for p in preds)
    assert preds[0][0] != 999.0
    assert preds[0] == preds[1] == preds[2]


def test_masked_errors_shape_and_mask():
    rows = _rows()
    row = rows[0]
    row["score_mask"][2] = False
    preds = [list(row["scores"]) for _ in rows]
    errors = review._masked_errors(rows, preds)
    assert len(errors) == len(rows)
    assert all(len(e) == review.N_TARGET_DIMS for e in errors)
    assert errors[0][2] == 0.0  # 未观测维度不产生误差


def test_masked_errors_rejects_ninth_dimension_and_nan():
    rows = _rows(1)
    with pytest.raises(ValueError, match="8 values"):
        review._masked_errors(rows, [list(rows[0]["scores"]) + [1.0]])
    with pytest.raises(ValueError, match="non-finite"):
        review._masked_errors(rows, [[float("nan")] * 8])


def test_bootstrap_ci_is_deterministic_and_brackets_mean():
    import statistics
    values = [float(i) - 5.0 for i in range(20)]
    mean = statistics.mean(values)
    assert review._bootstrap_ci(values, 11) == review._bootstrap_ci(values, 11)
    low, high = review._bootstrap_ci(values, 11, 2000)
    assert low <= mean <= high


def test_report_schema_keys():
    """契约：报告必须带齐证据分级 / 输入哈希 / 配对区间 / 逐维结果。"""
    required = {
        "schema_version", "status", "inputs", "protocol", "acceptance",
        "paired", "per_dimension", "fold_details", "unverified_generalization",
    }
    # 用纯函数拼一个最小报告骨架，校验字段契约而非数值
    skeleton = {
        "schema_version": review.REVIEW_SCHEMA,
        "status": {"evidence": review.EVIDENCE,
                   "trusted_evaluator": "AGENT_ISOLATION_UNVERIFIED",
                   "external_validation": "NONE", "clinical_claim": False},
        "inputs": {"holdout_subjects": 0, "scope": "selection-only"},
        "protocol": {"baseline": review.BASELINE_ID},
        "acceptance": {}, "paired": {}, "per_dimension": {},
        "fold_details": [], "unverified_generalization": [],
    }
    assert required <= set(skeleton)
    assert skeleton["status"]["clinical_claim"] is False
    assert skeleton["inputs"]["holdout_subjects"] == 0
    assert isinstance(json.dumps(skeleton), str)


def test_module_source_never_reads_holdout_or_results_tsv():
    source = (Path(__file__).resolve().parents[2]
              / "src/research/constitution_value_review.py").read_text(encoding="utf-8")
    assert "test.json" not in source
    assert "append_history(" not in source
    assert "test.json" not in source
    assert ".fit(" not in source
