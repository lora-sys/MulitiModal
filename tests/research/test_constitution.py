"""体质线契约测试（SYNTHETIC）：形状、mask、负例、冻结协议、评分器。

不读真实数据、不训练；真实数据验证在远程由 constitution_data verify 完成。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research import constitution_data as data
from research.constitution import (
    METRIC,
    METRIC_VERSION,
    score_predictions,
)

pytestmark = pytest.mark.synthetic


def _row(index: int = 0, missing: int | None = None) -> dict:
    features = [float(i + 1) for i in range(data.N_FEATURE_COLUMNS)]
    feature_mask = [True] * data.N_FEATURE_COLUMNS
    if missing is not None:
        features[missing] = None
        feature_mask[missing] = False
    return {
        "subject_index": index,
        "subject_id": data.subject_id(index),
        "features": features,
        "feature_mask": feature_mask,
        "scores": [10.0 + i for i in range(data.N_TARGET_DIMS)],
        "score_dims": list(data.OBSERVED_DIMS),
        "score_mask": [True] * data.N_TARGET_DIMS,
        "provenance": {
            "source_dataset": "mendeley-cxtph3tjsg-v1",
            "dataset_sha256": data.EXPECTED_SHA256,
            "label_tier": "PUBLISHED",
            "label_version": data.LABEL_VERSION_REGRESSION,
            "standard_version": data.STANDARD_VERSION,
            "score_column_order": list(data.OBSERVED_DIMS),
            "unobserved_dimensions": list(data.UNOBSERVED_DIMS),
            "identity": "row order is the only anonymous identity: UNVERIFIED",
            "feature_missing_policy": "null + feature_mask=False; never zero-filled",
        },
    }


def _manifest() -> dict:
    ids = [data.subject_id(i) for i in range(data.N_SUBJECTS)]
    return {
        "version": data.PROTOCOL_VERSION,
        "task": data.TASK,
        "dataset_sha256": data.EXPECTED_SHA256,
        "label_tier": "PUBLISHED",
        "label_version": data.LABEL_VERSION_REGRESSION,
        "n_subjects": data.N_SUBJECTS,
        "n_target_dims": data.N_TARGET_DIMS,
        "n_feature_columns": data.N_FEATURE_COLUMNS,
        "metric": "constitution/masked_mae",
        "direction": "minimize",
        "metric_version": "constitution-eval-v1",
        "evidence": data.EVIDENCE,
        "score_dims": list(data.OBSERVED_DIMS),
        "unobserved_dimensions": list(data.UNOBSERVED_DIMS),
        "split": {
            "seed": data.SEED, "folds": data.FOLDS, "repeats": data.REPEATS,
            "train": ids, "dev": [], "holdout": [], "selection_only": True,
        },
        "subject_row_by_id": {sid: i for i, sid in enumerate(ids)},
        "user_partition_by_id": {sid: "train" for sid in ids},
    }


# --------------------------------------------------------------- 冻结契约


def test_observed_dims_exclude_balanced_and_are_eight():
    assert data.N_TARGET_DIMS == 8
    assert data.OBSERVED_DIMS == tuple(data.CANONICAL_ORDER[1:])
    assert data.UNOBSERVED_DIMS == ("balanced",)
    assert "balanced" not in data.OBSERVED_DIMS


def test_manifest_rejects_any_holdout_or_fabricated_balanced():
    manifest = _manifest()
    data._validate_manifest(manifest)  # 正常路径
    for mutation in (
        lambda m: m["split"].update(holdout=["m0000"]),
        lambda m: m.update(score_dims=list(data.CANONICAL_ORDER)),
        lambda m: m.update(unobserved_dimensions=[]),
        lambda m: m.update(label_tier="RULE"),
        lambda m: m.update(n_target_dims=9),
        lambda m: m.update(metric_version="other"),
        lambda m: m["split"].update(selection_only=False),
        lambda m: m["split"].update(folds=5),
        lambda m: m["split"].update(seed=1),
        lambda m: m["split"].update(train=manifest["split"]["train"][:-1]),
        lambda m: m.update(subject_row_by_id={}),
    ):
        broken = copy.deepcopy(manifest)
        mutation(broken)
        with pytest.raises(ValueError):
            data._validate_manifest(broken)


def test_row_rejects_fabricated_balanced_label():
    row = _row()
    data._validate_row(row, 0, _manifest())  # 正常路径
    for mutation in (
        lambda r: r["score_dims"].insert(0, "balanced"),
        lambda r: r["scores"].append(0.0),
        lambda r: r.update(scores=[None] + r["scores"]),
        lambda r: r["provenance"].update(unobserved_dimensions=[]),
        lambda r: r.update(score_mask=[False] + [True] * 7),
    ):
        broken = copy.deepcopy(row)
        mutation(broken)
        with pytest.raises(ValueError):
            data._validate_row(broken, 0, _manifest())


def test_missing_feature_must_be_null_never_zero_filled():
    row = _row(missing=3)
    assert row["features"][3] is None and row["feature_mask"][3] is False
    data._validate_row(row, 0, _manifest())
    zero_filled = copy.deepcopy(row)
    zero_filled["features"][3] = 0.0  # mask 仍为 False：伪造缺失特征
    with pytest.raises(ValueError, match="missing feature must be null"):
        data._validate_row(zero_filled, 0, _manifest())
    # 观测位被写成 0.0 是合法数值，但缺失位必须保持 null + mask=False
    observed_zero = copy.deepcopy(row)
    observed_zero["features"][0] = 0.0
    data._validate_row(observed_zero, 0, _manifest())
    inconsistent = copy.deepcopy(row)
    inconsistent["feature_mask"][3] = True
    with pytest.raises(ValueError):
        data._validate_row(inconsistent, 0, _manifest())


def test_row_rejects_wrong_identity_and_non_finite_scores():
    for mutation in (
        lambda r: r.update(subject_index=7),
        lambda r: r.update(subject_id="wrong"),
        lambda r: r["scores"].__setitem__(0, float("nan")),
        lambda r: r["features"].__setitem__(0, True),
        lambda r: r["provenance"].update(label_tier="LLM_SILVER"),
    ):
        broken = _row()
        mutation(broken)
        with pytest.raises(ValueError):
            data._validate_row(broken, 0, _manifest())


# --------------------------------------------------------------- 分组折


def test_folds_are_deterministic_disjoint_and_cover_pool():
    manifest = _manifest()
    for repeat in range(data.REPEATS):
        seen: set[str] = set()
        for fold in range(data.FOLDS):
            train, dev = data.selection_fold_ids(manifest, fold, data.FOLDS, repeat)
            assert train == data.selection_fold_ids(manifest, fold, data.FOLDS, repeat)[0]
            assert set(train).isdisjoint(dev)
            assert set(train) | set(dev) == set(manifest["split"]["train"])
            assert not set(dev) & seen
            seen |= set(dev)
        assert seen == set(manifest["split"]["train"])
    for bad in ((-1, 0), (4, 0), (0, 2)):
        with pytest.raises(ValueError):
            data.selection_fold_ids(manifest, bad[0], data.FOLDS, bad[1])
    with pytest.raises(ValueError):
        data.selection_fold_ids(manifest, 0, 5, 0)


# --------------------------------------------------------------- 评分器


def test_score_predictions_masked_mae_and_perfect_score():
    rows = [_row(0), _row(1)]
    perfect = [row["scores"] for row in rows]
    value, per_dim = score_predictions(rows, perfect)
    assert value == 0.0 and per_dim == [0.0] * 8
    shifted = [[value + 2.0 for value in row["scores"]] for row in rows]
    value, per_dim = score_predictions(rows, shifted)
    assert value == 2.0 and per_dim == [2.0] * 8


def test_score_predictions_rejects_wrong_shape_and_fabricated_balanced():
    rows = [_row(0), _row(1)]
    for bad in (
        [[0.0] * 8],                                  # 行数不足
        [[0.0] * 9, [0.0] * 9],                       # 9 维＝伪造平和质
        [[0.0] * 8, [0.0] * 7],
        [[0.0] * 8, [float("nan")] * 8],
        [[0.0] * 8, [True] * 8],
        "not-a-list",
    ):
        with pytest.raises(ValueError):
            score_predictions(rows, bad)


# --------------------------------------------------------------- 冻结元数据


def test_record_requires_frozen_metadata(tmp_path: Path):
    from research.constitution import validate_result

    result = {
        "metric": METRIC, "direction": "minimize",
        "metric_version": METRIC_VERSION, "evidence": data.EVIDENCE,
        "promotion": "strict", "masked_mae": 12.5,
        "balanced_dimension": "UNOBSERVED_ABSTAIN_NEVER_FABRICATED",
        "holdout_subjects": 0, "manifest_sha256": "a" * 64,
        "code_sha256": {"candidate": "b" * 64},
    }
    assert validate_result(result) == 12.5
    for mutation in (
        lambda r: r.update(metric="syndrome/macro_f1"),
        lambda r: r.update(direction="maximize"),
        lambda r: r.update(metric_version="other"),
        lambda r: r.update(evidence="SMOKE_ONLY"),
        lambda r: r.update(promotion="loose"),
        lambda r: r.update(holdout_subjects=1),
        lambda r: r.update(balanced_dimension="FABRICATED"),
        lambda r: r.update(masked_mae=-1.0),
        lambda r: r.update(masked_mae=float("nan")),
        lambda r: r.pop("masked_mae"),
    ):
        broken = copy.deepcopy(result)
        mutation(broken)
        with pytest.raises(ValueError):
            validate_result(broken)


def test_candidate_baseline_outputs_eight_dims(tmp_path: Path):
    """基线候选在隔离风格下形状正确，且绝不输出第 9 维。"""
    import subprocess
    import sys

    train = {"rows": [_row(0), _row(1)], "score_dims": list(data.OBSERVED_DIMS)}
    predict = {"rows": [{"features": _row(2)["features"],
                         "feature_mask": _row(2)["feature_mask"]},
                        {"features": _row(3)["features"],
                         "feature_mask": _row(3)["feature_mask"]}],
               "score_dims": list(data.OBSERVED_DIMS)}
    train_path, predict_path, out_path = (tmp_path / "train.json",
                                          tmp_path / "predict.json",
                                          tmp_path / "output.json")
    train_path.write_text(json.dumps(train), encoding="utf-8")
    predict_path.write_text(json.dumps(predict), encoding="utf-8")
    candidate = Path(__file__).resolve().parents[2] / "src/research/constitution_train.py"
    subprocess.run([sys.executable, "-I", str(candidate), "--train", str(train_path),
                    "--predict", str(predict_path), "--output", str(out_path),
                    "--seed", "20261105"], check=True, timeout=30)
    output = json.loads(out_path.read_text(encoding="utf-8"))
    assert len(output) == 2 and all(len(row) == 8 for row in output)
    assert all(isinstance(value, (int, float)) for row in output for value in row)
