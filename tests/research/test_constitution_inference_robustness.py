"""任务 A 回归：体质推理的异常输入 / 缺失值 / 极端值 / 独立进程工件加载。

全部 SYNTHETIC（除最后一个 host-gated 用例），不需要真实数据或工件。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from research import constitution_delivery as delivery

pytestmark = pytest.mark.synthetic

REPO = Path(__file__).resolve().parents[2]


def _row(index: int = 0) -> dict:
    return {
        "subject_id": f"s{index}",
        "features": [float(i + 1) for i in range(63)],
        "feature_mask": [True] * 63,
        "scores": [float(10 + i) for i in range(8)],
        "score_dims": list(delivery.OBSERVED_DIMS),
        "score_mask": [True] * 8,
    }


def _inference():
    model = delivery.ConstitutionPredictor([_row(i) for i in range(40)], seed=20261105)
    return delivery.ConstitutionInference(model, {"model": "synthetic"}), model


@pytest.mark.parametrize("bad", [42, 1.5, True, "hello", [1, 2, 3], b"abc"])
def test_non_mapping_input_is_explicitly_rejected(bad):
    inf, _ = _inference()
    result = inf.predict_one(bad)  # 修复前 TypeError 崩溃
    assert result["status"] == "OUT_OF_DOMAIN"
    assert result["scores"] == []
    assert result["reason"]
    assert result["balanced"]["status"] == "ABSTAIN"


def test_missing_required_nine_features_abstains():
    """冠军只用前 9 项；全缺时不得凭补值给出误导性分数。"""
    inf, _ = _inference()
    row = _row()
    row["features"] = [None] * 9 + row["features"][9:]
    row["feature_mask"] = [False] * 9 + [True] * 54
    result = inf.predict_one(row)  # 修复前 status=OK
    assert result["status"] == "ABSTAIN"
    assert result["scores"] == []
    assert "9 项" in result["reason"]


@pytest.mark.parametrize("bad_value", [1e9, 1e308, -1e308])
def test_extreme_finite_values_never_produce_misleading_scores(bad_value):
    inf, _ = _inference()
    row = _row()
    row["features"] = [bad_value] * 63
    with np.errstate(all="ignore"):
        result = inf.predict_one(row)  # 修复前崩溃或给出 3.3e9 分
    assert result["status"] in ("OK", "OUT_OF_DOMAIN")
    if result["status"] == "OK":
        assert all(0.0 <= s["value"] <= 100.0 for s in result["scores"])
    else:
        assert result["reason"]


@pytest.mark.parametrize("bad", [None, 42, 1.5, {}, {"features": [1.0] * 63}])
def test_predict_batch_bad_container_does_not_crash(bad):
    inf, _ = _inference()
    results = inf.predict_batch(bad)
    assert isinstance(results, list) and len(results) >= 1
    assert all(r["status"] in ("ABSTAIN", "OUT_OF_DOMAIN", "OK") for r in results)


def test_balanced_dimension_always_abstains_even_on_rejection():
    inf, _ = _inference()
    for payload in (None, 42, _row()):
        result = inf.predict_one(payload)
        assert result["balanced"]["status"] == "ABSTAIN"
        assert result["balanced"]["value"] is None
        assert "balanced" not in [s["dim"] for s in result["scores"]]


def test_save_artifact_refuses_main_bound_class(tmp_path):
    """根因：python -m ... deliver 曾把类 pickle 成 __main__.ConstitutionPredictor。"""
    class StrayPredictor:
        pass

    StrayPredictor.__module__ = "__main__"
    with pytest.raises(ValueError, match="importable"):
        delivery.save_artifact(StrayPredictor(), tmp_path / "art", {})


def test_fresh_process_loads_artifact_without_any_alias(tmp_path):
    """独立进程正常加载：不注入 sys.modules[\"__main__\"] 别名。"""
    model = delivery.ConstitutionPredictor([_row(i) for i in range(40)], seed=20261105)
    art = tmp_path / "art"
    delivery.save_artifact(model, art, {"model": "synthetic"})
    code = (
        "from pathlib import Path; import numpy as np; "
        "from research.constitution_delivery import load_artifact; "
        f"p, e, l = load_artifact(Path({str(art)!r})); "
        "assert type(p).__module__ == 'research.constitution_delivery'; "
        "assert list(l['score_dims']) == list(p.dims)"
    )
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    proc = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env,
                          capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr


def test_cli_entry_delegates_to_importable_module():
    source = (REPO / "src/research/constitution_delivery.py").read_text(encoding="utf-8")
    assert "import research.constitution_delivery as _entry" in source
    assert "_entry.main()" in source
