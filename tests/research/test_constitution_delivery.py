"""体质冠军交付契约测试（SYNTHETIC）。

覆盖：第 9 维 ABSTAIN、形状/mask/负例、工件保存重载与哈希篡改、
报告 schema、registry provenance、verify 不训练。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from research import constitution_delivery as delivery

pytestmark = pytest.mark.synthetic


def _row(index: int = 0, missing: int | None = None) -> dict:
    features = [float(i + 1) for i in range(63)]
    feature_mask = [True] * 63
    if missing is not None:
        features[missing] = None
        feature_mask[missing] = False
    return {
        "subject_index": index,
        "subject_id": f"m{index:04d}",
        "features": features,
        "feature_mask": feature_mask,
        "scores": [10.0 + j for j in range(8)],
        "score_dims": ["qi_deficiency", "yang_deficiency", "yin_deficiency",
                       "phlegm_damp", "damp_heat", "blood_stasis",
                       "qi_depression", "special_diathesis"],
        "score_mask": [True] * 8,
    }


def _predictor():
    rows = [_row(i) for i in range(40)]
    return delivery.ConstitutionPredictor(rows, seed=20261105), rows


def test_predictor_outputs_eight_dims_and_never_ninth():
    model, rows = _predictor()
    output = model.predict(rows[:3])
    assert output.shape == (3, 8)
    assert np.isfinite(output).all()
    assert tuple(model.dims) == tuple(delivery.OBSERVED_DIMS)
    assert "balanced" not in model.dims
    assert len(model.provenance()["config"]["K_GRID"]) == 6
    assert model.predict([]).shape == (0, 8)


def test_inference_balanced_is_always_abstain():
    model, rows = _predictor()
    inference = delivery.ConstitutionInference(model, {"model": "synthetic"})
    for row in (rows[0], None, _row(1, missing=12),
                {"features": [None] * 63, "feature_mask": [False] * 63},
                {"features": [1.0] * 3, "feature_mask": [True] * 3}):
        result = inference.predict_one(row)
        assert result["balanced"]["status"] == "ABSTAIN"
        assert result["balanced"]["value"] is None
        assert result["calibrated"] is False
        assert all(c["calibrated"] is False for c in result["scores"])
        assert len(result["scores"]) in (0, 8)


def test_inference_statuses_and_missing_feature_policy():
    model, rows = _predictor()
    inference = delivery.ConstitutionInference(model, {})
    assert inference.predict_one(rows[0])["status"] == "OK"
    assert inference.predict_one(None)["status"] == "ABSTAIN"
    assert inference.predict_one({"features": [None] * 63,
                                   "feature_mask": [False] * 63})["status"] == "ABSTAIN"
    assert inference.predict_one({"features": [1.0] * 3,
                                   "feature_mask": [True] * 3})["status"] == "OUT_OF_DOMAIN"
    zero_filled = copy.deepcopy(rows[0])
    zero_filled["features"][5] = 0.0
    zero_filled["feature_mask"][5] = False
    assert inference.predict_one(zero_filled)["status"] == "OUT_OF_DOMAIN"
    partial = _row(2, missing=12)
    ok = inference.predict_one(partial)
    assert ok["status"] == "OK" and partial["features"][12] is None
    assert "feature 12" in ok["missing_information"]


def test_batch_matches_single():
    model, rows = _predictor()
    inference = delivery.ConstitutionInference(model, {})
    batch = inference.predict_batch([rows[0], None, rows[1]])
    single = [inference.predict_one(r) for r in (rows[0], None, rows[1])]
    assert [b["status"] for b in batch] == [s["status"] for s in single]
    assert [len(b["scores"]) for b in batch] == [len(s["scores"]) for s in single]


def test_verify_champion_source_rejects_drift(monkeypatch):
    source = (Path(__file__).resolve().parents[2] / delivery.CANDIDATE_PATH).read_bytes()
    assert delivery.verify_champion_source(source) == delivery.CHAMPION_SOURCE_SHA256
    with pytest.raises(ValueError, match="sha256 mismatch"):
        delivery.verify_champion_source(source.replace(b"RIDGE = 100.0", b"RIDGE = 99.0"))
    tampered = source.replace(b"RIDGE = 100.0", b"RIDGE = 99.0")
    monkeypatch.setattr(delivery, "CHAMPION_SOURCE_SHA256",
                        __import__("hashlib").sha256(tampered).hexdigest())
    with pytest.raises(ValueError, match="missing frozen token"):
        delivery.verify_champion_source(tampered)


def test_artifact_save_load_roundtrip_and_tamper(tmp_path: Path):
    model, rows = _predictor()
    directory = tmp_path / "artifact"
    entry = delivery.save_artifact(model, directory, {"source": "SYNTHETIC"})
    assert sorted(entry["artifacts"]) == ["labels.json", "model.joblib"]
    loaded, reloaded_entry, labels = delivery.load_artifact(directory)
    assert reloaded_entry == entry
    assert tuple(labels["score_dims"]) == delivery.OBSERVED_DIMS
    assert tuple(labels["unobserved_dimensions"]) == delivery.UNOBSERVED_DIMS
    assert np.allclose(loaded.predict(rows[:5]), model.predict(rows[:5]), atol=1e-12)
    tampered = tmp_path / "tampered"
    import shutil
    shutil.copytree(directory, tampered)
    model_path = tampered / "model.joblib"
    payload = bytearray(model_path.read_bytes())
    payload[0] ^= 0x01
    model_path.write_bytes(bytes(payload))
    with pytest.raises(ValueError, match="hash mismatch"):
        delivery.load_artifact(tampered)


def test_artifact_rejects_wrong_schema_and_taxonomy(tmp_path: Path):
    model, _ = _predictor()
    directory = tmp_path / "artifact"
    delivery.save_artifact(model, directory, {})
    entry_path = directory / "registry_entry.json"
    entry = json.loads(entry_path.read_text(encoding="utf-8"))
    for key, value in (("schema_version", "other"), ("taxonomy_version", "other")):
        broken = dict(entry)
        broken[key] = value
        entry_path.write_text(json.dumps(broken), encoding="utf-8")
        with pytest.raises(ValueError):
            delivery.load_artifact(directory)
    labels_path = directory / "labels.json"
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    labels["score_dims"] = ["balanced"] + list(labels["score_dims"])[:-1]
    entry_path.write_text(json.dumps(entry), encoding="utf-8")
    labels_path.write_text(json.dumps(labels), encoding="utf-8")
    # labels.json 哈希也随之改变，加载必须拒绝
    with pytest.raises(ValueError):
        delivery.load_artifact(directory)


def test_validate_report_schema():
    complete = {key: {} for key in delivery.REQUIRED_REPORT_KEYS}
    complete["schema_version"] = delivery.REPORT_SCHEMA
    delivery.validate_report_schema(complete)
    del complete["per_dimension"]
    with pytest.raises(ValueError, match="missing keys"):
        delivery.validate_report_schema(complete)
    complete["per_dimension"] = {}
    complete["schema_version"] = "wrong"
    with pytest.raises(ValueError, match="unsupported"):
        delivery.validate_report_schema(complete)


def test_deliver_never_reads_holdout_or_fabricates_balanced():
    source = (Path(__file__).resolve().parents[2] /
              "src/research/constitution_delivery.py").read_text(encoding="utf-8")
    assert "test.json" not in source
    assert "holdout" in source
    assert "ABSTAIN" in source
    assert "UNOBSERVED_ABSTAIN" not in source or "balanced" in source
    verify_body = source.split("def verify_only(", 1)[1].split("def main(", 1)[0]
    assert "ConstitutionPredictor(" not in verify_body
    assert "joblib.dump" not in verify_body
    assert "retraining_performed\": False" in verify_body


def test_champion_constants_are_frozen():
    assert len(delivery.CHAMPION_COMMIT) == 40
    assert delivery.CHAMPION_COMMIT.startswith("16c26582")
    assert delivery.CHAMPION_MASKED_MAE == 14.14368356353204
    assert delivery.N_TARGET_DIMS == 8
    assert delivery.UNOBSERVED_DIMS == ("balanced",)
    assert delivery.FEATURE_KEEP == slice(0, 9)
    assert delivery.CANDIDATE_PATH == "src/research/constitution_train.py"


def test_go_gates_evaluate_independently():
    from pathlib import Path as P
    registry = P(__file__).resolve().parents[2] / "docs/evidence/registry.json"
    if not registry.is_file():
        pytest.skip("evidence registry not present")
    from evidence.registry import EvidenceRegistry, evaluate_gates
    gates = evaluate_gates(EvidenceRegistry.load(registry), "tcm_constitution")
    assert gates["GO_RESEARCH_CLAIMS"].state == "NOT_GRANTED"
    assert gates["GO_ENGINEERING"].state == "GRANTED"
