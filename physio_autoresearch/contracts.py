"""Frozen research contracts and declarative candidate validation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import yaml


class ResearchError(ValueError):
    """A rejected research operation with a machine-readable code."""

    def __init__(self, code: str, detail: str):
        self.code = code
        super().__init__(f"{code}: {detail}")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def future_feature(name: str) -> bool:
    tokens = name.lower().replace("-", "_").split("_")
    return bool(set(tokens) & {"post", "after", "outcome", "target", "efri"})


def validate_contract(contract: dict) -> dict:
    try:
        if contract["schema_version"] != 1 or contract["prediction_time"] != "pre_massage":
            raise ResearchError("BAD_CONTRACT", "unsupported schema or prediction time")
        if contract["evaluation"] != "leave_one_subject_out" or contract["status"] not in {"smoke", "development"}:
            raise ResearchError("BAD_CONTRACT", "only subject-level development/smoke evaluation is supported")
        if contract["metric"] != "subject_macro_normalized_mae":
            raise ResearchError("WRONG_METRIC", "the evaluator owns metric calculation")
        features, targets = contract["features"], contract["targets"]
        if not features or not targets:
            raise ResearchError("BAD_CONTRACT", "features and targets must be explicit")
        names = [contract["record_id"], contract["subject_id"]] + [v["name"] for v in features + targets]
        if len(set(names)) != len(names):
            raise ResearchError("SCHEMA_COLLISION", "column names must be unique")
        prediction_names = {f"prediction_{target['name']}" for target in targets}
        if prediction_names.intersection(names):
            raise ResearchError("SCHEMA_COLLISION", "input columns collide with generated prediction columns")
        for feature in features:
            if feature["phase"] != "pre" or future_feature(feature["name"]):
                raise ResearchError("FUTURE_FEATURE_LEAKAGE", feature["name"])
            if not feature.get("unit") or not feature.get("group"):
                raise ResearchError("BAD_CONTRACT", "feature units and groups are required")
        for target in targets:
            scale = target["scale"]
            if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
                raise ResearchError("BAD_CONTRACT", "target scale must be a fixed positive finite number")
            if target["phase"] != "post" or not target.get("unit"):
                raise ResearchError("BAD_CONTRACT", "targets must identify measured post outcomes")
        if contract["source"] not in {"synthetic", "public", "company"}:
            raise ResearchError("UNKNOWN_PROVENANCE", "explicit data source required")
        budget = contract["budget"]
        for key in ("max_experiments", "max_seconds", "candidate_timeout_seconds", "patience"):
            if type(budget[key]) is not int or budget[key] < 1:
                raise ResearchError("BAD_BUDGET", f"{key} must be a positive integer")
        if budget["max_experiments"] > 20:
            raise ResearchError("BAD_BUDGET", "V0 caps a data version at 20 attempts")
        if type(contract["seed"]) is not int or not 0 <= contract["seed"] < 2**31:
            raise ResearchError("BAD_CONTRACT", "invalid seed")
        minimum = contract["min_improvement"]
        if isinstance(minimum, bool) or not isinstance(minimum, (int, float)) or not math.isfinite(minimum) or minimum < 0:
            raise ResearchError("BAD_CONTRACT", "invalid promotion threshold")
    except (KeyError, TypeError) as exc:
        raise ResearchError("BAD_CONTRACT", str(exc)) from exc
    return contract


def load_contract(path: Path) -> dict:
    return validate_contract(yaml.safe_load(path.read_text(encoding="utf-8")))


PARAMETERS = {
    "naive": {},
    "elastic_net": {"alpha": (0.001, 100.0), "l1_ratio": (0.0, 1.0)},
    "random_forest": {"n_estimators": (10, 200), "max_depth": (1, 6), "min_samples_leaf": (1, 10)},
    "xgboost": {"n_estimators": (5, 100), "max_depth": (1, 3), "learning_rate": (0.01, 0.3), "reg_lambda": (0.1, 100.0)},
}
INTEGER_PARAMS = {"n_estimators", "max_depth", "min_samples_leaf"}


def validate_candidate(spec: dict, contract: dict) -> dict:
    if not isinstance(spec, dict) or set(spec) - {"model", "params", "features", "hypothesis"}:
        raise ResearchError("BAD_MODEL_CONFIG", "only declarative model, params, features, hypothesis are allowed")
    model = spec.get("model")
    if model not in PARAMETERS:
        raise ResearchError("BAD_MODEL_CONFIG", "model family is outside V0")
    params = spec.get("params", {})
    if not isinstance(params, dict):
        raise ResearchError("BAD_MODEL_CONFIG", "params must be an object")
    if "max_depth" in PARAMETERS[model]:
        # Library defaults (unbounded RF / depth-6 XGBoost) exceed the V0 caps.
        params = {"max_depth": PARAMETERS[model]["max_depth"][1], **params}
    for name, value in params.items():
        if name not in PARAMETERS[model]:
            raise ResearchError("BAD_MODEL_CONFIG", f"unsupported parameter: {name}")
        low, high = PARAMETERS[model][name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
            raise ResearchError("BAD_MODEL_CONFIG", f"parameter out of bounds: {name}")
        if name in INTEGER_PARAMS and type(value) is not int:
            raise ResearchError("BAD_MODEL_CONFIG", f"integer required: {name}")
    allowed = [f["name"] for f in contract["features"]]
    selected = spec.get("features", allowed)
    if not isinstance(selected, list) or not selected or any(not isinstance(v, str) for v in selected):
        raise ResearchError("BAD_MODEL_CONFIG", "feature list must be nonempty")
    if len(set(selected)) != len(selected):
        raise ResearchError("BAD_MODEL_CONFIG", "duplicate feature selection")
    for name in selected:
        if future_feature(name):
            raise ResearchError("FUTURE_FEATURE_LEAKAGE", name)
        if name not in allowed:
            raise ResearchError("UNKNOWN_FEATURE", name)
    hypothesis = spec.get("hypothesis", "fixed-search baseline")
    if not isinstance(hypothesis, str) or len(hypothesis) > 2000:
        raise ResearchError("BAD_MODEL_CONFIG", "invalid hypothesis")
    # Hypotheses are memory, not part of experiment identity; column order is fixed.
    return {"model": model, "params": dict(sorted(params.items())),
            "features": [name for name in allowed if name in selected], "hypothesis": hypothesis}


def candidate_id(spec: dict) -> str:
    return digest({k: v for k, v in spec.items() if k != "hypothesis"})
