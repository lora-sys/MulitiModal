"""#97 scaler tests: mask-aware fitting, artifact integrity and contracts."""

from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import replace

import pytest
from physio_testkit import ts

from physio.preprocessing import (
    PHYSIO_WINDOW_SCALER_SCHEMA,
    PreprocessingError,
    WindowScaler,
)
from physio.windows import WINDOW_FEATURE_NAMES, WindowFeatures

pytestmark = pytest.mark.synthetic

NAMES = WINDOW_FEATURE_NAMES
HR_MEAN = NAMES.index("heart_rate.mean")
HR_COVERAGE = NAMES.index("heart_rate.coverage")
SPO2_MEAN = NAMES.index("spo2.mean")


def make_window(
    observed: dict[str, float] | None = None,
    *,
    masked_values: dict[str, float] | None = None,
) -> WindowFeatures:
    observed = observed or {}
    masked_values = masked_values or {}
    values = [0.0] * len(NAMES)
    mask = [False] * len(NAMES)
    for name, value in masked_values.items():
        values[NAMES.index(name)] = value
    for name, value in observed.items():
        index = NAMES.index(name)
        values[index] = value
        mask[index] = True
    return WindowFeatures(
        usable=any(mask),
        feature_names=NAMES,
        values=tuple(values),
        mask=tuple(mask),
        coverage=1.0,
        n_events=2,
        start=ts(0),
        end=ts(30),
        quality="ok",
        metrics_present=tuple(
            name.removesuffix(".mean") for name in observed if name.endswith(".mean")
        ),
    )


def training_windows() -> list[WindowFeatures]:
    return [
        make_window({"heart_rate.mean": 60.0, "heart_rate.coverage": 1.0}),
        make_window({"heart_rate.mean": 70.0, "heart_rate.coverage": 1.0}),
        make_window({"heart_rate.mean": 80.0, "heart_rate.coverage": 1.0}),
    ]


def test_fit_ignores_masked_placeholders():
    rows = training_windows()
    with_placeholders = [
        make_window(
            {"heart_rate.mean": 60.0, "heart_rate.coverage": 1.0},
            masked_values={"heart_rate.mean": 9999.0},
        ),
        make_window({"heart_rate.mean": 70.0, "heart_rate.coverage": 1.0}),
        make_window({"heart_rate.mean": 80.0, "heart_rate.coverage": 1.0}),
    ]
    scaler = WindowScaler.fit(rows)
    placeholder_scaler = WindowScaler.fit(with_placeholders)
    assert scaler.mean[HR_MEAN] == pytest.approx(70.0)
    assert placeholder_scaler.mean[HR_MEAN] == pytest.approx(70.0)
    assert scaler.to_dict() == placeholder_scaler.to_dict()


def test_transform_normalizes_observed_and_keeps_missing_masked():
    scaler = WindowScaler.fit(training_windows())
    scaled = scaler.transform(
        make_window(
            {"heart_rate.mean": 80.0},
            masked_values={"heart_rate.coverage": 123.0},
        )
    )
    assert scaled.feature_names == NAMES
    assert scaled.mask[HR_MEAN] is True
    assert scaled.values[HR_MEAN] == pytest.approx(10.0 / (200.0 / 3.0) ** 0.5)
    assert scaled.mask[HR_COVERAGE] is False
    assert scaled.values[HR_COVERAGE] == 0.0


def test_fit_normalizes_training_values_to_zero_mean_unit_std():
    scaler = WindowScaler.fit(training_windows())
    normalized = [
        scaler.transform(row).values[HR_MEAN] for row in training_windows()
    ]
    assert statistics.fmean(normalized) == pytest.approx(0.0, abs=1e-12)
    assert statistics.pstdev(normalized) == pytest.approx(1.0)


def test_fit_is_deterministic():
    first = WindowScaler.fit(training_windows())
    second = WindowScaler.fit(training_windows())
    assert first.to_dict() == second.to_dict()


def test_save_and_load_roundtrip(tmp_path):
    scaler = WindowScaler.fit(training_windows())
    path = tmp_path / "scaler.json"
    scaler.save(path)
    loaded = WindowScaler.load(path)
    assert loaded.to_dict() == scaler.to_dict()
    window = make_window({"heart_rate.mean": 80.0})
    assert loaded.transform(window).values == scaler.transform(window).values


def test_artifact_hash_tampering_is_rejected():
    artifact = WindowScaler.fit(training_windows()).to_dict()
    artifact["mean"][HR_MEAN] += 5.0
    with pytest.raises(PreprocessingError, match="sha256"):
        WindowScaler.from_dict(artifact)


def test_wrong_artifact_schema_is_rejected():
    artifact = WindowScaler.fit(training_windows()).to_dict()
    artifact["schema"] = "some-other-scaler"
    with pytest.raises(PreprocessingError, match="schema"):
        WindowScaler.from_dict(artifact)


def test_artifact_feature_version_is_validated():
    artifact = WindowScaler.fit(training_windows()).to_dict()
    artifact["feature_version"] = "physio-window-v0"
    payload = {key: value for key, value in artifact.items() if key != "sha256"}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    artifact["sha256"] = hashlib.sha256(encoded).hexdigest()
    with pytest.raises(PreprocessingError, match="feature_version"):
        WindowScaler.from_dict(artifact)


def test_transform_rejects_a_different_feature_contract():
    scaler = WindowScaler.fit(training_windows())
    window = make_window({"heart_rate.mean": 80.0})
    wrong = replace(window, feature_names=tuple(reversed(NAMES)))
    with pytest.raises(PreprocessingError, match="feature contract"):
        scaler.transform(wrong)


def test_unobserved_feature_never_gets_a_fabricated_normalization():
    scaler = WindowScaler.fit(training_windows())
    with pytest.raises(PreprocessingError, match="never observed"):
        scaler.transform(make_window({"spo2.mean": 97.0}))


def test_fit_requires_training_windows():
    with pytest.raises(PreprocessingError, match="without training windows"):
        WindowScaler.fit([])


def test_artifact_feature_names_are_checked_on_load():
    scaler = WindowScaler(
        feature_names=NAMES[:-1],
        mean=tuple(0.0 for _ in NAMES[:-1]),
        std=tuple(1.0 for _ in NAMES[:-1]),
        observed_counts=tuple(0 for _ in NAMES[:-1]),
    )
    with pytest.raises(PreprocessingError, match="feature contract"):
        WindowScaler.from_dict(scaler.to_dict())


def test_artifact_declares_the_expected_schema():
    artifact = WindowScaler.fit(training_windows()).to_dict()
    assert artifact["schema"] == PHYSIO_WINDOW_SCALER_SCHEMA
    assert artifact["feature_names"] == list(NAMES)
