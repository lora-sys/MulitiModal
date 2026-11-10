"""#108 train-ready engineering: stage gates, fold isolation, honesty."""

from __future__ import annotations

import json

import pytest

from lifecycle import training as tr
from lifecycle.contracts import (
    APPROVAL_APPROVED,
    APPROVAL_PENDING,
    EVIDENCE_SYNTHETIC,
    DatasetManifest,
    EvalSpec,
    FeatureSpec,
    FieldSpec,
    SplitManifest,
    TargetSpec,
    TrainConfig,
    canonical_json,
    digest,
)

pytestmark = pytest.mark.synthetic


def _features() -> FeatureSpec:
    return FeatureSpec(
        version="f-v1",
        fields=(
            FieldSpec(
                "a", "unitless", "float", "pre", "none", "synthetic", "input", "none", "observed"
            ),
            FieldSpec(
                "b", "unitless", "float", "pre", "none", "synthetic", "input", "none", "observed"
            ),
        ),
        transform_id="identity",
        evidence_level=EVIDENCE_SYNTHETIC,
    )


def _target() -> TargetSpec:
    return TargetSpec(
        name="synthetic_response",
        version="t-v1",
        head="linear_regression",
        legal_range=(0.0, 1.0),
        unit="index",
        timepoint="derived",
        provenance="synthetic declared rule",
        approval="UNVERIFIED",
        unavailable_reason="no real paired fatigue label; synthetic smoke only",
    )


def _dataset() -> DatasetManifest:
    return DatasetManifest(
        dataset_id="synth-ds",
        source_sha256=(digest({"synthetic": True}),),
        schema_version="lifecycle-batch-csv-v1",
        label_version="t-v1",
        feature_version="f-v1",
        qc_sha256=digest({"qc": "clean"}),
        n_rows=30,
        n_subjects=6,
        identity_trust="SHARED_ACCOUNT_UNKNOWN",
        evidence_level=EVIDENCE_SYNTHETIC,
        approval=APPROVAL_PENDING,
        frozen=False,
        created_at="2026-11-01T00:00:00+00:00",
    )


def _split() -> SplitManifest:
    return SplitManifest(
        version="s-v1",
        seed=7,
        n_folds=3,
        assignment_sha256=digest({"fold": 3}),
        trainable_subjects=tuple(f"subject:{i}" for i in range(6)),
        held_out=(),
    )


def _eval() -> EvalSpec:
    return EvalSpec(version="e-v1", metric="mae", min_samples=30, min_subjects=2)


def _config(**kwargs) -> TrainConfig:
    base = dict(
        seed=7, model_kind="ridge", epochs=1, evidence_level=EVIDENCE_SYNTHETIC,
        commit="test-commit", environment="local-cpu", license="SMOKE_ONLY",
        approval=APPROVAL_PENDING,
    )
    base.update(kwargs)
    return TrainConfig(**base)


def test_prepare_binds_every_immutable_reference():
    run = tr.prepare(
        dataset=_dataset(), split=_split(), features=_features(), target=_target(),
        evaluation=_eval(), config=_config(),
    )
    assert run.config_sha256 == _config().sha256
    assert run.dataset_sha256 == _dataset().sha256
    assert run.feature_sha256 == _features().sha256
    assert run.target_sha256 == _target().sha256
    assert len(run.sha256) == 64


def test_prepare_refuses_unknown_model_kind():
    with pytest.raises(tr.TrainError, match="unknown model_kind"):
        tr.prepare(
            dataset=_dataset(), split=_split(), features=_features(), target=_target(),
            evaluation=_eval(), config=_config(model_kind="invented"),
        )


def test_synthetic_dataset_is_deterministic():
    rows, targets, subjects = tr.synthetic_dataset(seed=11)
    rows2, targets2, subjects2 = tr.synthetic_dataset(seed=11)
    assert rows == rows2 and targets == targets2 and subjects == subjects2
    assert tr.synthetic_dataset(seed=12)[1] != targets


def test_subject_folds_never_split_a_subject():
    _, _, subjects = tr.synthetic_dataset(n_subjects=6, seed=3)
    for train_idx, test_idx in tr.subject_folds(subjects, n_folds=3, seed=3):
        train_subjects = {subjects[i] for i in train_idx}
        test_subjects = {subjects[i] for i in test_idx}
        assert not train_subjects & test_subjects


def test_subject_folds_refuse_more_folds_than_subjects():
    with pytest.raises(tr.TrainError, match="more folds"):
        tr.subject_folds(["subject:0"], n_folds=3)


def test_smoke_run_covers_all_five_stages():
    report = tr.run_synthetic_smoke(
        features=_features(), target=_target(), dataset=_dataset(), split=_split(),
        evaluation=_eval(), config=_config(),
    )
    assert [s.stage for s in report.stages] == [
        "prepare", "train", "validate_model", "compare", "package"
    ]
    assert all(s.passed for s in report.stages)


def test_smoke_run_never_promotes_research_gates():
    report = tr.run_synthetic_smoke(
        features=_features(), target=_target(), dataset=_dataset(), split=_split(),
        evaluation=_eval(), config=_config(),
    )
    assert report.gates["GO_ENGINEERING"] == "GRANTED"
    assert report.gates["GO_REAL_DATA_TRAINING"] == "NOT_GRANTED"
    assert report.gates["GO_RESEARCH_CLAIMS"] == "NOT_GRANTED"
    assert report.evaluation_status == "INSUFFICIENT_EVIDENCE"
    assert report.status == "ABSTAIN"
    assert report.decision == "REVERT"


def test_smoke_run_repeatable_and_metrics_are_local():
    first = tr.run_synthetic_smoke(
        features=_features(), target=_target(), dataset=_dataset(), split=_split(),
        evaluation=_eval(), config=_config(),
    )
    second = tr.run_synthetic_smoke(
        features=_features(), target=_target(), dataset=_dataset(), split=_split(),
        evaluation=_eval(), config=_config(),
    )
    assert first.stages[1].metrics == second.stages[1].metrics
    assert "Synthetic engineering smoke only" in first.notes[0]
    assert "not a research claim" in first.stages[3].reason


def test_smoke_config_cannot_claim_approved_real_training():
    with pytest.raises(tr.TrainError, match="SMOKE_ONLY config"):
        tr.run_synthetic_smoke(
            features=_features(), target=_target(), dataset=_dataset(), split=_split(),
            evaluation=_eval(),
            config=_config(evidence_level="SMOKE_ONLY", approval=APPROVAL_APPROVED),
        )


def test_artifact_from_smoke_never_validated():
    report = tr.run_synthetic_smoke(
        features=_features(), target=_target(), dataset=_dataset(), split=_split(),
        evaluation=_eval(), config=_config(),
    )
    assert "NOT_VALIDATED" in report.notes[1]
    artifact = report.notes[1]
    assert "never promoted" in artifact


def test_report_is_canonical_json_and_writable(tmp_path):
    report = tr.run_synthetic_smoke(
        features=_features(), target=_target(), dataset=_dataset(), split=_split(),
        evaluation=_eval(), config=_config(),
    )
    payload = json.loads(canonical_json(report.to_dict()))
    assert payload["schema"] == tr.TRAIN_VERSION
    assert payload["decision"] == "REVERT"
    path = tr.write_report(report, tmp_path / "runs")
    assert path.is_file()
    assert json.loads(path.read_text())["evaluation_status"] == "INSUFFICIENT_EVIDENCE"


def test_crash_record_is_durable_and_reasoned():
    record = tr.crash_record("torch unavailable", run_id="r1")
    assert record["decision"] == "CRASH"
    assert record["reason"] == "torch unavailable"
    assert len(record["sha256"]) == 64
    assert tr.crash_record("torch unavailable", run_id="r1") == record


def test_dataset_manifest_hard_gate_blocks_real_training():
    dataset = _dataset()
    ok, reason = dataset.trainable()
    assert ok is False
    assert reason
    assert "approval" in reason or "identity" in reason
