"""#112 fast contract tests: pure stdlib, no file I/O, no network, no torch.

These are the first CI gate. They deliberately exercise only immutable contract
logic, so the same assertions hold on every supported Python and in a clean
environment with none of the optional dependencies installed.
"""

from __future__ import annotations

import pytest

from lifecycle.contracts import (
    APPROVAL_APPROVED,
    APPROVAL_UNVERIFIED,
    CAP_CONNECTED,
    CAP_NOT_CONNECTED,
    CONTRACTS_VERSION,
    EVIDENCE_LOCAL,
    EVIDENCE_SMOKE,
    EVIDENCE_SYNTHETIC,
    GATE_GRANTED,
    GATE_NOT_GRANTED,
    GATE_REAL_DATA_TRAINING,
    GATE_RESEARCH_CLAIMS,
    SOURCE_DEVICE,
    SOURCE_OFFLINE,
    STATUS_ABSTAIN,
    STATUS_OK,
    TRUST_SHARED,
    TRUST_TRUSTED,
    ArtifactManifest,
    ContractError,
    DatasetManifest,
    Envelope,
    EvalSpec,
    FeatureSpec,
    FieldSpec,
    PipelineState,
    QCReport,
    SourceManifest,
    SplitManifest,
    StageEvidence,
    TargetSpec,
    TrainConfig,
    VersionRef,
    canonical_json,
    digest,
    gate_bundle,
)

pytestmark = [pytest.mark.fast, pytest.mark.synthetic]


def _sha(value: str) -> str:
    return digest({"value": value})


def test_contracts_version_is_declared():
    assert CONTRACTS_VERSION == "ml-lifecycle-contracts-v1"


def test_canonical_json_is_stable_regardless_of_key_order():
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})
    assert digest({"b": 1, "a": 2}) == digest({"a": 2, "b": 1})


def test_digest_is_64_hex():
    value = digest({"x": 1})
    assert len(value) == 64
    assert set(value) <= set("0123456789abcdef")


def test_version_ref_rejects_bad_hash():
    with pytest.raises(ContractError, match="sha256"):
        VersionRef("name", "v1", "not-a-hash")


def test_source_manifest_refuses_fake_connected_real_device():
    with pytest.raises(ContractError, match="vendor conformance record"):
        SourceManifest(
            source_id="dev1", kind=SOURCE_DEVICE, uri="usb://0",
            sha256=_sha("a"), license="PRIVATE", provenance="vendor",
            identity_trust=TRUST_TRUSTED, capability=CAP_CONNECTED,
            schema_version="v1",
        )
    offline = SourceManifest(
        source_id="b1", kind=SOURCE_OFFLINE, uri="b.csv",
        sha256=_sha("a"), license="PRIVATE", provenance="import",
        identity_trust=TRUST_SHARED, capability=CAP_NOT_CONNECTED,
        schema_version="v1",
    )
    assert offline.id != ""


def test_feature_spec_requires_unique_fields():
    field = FieldSpec("a", "u", "float", "pre", "none", "s", "in", "none", "observed")
    with pytest.raises(ContractError, match="unique"):
        FeatureSpec("v1", (field, field), "identity")


def test_unapproved_target_must_state_unavailable_reason():
    with pytest.raises(ContractError, match="unavailable"):
        TargetSpec(
            "t", "v1", "linear", (0.0, 1.0), "index", "derived", "synthetic",
            APPROVAL_UNVERIFIED, unavailable_reason="",
        )
    ok = TargetSpec(
        "t", "v1", "linear", (0.0, 1.0), "index", "derived", "synthetic",
        APPROVAL_UNVERIFIED, unavailable_reason="no real label",
    )
    assert ok.approval == APPROVAL_UNVERIFIED


def test_qc_report_rejects_silent_skips():
    with pytest.raises(ContractError, match="reason"):
        QCReport(3, 1, 2, 0, 0, rejections=({"key": "k"},))
    with pytest.raises(ContractError, match="equal"):
        QCReport(3, 1, 1, 0, 0)


def test_pipeline_cache_key_depends_on_every_declared_input():
    base = dict(
        run_id="r", stage="ingest", cursor="c", processed_keys=(),
        dataset_sha256=_sha("d"), schema_version="s", code_version="code",
        config_sha256=_sha("cfg"), rules_sha256=_sha("rules"),
        status="RUNNING", started_at="t", updated_at="t",
    )
    state = PipelineState(**base)
    assert state.cache_key == PipelineState(**base).cache_key
    assert state.cache_key != PipelineState(**{**base, "code_version": "v2"}).cache_key
    assert state.cache_key != PipelineState(**{**base, "config_sha256": _sha("c2")}).cache_key
    assert state.cache_key != PipelineState(**{**base, "rules_sha256": _sha("r2")}).cache_key
    assert state.cache_key != PipelineState(**{**base, "stage": "snapshot"}).cache_key


def test_dataset_manifest_trainable_gate_is_hard():
    base = dict(
        dataset_id="d", source_sha256=(_sha("s"),), schema_version="s",
        label_version="l", feature_version="f", qc_sha256=_sha("q"),
        n_rows=30, n_subjects=6, identity_trust=TRUST_TRUSTED,
        evidence_level=EVIDENCE_SYNTHETIC, approval=APPROVAL_APPROVED,
        frozen=True, created_at="t",
    )
    assert DatasetManifest(**base).trainable() == (True, "admitted")
    assert DatasetManifest(**{**base, "identity_trust": TRUST_SHARED}).trainable()[0] is False
    # An unapproved dataset cannot even be declared frozen; construct it un-frozen.
    assert DatasetManifest(
        **{**base, "approval": APPROVAL_UNVERIFIED, "frozen": False}
    ).trainable()[0] is False
    assert DatasetManifest(**{**base, "frozen": False}).trainable()[0] is False
    assert DatasetManifest(**{**base, "evidence_level": EVIDENCE_SMOKE}).trainable()[0] is False


def test_frozen_dataset_must_be_approved():
    with pytest.raises(ContractError, match="approved"):
        DatasetManifest(
            dataset_id="d", source_sha256=(_sha("s"),), schema_version="s",
            label_version="l", feature_version="f", qc_sha256=_sha("q"),
            n_rows=1, n_subjects=1, identity_trust=TRUST_TRUSTED,
            evidence_level=EVIDENCE_SYNTHETIC, approval=APPROVAL_UNVERIFIED,
            frozen=True, created_at="t",
        )


def test_split_manifest_refuses_subject_leakage():
    with pytest.raises(ContractError, match="leakage"):
        SplitManifest("v1", 7, 3, _sha("a"), ("P1",), ("P1",))


def test_eval_spec_has_positive_minimums():
    with pytest.raises(ContractError, match="positive"):
        EvalSpec("v1", "mae", min_samples=0, min_subjects=2)


def test_train_config_rejects_unknown_evidence():
    with pytest.raises(ContractError, match="evidence_level"):
        TrainConfig(
            seed=1, model_kind="ridge", epochs=1, evidence_level="INVENTED",
            commit="c", environment="e", license="l", approval=APPROVAL_UNVERIFIED,
        )


def test_artifact_manifest_refuses_validated_smoke():
    base = dict(
        artifact_id="a", model_sha256=_sha("m"), feature_sha256=_sha("f"),
        target_sha256=_sha("t"), scaler_sha256=_sha("s"), dataset_sha256=_sha("d"),
        split_sha256=_sha("sp"), eval_sha256=_sha("e"), commit="c",
        dependencies=(), license="l", approval=APPROVAL_APPROVED,
        validation_status="VALIDATED", evidence_level=EVIDENCE_SMOKE,
    )
    with pytest.raises(ContractError, match="cannot be VALIDATED"):
        ArtifactManifest(**base)
    # VALIDATED requires non-synthetic evidence AND approval; LOCAL is allowed.
    ok = ArtifactManifest(**{**base, "evidence_level": EVIDENCE_LOCAL})
    assert ok.validation_status == "VALIDATED"


def test_validated_artifact_requires_approval():
    with pytest.raises(ContractError, match="APPROVED"):
        ArtifactManifest(
            artifact_id="a", model_sha256=_sha("m"), feature_sha256=_sha("f"),
            target_sha256=_sha("t"), scaler_sha256=_sha("s"), dataset_sha256=_sha("d"),
            split_sha256=_sha("sp"), eval_sha256=_sha("e"), commit="c",
            dependencies=(), license="l", approval=APPROVAL_UNVERIFIED,
            validation_status="VALIDATED", evidence_level=EVIDENCE_LOCAL,
        )


def test_envelope_refuses_smoke_model_answering_product_modes():
    base = dict(
        schema_version="v1", request_id="r", session_id="s", trace_id="t",
        event_time="e", feature_version="f", model_version="m", target_version="t",
        quality="ok", provenance="p", license="l", evidence_level=EVIDENCE_SMOKE,
        display_gate="g",
    )
    Envelope(mode="A_pre_only", status=STATUS_OK, **base)
    for mode in ("B_realtime", "C_tcm"):
        with pytest.raises(ContractError, match="only answer research mode A"):
            Envelope(mode=mode, status=STATUS_OK, **base)
    # A non-OK status is always allowed, regardless of mode/evidence.
    assert Envelope(mode="B_realtime", status=STATUS_ABSTAIN, **base).status == STATUS_ABSTAIN


def test_stage_evidence_rejects_unknown_stage():
    with pytest.raises(ContractError, match="unknown stage"):
        StageEvidence("invented_stage", True, EVIDENCE_SYNTHETIC, GATE_GRANTED, "ok")


def test_three_gates_are_independent():
    gates = gate_bundle(True)
    assert gates["GO_ENGINEERING"] == GATE_GRANTED
    assert gates[GATE_REAL_DATA_TRAINING] == GATE_NOT_GRANTED
    assert gates[GATE_RESEARCH_CLAIMS] == GATE_NOT_GRANTED
    assert gate_bundle(False)["GO_ENGINEERING"] == GATE_NOT_GRANTED
