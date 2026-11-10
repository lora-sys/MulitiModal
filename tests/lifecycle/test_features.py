"""Tests for lifecycle features and target registry (#107).

Mutation-sensitive validation of:
- FeatureSpec / TargetSpec field contracts, legal ranges, and unavailable reasons.
- BPSI and EFRI hand-computable golden rules.
- Structural head compatibility and regression on negative delta targets.
- Fold-internal median imputation (held-out subjects never affect train medians).
- Temporal leakage prevention and same-record pre/post spoof detection.
- Explicit masks: missing is not silently 0, imputed is valid but not observed.
- Feature version sha256 sensitivity to any field change.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from core.specs import make_physio_input_spec
from lifecycle.contracts import (
    APPROVAL_APPROVED,
    APPROVAL_UNVERIFIED,
    EVIDENCE_SYNTHETIC,
    ContractError,
    FeatureSpec,
    FieldSpec,
    TargetSpec,
)
from lifecycle.features import (
    BODY_DIM,
    DEFAULT_TARGET_SPECS,
    FROZEN_PRE_ONLY_FEATURE_SPEC,
    PHYSIO_DIM,
    FeatureRegistry,
    TargetCompatibilityError,
    TargetHead,
    TargetRegistry,
    TemporalLeakageError,
    assert_no_temporal_leakage,
    bpsi,
    build_feature_rows,
    check_target_head_compatibility,
    efri,
    head_for,
    unavailable_target,
    verify_against_core_spec,
)

pytestmark = pytest.mark.synthetic


def test_every_field_has_version_source_timepoint_and_legal_range() -> None:
    spec = FROZEN_PRE_ONLY_FEATURE_SPEC
    assert spec.version == "physio-features-pre-only-v1"
    assert len(spec.fields) == BODY_DIM + PHYSIO_DIM

    for field in spec.fields:
        assert field.name
        assert field.unit
        assert field.dtype == "float"
        assert field.timepoint in ("pre", "static")
        assert field.source in ("body_pre", "physio_pre")
        assert field.direction in ("neutral", "descriptive")
        assert field.fit_scope == "train_fold"
        assert field.mask == "observed"
        assert field.legal_range is not None
        low, high = field.legal_range
        assert low < high


def test_target_spec_carries_version_timepoint_and_unavailable_reason() -> None:
    for target in DEFAULT_TARGET_SPECS:
        assert target.name
        assert target.version
        assert target.timepoint in ("pre", "during", "post", "derived")
        assert target.approval == APPROVAL_UNVERIFIED
        assert target.unavailable_reason
        assert len(target.unavailable_reason.strip()) > 0
        if target.legal_range is not None:
            low, high = target.legal_range
            assert low < high


def test_unapproved_target_requires_explicit_unavailable_reason() -> None:
    with pytest.raises(ContractError, match="unavailable_reason"):
        unavailable_target(
            name="fake_fatigue_label",
            reason="",
        )


def test_verify_against_core_spec_enforces_body_and_physio_dims() -> None:
    core_spec = make_physio_input_spec(BODY_DIM, PHYSIO_DIM)
    assert verify_against_core_spec(FROZEN_PRE_ONLY_FEATURE_SPEC, core_spec) is True

    # Mutated spec with only 11 body fields must fail
    bad_spec = FeatureSpec(
        version="bad-dim-v1",
        fields=FROZEN_PRE_ONLY_FEATURE_SPEC.fields[1:],
        transform_id="t1",
    )
    with pytest.raises(ContractError, match="Expected exactly 12 body fields"):
        verify_against_core_spec(bad_spec, core_spec)


def test_verify_against_core_spec_rejects_during_and_post_fields() -> None:
    leaked_field = FieldSpec(
        "eeg_during", "microvolt", "float", "during", "none", "during", "neutral", "none",
        "observed", (0.0, 100.0),
    )
    leaked_spec = FeatureSpec(
        version="leaked-v1",
        fields=FROZEN_PRE_ONLY_FEATURE_SPEC.fields[:18] + (leaked_field,),
        transform_id="t1",
    )
    with pytest.raises(ContractError, match="Pre-only contract violated"):
        verify_against_core_spec(leaked_spec)


def test_bpsi_hand_computed_golden_values() -> None:
    # 1. Equal weighting: (10 + 20 + 30) / 3 = 20.0
    values_dict = {"f1": 10.0, "f2": 20.0, "f3": 30.0}
    assert bpsi(values_dict) == pytest.approx(20.0)

    # Sequence with equal weighting: (12 + 24) / 2 = 18.0
    assert bpsi([12.0, 24.0]) == pytest.approx(18.0)

    # 2. Explicit weights: 2.0 * 0.4 + 5.0 * 0.6 = 0.8 + 3.0 = 3.8
    assert bpsi({"a": 2.0, "b": 5.0}, {"a": 0.4, "b": 0.6}) == pytest.approx(3.8)

    # 3. Rescaling to 0-100: 100 * (3.8 - 0) / 10 = 38.0
    scaled = bpsi({"a": 2.0, "b": 5.0}, {"a": 0.4, "b": 0.6}, rescale_range=(0.0, 10.0))
    assert scaled == pytest.approx(38.0)

    # Empty inputs must fail
    with pytest.raises(ValueError, match="non-empty"):
        bpsi([])


def test_efri_hand_computed_golden_values() -> None:
    eeg = [2.0, 4.0]  # mean = 3.0
    post = [10.0, 20.0]  # mean = 15.0

    # 1. EEG-only variant
    assert efri(eeg_responses=eeg, variant="eeg_only") == pytest.approx(3.0)

    # 2. Post-only variant
    assert efri(post_responses=post, variant="post_only") == pytest.approx(15.0)

    # 3. Joint variant with default equal block weights (0.5, 0.5): 0.5 * 3.0 + 0.5 * 15.0 = 9.0
    assert efri(eeg_responses=eeg, post_responses=post, variant="joint") == pytest.approx(9.0)

    # 4. Joint variant with custom block weights: 0.2 * 3.0 + 0.8 * 15.0 = 0.6 + 12.0 = 12.6
    val_custom = efri(
        eeg_responses=eeg,
        post_responses=post,
        variant="joint",
        joint_block_weights=(0.2, 0.8),
    )
    assert val_custom == pytest.approx(12.6)

    # 5. Missing required response blocks must fail loudly
    with pytest.raises(ValueError, match="eeg_responses"):
        efri(post_responses=post, variant="eeg_only")
    with pytest.raises(ValueError, match="post_responses"):
        efri(eeg_responses=eeg, variant="post_only")
    with pytest.raises(ValueError, match="Both eeg_responses and post_responses"):
        efri(eeg_responses=eeg, variant="joint")


def test_bpsi_and_efri_are_unapproved_and_never_claim_clinical_truth() -> None:
    registry = TargetRegistry()
    bpsi_spec = registry.get("bpsi")
    efri_joint = registry.get("efri_joint")

    assert bpsi_spec.approval != APPROVAL_APPROVED
    assert bpsi_spec.approval == APPROVAL_UNVERIFIED
    assert "clinically validated" in bpsi_spec.unavailable_reason

    assert efri_joint.approval != APPROVAL_APPROVED
    assert efri_joint.approval == APPROVAL_UNVERIFIED
    assert "clinically validated" in efri_joint.unavailable_reason


def test_structural_head_mismatch_raises_on_negative_delta() -> None:
    shared_realtime_head = TargetHead(
        name="realtime_head_0_100",
        kind="clamped_100",
        legal_range=(0.0, 100.0),
        description="Shared realtime model head clamped to [0, 100]",
    )
    delta_target = TargetSpec(
        name="delta_vendor_fatigue",
        version="delta-v1",
        head="linear_delta",
        legal_range=(-18.0, 0.0),
        unit="delta_points",
        timepoint="derived",
        provenance="vendor post-pre difference",
        approval=APPROVAL_UNVERIFIED,
        unavailable_reason="unverified vendor delta",
    )

    # Structural mismatch: [-18.0, 0.0] cannot fit in [0, 100] clamped output head
    with pytest.raises(TargetCompatibilityError, match="Negative target values cannot be produced"):
        check_target_head_compatibility(delta_target, shared_realtime_head)

    # Correct adapter head covers negative range
    adapted_head = head_for(delta_target)
    assert adapted_head.legal_range is not None
    assert adapted_head.legal_range[0] <= delta_target.legal_range[0]
    assert adapted_head.legal_range[1] >= delta_target.legal_range[1]
    assert check_target_head_compatibility(delta_target, adapted_head) is True


def test_post_state_adapter_head_compatibility() -> None:
    post_state_target = TargetSpec(
        name="post_state",
        version="v1",
        head="unit_interval",
        legal_range=(0.25, 0.48),
        unit="ratio",
        timepoint="post",
        provenance="post-massage state ratio",
        approval=APPROVAL_UNVERIFIED,
        unavailable_reason="unverified ratio",
    )
    head = head_for(post_state_target)
    assert head.legal_range == (0.0, 1.0)
    assert check_target_head_compatibility(post_state_target, head) is True

    too_narrow_head = TargetHead("narrow", "sigmoid", legal_range=(0.0, 0.2))
    with pytest.raises(TargetCompatibilityError, match="max 0.48 is outside"):
        check_target_head_compatibility(post_state_target, too_narrow_head)


def test_fold_internal_imputation_isolates_held_out_subject() -> None:
    spec = FeatureSpec(
        version="f-test-v1",
        fields=(
            FieldSpec(
                "hr", "bpm", "float", "pre", "none", "physio_pre", "neutral", "train_fold",
                "observed", (30.0, 200.0),
            ),
        ),
        transform_id="t1",
    )

    # Training subjects: sub_A (hr=60), sub_B (hr=80) -> median must be 70.0
    # Held-out subject: sub_C
    base_events = [
        {"subject_ref": "sub_A", "session_id": "s1", "hr": 60.0, "cutoff": datetime.now(UTC)},
        {"subject_ref": "sub_B", "session_id": "s2", "hr": 80.0, "cutoff": datetime.now(UTC)},
        # Missing hr for sub_A in session 3 -> should be imputed to train median
        {"subject_ref": "sub_A", "session_id": "s3", "hr": None, "cutoff": datetime.now(UTC)},
    ]

    # Run 1: sub_C has hr = 1000.0
    events_run1 = list(base_events) + [
        {"subject_ref": "sub_C", "session_id": "s4", "hr": 1000.0, "cutoff": datetime.now(UTC)}
    ]
    rows_run1 = build_feature_rows(events_run1, spec, train_subjects={"sub_A", "sub_B"})

    # Run 2: sub_C has hr = -500.0
    events_run2 = list(base_events) + [
        {"subject_ref": "sub_C", "session_id": "s4", "hr": -500.0, "cutoff": datetime.now(UTC)}
    ]
    rows_run2 = build_feature_rows(events_run2, spec, train_subjects={"sub_A", "sub_B"})

    # Imputed value for sub_A session 3 must be identical (70.0) in both runs
    assert rows_run1[2].values[0] == pytest.approx(70.0)
    assert rows_run2[2].values[0] == pytest.approx(70.0)
    assert rows_run1[2].imputed_mask[0] is True
    assert rows_run1[2].imputation[0] == "train_median"


def test_temporal_leakage_and_same_record_spoof_rejection() -> None:
    spec = FROZEN_PRE_ONLY_FEATURE_SPEC

    # 1. During event leaked into pre spec
    leak_during = [
        {"subject_ref": "sub1", "phase": "during", "hr": 70.0, "cutoff": datetime.now(UTC)}
    ]
    with pytest.raises(TemporalLeakageError, match="cannot extract pre features from during event"):
        build_feature_rows(leak_during, spec, train_subjects={"sub1"})

    # 2. Post event leaked into pre spec
    leak_post = [
        {"subject_ref": "sub1", "phase": "post", "hr": 70.0, "cutoff": datetime.now(UTC)}
    ]
    with pytest.raises(TemporalLeakageError, match="cannot extract pre features from post event"):
        build_feature_rows(leak_post, spec, train_subjects={"sub1"})

    # 3. Same-record pre/post spoofing
    spoofed = [
        {
            "subject_ref": "sub1",
            "pre_record": "rec_001",
            "post_record": "rec_001",
            "cutoff": datetime.now(UTC),
        }
    ]
    with pytest.raises(TemporalLeakageError, match="same-record spoofing"):
        build_feature_rows(spoofed, spec, train_subjects={"sub1"})

    # 4. assert_no_temporal_leakage standalone checks
    with pytest.raises(TemporalLeakageError, match="same-record spoofing"):
        assert_no_temporal_leakage([{"pre_record": "r1", "post_record": "r1"}])

    with pytest.raises(TemporalLeakageError, match="phase 'post'"):
        assert_no_temporal_leakage([{"phase": "post"}])

    # Clean rows pass
    clean_events = [
        {"subject_ref": "sub1", "phase": "pre", "heart_rate": 70.0, "cutoff": datetime.now(UTC)}
    ]
    clean_rows = build_feature_rows(clean_events, spec, train_subjects={"sub1"})
    assert_no_temporal_leakage(clean_rows)


def test_masks_observed_valid_imputed_contract() -> None:
    spec = FeatureSpec(
        version="f-masks-v1",
        fields=(
            FieldSpec(
                "observed_feat", "bpm", "float", "pre", "none", "physio_pre", "neutral",
                "train_fold", "observed", (30.0, 200.0),
            ),
            FieldSpec(
                "imputed_feat", "score", "float", "pre", "none", "physio_pre", "neutral",
                "train_fold", "observed", (0.0, 100.0),
            ),
            FieldSpec(
                "unusable_feat", "level", "float", "pre", "none", "body_pre", "neutral",
                "none", "observed", (1.0, 10.0),
            ),
        ),
        transform_id="t1",
    )

    events = [
        # Training row supplying median for imputed_feat (50.0)
        {
            "subject_ref": "s1",
            "session_id": "sess1",
            "observed_feat": 72.0,
            "imputed_feat": 50.0,
            "unusable_feat": None,
            "cutoff": datetime.now(UTC),
        },
        # Row with one observed, one missing-imputed, one unusable (fit_scope=none, missing)
        {
            "subject_ref": "s2",
            "session_id": "sess2",
            "observed_feat": 80.0,
            "imputed_feat": None,
            "unusable_feat": None,
            "cutoff": datetime.now(UTC),
        },
    ]

    rows = build_feature_rows(events, spec, train_subjects={"s1"})
    row = rows[1]

    # observed_feat: observed=True, valid=True, imputed=False, value=80.0
    assert row.observed_mask[0] is True
    assert row.valid_mask[0] is True
    assert row.imputed_mask[0] is False
    assert row.values[0] == pytest.approx(80.0)
    assert row.imputation[0] == "none"

    # imputed_feat: observed=False, valid=True, imputed=True, value=50.0
    assert row.observed_mask[1] is False
    assert row.valid_mask[1] is True
    assert row.imputed_mask[1] is True
    assert row.values[1] == pytest.approx(50.0)
    assert row.imputation[1] == "train_median"

    # unusable_feat: observed=False, valid=False, imputed=False, value=0.0 (placeholder)
    assert row.observed_mask[2] is False
    assert row.valid_mask[2] is False
    assert row.imputed_mask[2] is False
    assert row.values[2] == 0.0
    assert row.imputation[2] == "none"


def test_feature_version_hash_changes_when_field_changes() -> None:
    f1 = FieldSpec(
        "a", "bpm", "float", "pre", "none", "p", "neutral", "none", "observed", (0.0, 10.0)
    )
    f2 = FieldSpec(
        "b", "bpm", "float", "pre", "none", "p", "neutral", "none", "observed", (0.0, 10.0)
    )

    spec1 = FeatureSpec("v1", (f1, f2), "t1", EVIDENCE_SYNTHETIC)

    # Change field unit
    f1_modified = FieldSpec(
        "a", "mmHg", "float", "pre", "none", "p", "neutral", "none", "observed", (0.0, 10.0)
    )
    spec2 = FeatureSpec("v1", (f1_modified, f2), "t1", EVIDENCE_SYNTHETIC)

    # Change legal_range
    f2_modified = FieldSpec(
        "b", "bpm", "float", "pre", "none", "p", "neutral", "none", "observed", (0.0, 20.0)
    )
    spec3 = FeatureSpec("v1", (f1, f2_modified), "t1", EVIDENCE_SYNTHETIC)

    # Change transform_id
    spec4 = FeatureSpec("v1", (f1, f2), "t2", EVIDENCE_SYNTHETIC)

    # Change version
    spec5 = FeatureSpec("v2", (f1, f2), "t1", EVIDENCE_SYNTHETIC)

    assert spec1.sha256 != spec2.sha256
    assert spec1.sha256 != spec3.sha256
    assert spec1.sha256 != spec4.sha256
    assert spec1.sha256 != spec5.sha256


def test_registries_and_data_dictionaries() -> None:
    f_reg = FeatureRegistry()
    assert "physio-features-pre-only-v1" in f_reg
    f_spec = f_reg["physio-features-pre-only-v1"]
    assert len(f_spec.fields) == 19

    f_dict = f_reg.data_dictionary("physio-features-pre-only-v1")
    assert len(f_dict) == 19
    for entry in f_dict:
        for key in (
            "name", "unit", "dtype", "timepoint", "window", "source",
            "direction", "fit_scope", "mask", "legal_range", "unavailable_reason",
        ):
            assert key in entry

    t_reg = TargetRegistry()
    assert "bpsi" in t_reg
    assert "efri_joint" in t_reg
    assert "delta_vendor_fatigue" in t_reg
    t_dict = t_reg.data_dictionary()
    assert len(t_dict) >= 5
    for entry in t_dict:
        for key in (
            "name", "version", "head", "legal_range", "unit",
            "timepoint", "provenance", "approval", "unavailable_reason",
        ):
            assert key in entry
