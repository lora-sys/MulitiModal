"""Synthetic tests for Post-Pre deltas, y_pri guards and masked multi-target loss (#98)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from physio.metrics import route_measurement
from physio.programs import (
    ProgramCapability,
    ProgramCapabilityCatalog,
    ProgramContext,
    ProgramValidationError,
)
from physio.response import (
    DELTA_ACCEPTED,
    DELTA_INVALID_VALUE,
    DELTA_MISSING_POST,
    DELTA_MISSING_PRE,
    DELTA_NON_POSITIVE_INTERVAL,
    DELTA_SCHEMA_VERSION,
    DELTA_UNIT_MISMATCH,
    PRI_TARGET_KEY,
    PRI_TARGET_VERSION,
    RULE_REGISTRY,
    RULE_STATUS_APPROVED,
    RULE_STATUS_UNAPPROVED,
    DeltaMeasure,
    PriTarget,
    RuleSpec,
    build_delta_measures,
    build_response_sample,
    fit_only_on_train,
    masked_multi_target_loss,
    pri_target,
    validate_pri_claim,
)
from physio.schema import (
    PHASE_DURING,
    PHASE_POST,
    PHASE_PRE,
    QUALITY_STALE,
    SOURCE_DEVICE,
    MeasurementV1,
    PhysioEventV1,
)

pytestmark = pytest.mark.synthetic

BASE = datetime(2026, 11, 7, 10, 0, tzinfo=UTC)
CATALOG_VERSION = "synth-catalog-v1"
DEVICE = "chair-synth-01"


def ts(seconds: int = 0) -> datetime:
    return BASE + timedelta(seconds=seconds)


def event(phase, at, measurements, *, sequence=0, subject="p1"):
    return PhysioEventV1(
        session_id="s1",
        subject_ref=subject,
        device_ref=DEVICE,
        source=SOURCE_DEVICE,
        phase=phase,
        measured_at=ts(at),
        received_at=ts(at + 1),
        sequence=sequence,
        measurements=tuple(measurements),
    )


def hr(value, unit="bpm"):
    return route_measurement("heart_rate", value, unit)


def spo2(value):
    return route_measurement("spo2", value, "%")


def by_metric(deltas):
    return {delta.metric: delta for delta in deltas}


def capability(**overrides):
    values = {
        "program_id": "SYNTH-P01",
        "name": "Synthetic knead program",
        "intensity_min": 1,
        "intensity_max": 5,
        "duration_min_s": 300,
        "duration_max_s": 1200,
        "actions": ("knead",),
        "selectable": True,
        "device_compat": (DEVICE,),
        "reviewed": True,
        "catalog_version": CATALOG_VERSION,
    }
    values.update(overrides)
    return ProgramCapability(**values)


def catalog():
    return ProgramCapabilityCatalog(
        capabilities=(capability(),), catalog_version=CATALOG_VERSION, reviewed=True
    )


def program_context(**overrides):
    values = {
        "program_id": "SYNTH-P01",
        "intensity": 3,
        "duration_s": 600,
        "actions": ("knead",),
        "catalog_version": CATALOG_VERSION,
        "reviewed": True,
    }
    values.update(overrides)
    return ProgramContext(**values)


def test_delta_golden_case_is_plain_post_minus_pre():
    pre = event(PHASE_PRE, 0, [hr(70), spo2(97)])
    post = event(PHASE_POST, 600, [hr(78), spo2(96)])
    deltas = build_delta_measures(pre, post)
    assert [delta.metric for delta in deltas] == ["heart_rate", "spo2"]
    heart_rate = by_metric(deltas)["heart_rate"]
    assert heart_rate.valid
    assert heart_rate.status == DELTA_ACCEPTED
    assert heart_rate.delta == pytest.approx(8.0)
    assert heart_rate.unit == "bpm"
    assert heart_rate.pre_time == ts(0)
    assert heart_rate.post_time == ts(600)
    assert by_metric(deltas)["spo2"].delta == pytest.approx(-1.0)
    assert heart_rate.provenance["pre"]["source"] == SOURCE_DEVICE
    assert heart_rate.provenance["post"]["sequence"] == 0


def test_negative_delta_is_not_interpreted_as_improvement():
    pre = event(PHASE_PRE, 0, [hr(80)])
    post = event(PHASE_POST, 600, [hr(70)])
    delta = by_metric(build_delta_measures(pre, post))["heart_rate"]
    assert delta.valid and delta.delta == pytest.approx(-10.0)
    assert "efficacy meaning" in " ".join(delta.notes)
    assert "improved" not in str(delta.to_dict()).lower()


def test_missing_pre_and_missing_post_are_explicit():
    pre = event(PHASE_PRE, 0, [hr(70)])
    post = event(PHASE_POST, 600, [spo2(96)])
    deltas = by_metric(build_delta_measures(pre, post))
    assert deltas["heart_rate"].status == DELTA_MISSING_POST
    assert not deltas["heart_rate"].valid
    assert deltas["heart_rate"].delta is None
    assert deltas["spo2"].status == DELTA_MISSING_PRE
    assert not deltas["spo2"].valid
    assert deltas["spo2"].delta is None


def test_unit_mismatch_is_rejected():
    pre = event(
        PHASE_PRE,
        0,
        [
            MeasurementV1(
                metric="heart_rate",
                value=70,
                unit="bpm",
                data_type="float",
                valid=True,
                status="accepted",
            )
        ],
    )
    post = event(
        PHASE_POST,
        600,
        [
            MeasurementV1(
                metric="heart_rate",
                value=72,
                unit="1/min",
                data_type="float",
                valid=True,
                status="accepted",
            )
        ],
    )
    delta = by_metric(build_delta_measures(pre, post))["heart_rate"]
    assert delta.status == DELTA_UNIT_MISMATCH
    assert not delta.valid
    assert delta.delta is None


@pytest.mark.parametrize("post_at", [0, -10])
def test_non_positive_interval_is_rejected(post_at):
    pre = event(PHASE_PRE, 0, [hr(70)])
    post = event(PHASE_POST, post_at, [hr(75)])
    delta = by_metric(build_delta_measures(pre, post))["heart_rate"]
    assert delta.status == DELTA_NON_POSITIVE_INTERVAL
    assert not delta.valid
    assert delta.delta is None


def test_invalid_measurement_and_stale_event_are_rejected():
    pre = event(PHASE_PRE, 0, [hr(0)])
    post = event(PHASE_POST, 600, [hr(75)])
    invalid = by_metric(build_delta_measures(pre, post))["heart_rate"]
    assert invalid.status == DELTA_INVALID_VALUE
    assert not invalid.valid

    stale = event(PHASE_PRE, 0, [hr(70)]).with_quality(QUALITY_STALE)
    stale_delta = by_metric(build_delta_measures(stale, post))["heart_rate"]
    assert stale_delta.status == DELTA_INVALID_VALUE
    assert "event quality" in " ".join(stale_delta.notes)


def test_unknown_and_string_metrics_are_rejected():
    unknown = MeasurementV1(
        metric="vendor_magic",
        value=3.0,
        unit="index",
        data_type="float",
        valid=True,
        status="accepted",
    )
    pre = event(PHASE_PRE, 0, [unknown])
    post = event(PHASE_POST, 600, [unknown])
    assert by_metric(build_delta_measures(pre, post))["vendor_magic"].status == (
        DELTA_INVALID_VALUE
    )

    sex_pre = event(PHASE_PRE, 0, [route_measurement("sex", "female", None)])
    sex_post = event(PHASE_POST, 600, [route_measurement("sex", "female", None)])
    sex_delta = by_metric(build_delta_measures(sex_pre, sex_post))["sex"]
    assert sex_delta.status == DELTA_INVALID_VALUE
    assert sex_delta.delta is None


def test_duplicate_metric_is_rejected():
    pre = event(PHASE_PRE, 0, [hr(70), hr(71)])
    post = event(PHASE_POST, 600, [hr(75)])
    delta = by_metric(build_delta_measures(pre, post))["heart_rate"]
    assert delta.status == DELTA_INVALID_VALUE
    assert "duplicate" in " ".join(delta.notes)


def test_delta_requires_matching_phases_and_subject():
    pre = event(PHASE_PRE, 0, [hr(70)])
    post = event(PHASE_POST, 600, [hr(75)])
    with pytest.raises(ValueError, match="phase"):
        build_delta_measures(event(PHASE_DURING, 0, [hr(70)]), post)
    with pytest.raises(ValueError, match="phase"):
        build_delta_measures(pre, event(PHASE_DURING, 600, [hr(75)]))
    other_subject = event(PHASE_POST, 600, [hr(75)], subject="p2")
    with pytest.raises(ValueError, match="subject_ref"):
        build_delta_measures(pre, other_subject)


def test_pri_is_never_available_and_never_auto_filled():
    label = pri_target()
    assert label.value is None
    assert label.available is False
    assert label.score_version == PRI_TARGET_VERSION
    assert label.rule_status == RULE_STATUS_UNAPPROVED
    validate_pri_claim(label)

    valid_delta = by_metric(
        build_delta_measures(
            event(PHASE_PRE, 0, [hr(70)]), event(PHASE_POST, 600, [hr(78)])
        )
    )["heart_rate"]
    still_unavailable = pri_target((valid_delta,))
    assert still_unavailable.value is None
    assert still_unavailable.available is False


def test_pri_constructor_rejects_fabricated_availability():
    with pytest.raises(ValueError, match="approved rule"):
        PriTarget(value=12.0, available=True)
    with pytest.raises(ValueError, match="must carry a value"):
        PriTarget(value=None, available=True, rule_status=RULE_STATUS_APPROVED)
    with pytest.raises(ValueError, match="must not carry a value"):
        PriTarget(value=0.0, available=False)


def test_approved_pri_rule_without_an_implementation_still_abstains():
    approved = RuleSpec(
        name="PRI",
        version="pri-future-v1",
        source="synthetic approval for the guardrail test only",
        direction="descriptive synthetic direction",
        status=RULE_STATUS_APPROVED,
    )
    label = pri_target(registry={"PRI": approved})
    assert label.value is None
    assert label.available is False
    assert "no implemented scoring formula" in label.reason


def test_validate_pri_claim_rejects_bypassed_objects():
    available_without_value = PriTarget()
    object.__setattr__(available_without_value, "available", True)
    object.__setattr__(available_without_value, "rule_status", RULE_STATUS_APPROVED)
    with pytest.raises(ValueError, match="no value"):
        validate_pri_claim(available_without_value)

    unapproved_with_value = PriTarget()
    object.__setattr__(unapproved_with_value, "available", True)
    object.__setattr__(unapproved_with_value, "value", 9.0)
    with pytest.raises(ValueError, match="unapproved"):
        validate_pri_claim(unapproved_with_value)


def test_rule_registry_preserves_unapproved_bpsi_efri_pri():
    assert set(RULE_REGISTRY) == {"BPSI", "EFRI", "PRI"}
    for name, rule in RULE_REGISTRY.items():
        assert rule.name == name
        assert rule.status == RULE_STATUS_UNAPPROVED
        assert rule.version
        assert rule.source
        assert rule.direction
        assert rule.weights == ()
    assert RULE_REGISTRY["PRI"].version == PRI_TARGET_VERSION


def test_rule_spec_rejects_unknown_status_and_bad_weights():
    with pytest.raises(ValueError, match="unknown rule status"):
        RuleSpec(name="X", version="v1", source="s", direction="d", status="VALIDATED")
    with pytest.raises(ValueError, match="pair"):
        RuleSpec(name="X", version="v1", source="s", direction="d", weights=(("a",),))
    with pytest.raises(ValueError, match="finite"):
        RuleSpec(
            name="X",
            version="v1",
            source="s",
            direction="d",
            weights=(("a", float("nan")),),
        )


def test_response_sample_masks_available_deltas_and_keeps_pri_missing():
    sample = build_response_sample(
        event(PHASE_PRE, 0, [hr(70), spo2(97)]),
        event(PHASE_DURING, 300, [hr(74)]),
        event(PHASE_POST, 600, [hr(78), spo2(96)]),
        program_context(),
        catalog(),
    )
    assert sample.program.program_id == "SYNTH-P01"
    assert sample.delta_schema_version == DELTA_SCHEMA_VERSION
    assert sample.targets["y_delta_heart_rate"] == pytest.approx(8.0)
    assert sample.target_mask["y_delta_heart_rate"] is True
    assert sample.targets[PRI_TARGET_KEY] is None
    assert sample.target_mask[PRI_TARGET_KEY] is False
    assert set(sample.available_dimensions) == {"y_delta_heart_rate", "y_delta_spo2"}
    assert sample.missing_dimensions == (PRI_TARGET_KEY,)


def test_response_sample_without_available_target_cannot_train():
    sample = build_response_sample(
        event(PHASE_PRE, 0, [hr(0)]),
        None,
        event(PHASE_POST, 600, [hr(75)]),
        program_context(),
        catalog(),
    )
    assert sample.available_dimensions == ()
    assert sample.missing_dimensions == ("y_delta_heart_rate", PRI_TARGET_KEY)
    with pytest.raises(ValueError, match="training cannot start"):
        masked_multi_target_loss([1.0], [1.0], [sample.target_mask["y_delta_heart_rate"]])


def test_response_sample_validates_phase_identity_and_order():
    pre = event(PHASE_PRE, 0, [hr(70)])
    post = event(PHASE_POST, 600, [hr(75)])
    with pytest.raises(ValueError, match="phase"):
        build_response_sample(
            pre, event(PHASE_POST, 300, [hr(72)]), post, program_context(), catalog()
        )
    with pytest.raises(ValueError, match="between"):
        build_response_sample(
            pre, event(PHASE_DURING, 900, [hr(72)]), post, program_context(), catalog()
        )
    with pytest.raises(ValueError, match="session_id"):
        build_response_sample(
            pre,
            None,
            PhysioEventV1(
                session_id="other",
                subject_ref="p1",
                device_ref=DEVICE,
                source=SOURCE_DEVICE,
                phase=PHASE_POST,
                measured_at=ts(600),
                received_at=ts(601),
                sequence=1,
                measurements=(hr(75),),
            ),
            program_context(),
            catalog(),
        )


def test_response_sample_revalidates_program():
    pre = event(PHASE_PRE, 0, [hr(70)])
    post = event(PHASE_POST, 600, [hr(75)])
    with pytest.raises(ProgramValidationError, match="unknown program_id"):
        build_response_sample(
            pre, None, post, program_context(program_id="M01"), catalog()
        )
    with pytest.raises(ProgramValidationError, match="unreviewed"):
        build_response_sample(
            pre, None, post, program_context(reviewed=False), catalog()
        )
    with pytest.raises(ProgramValidationError, match="actions"):
        build_response_sample(
            pre, None, post, program_context(actions=("different",)), catalog()
        )
    with pytest.raises(ProgramValidationError, match="catalog is required"):
        build_response_sample(pre, None, post, program_context(), None)


def test_delta_measure_never_exposes_delta_when_invalid():
    with pytest.raises(ValueError, match="must not expose"):
        DeltaMeasure(
            metric="heart_rate",
            pre_value=70,
            post_value=75,
            delta=5.0,
            unit="bpm",
            valid=False,
            status=DELTA_MISSING_POST,
            pre_time=ts(0),
            post_time=ts(600),
        )
    with pytest.raises(ValueError, match="status"):
        DeltaMeasure(
            metric="heart_rate",
            pre_value=70,
            post_value=75,
            delta=5.0,
            unit="bpm",
            valid=True,
            status=DELTA_MISSING_POST,
            pre_time=ts(0),
            post_time=ts(600),
        )
    with pytest.raises(ValueError, match="post_value - pre_value"):
        DeltaMeasure(
            metric="heart_rate",
            pre_value=70,
            post_value=75,
            delta=6.0,
            unit="bpm",
            valid=True,
            status=DELTA_ACCEPTED,
            pre_time=ts(0),
            post_time=ts(600),
        )


def test_masked_loss_ignores_masked_dimensions():
    predictions = [1.0, 2.0, 3.0]
    targets = [1.0, 4.0, 100.0]
    mask = [True, True, False]
    assert masked_multi_target_loss(predictions, targets, mask) == pytest.approx(1.0)
    weighted = masked_multi_target_loss(predictions, targets, mask, weights=[1.0, 3.0, 1.0])
    assert weighted == pytest.approx(1.5)


def test_masked_loss_supports_batches_and_ignores_masked_nan():
    detections = np.array([[1.0, 2.0], [3.0, 4.0]])
    labels = np.array([[1.0, 9.0], [3.0, 7.0]])
    mask = np.array([[True, False], [True, True]])
    assert masked_multi_target_loss(detections, labels, mask) == pytest.approx(1.0)
    assert masked_multi_target_loss([1.0, 0.0], [1.0, np.nan], [True, False]) == 0.0


@pytest.mark.parametrize(
    "mask", [[False, False, False], np.zeros((2, 3), dtype=bool)]
)
def test_all_masked_or_empty_mask_raises(mask):
    size = np.asarray(mask).shape
    with pytest.raises(ValueError, match="training cannot start"):
        masked_multi_target_loss(np.zeros(size), np.zeros(size), mask)


def test_masked_loss_rejects_shape_weight_and_value_problems():
    with pytest.raises(ValueError, match="shape"):
        masked_multi_target_loss([1.0], [1.0, 2.0], [True, True])
    with pytest.raises(ValueError, match="weights"):
        masked_multi_target_loss([1.0, 2.0], [1.0, 2.0], [True, True], weights=[1.0])
    with pytest.raises(ValueError, match="non-negative"):
        masked_multi_target_loss([1.0], [1.0], [True], weights=[-1.0])
    with pytest.raises(ValueError, match="sum to a positive"):
        masked_multi_target_loss([1.0], [1.0], [True], weights=[0.0])
    with pytest.raises(ValueError, match="NaN/Inf"):
        masked_multi_target_loss([np.nan], [1.0], [True])
    with pytest.raises(ValueError, match="NaN/Inf"):
        masked_multi_target_loss([1.0], [np.nan], [True])
    with pytest.raises(ValueError, match="1-D vectors or 2-D"):
        masked_multi_target_loss(np.zeros((1, 1, 1)), np.zeros((1, 1, 1)), np.ones((1, 1, 1)))
    with pytest.raises(ValueError, match="no targets"):
        masked_multi_target_loss([], [], [])


def test_fit_only_on_train_returns_a_validated_mask():
    assert fit_only_on_train([0, 2], 4).tolist() == [True, False, True, False]
    with pytest.raises(ValueError, match="duplicates"):
        fit_only_on_train([0, 0], 2)
    with pytest.raises(ValueError, match="within"):
        fit_only_on_train([2], 2)
    with pytest.raises(ValueError, match="non-empty"):
        fit_only_on_train([], 2)
    with pytest.raises(ValueError, match="integers"):
        fit_only_on_train([0.5], 2)
