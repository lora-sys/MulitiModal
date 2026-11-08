"""#95/#98 dataset discipline: masks, fold imputation, subject splits and labels."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest
from physio_testkit import event, measurement

from physio.dataset import (
    ADMISSION_IMPUTED_ENDPOINT,
    ADMISSION_INTERVAL_TOO_LONG,
    ADMISSION_SAME_RECORD,
    ADMISSION_UNVERIFIED_RECORD,
    IDENTITY_SHARED,
    IDENTITY_TRUSTED,
    DatasetError,
    FeatureRow,
    FoldImputer,
    IdentityTrustError,
    admit_delta_targets,
    assert_loss_uses_only_observed_labels,
    dataset_qc,
    feature_row_from_window,
    make_subject_folds,
)
from physio.schema import MeasurementV1, ProvenanceV1
from physio.windows import WindowFeatures

pytestmark = pytest.mark.synthetic

CUTOFF = datetime(2026, 11, 7, 10, 1, tzinfo=UTC)


def window(values=(1.0, 2.0, 0.0), mask=(True, True, False)):
    return WindowFeatures(
        usable=True,
        feature_names=("f1", "f2", "f3"),
        values=values,
        mask=mask,
        coverage=sum(mask) / len(mask),
        n_events=2,
        start=CUTOFF,
        end=CUTOFF,
        quality="ok",
        metrics_present=("heart_rate",),
    )


def row(subject="s1", values=(1.0, 2.0, 0.0), mask=(True, True, False)):
    return feature_row_from_window(
        window(values, mask), subject_ref=subject, session_id="sess", cutoff=CUTOFF
    )


def test_feature_row_keeps_three_masks_separate():
    item = row()
    assert item.observed_mask == (True, True, False)
    assert item.valid_mask == (True, True, False)
    assert item.imputed_mask == (False, False, False)
    assert item.value_map() == {"f1": 1.0, "f2": 2.0}


def test_imputer_uses_training_fold_medians_only():
    train = [row(values=(10.0, 0.0, 0.0), mask=(True, False, False)),
             row(values=(20.0, 0.0, 0.0), mask=(True, False, False)),
             row(values=(30.0, 0.0, 0.0), mask=(True, False, False))]
    imputer = FoldImputer.fit(train)
    test_row = row(values=(0.0, 999.0, 0.0), mask=(False, True, False))
    transformed = imputer.transform(test_row)
    assert transformed.values[0] == 20.0
    assert transformed.imputed_mask[0] is True
    assert transformed.imputation[0] == "train_median"
    # f2 was never observed in training; it must stay masked despite a test value.
    assert transformed.valid_mask[1] is True  # the test row's own observed f2 is valid
    assert transformed.imputed_mask[1] is False
    # f3 was never observed anywhere; no median exists and it stays masked.
    assert transformed.valid_mask[2] is False
    assert transformed.values[2] == 0.0
    assert imputer.medians[2] is None


def test_imputer_never_imputes_qc_invalid_observation():
    train = [row(values=(10.0, 0.0, 0.0), mask=(True, False, False)),
             row(values=(20.0, 0.0, 0.0), mask=(True, False, False))]
    imputer = FoldImputer.fit(train)
    invalid = FeatureRow(
        feature_names=("f1", "f2", "f3"),
        values=(0.0, 0.0, 0.0),
        observed_mask=(True, False, False),
        valid_mask=(False, False, False),
        imputed_mask=(False, False, False),
        imputation=("none", "none", "none"),
        subject_ref="s1",
        session_id="sess",
        cutoff=CUTOFF,
    )
    transformed = imputer.transform(invalid)
    assert transformed.valid_mask[0] is False
    assert transformed.imputed_mask[0] is False


def test_subject_folds_never_split_a_subject():
    subjects = ["s1", "s2", "s3", "s4", "s5", "s6"]
    folds = make_subject_folds(subjects, n_folds=3, seed=7)
    rows = [row(subject=subject) for subject in subjects for _ in range(2)]
    folds.validate_rows(rows)
    assert set(folds.folds) == set(subjects)
    assert set(folds.folds.values()) == {0, 1, 2}
    again = make_subject_folds(subjects, n_folds=3, seed=7)
    assert folds.folds == again.folds


def test_shared_identity_cannot_claim_generalization():
    with pytest.raises(IdentityTrustError):
        make_subject_folds(["m1", "m2", "m3"], n_folds=2, seed=1,
                           identity_trust=IDENTITY_SHARED)
    folds = make_subject_folds(
        ["m1", "m2", "m3"], n_folds=2, seed=1,
        identity_trust=IDENTITY_SHARED, allow_untrusted=True,
    )
    assert folds.trusted is False


def test_delta_label_accepts_independent_pre_post():
    pre = event(sequence=0, at=0, phase="pre", measurements=[measurement(value=70)])
    post = event(sequence=1, at=300, phase="post", measurements=[measurement(value=66)])
    admission = admit_delta_targets(pre, post)
    assert admission.mask["y_delta_heart_rate"] is True
    assert admission.values["y_delta_heart_rate"] == -4.0
    assert admission.y_pri_available is False
    assert_loss_uses_only_observed_labels(admission)


def test_same_record_cannot_masquerade_as_pre_and_post():
    pre = event(sequence=0, at=0, phase="pre", measurements=[measurement(value=70)])
    post = event(sequence=0, at=300, phase="post", measurements=[measurement(value=66)])
    admission = admit_delta_targets(pre, post)
    assert admission.mask["y_delta_heart_rate"] is False
    assert admission.reasons["y_delta_heart_rate"] == ADMISSION_SAME_RECORD


def test_independent_phases_may_restart_sequence_at_zero():
    provenance_pre = ProvenanceV1(adapter="test", adapter_version="1", source_record="pre-001")
    provenance_post = ProvenanceV1(adapter="test", adapter_version="1", source_record="post-001")
    pre = replace(
        event(sequence=0, at=0, phase="pre", measurements=[measurement(value=70)]),
        provenance=provenance_pre,
    )
    post = replace(
        event(sequence=0, at=300, phase="post", measurements=[measurement(value=66)]),
        provenance=provenance_post,
    )
    admission = admit_delta_targets(pre, post)
    assert admission.mask["y_delta_heart_rate"] is True
    assert admission.values["y_delta_heart_rate"] == -4.0


def test_imputed_endpoint_never_creates_a_label():
    pre = event(sequence=0, at=0, phase="pre", measurements=[measurement(value=70)])
    imputed = MeasurementV1(
        metric="heart_rate", value=66, unit="bpm", data_type="float",
        valid=True, status="accepted", notes=("value_origin=imputed",),
    )
    post = event(sequence=1, at=300, phase="post", measurements=[imputed])
    admission = admit_delta_targets(pre, post)
    assert admission.mask["y_delta_heart_rate"] is False
    assert admission.reasons["y_delta_heart_rate"] == ADMISSION_IMPUTED_ENDPOINT


def test_interval_too_long_rejects_all_metrics():
    pre = event(sequence=0, at=0, phase="pre", measurements=[measurement(value=70)])
    post = event(sequence=1, at=10000, phase="post", measurements=[measurement(value=66)])
    admission = admit_delta_targets(pre, post, max_interval_s=600)
    assert not any(admission.mask.values())
    assert admission.reasons["y_delta_heart_rate"] == ADMISSION_INTERVAL_TOO_LONG
    assert admission.values["y_delta_heart_rate"] is None
    assert admission.reasons["y_pri"] == ADMISSION_INTERVAL_TOO_LONG


def test_dataset_qc_is_reproducible():
    rows = [row("s1"), row("s2")]
    first = dataset_qc(rows, identity_trust=IDENTITY_TRUSTED)
    second = dataset_qc(list(reversed(rows)), identity_trust=IDENTITY_TRUSTED)
    assert first.to_dict() == second.to_dict()
    assert first.rows == 2 and first.subjects == 2
    assert first.observed_entries == 4


def test_invalid_feature_row_is_rejected():
    with pytest.raises(DatasetError):
        FeatureRow(
            feature_names=("f1",),
            values=(1.0,),
            observed_mask=(True,),
            valid_mask=(False,),
            imputed_mask=(False,),
            imputation=("none",),
            subject_ref="s1",
            session_id="sess",
            cutoff=CUTOFF,
        )


def test_half_identified_records_are_refused():
    provenance = ProvenanceV1(adapter="test", adapter_version="1", source_record="pre-001")
    pre = replace(
        event(sequence=0, at=0, phase="pre", measurements=[measurement(value=70)]),
        provenance=provenance,
    )
    post = event(sequence=1, at=300, phase="post", measurements=[measurement(value=66)])
    admission = admit_delta_targets(pre, post)
    assert admission.mask["y_delta_heart_rate"] is False
    assert admission.reasons["y_delta_heart_rate"] == ADMISSION_UNVERIFIED_RECORD
    assert admission.reasons["y_pri"] == ADMISSION_UNVERIFIED_RECORD
