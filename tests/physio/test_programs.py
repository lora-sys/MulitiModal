"""Synthetic tests for reviewed program capabilities and context parsing (#98)."""

from __future__ import annotations

import pytest

from physio.programs import (
    ProgramCapability,
    ProgramCapabilityCatalog,
    ProgramValidationError,
    parse_program_context,
    validate_program_payload,
)
from physio.schema import ProtocolContextV1

pytestmark = pytest.mark.synthetic

CATALOG_VERSION = "synth-catalog-v1"
DEVICE = "chair-synth-01"


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
        "contraindications": (),
        "device_compat": (DEVICE,),
        "reviewed": True,
        "catalog_version": CATALOG_VERSION,
    }
    values.update(overrides)
    return ProgramCapability(**values)


def catalog(*capabilities, reviewed=True, version=CATALOG_VERSION):
    items = capabilities or (capability(),)
    return ProgramCapabilityCatalog(
        capabilities=tuple(items),
        catalog_version=version,
        reviewed=reviewed,
    )


def protocol(**overrides):
    values = {"program_id": "SYNTH-P01", "intensity": 3, "duration_s": 600}
    values.update(overrides)
    return ProtocolContextV1(**values)


def test_catalog_roundtrip_and_lookup():
    source = catalog()
    restored = ProgramCapabilityCatalog.from_dict(source.to_dict())
    assert restored == source
    assert source.program_ids == ("SYNTH-P01",)
    assert source.get("SYNTH-P01") is not None
    assert source.get("M01") is None
    # #99 calls the collection as a method; both shapes must work.
    assert source.capabilities() == tuple(source.capabilities)


def test_missing_catalog_is_rejected():
    with pytest.raises(ProgramValidationError, match="catalog is required"):
        parse_program_context(protocol(), None)


def test_missing_program_id_is_rejected():
    with pytest.raises(ProgramValidationError, match="missing program_id"):
        parse_program_context(protocol(program_id=None), catalog())
    with pytest.raises(ProgramValidationError, match="missing program_id"):
        parse_program_context(protocol(program_id=123), catalog())


def test_unknown_program_is_never_mapped_to_m01_m04():
    with pytest.raises(ProgramValidationError, match="never mapped"):
        parse_program_context(protocol(program_id="M01"), catalog())
    assert catalog().get("M01") is None
    for legacy_id in ("M01", "M02", "M03", "M04"):
        assert legacy_id not in catalog().program_ids


def test_unreviewed_catalog_and_capability_are_rejected():
    with pytest.raises(ProgramValidationError, match="catalog is unreviewed"):
        parse_program_context(protocol(), catalog(reviewed=False))
    with pytest.raises(ProgramValidationError, match="not reviewed"):
        parse_program_context(protocol(), catalog(capability(reviewed=False)))


def test_non_selectable_program_is_rejected():
    with pytest.raises(ProgramValidationError, match="not selectable"):
        parse_program_context(protocol(), catalog(capability(selectable=False)))


@pytest.mark.parametrize("intensity", [0, 6])
def test_intensity_outside_capability_bounds_is_rejected(intensity):
    with pytest.raises(ProgramValidationError, match="intensity"):
        parse_program_context(protocol(intensity=intensity), catalog())


def test_intensity_must_be_an_integer():
    with pytest.raises(ProgramValidationError, match="integer"):
        parse_program_context(protocol(intensity=2.5), catalog())


def test_intensity_bounds_are_inclusive():
    accepted = parse_program_context(protocol(intensity=1), catalog())
    assert accepted.intensity == 1
    assert parse_program_context(protocol(intensity=5), catalog()).intensity == 5


@pytest.mark.parametrize("duration", [299, 1201])
def test_duration_outside_capability_bounds_is_rejected(duration):
    with pytest.raises(ProgramValidationError, match="duration"):
        parse_program_context(protocol(duration_s=duration), catalog())


def test_omitted_intensity_and_duration_have_no_default():
    with pytest.raises(ProgramValidationError, match="omission is rejected"):
        parse_program_context(protocol(intensity=None), catalog())
    with pytest.raises(ProgramValidationError, match="omission is rejected"):
        parse_program_context(protocol(duration_s=None), catalog())


def test_optional_fields_are_accepted_without_a_default_value():
    optional = capability(intensity_optional=True, duration_optional=True)
    context = parse_program_context(
        protocol(intensity=None, duration_s=None), catalog(optional)
    )
    assert context.intensity is None
    assert context.duration_s is None


def test_optional_flag_does_not_permit_the_other_field_to_default():
    optional = capability(intensity_optional=True)
    with pytest.raises(ProgramValidationError, match="duration"):
        parse_program_context(protocol(intensity=None, duration_s=None), catalog(optional))
    with pytest.raises(ProgramValidationError, match="intensity"):
        parse_program_context(
            protocol(intensity=None, duration_s=600),
            catalog(capability(intensity_optional=False)),
        )


def test_device_mismatch_and_missing_compatibility_are_rejected():
    compatible = parse_program_context(protocol(), catalog(), device_ref=DEVICE)
    assert compatible.program_id == "SYNTH-P01"
    with pytest.raises(ProgramValidationError, match="not compatible"):
        parse_program_context(protocol(), catalog(), device_ref="chair-other")
    with pytest.raises(ProgramValidationError, match="no device compatibility"):
        parse_program_context(
            protocol(), catalog(capability(device_compat=())), device_ref=DEVICE
        )


def test_device_is_not_checked_when_absent():
    context = parse_program_context(protocol(), catalog(capability(device_compat=())))
    assert context.program_id == "SYNTH-P01"


def test_catalog_version_mismatch_is_rejected():
    with pytest.raises(ProgramValidationError, match="catalog_version"):
        parse_program_context(
            protocol(catalog_version="other-version"),
            catalog(),
        )


def test_catalog_rejects_empty_duplicate_and_mismatched_entries():
    with pytest.raises(ProgramValidationError, match="must not be empty"):
        ProgramCapabilityCatalog(capabilities=(), catalog_version=CATALOG_VERSION)
    with pytest.raises(ProgramValidationError, match="duplicate"):
        catalog(capability(), capability())
    with pytest.raises(ProgramValidationError, match="carries catalog_version"):
        catalog(capability(catalog_version="other-version"))


def test_catalog_from_dict_rejects_missing_or_empty_programs():
    with pytest.raises(ProgramValidationError, match="missing catalog_version"):
        ProgramCapabilityCatalog.from_dict({"programs": []})
    with pytest.raises(ProgramValidationError, match="never guessed"):
        ProgramCapabilityCatalog.from_dict(
            {"catalog_version": CATALOG_VERSION, "reviewed": True}
        )
    with pytest.raises(ProgramValidationError, match="never guessed"):
        ProgramCapabilityCatalog.from_dict(
            {"catalog_version": CATALOG_VERSION, "reviewed": True, "programs": []}
        )


def test_capability_from_dict_requires_core_fields():
    with pytest.raises(ProgramValidationError, match="missing fields"):
        ProgramCapability.from_dict({"program_id": "SYNTH-P01"})


def test_invalid_capability_bounds_are_rejected():
    with pytest.raises(ProgramValidationError, match="intensity bounds"):
        capability(intensity_min=6, intensity_max=5)
    with pytest.raises(ProgramValidationError, match="duration bounds"):
        capability(duration_min_s=0)


def test_validate_program_payload_accepts_device_json_shape():
    payload = {
        "protocol": {"program_id": "SYNTH-P01", "intensity": 2, "duration_s": 600},
        "device_ref": DEVICE,
    }
    context = validate_program_payload(payload, catalog())
    assert context.intensity == 2
    assert context.actions == ("knead",)


def test_validate_program_payload_accepts_flat_shape_without_device():
    context = validate_program_payload(
        {"program_id": "SYNTH-P01", "intensity": 2, "duration_s": 600},
        catalog(),
    )
    assert context.program_id == "SYNTH-P01"


def test_validate_program_payload_rejects_non_mapping_and_unknown_program():
    with pytest.raises(ProgramValidationError, match="JSON object"):
        validate_program_payload("SYNTH-P01,2,600", catalog())
    with pytest.raises(ProgramValidationError, match="unknown program_id"):
        validate_program_payload(
            {"program_id": "M02", "intensity": 2, "duration_s": 600}, catalog()
        )
    with pytest.raises(ProgramValidationError, match="invalid protocol payload"):
        validate_program_payload(
            {"program_id": "SYNTH-P01", "intensity": 11, "duration_s": 600}, catalog()
        )
