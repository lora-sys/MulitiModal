"""Reviewed massage-program capabilities and typed program context (#98).

The capability catalog is the only source of program vocabulary. A protocol
that names an unknown program is rejected; it is never mapped onto M01--M04 or
any other "closest" option. Omitted intensity/duration are also rejected
unless the capability explicitly marks them optional, so no hidden default is
invented for training or inference.

A catalog is a reviewed artifact. Both the catalog and the individual
capability must be marked ``reviewed`` before they can condition a prediction.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .schema import ProtocolContextV1, SchemaError

PROGRAM_SCHEMA_VERSION = "physio-program-context-v1"
PROGRAM_CATALOG_VERSION = "physio-program-catalog-v1"

INTENSITY_MIN = 0
INTENSITY_MAX = 10


class ProgramValidationError(ValueError):
    """A program context cannot be resolved without guessing."""


class _CapabilityList(tuple):
    """Immutable capability tuple that also answers ``catalog.capabilities()``."""

    def __call__(self) -> tuple[ProgramCapability, ...]:
        return tuple(self)


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProgramValidationError(f"{field} must be a non-empty string")
    return value.strip()


def _optional_text(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _text(value, field)


def _tuple_of_text(value: Any, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, (tuple, list)):
        raise ProgramValidationError(f"{field} must be a sequence of strings")
    return tuple(_text(item, f"{field} item") for item in value)


def _int_value(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProgramValidationError(f"{field} must be an integer")
    return value


@dataclass(frozen=True)
class ProgramCapability:
    """One reviewed program entry from the device/training catalog.

    ``intensity_optional`` / ``duration_optional`` only permit an omitted
    protocol value; they never supply a default.
    """

    program_id: str
    name: str
    intensity_min: int
    intensity_max: int
    duration_min_s: int
    duration_max_s: int
    actions: tuple[str, ...]
    selectable: bool
    contraindications: tuple[str, ...] = ()
    device_compat: tuple[str, ...] = ()
    reviewed: bool = False
    catalog_version: str = PROGRAM_CATALOG_VERSION
    intensity_optional: bool = False
    duration_optional: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "program_id", _text(self.program_id, "program_id"))
        object.__setattr__(self, "name", _text(self.name, "name"))
        object.__setattr__(self, "catalog_version", _text(self.catalog_version, "catalog_version"))
        object.__setattr__(self, "actions", _tuple_of_text(self.actions, "actions"))
        object.__setattr__(
            self, "contraindications", _tuple_of_text(self.contraindications, "contraindications")
        )
        object.__setattr__(
            self, "device_compat", _tuple_of_text(self.device_compat, "device_compat")
        )
        if type(self.selectable) is not bool:
            raise ProgramValidationError("selectable must be a bool")
        if type(self.reviewed) is not bool:
            raise ProgramValidationError("reviewed must be a bool")
        if type(self.intensity_optional) is not bool:
            raise ProgramValidationError("intensity_optional must be a bool")
        if type(self.duration_optional) is not bool:
            raise ProgramValidationError("duration_optional must be a bool")
        low = _int_value(self.intensity_min, "intensity_min")
        high = _int_value(self.intensity_max, "intensity_max")
        if not INTENSITY_MIN <= low <= high <= INTENSITY_MAX:
            raise ProgramValidationError(
                "intensity bounds must satisfy "
                f"{INTENSITY_MIN} <= min <= max <= {INTENSITY_MAX}, got [{low}, {high}]"
            )
        duration_low = _int_value(self.duration_min_s, "duration_min_s")
        duration_high = _int_value(self.duration_max_s, "duration_max_s")
        if not 0 < duration_low <= duration_high:
            raise ProgramValidationError(
                "duration bounds must satisfy 0 < min <= max, "
                f"got [{duration_low}, {duration_high}]"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "program_id": self.program_id,
            "name": self.name,
            "intensity_min": self.intensity_min,
            "intensity_max": self.intensity_max,
            "duration_min_s": self.duration_min_s,
            "duration_max_s": self.duration_max_s,
            "actions": list(self.actions),
            "selectable": self.selectable,
            "contraindications": list(self.contraindications),
            "device_compat": list(self.device_compat),
            "reviewed": self.reviewed,
            "catalog_version": self.catalog_version,
            "intensity_optional": self.intensity_optional,
            "duration_optional": self.duration_optional,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProgramCapability:
        if not isinstance(data, Mapping):
            raise ProgramValidationError("program capability must be a mapping")
        required = (
            "program_id",
            "name",
            "intensity_min",
            "intensity_max",
            "duration_min_s",
            "duration_max_s",
            "actions",
            "selectable",
        )
        missing = [field for field in required if field not in data]
        if missing:
            raise ProgramValidationError(f"program capability is missing fields: {missing}")
        return cls(
            program_id=data["program_id"],
            name=data["name"],
            intensity_min=data["intensity_min"],
            intensity_max=data["intensity_max"],
            duration_min_s=data["duration_min_s"],
            duration_max_s=data["duration_max_s"],
            actions=_tuple_of_text(data["actions"], "actions"),
            selectable=data["selectable"],
            contraindications=data.get("contraindications", ()),
            device_compat=data.get("device_compat", ()),
            reviewed=data.get("reviewed", False),
            catalog_version=data.get("catalog_version", PROGRAM_CATALOG_VERSION),
            intensity_optional=data.get("intensity_optional", False),
            duration_optional=data.get("duration_optional", False),
        )


@dataclass(frozen=True)
class ProgramCapabilityCatalog:
    """Versioned, reviewed set of programs. An empty catalog is invalid.

    ``capabilities`` is an immutable tuple that is also callable, so both
    ``catalog.capabilities`` and ``catalog.capabilities()`` are valid access
    patterns for existing consumers.
    """

    capabilities: tuple[ProgramCapability, ...]
    catalog_version: str
    reviewed: bool = False

    def __post_init__(self) -> None:
        capabilities = _CapabilityList(self.capabilities)
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "catalog_version", _text(self.catalog_version, "catalog_version"))
        if type(self.reviewed) is not bool:
            raise ProgramValidationError("catalog reviewed must be a bool")
        if not capabilities:
            raise ProgramValidationError("program capability catalog must not be empty")
        identifiers = [capability.program_id for capability in capabilities]
        if len(identifiers) != len(set(identifiers)):
            raise ProgramValidationError(f"duplicate program_id entries: {identifiers}")
        for capability in capabilities:
            if capability.catalog_version != self.catalog_version:
                raise ProgramValidationError(
                    f"program {capability.program_id!r} carries catalog_version "
                    f"{capability.catalog_version!r}, expected {self.catalog_version!r}"
                )

    @property
    def program_ids(self) -> tuple[str, ...]:
        return tuple(capability.program_id for capability in self.capabilities)

    def get(self, program_id: str | None) -> ProgramCapability | None:
        """Return the matching capability or ``None``; never a fallback."""
        if program_id is None:
            return None
        for capability in self.capabilities:
            if capability.program_id == program_id:
                return capability
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "catalog_version": self.catalog_version,
            "reviewed": self.reviewed,
            "programs": [capability.to_dict() for capability in self.capabilities],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProgramCapabilityCatalog:
        if not isinstance(data, Mapping):
            raise ProgramValidationError("program catalog must be a mapping")
        if "catalog_version" not in data:
            raise ProgramValidationError("program catalog is missing catalog_version")
        catalog_version = _text(data["catalog_version"], "catalog_version")
        raw_programs = data.get("programs", data.get("capabilities"))
        if not isinstance(raw_programs, (tuple, list)) or not raw_programs:
            raise ProgramValidationError(
                "program catalog must declare a non-empty 'programs' list; "
                "a missing/empty catalog is never guessed"
            )
        programs = []
        for item in raw_programs:
            if not isinstance(item, Mapping):
                raise ProgramValidationError("each program capability must be a mapping")
            enriched = dict(item)
            enriched.setdefault("catalog_version", catalog_version)
            programs.append(ProgramCapability.from_dict(enriched))
        return cls(
            capabilities=tuple(programs),
            catalog_version=catalog_version,
            reviewed=data.get("reviewed", False),
        )


@dataclass(frozen=True)
class ProgramContext:
    """Validated program context used to condition response labels."""

    program_id: str
    intensity: int | None
    duration_s: int | None
    actions: tuple[str, ...]
    catalog_version: str
    reviewed: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "program_id", _text(self.program_id, "program_id"))
        object.__setattr__(self, "catalog_version", _text(self.catalog_version, "catalog_version"))
        object.__setattr__(self, "actions", _tuple_of_text(self.actions, "actions"))
        if self.intensity is not None:
            value = _int_value(self.intensity, "intensity")
            if not INTENSITY_MIN <= value <= INTENSITY_MAX:
                raise ProgramValidationError("program context intensity is outside 0..10")
        if self.duration_s is not None:
            if _int_value(self.duration_s, "duration_s") <= 0:
                raise ProgramValidationError("program context duration_s must be positive")
        if type(self.reviewed) is not bool:
            raise ProgramValidationError("program context reviewed must be a bool")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PROGRAM_SCHEMA_VERSION,
            "program_id": self.program_id,
            "intensity": self.intensity,
            "duration_s": self.duration_s,
            "actions": list(self.actions),
            "catalog_version": self.catalog_version,
            "reviewed": self.reviewed,
        }


def _check_intensity(value: int | None, capability: ProgramCapability) -> int | None:
    if value is None:
        if capability.intensity_optional:
            return None
        raise ProgramValidationError(
            f"program {capability.program_id!r} requires intensity within "
            f"[{capability.intensity_min}, {capability.intensity_max}]; "
            "omission is rejected, not defaulted"
        )
    intensity = _int_value(value, "protocol intensity")
    if not capability.intensity_min <= intensity <= capability.intensity_max:
        raise ProgramValidationError(
            f"program {capability.program_id!r} intensity {intensity} is outside "
            f"[{capability.intensity_min}, {capability.intensity_max}]"
        )
    return intensity


def _check_duration(value: int | None, capability: ProgramCapability) -> int | None:
    if value is None:
        if capability.duration_optional:
            return None
        raise ProgramValidationError(
            f"program {capability.program_id!r} requires duration_s within "
            f"[{capability.duration_min_s}, {capability.duration_max_s}]; "
            "omission is rejected, not defaulted"
        )
    duration = _int_value(value, "protocol duration_s")
    if not capability.duration_min_s <= duration <= capability.duration_max_s:
        raise ProgramValidationError(
            f"program {capability.program_id!r} duration_s {duration} is outside "
            f"[{capability.duration_min_s}, {capability.duration_max_s}]"
        )
    return duration


def _check_device(device_ref: str | None, capability: ProgramCapability) -> None:
    if device_ref is None:
        return
    device = _text(device_ref, "device_ref")
    if not capability.device_compat:
        raise ProgramValidationError(
            f"program {capability.program_id!r} declares no device compatibility; "
            "a supplied device cannot be verified"
        )
    if device not in capability.device_compat:
        raise ProgramValidationError(
            f"device {device!r} is not compatible with program "
            f"{capability.program_id!r}; expected one of {capability.device_compat}"
        )


def parse_program_context(
    protocol: ProtocolContextV1 | None,
    catalog: ProgramCapabilityCatalog | None,
    *,
    device_ref: str | None = None,
) -> ProgramContext:
    """Resolve a protocol into a reviewed program context or reject it.

    Rejections are explicit for: missing catalog, unreviewed catalog, missing or
    unknown program_id, unreviewed/non-selectable capability, catalog version
    mismatch, intensity/duration outside the capability bounds (or omitted
    without an explicit optional flag), and an incompatible device.
    """
    if catalog is None:
        raise ProgramValidationError(
            "program capability catalog is required; no program is guessed"
        )
    if not catalog.reviewed:
        raise ProgramValidationError(
            "program capability catalog is unreviewed; refusing to condition labels"
        )
    if protocol is None:
        raise ProgramValidationError("protocol context is required; program_id is missing")
    if not isinstance(protocol.program_id, str) or not protocol.program_id.strip():
        raise ProgramValidationError("protocol is missing program_id")
    program_id = protocol.program_id.strip()
    if (
        protocol.catalog_version is not None
        and protocol.catalog_version != catalog.catalog_version
    ):
        raise ProgramValidationError(
            f"protocol catalog_version {protocol.catalog_version!r} does not match "
            f"catalog {catalog.catalog_version!r}"
        )
    capability = catalog.get(program_id)
    if capability is None:
        raise ProgramValidationError(
            f"unknown program_id {program_id!r}; known ids are {catalog.program_ids}. "
            "Unknown programs are never mapped to M01--M04 or any other program"
        )
    if not capability.reviewed:
        raise ProgramValidationError(
            f"program {program_id!r} is not reviewed; refusing to condition labels"
        )
    if not capability.selectable:
        raise ProgramValidationError(
            f"program {program_id!r} is not selectable; refusing to condition labels"
        )
    intensity = _check_intensity(protocol.intensity, capability)
    duration = _check_duration(protocol.duration_s, capability)
    _check_device(device_ref, capability)
    return ProgramContext(
        program_id=program_id,
        intensity=intensity,
        duration_s=duration,
        actions=capability.actions,
        catalog_version=catalog.catalog_version,
        reviewed=True,
    )


def validate_program_payload(
    payload: Mapping[str, Any],
    catalog: ProgramCapabilityCatalog | None,
) -> ProgramContext:
    """Thin wrapper for device/service JSON; CSV rows are not accepted here."""
    if not isinstance(payload, Mapping):
        raise ProgramValidationError(
            "program payload must be a JSON object; CSV rows are not a realtime input"
        )
    protocol_data = payload.get("protocol", payload)
    if not isinstance(protocol_data, Mapping):
        raise ProgramValidationError("protocol must be a JSON object")
    try:
        protocol = ProtocolContextV1.from_dict(dict(protocol_data))
    except SchemaError as exc:
        raise ProgramValidationError(f"invalid protocol payload: {exc}") from exc
    device_ref = payload.get("device_ref")
    return parse_program_context(
        protocol,
        catalog,
        device_ref=str(device_ref) if device_ref is not None else None,
    )


__all__ = [
    "INTENSITY_MAX",
    "INTENSITY_MIN",
    "PROGRAM_CATALOG_VERSION",
    "PROGRAM_SCHEMA_VERSION",
    "ProgramCapability",
    "ProgramCapabilityCatalog",
    "ProgramContext",
    "ProgramValidationError",
    "parse_program_context",
    "validate_program_payload",
]
