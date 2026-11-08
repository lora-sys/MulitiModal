"""Device TCM exam mapping and reviewed wellness knowledge sidecar (#100).

This module is a thin product-side adapter. It consumes structured tongue /
face / constitution fields that a device (or vendor service) already produced.
It never infers a personal constitution from a few tongue or face signs, never
emits a diagnosis, and never prescribes tea, acupoints or a constitution-derived
massage intensity.

The Mendeley-274 constitution model is deliberately absent: this path works
with no research artifact installed. Vendor ``constitution_names`` are carried
as device-provided findings with explicit source/provenance and are always
``is_clinical_truth=False``.

Knowledge text is served only from a reviewed registry. An entry missing
``source`` / ``license`` / ``version`` / ``review_status`` / ``reviewed_by`` is
excluded, and an entry that is not ``APPROVED`` is never displayed.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .adapters import parse_timestamp
from .schema import (
    QUALITY_DEGRADED,
    QUALITY_OK,
    QUALITY_REJECTED,
    QUALITY_VALUES,
    SOURCE_DEVICE,
    SOURCES,
    ProvenanceV1,
    SchemaError,
)

TCM_EXAM_SCHEMA_VERSION = "tcm-exam-v1"

STATUS_OK = "OK"
STATUS_PARTIAL = "PARTIAL"
STATUS_ABSTAIN = "ABSTAIN"
STATUS_UNKNOWN = "UNKNOWN"

PARSE_STATUSES = (STATUS_OK, STATUS_PARTIAL, STATUS_UNKNOWN)
SIDECAR_STATUSES = (STATUS_OK, STATUS_PARTIAL, STATUS_ABSTAIN, STATUS_UNKNOWN)

CONSTITUTION_SOURCE_DEVICE = "DEVICE"
CONSTITUTION_SOURCE_PROJECT_MODEL = "PROJECT_MODEL"
CONSTITUTION_SOURCE_QUESTIONNAIRE = "QUESTIONNAIRE"
CONSTITUTION_SOURCES = (
    CONSTITUTION_SOURCE_DEVICE,
    CONSTITUTION_SOURCE_PROJECT_MODEL,
    CONSTITUTION_SOURCE_QUESTIONNAIRE,
)

REVIEW_APPROVED = "APPROVED"

#: Reviewed vocabulary. Canonical value -> accepted surface forms. Unknown
#: surface forms stay unknown; this table is intentionally small and auditable.
TONGUE_BODY_COLOR_SYNONYMS: dict[str, tuple[str, ...]] = {
    "淡红": ("淡红", "淡红色", "淡红舌", "舌质淡红"),
    "淡白": ("淡白", "淡白色", "淡白舌", "舌质淡白"),
    "红": ("红", "红色", "红舌", "舌质红"),
    "绛": ("绛", "绛色", "绛红", "深红", "舌绛"),
    "紫": ("紫", "紫色", "紫暗", "青紫", "紫舌"),
}

TONGUE_SHAPE_SYNONYMS: dict[str, tuple[str, ...]] = {
    "正常": ("正常", "适中", "正常舌形"),
    "胖大": ("胖大", "胖大舌", "舌体胖大", "肿大"),
    "齿痕": ("齿痕", "齿痕舌", "边有齿痕", "舌边齿痕"),
    "裂纹": ("裂纹", "裂纹舌", "舌有裂纹"),
    "瘦薄": ("瘦薄", "瘦小", "瘦薄舌"),
}

COATING_COLOR_SYNONYMS: dict[str, tuple[str, ...]] = {
    "白": ("白", "白色", "白苔", "苔白"),
    "黄": ("黄", "黄色", "黄苔", "苔黄"),
    "灰": ("灰", "灰色", "灰苔"),
    "黑": ("黑", "黑色", "黑苔"),
}

COATING_TEXTURE_SYNONYMS: dict[str, tuple[str, ...]] = {
    "薄": ("薄", "薄苔", "苔薄"),
    "厚": ("厚", "厚苔", "苔厚"),
    "腻": ("腻", "腻苔", "苔腻"),
    "腐": ("腐", "腐苔"),
    "燥": ("燥", "燥苔", "干燥"),
    "滑": ("滑", "滑苔", "湿润"),
}

FACE_COLOR_SYNONYMS: dict[str, tuple[str, ...]] = {
    "红润": ("红润", "面色红润", "红润有泽"),
    "苍白": ("苍白", "面色苍白", "白"),
    "萎黄": ("萎黄", "面色萎黄", "黄"),
    "潮红": ("潮红", "面红", "面赤", "面色潮红"),
    "青紫": ("青紫", "面色青紫", "青"),
    "黧黑": ("黧黑", "面色黧黑", "黑"),
}

COMPLEXION_SYNONYMS: dict[str, tuple[str, ...]] = {
    "荣润": ("荣润", "润泽", "有光泽", "面色荣润"),
    "少华": ("少华", "面色少华", "无华", "面色无华"),
    "晦暗": ("晦暗", "暗滞", "晦滞", "面色晦暗"),
}

CONSTITUTION_SYNONYMS: dict[str, tuple[str, ...]] = {
    "平和": ("平和", "平和质", "平和型"),
    "气虚": ("气虚", "气虚质", "气虚型"),
    "阳虚": ("阳虚", "阳虚质", "阳虚型"),
    "阴虚": ("阴虚", "阴虚质", "阴虚型"),
    "痰湿": ("痰湿", "痰湿质", "痰湿型"),
    "湿热": ("湿热", "湿热质", "湿热型"),
    "血瘀": ("血瘀", "血瘀质", "血瘀型"),
    "气郁": ("气郁", "气郁质", "气郁型"),
    "特禀": ("特禀", "特禀质", "特禀型"),
}

_COMPOUND_SEPARATORS = ("兼", "夹", "伴")
_TOKEN_SPLIT_RE = re.compile(r"[、,，;；/|\\\n\r\t]+")
_EDGE_CHARS = "。.、,，;；:：()（）[]【】 \t\u3000"
_WHITESPACE_RE = re.compile(r"\s+")

_EXAM_TEXT_FIELDS = (
    "tongue_body_color",
    "tongue_shape",
    "coating_color",
    "coating_texture",
    "face_color",
    "complexion",
    "eye_description",
)


def _normalize_token(token: str) -> str:
    return _WHITESPACE_RE.sub("", str(token).strip()).lower()


def _dedupe(items: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            ordered.append(item)
    return tuple(ordered)


@dataclass(frozen=True)
class ParsedValues:
    """Result of parsing one free-text vocabulary field.

    ``values`` are canonical labels only. Unrecognised tokens are kept in
    ``unknown`` and contradictory single-valued labels are kept in
    ``conflicts``; neither is resolved by choosing a winner.
    """

    values: tuple[str, ...] = ()
    unknown: tuple[str, ...] = ()
    status: str = STATUS_UNKNOWN
    conflicts: tuple[str, ...] = ()
    reason: str = ""

    def __post_init__(self) -> None:
        if self.status not in PARSE_STATUSES:
            raise ValueError(f"unknown parse status: {self.status!r}")
        object.__setattr__(self, "values", tuple(self.values))
        object.__setattr__(self, "unknown", tuple(self.unknown))
        object.__setattr__(self, "conflicts", tuple(self.conflicts))

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": list(self.values),
            "unknown": list(self.unknown),
            "status": self.status,
            "conflicts": list(self.conflicts),
            "reason": self.reason,
        }


def _alias_lookup(synonyms: Mapping[str, Sequence[str]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for canonical, aliases in synonyms.items():
        for alias in (canonical, *aliases):
            key = _normalize_token(alias)
            if key:
                lookup.setdefault(key, canonical)
    return lookup


def _tokens(text: str) -> tuple[str, ...]:
    tokens: list[str] = []
    for chunk in _TOKEN_SPLIT_RE.split(text):
        for token in chunk.split():
            cleaned = token.strip(_EDGE_CHARS)
            if cleaned:
                tokens.append(cleaned)
    return tuple(tokens)


def _resolve_token(
    token: str,
    lookup: Mapping[str, str],
    compound_separators: Sequence[str],
) -> tuple[list[str], list[str]]:
    canonical = lookup.get(_normalize_token(token))
    if canonical is not None:
        return [canonical], []
    if compound_separators:
        pattern = "|".join(re.escape(separator) for separator in compound_separators)
        parts = [part.strip() for part in re.split(pattern, token) if part.strip()]
        if len(parts) > 1:
            values: list[str] = []
            unknown: list[str] = []
            for part in parts:
                resolved = lookup.get(_normalize_token(part))
                if resolved is None:
                    unknown.append(part)
                else:
                    values.append(resolved)
            return values, unknown
    return [], [token]


def parse_multi_value(
    text: str | None,
    synonyms: Mapping[str, Sequence[str]],
    *,
    multi_valued: bool = False,
    compound_separators: Sequence[str] = (),
) -> ParsedValues:
    """Split and normalise a free-text label field without guessing.

    Separators are the common device spellings ``、,，;；/|`` plus whitespace.
    For a multi-valued field (e.g. tongue shape or constitution names) several
    canonical values are valid; for a single-valued field two different values
    are a conflict and the result is ``PARTIAL`` rather than a silent pick.
    """
    if text is None or (isinstance(text, str) and not text.strip()):
        return ParsedValues(status=STATUS_UNKNOWN, reason="blank input")
    if not isinstance(text, str):
        raise TypeError("parse_multi_value expects a string or None")

    lookup = _alias_lookup(synonyms)
    values: list[str] = []
    unknown: list[str] = []
    for token in _tokens(text):
        resolved_values, resolved_unknown = _resolve_token(
            token, lookup, compound_separators
        )
        values.extend(resolved_values)
        unknown.extend(resolved_unknown)

    deduped_values = _dedupe(values)
    deduped_unknown = _dedupe(unknown)
    conflicts = (
        deduped_values
        if not multi_valued and len(deduped_values) > 1
        else ()
    )

    if deduped_values and not deduped_unknown and not conflicts:
        return ParsedValues(
            values=deduped_values,
            status=STATUS_OK,
            reason="all tokens matched the reviewed vocabulary",
        )
    if deduped_values:
        notes = []
        if deduped_unknown:
            notes.append(f"unrecognised tokens: {', '.join(deduped_unknown)}")
        if conflicts:
            notes.append(f"conflicting single-valued labels: {', '.join(conflicts)}")
        return ParsedValues(
            values=deduped_values,
            unknown=deduped_unknown,
            status=STATUS_PARTIAL,
            conflicts=conflicts,
            reason="; ".join(notes),
        )
    return ParsedValues(
        unknown=deduped_unknown,
        status=STATUS_UNKNOWN,
        reason="no token matched the reviewed vocabulary",
    )


def parse_constitution_names(text: str | None) -> ParsedValues:
    """Parse device/legacy constitution text, including ``气虚兼痰湿`` compounds."""
    return parse_multi_value(
        text,
        CONSTITUTION_SYNONYMS,
        multi_valued=True,
        compound_separators=_COMPOUND_SEPARATORS,
    )


def canonical_constitution_name(name: str) -> str:
    """Map a reviewed synonym to its canonical name; unknown names pass through."""
    lookup = _alias_lookup(CONSTITUTION_SYNONYMS)
    key = _normalize_token(name)
    return lookup.get(key, str(name).strip())


@dataclass(frozen=True)
class ConstitutionFinding:
    """A constitution label reported by a device, project model or questionnaire.

    ``is_clinical_truth`` is always ``False``: a vendor result is a label with
    provenance, not a diagnosis and not a validated clinical truth.
    """

    name: str
    source: str
    confidence: float | None = None
    algorithm: str | None = None
    version: str | None = None
    is_clinical_truth: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise SchemaError("constitution finding name must be non-empty")
        if self.source not in CONSTITUTION_SOURCES:
            raise SchemaError(f"unknown constitution source: {self.source!r}")
        if self.confidence is not None:
            if isinstance(self.confidence, bool) or not isinstance(
                self.confidence, (int, float)
            ):
                raise SchemaError("constitution confidence must be numeric or None")
            if not 0.0 <= float(self.confidence) <= 1.0:
                raise SchemaError(
                    f"constitution confidence must be within 0..1, got {self.confidence}"
                )
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "is_clinical_truth", False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "confidence": self.confidence,
            "algorithm": self.algorithm,
            "version": self.version,
            "is_clinical_truth": self.is_clinical_truth,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ConstitutionFinding:
        return cls(
            name=str(data["name"]).strip(),
            source=str(data["source"]).strip().upper(),
            confidence=data.get("confidence"),
            algorithm=data.get("algorithm"),
            version=data.get("version"),
            is_clinical_truth=False,
        )


def _exam_quality(fields: Sequence[str | None], findings: Sequence[ConstitutionFinding]) -> str:
    present = sum(1 for value in fields if isinstance(value, str) and value.strip())
    has_findings = any(finding.name for finding in findings)
    if present == 0 and not has_findings:
        return QUALITY_REJECTED
    if present == len(_EXAM_TEXT_FIELDS):
        return QUALITY_OK
    return QUALITY_DEGRADED


@dataclass(frozen=True)
class TCMExamV1:
    """One structured device TCM exam event."""

    exam_id: str
    session_id: str
    subject_ref: str
    device_ref: str
    source: str
    measured_at: datetime
    received_at: datetime
    tongue_body_color: str | None = None
    tongue_shape: str | None = None
    coating_color: str | None = None
    coating_texture: str | None = None
    face_color: str | None = None
    complexion: str | None = None
    eye_description: str | None = None
    constitution_names: tuple[ConstitutionFinding, ...] = ()
    device_algorithm: str | None = None
    vocabulary_version: str | None = None
    quality: str = ""
    provenance: ProvenanceV1 | None = None
    schema_version: str = TCM_EXAM_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("exam_id", "session_id", "subject_ref", "device_ref"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise SchemaError(f"{name} must be a non-empty string")
        if self.schema_version != TCM_EXAM_SCHEMA_VERSION:
            raise SchemaError(
                f"unsupported TCM exam schema {self.schema_version!r}; "
                f"expected {TCM_EXAM_SCHEMA_VERSION!r}"
            )
        if self.source not in SOURCES:
            raise SchemaError(f"unknown exam source: {self.source!r}")
        if not isinstance(self.measured_at, datetime) or not isinstance(
            self.received_at, datetime
        ):
            raise SchemaError("measured_at and received_at must be datetime values")
        if self.measured_at.tzinfo is None or self.received_at.tzinfo is None:
            raise SchemaError("measured_at and received_at must be timezone-aware")
        if self.received_at < self.measured_at:
            raise SchemaError(
                "received_at precedes measured_at: reject reversed timestamps "
                "instead of inventing latency"
            )
        for name in _EXAM_TEXT_FIELDS:
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, str):
                raise SchemaError(f"{name} must be text or None")
            cleaned = value.strip()
            object.__setattr__(self, name, cleaned or None)
        for name in ("device_algorithm", "vocabulary_version"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise SchemaError(f"{name} must be text or None")
        findings = tuple(self.constitution_names)
        if any(not isinstance(item, ConstitutionFinding) for item in findings):
            raise SchemaError("constitution_names must contain ConstitutionFinding items")
        object.__setattr__(self, "constitution_names", findings)
        if self.provenance is not None and not isinstance(self.provenance, ProvenanceV1):
            raise SchemaError("provenance must be ProvenanceV1 or None")

        text_values = [getattr(self, name) for name in _EXAM_TEXT_FIELDS]
        derived = _exam_quality(text_values, findings)
        if isinstance(self.quality, str):
            object.__setattr__(self, "quality", self.quality.strip().lower())
        if not self.quality:
            object.__setattr__(self, "quality", derived)
        elif self.quality not in QUALITY_VALUES:
            raise SchemaError(f"unknown exam quality: {self.quality!r}")
        elif self.quality == QUALITY_OK and derived != QUALITY_OK:
            raise SchemaError("exam quality cannot be 'ok' when fields are missing")

    def parsed_fields(self) -> dict[str, ParsedValues]:
        return parse_exam_fields(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "exam_id": self.exam_id,
            "session_id": self.session_id,
            "subject_ref": self.subject_ref,
            "device_ref": self.device_ref,
            "source": self.source,
            "measured_at": self.measured_at.isoformat(),
            "received_at": self.received_at.isoformat(),
            "tongue_body_color": self.tongue_body_color,
            "tongue_shape": self.tongue_shape,
            "coating_color": self.coating_color,
            "coating_texture": self.coating_texture,
            "face_color": self.face_color,
            "complexion": self.complexion,
            "eye_description": self.eye_description,
            "constitution_names": [item.to_dict() for item in self.constitution_names],
            "device_algorithm": self.device_algorithm,
            "vocabulary_version": self.vocabulary_version,
            "quality": self.quality,
            "provenance": self.provenance.to_dict() if self.provenance else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TCMExamV1:
        if data.get("schema_version") != TCM_EXAM_SCHEMA_VERSION:
            raise SchemaError(
                f"unsupported TCM exam schema: {data.get('schema_version')!r}"
            )
        tongue = data.get("tongue") or {}
        face = data.get("face") or {}
        if not isinstance(tongue, Mapping) or not isinstance(face, Mapping):
            raise SchemaError("tongue and face must be objects")
        try:
            measured_at = datetime.fromisoformat(str(data["measured_at"]))
            received_at = datetime.fromisoformat(str(data["received_at"]))
        except (KeyError, ValueError) as exc:
            raise SchemaError(f"invalid exam timestamps: {exc}") from exc
        raw_provenance = data.get("provenance")
        if isinstance(raw_provenance, ProvenanceV1):
            provenance = raw_provenance
        elif isinstance(raw_provenance, Mapping):
            provenance = ProvenanceV1.from_dict(raw_provenance)
        else:
            provenance = None
        return cls(
            schema_version=TCM_EXAM_SCHEMA_VERSION,
            exam_id=str(data["exam_id"]),
            session_id=str(data["session_id"]),
            subject_ref=str(data["subject_ref"]),
            device_ref=str(data["device_ref"]),
            source=str(data["source"]),
            measured_at=measured_at,
            received_at=received_at,
            tongue_body_color=data.get("tongue_body_color", tongue.get("body_color")),
            tongue_shape=data.get("tongue_shape", tongue.get("shape")),
            coating_color=data.get("coating_color", tongue.get("coating_color")),
            coating_texture=data.get("coating_texture", tongue.get("coating_texture")),
            face_color=data.get("face_color", face.get("color")),
            complexion=data.get("complexion", face.get("complexion")),
            eye_description=data.get("eye_description", face.get("eye_description")),
            constitution_names=tuple(
                ConstitutionFinding.from_dict(item)
                for item in data.get("constitution_names", ())
            ),
            device_algorithm=data.get("device_algorithm"),
            vocabulary_version=data.get("vocabulary_version"),
            quality=str(data.get("quality", "")),
            provenance=provenance,
        )


def parse_exam_fields(exam: TCMExamV1) -> dict[str, ParsedValues]:
    """Parse the six reviewed tongue/face dimensions of an exam.

    ``eye_description`` stays verbatim because left/right eye descriptors are
    free text and no reviewed eye vocabulary is frozen yet.
    """
    if not isinstance(exam, TCMExamV1):
        raise TypeError("parse_exam_fields expects a TCMExamV1")
    return {
        "tongue_body_color": parse_multi_value(
            exam.tongue_body_color, TONGUE_BODY_COLOR_SYNONYMS
        ),
        "tongue_shape": parse_multi_value(
            exam.tongue_shape, TONGUE_SHAPE_SYNONYMS, multi_valued=True
        ),
        "coating_color": parse_multi_value(
            exam.coating_color, COATING_COLOR_SYNONYMS
        ),
        "coating_texture": parse_multi_value(
            exam.coating_texture, COATING_TEXTURE_SYNONYMS, multi_valued=True
        ),
        "face_color": parse_multi_value(exam.face_color, FACE_COLOR_SYNONYMS),
        "complexion": parse_multi_value(exam.complexion, COMPLEXION_SYNONYMS),
    }


def _first_text(mapping: Mapping[str, Any], names: Sequence[str]) -> str | None:
    for name in names:
        value = mapping.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _nested_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, Mapping) else {}


def _raw_constitution_value(payload: Mapping[str, Any]) -> Any:
    for key in ("constitution_names", "constitution_findings", "constitutions"):
        if key in payload:
            return payload[key]
    device = _nested_mapping(payload, "device")
    for key in ("constitution_names", "constitution_findings", "constitutions"):
        if key in device:
            return device[key]
    return None


def _constitution_findings(
    raw: Any,
    *,
    default_algorithm: str | None,
    default_version: str | None,
) -> tuple[ConstitutionFinding, ...]:
    if raw is None:
        return ()
    if isinstance(raw, Mapping):
        raw = [raw]
    if isinstance(raw, str):
        parsed = parse_constitution_names(raw)
        names = (*parsed.values, *parsed.unknown)
        return tuple(
            ConstitutionFinding(
                name=name,
                source=CONSTITUTION_SOURCE_DEVICE,
                algorithm=default_algorithm,
                version=default_version,
            )
            for name in _dedupe(names)
        )
    if not isinstance(raw, Sequence) or isinstance(raw, (bytes, bytearray)):
        raise SchemaError("constitution_names must be text or a list")
    findings: list[ConstitutionFinding] = []
    for item in raw:
        if isinstance(item, str):
            parsed = parse_constitution_names(item)
            names = (*parsed.values, *parsed.unknown)
            findings.extend(
                ConstitutionFinding(
                    name=name,
                    source=CONSTITUTION_SOURCE_DEVICE,
                    algorithm=default_algorithm,
                    version=default_version,
                )
                for name in _dedupe(names)
            )
            continue
        if not isinstance(item, Mapping):
            raise SchemaError("constitution_names items must be text or objects")
        name = _first_text(item, ("name", "constitution", "tag"))
        if name is None:
            raise SchemaError("constitution finding is missing a name")
        source = str(item.get("source", CONSTITUTION_SOURCE_DEVICE)).strip().upper()
        findings.append(
            ConstitutionFinding(
                name=name,
                source=source,
                confidence=item.get("confidence"),
                algorithm=item.get("algorithm", default_algorithm),
                version=item.get("version", default_version),
            )
        )
    return tuple(findings)


def parse_tcm_exam(payload: Mapping[str, Any]) -> TCMExamV1:
    """Parse a device/service exam event directly, with no CSV round trip."""
    if not isinstance(payload, Mapping):
        raise SchemaError("TCM exam payload must be a mapping")

    tongue = _nested_mapping(payload, "tongue")
    face = _nested_mapping(payload, "face")
    device = _nested_mapping(payload, "device")

    measured_at = parse_timestamp(
        payload.get("measured_at", payload.get("timestamp", payload.get("ts")))
    )
    received_raw = payload.get("received_at", payload.get("ingest_at"))
    received_at = (
        measured_at if received_raw in (None, "") else parse_timestamp(received_raw)
    )

    session_id = _first_text(payload, ("session_id", "session"))
    subject_ref = _first_text(payload, ("subject_ref", "subject", "subject_id", "user_id"))
    device_ref = _first_text(payload, ("device_ref", "device_id"))
    if device_ref is None and isinstance(payload.get("device"), str):
        device_ref = _first_text(payload, ("device",))
    if session_id is None or subject_ref is None or device_ref is None:
        raise SchemaError("exam payload needs session_id, subject_ref and device_ref")

    device_algorithm = _first_text(
        payload, ("device_algorithm", "algorithm")
    ) or _first_text(device, ("algorithm", "device_algorithm"))
    vocabulary_version = _first_text(
        payload, ("vocabulary_version", "vocab_version")
    ) or _first_text(device, ("vocabulary_version", "vocab_version", "version"))

    exam_id = _first_text(payload, ("exam_id", "record_id", "message_id"))
    if exam_id is None:
        exam_id = f"{session_id}@{measured_at.isoformat()}"

    raw_provenance = payload.get("provenance")
    if isinstance(raw_provenance, ProvenanceV1):
        provenance = raw_provenance
    elif isinstance(raw_provenance, Mapping):
        provenance = ProvenanceV1.from_dict(raw_provenance)
    else:
        provenance = ProvenanceV1(
            adapter="tcm-exam",
            adapter_version=TCM_EXAM_SCHEMA_VERSION,
            source_record=payload.get("message_id"),
            data_license=str(payload.get("data_license", "PRIVATE_NOT_FOR_COMMIT")),
            notes=("parsed directly from a device/service exam payload",),
        )

    findings = _constitution_findings(
        _raw_constitution_value(payload),
        default_algorithm=device_algorithm,
        default_version=vocabulary_version,
    )

    return TCMExamV1(
        exam_id=exam_id,
        session_id=session_id,
        subject_ref=subject_ref,
        device_ref=device_ref,
        source=str(payload.get("source", SOURCE_DEVICE)).strip().lower(),
        measured_at=measured_at,
        received_at=received_at,
        tongue_body_color=_first_text(
            payload, ("tongue_body_color", "body_color")
        ) or _first_text(tongue, ("body_color", "tongue_body_color", "color")),
        tongue_shape=_first_text(payload, ("tongue_shape", "shape"))
        or _first_text(tongue, ("shape", "tongue_shape")),
        coating_color=_first_text(
            payload, ("coating_color", "tongue_coating_color")
        ) or _first_text(tongue, ("coating_color", "color")),
        coating_texture=_first_text(
            payload, ("coating_texture", "tongue_coating_texture")
        ) or _first_text(tongue, ("coating_texture", "texture")),
        face_color=_first_text(payload, ("face_color",))
        or _first_text(face, ("color", "face_color")),
        complexion=_first_text(payload, ("complexion", "qi_color"))
        or _first_text(face, ("complexion", "qi_color")),
        eye_description=_first_text(payload, ("eye_description", "eyes", "eye"))
        or _first_text(face, ("eye_description", "eyes", "eye")),
        constitution_names=findings,
        device_algorithm=device_algorithm,
        vocabulary_version=vocabulary_version,
        quality=_first_text(payload, ("quality",)) or "",
        provenance=provenance,
    )


@dataclass(frozen=True)
class KnowledgeEntry:
    """One reviewed wellness knowledge entry (never a prescription)."""

    topic_id: str
    title: str
    text: str
    source: str
    license: str
    version: str
    review_status: str
    reviewed_by: str
    constitution_tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "constitution_tags",
            _dedupe(
                str(tag).strip() for tag in self.constitution_tags if str(tag).strip()
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic_id": self.topic_id,
            "title": self.title,
            "text": self.text,
            "source": self.source,
            "license": self.license,
            "version": self.version,
            "review_status": self.review_status,
            "reviewed_by": self.reviewed_by,
            "constitution_tags": list(self.constitution_tags),
        }


_KNOWLEDGE_REQUIRED = (
    "topic_id",
    "title",
    "text",
    "source",
    "license",
    "version",
    "review_status",
    "reviewed_by",
)


def _tags(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return _dedupe(tag for tag in _tokens(value))
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return _dedupe(str(tag).strip() for tag in value if str(tag).strip())
    return ()


class KnowledgeRegistry:
    """Fail-closed reviewed knowledge registry.

    An entry is displayable only when every provenance field is present and
    ``review_status`` is ``APPROVED``. Malformed / unreviewed entries are kept
    in ``excluded`` for audit and are never served.
    """

    def __init__(
        self,
        entries: Sequence[KnowledgeEntry] = (),
        excluded: Sequence[str] = (),
    ) -> None:
        self.entries = tuple(entries)
        self.excluded = tuple(excluded)

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    @classmethod
    def from_json(cls, source: str | Path | Mapping[str, Any]) -> KnowledgeRegistry:
        """Load from a JSON path or an already-decoded mapping."""
        if isinstance(source, Mapping):
            data: Any = source
        elif isinstance(source, (str, Path)):
            data = json.loads(Path(source).read_text(encoding="utf-8"))
        else:
            raise TypeError("KnowledgeRegistry.from_json expects a path or mapping")
        return cls._from_data(data)

    @classmethod
    def _from_data(cls, data: Any) -> KnowledgeRegistry:
        if isinstance(data, Mapping):
            if "entries" in data:
                raw_entries = data.get("entries")
            elif "topic_id" in data:
                raw_entries = [data]
            else:
                raw_entries = []
        elif isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
            raw_entries = list(data)
        else:
            raw_entries = []

        approved: list[KnowledgeEntry] = []
        excluded: list[str] = []
        if not isinstance(raw_entries, Sequence) or isinstance(
            raw_entries, (str, bytes, bytearray)
        ):
            return cls((), (f"entries: expected a list, got {type(raw_entries).__name__}",))

        for index, raw in enumerate(raw_entries):
            if not isinstance(raw, Mapping):
                excluded.append(f"entry[{index}]: expected an object")
                continue
            missing = [
                name
                for name in _KNOWLEDGE_REQUIRED
                if not str(raw.get(name, "")).strip()
            ]
            topic_id = str(raw.get("topic_id", f"entry[{index}]"))
            if missing:
                excluded.append(f"{topic_id}: missing {','.join(missing)}")
                continue
            review_status = str(raw["review_status"]).strip().upper()
            if review_status != REVIEW_APPROVED:
                excluded.append(
                    f"{topic_id}: review_status {raw['review_status']!r} is not APPROVED"
                )
                continue
            approved.append(
                KnowledgeEntry(
                    topic_id=str(raw["topic_id"]).strip(),
                    title=str(raw["title"]).strip(),
                    text=str(raw["text"]).strip(),
                    source=str(raw["source"]).strip(),
                    license=str(raw["license"]).strip(),
                    version=str(raw["version"]).strip(),
                    review_status=REVIEW_APPROVED,
                    reviewed_by=str(raw["reviewed_by"]).strip(),
                    constitution_tags=_tags(raw.get("constitution_tags")),
                )
            )
        return cls(approved, excluded)


@dataclass(frozen=True)
class TCMSidecarResult:
    """Structured findings plus reviewed knowledge, never a diagnosis."""

    status: str
    constitution_findings: list[dict[str, Any]] = field(default_factory=list)
    knowledge: list[dict[str, Any]] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def __post_init__(self) -> None:
        if self.status not in SIDECAR_STATUSES:
            raise ValueError(f"unknown sidecar status: {self.status!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "constitution_findings": [dict(item) for item in self.constitution_findings],
            "knowledge": [dict(item) for item in self.knowledge],
            "provenance": dict(self.provenance),
            "reason": self.reason,
        }


def _finding_dict(finding: ConstitutionFinding) -> dict[str, Any]:
    return {
        **finding.to_dict(),
        "source_tag": f"{finding.source}_FINDING",
        "is_clinical_truth": False,
    }


def _matches_device_constitution(
    entry: KnowledgeEntry, device_names: Sequence[str]
) -> list[str]:
    tags = {
        canonical_constitution_name(tag)
        for tag in entry.constitution_tags
    }
    return [name for name in device_names if name in tags]


def build_tcm_sidecar(
    exam: TCMExamV1,
    registry: KnowledgeRegistry | None = None,
) -> TCMSidecarResult:
    """Map device-provided constitution names to approved wellness knowledge.

    Only ``source=DEVICE`` findings are eligible for knowledge matching. Project
    model or questionnaire findings are echoed with their own source, but they
    never trigger knowledge and are never presented as clinical truth. Tongue
    and face signs alone never produce a constitution conclusion.
    """
    if not isinstance(exam, TCMExamV1):
        raise TypeError("build_tcm_sidecar expects a TCMExamV1")
    if registry is None:
        registry = KnowledgeRegistry()
    if not isinstance(registry, KnowledgeRegistry):
        raise TypeError("registry must be a KnowledgeRegistry")

    findings = [_finding_dict(finding) for finding in exam.constitution_names]
    device_names = _dedupe(
        canonical_constitution_name(finding.name)
        for finding in exam.constitution_names
        if finding.source == CONSTITUTION_SOURCE_DEVICE
    )
    provenance = {
        "exam_id": exam.exam_id,
        "session_id": exam.session_id,
        "device_ref": exam.device_ref,
        "source": exam.source,
        "device_algorithm": exam.device_algorithm,
        "vocabulary_version": exam.vocabulary_version,
        "constitution_source": CONSTITUTION_SOURCE_DEVICE,
        "knowledge_registry_entries": len(registry),
        "knowledge_entries_matched": 0,
        "device_constitutions": list(device_names),
        "matched_constitutions": [],
        "not_a_diagnosis": True,
        "not_a_prescription": True,
        "exam_provenance": exam.provenance.to_dict() if exam.provenance else None,
    }

    if not device_names:
        return TCMSidecarResult(
            status=STATUS_UNKNOWN,
            constitution_findings=findings,
            provenance=provenance,
            reason=(
                "device did not provide a constitution result; tongue/face signs are "
                "not used to infer a personal constitution"
            ),
        )

    if not registry.entries:
        return TCMSidecarResult(
            status=STATUS_ABSTAIN,
            constitution_findings=findings,
            provenance=provenance,
            reason="no approved knowledge entries; nothing is displayed",
        )

    knowledge: dict[str, dict[str, Any]] = {}
    matched_names: set[str] = set()
    for name in device_names:
        for entry in registry.entries:
            if name not in _matches_device_constitution(entry, (name,)):
                continue
            matched_names.add(name)
            record = knowledge.get(entry.topic_id)
            if record is None:
                record = {
                    **entry.to_dict(),
                    "matched_constitution": [],
                    "source_tag": "KNOWLEDGE",
                }
                knowledge[entry.topic_id] = record
            if name not in record["matched_constitution"]:
                record["matched_constitution"].append(name)

    provenance["knowledge_entries_matched"] = len(knowledge)
    provenance["matched_constitutions"] = sorted(matched_names)

    if not knowledge:
        return TCMSidecarResult(
            status=STATUS_ABSTAIN,
            constitution_findings=findings,
            provenance=provenance,
            reason=(
                "no approved knowledge entry matches the device constitution "
                "result; nothing is displayed"
            ),
        )

    status = STATUS_OK if matched_names == set(device_names) else STATUS_PARTIAL
    reason = (
        "reviewed wellness knowledge matched device-provided constitution labels"
        if status == STATUS_OK
        else "only part of the device constitution labels have reviewed knowledge"
    )
    return TCMSidecarResult(
        status=status,
        constitution_findings=findings,
        knowledge=list(knowledge.values()),
        provenance=provenance,
        reason=reason,
    )


__all__ = [
    "COATING_COLOR_SYNONYMS",
    "COATING_TEXTURE_SYNONYMS",
    "COMPLEXION_SYNONYMS",
    "CONSTITUTION_SOURCES",
    "CONSTITUTION_SOURCE_DEVICE",
    "CONSTITUTION_SOURCE_PROJECT_MODEL",
    "CONSTITUTION_SOURCE_QUESTIONNAIRE",
    "CONSTITUTION_SYNONYMS",
    "FACE_COLOR_SYNONYMS",
    "KnowledgeEntry",
    "KnowledgeRegistry",
    "PARSE_STATUSES",
    "ParsedValues",
    "REVIEW_APPROVED",
    "SIDECAR_STATUSES",
    "STATUS_ABSTAIN",
    "STATUS_OK",
    "STATUS_PARTIAL",
    "STATUS_UNKNOWN",
    "TCMExamV1",
    "TCM_EXAM_SCHEMA_VERSION",
    "TCMSidecarResult",
    "TONGUE_BODY_COLOR_SYNONYMS",
    "TONGUE_SHAPE_SYNONYMS",
    "ConstitutionFinding",
    "build_tcm_sidecar",
    "canonical_constitution_name",
    "parse_constitution_names",
    "parse_exam_fields",
    "parse_multi_value",
    "parse_tcm_exam",
]
