"""Field-by-field audit of a real export against its reference rows (#102).

This module does not decide whether the data is good. It answers one question:
can every value in the reference training set and the reference quality report be
rebuilt from the raw records using only declared rules? Every field is reported
as ``match``, ``mismatch``, ``explained`` or ``not_recoverable``; nothing is
silently accepted and nothing is repaired.

``mismatch`` on a *value* is a mapping failure. ``mismatch`` on a *contract*
field (a label mask) is reported as an open item with the difference spelled out,
because a mask reflects a vendor rule that may not be published.
"""

from __future__ import annotations

import ast
import csv
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .real_export import (
    PHASE_POST,
    PHASE_PRE,
    QC_BP_DELTA_ANOMALY,
    QC_HR_CROSS_MODULE_MISMATCH,
    QC_IDENTITY_SHARED,
    QC_POINT_STREAM_EXCLUDED,
    QC_SNAPSHOT_ENDPOINT,
    REFERENCE_LABEL_KEYS,
    RealQCIssue,
    RealSession,
    RealSessionLabels,
    reconcile_reference_contract,
    session_feature_row,
    session_labels,
)

AUDIT_SCHEMA_VERSION = "physio-real-data-audit-v1"

VERDICT_MATCH = "match"
VERDICT_MISMATCH = "mismatch"
VERDICT_NOT_RECOVERABLE = "not_recoverable"
VERDICT_EXPLAINED = "explained"
#: A reconstruction the reference report has no counterpart for. It is reported
#: as an observation and never counted as agreement.
VERDICT_OBSERVATION = "observation"

STATUS_VERIFIED = "VERIFIED"
STATUS_VERIFIED_WITH_EXCEPTIONS = "VERIFIED_WITH_EXCEPTIONS"
STATUS_FAILED = "FAILED"

#: Fields whose mismatch is a contract question rather than a value failure.
CONTRACT_FIELDS: frozenset[str] = frozenset({"label_masks"})

TOLERANCE = 1e-6

ENDPOINT_FIELD_MAP: dict[str, tuple[str, str]] = {
    "sbp": ("bp", "systolic_bp"),
    "dbp": ("bp", "diastolic_bp"),
    "hr": ("health", "heart_rate"),
    "spo2": ("health", "spo2"),
    "micro": ("health", "microcirculation_index"),
    "rr": ("health", "respiration_rate"),
    "fatigue": ("health", "fatigue_index"),
    "vendor_score": ("health", "vendor_health_index"),
}

#: Reference columns the provided de-identified extract does not carry. They are
#: reported as ``not_recoverable`` instead of being assumed.
EXTRACT_ABSENT_REFERENCE_COLUMNS: tuple[str, ...] = (
    "age",
    "age_group",
    "height_cm",
    "body_fat_pct",
    "muscle_kg",
    "protein_kg",
    "bone_mineral_kg",
    "skeletal_muscle_kg",
    "water_pct",
    "visceral_fat",
    "impedance",
    "body_bmr_kcal",
    "body_age",
    "body_score",
    "tongue_color",
    "moss_color",
    "tongue_shape",
    "moss_quality",
    "lip_color",
    "sublingual_vein",
    "eye_circle",
    "face_color",
    "face_luster",
    "face_shen",
    "main_face_color",
    "bodyfluid",
    "brain_deviate_count",
    "brain_deviate_index",
    "brain_deviate_second",
    "brain_lowbeta_avg",
    "brain_theta_avg",
    "brain_gamma_avg",
    "brain_highbeta_avg",
    "brain_alpha_avg",
    "brain_mindfulness_avg",
    "brain_meditation_max",
    "brain_meditation_min",
    "brain_fatigue",
    "brain_level",
    "brain_delta_avg",
    "brain_decompression",
    "brain_zoned_in_pct",
    "brain_relaxed_pct",
    "brain_neutral_pct",
    "brain_active_pct",
    "feature_route",
    "missing_features",
    "qc_note",
)

#: Reference columns that must equal a declared constant of the frozen contract.
DECLARED_CONSTANT_COLUMNS: dict[str, str] = {
    "protocol_version": "UNRECORDED",
    "pipeline_version": "train-pipeline-v1",
    "score_version": "pri-v1-pending",
    "label_version": "label-v1",
    "y_pri": "",
    "y_pri_available": "0",
}

#: Reference columns that are expected to stay empty in this extract: they carry
#: no independent verification and must not be fabricated.
RESPONSE_HASH_COLUMNS: tuple[str, ...] = ("response_hash",)

#: Declared rule that reproduces the reference ``qc_status`` column.
QC_STATUS_COMPLETE = "COMPLETE"
QC_STATUS_INCOMPLETE = "INCOMPLETE"

#: Reference columns derived from the session window rather than a raw field.
DATE_COLUMNS: tuple[str, ...] = ("date", "batch_date")


@dataclass(frozen=True)
class AuditFinding:
    """One field-level comparison between the reconstruction and the reference."""

    session_id: str
    field: str
    verdict: str
    reference: str
    reconstructed: str
    note: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "session_id": self.session_id,
            "field": self.field,
            "verdict": self.verdict,
            "reference": self.reference,
            "reconstructed": self.reconstructed,
            "note": self.note,
        }


@dataclass(frozen=True)
class QCCrossCheck:
    """One comparison against the reference quality report."""

    session_id: str
    item: str
    verdict: str
    reference: str
    reconstructed: str
    note: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "session_id": self.session_id,
            "item": self.item,
            "verdict": self.verdict,
            "reference": self.reference,
            "reconstructed": self.reconstructed,
            "note": self.note,
        }


@dataclass(frozen=True)
class RealDataAudit:
    """The complete field-level audit result."""

    schema: str
    status: str
    sessions: int
    records: int
    findings: tuple[AuditFinding, ...]
    qc_cross_checks: tuple[QCCrossCheck, ...]
    issues: tuple[RealQCIssue, ...]
    open_items: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.status != STATUS_FAILED

    def verdict_counts(self) -> dict[str, int]:
        counts = {
            VERDICT_MATCH: 0,
            VERDICT_MISMATCH: 0,
            VERDICT_EXPLAINED: 0,
            VERDICT_NOT_RECOVERABLE: 0,
            VERDICT_OBSERVATION: 0,
        }
        for finding in self.findings:
            counts[finding.verdict] = counts.get(finding.verdict, 0) + 1
        return counts

    def qc_verdict_counts(self) -> dict[str, int]:
        """Verdicts of the quality-report cross-checks, observations included."""
        counts = {VERDICT_MATCH: 0, VERDICT_MISMATCH: 0, VERDICT_OBSERVATION: 0}
        for check in self.qc_cross_checks:
            counts[check.verdict] = counts.get(check.verdict, 0) + 1
        return counts

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "status": self.status,
            "sessions": self.sessions,
            "records": self.records,
            "verdicts": self.verdict_counts(),
            "qc_verdicts": self.qc_verdict_counts(),
            "open_items": list(self.open_items),
            "findings": [finding.to_dict() for finding in self.findings],
            "qc_cross_checks": [check.to_dict() for check in self.qc_cross_checks],
            "issues": [issue.to_dict() for issue in self.issues],
            "notes": list(self.notes),
        }


def load_reference_rows(path: str | Path) -> tuple[dict[str, str], ...]:
    """Read a reference CSV (training set or quality report) into plain rows."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"reference rows not found: {file_path}")
    with file_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return tuple(
            {key: ("" if value is None else str(value)) for key, value in row.items()}
            for row in csv.DictReader(handle)
        )


def _number(text: str | None) -> float | None:
    if text is None:
        return None
    stripped = str(text).strip()
    if not stripped:
        return None
    try:
        value = float(stripped)
    except ValueError:
        return None
    return value


def _close(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return abs(float(left) - float(right)) <= TOLERANCE * max(1.0, abs(float(right)))


def _as_number(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return _number(str(value))


def _parse_list(text: str) -> tuple[str, ...]:
    stripped = str(text).strip()
    if not stripped:
        return ()
    try:
        parsed = ast.literal_eval(stripped)
    except (ValueError, SyntaxError):
        return tuple(part.strip() for part in stripped.strip("[]").split(",") if part.strip())
    if isinstance(parsed, (list, tuple)):
        return tuple(str(item) for item in parsed)
    return (str(parsed),)


def _split_keys(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(text).split(",") if part.strip())


def _endpoint_value(session: RealSession, module: str, metric: str, phase: str) -> float | None:
    endpoint = session.endpoints.get(module, {}).get(phase)
    if endpoint is None:
        return None
    measurement = endpoint.measurement(metric)
    if measurement is None or not measurement.valid or measurement.value is None:
        return None
    return float(measurement.value)


def _not_recoverable(session_id: str, field: str, reference: str, note: str) -> AuditFinding:
    return AuditFinding(session_id, field, VERDICT_NOT_RECOVERABLE, reference, "", note)


def _compare(
    session_id: str,
    name: str,
    reference: str,
    reconstructed: object,
    *,
    note: str = "",
) -> AuditFinding:
    reference_number = _number(reference)
    reconstructed_number = _as_number(reconstructed)
    if reference_number is not None or reconstructed_number is not None:
        if _close(reconstructed_number, reference_number):
            return AuditFinding(
                session_id, name, VERDICT_MATCH, reference, str(reconstructed), note
            )
        return AuditFinding(
            session_id, name, VERDICT_MISMATCH, reference, str(reconstructed), note
        )
    if str(reference).strip() == str(reconstructed).strip():
        return AuditFinding(session_id, name, VERDICT_MATCH, reference, str(reconstructed), note)
    return AuditFinding(session_id, name, VERDICT_MISMATCH, reference, str(reconstructed), note)


def _expected_reference_qc_status(qc_row: Mapping[str, str] | None) -> str:
    """What the reference qc_status should be, given its own 缺失字段 column."""
    missing = str((qc_row or {}).get("缺失字段", "")).strip()
    return QC_STATUS_INCOMPLETE if missing not in ("", "none") else QC_STATUS_COMPLETE


def _audit_session(
    session: RealSession,
    reference: Mapping[str, str],
    qc_row: Mapping[str, str] | None,
) -> tuple[list[AuditFinding], list[QCCrossCheck]]:
    session_id = session.session_id
    findings: list[AuditFinding] = []
    checks: list[QCCrossCheck] = []
    labels = session_labels(session)
    label_codes = tuple(sorted({issue.code for issue in labels.issues}))
    row = session_feature_row(session)
    row_values = dict(zip(row.feature_names, row.values, strict=True))
    row_valid = dict(zip(row.feature_names, row.valid_mask, strict=True))

    findings.append(_compare(session_id, "session_id", reference.get("session_id", ""), session_id))

    member_hash = session.member_ref.removeprefix("member:")
    user_hash = reference.get("user_id_hash", "")
    findings.append(
        AuditFinding(
            session_id,
            "user_id_hash",
            VERDICT_MATCH if user_hash.startswith(member_hash) else VERDICT_MISMATCH,
            user_hash,
            f"{member_hash} (prefix relation)",
            "the reference hash extends the raw member hash; no identity is inferred",
        )
    )

    reference_records = set(_split_keys(reference.get("source_record_ids", "")))
    raw_records = {record.record_id for record in session.records}
    if reference_records == raw_records:
        provenance_note = "provenance covers exactly the provided records"
    else:
        provenance_note = (
            "raw-only=" + ",".join(sorted(raw_records - reference_records))
            + " reference-only="
            + ",".join(sorted(reference_records - raw_records))
        )
    findings.append(
        AuditFinding(
            session_id,
            "source_record_ids",
            VERDICT_MATCH if reference_records == raw_records else VERDICT_MISMATCH,
            f"{len(reference_records)} ids",
            f"{len(raw_records)} ids",
            provenance_note,
        )
    )

    for suffix, (module, metric) in ENDPOINT_FIELD_MAP.items():
        for phase in ("pre", "post"):
            name = f"{suffix}_{phase}"
            reference_text = reference.get(name, "")
            if reference_text == "":
                findings.append(
                    _not_recoverable(session_id, name, reference_text, "absent in reference")
                )
                continue
            reconstructed = _endpoint_value(session, module, metric, phase)
            if reconstructed is not None:
                reconstructed = round(reconstructed * (0.01 if metric == "spo2" else 1.0), 6)
            if reconstructed is None:
                findings.append(
                    AuditFinding(
                        session_id,
                        name,
                        VERDICT_EXPLAINED,
                        reference_text,
                        "null",
                        "the raw endpoint has no usable value; the reference value must be "
                        "imputed or come from outside the provided records",
                    )
                )
                continue
            findings.append(
                _compare(
                    session_id,
                    name,
                    reference_text,
                    reconstructed,
                    note=f"module={module}, metric={metric}",
                )
            )

    for name in ("weight_kg", "bmi"):
        reference_text = reference.get(name, "")
        if reference_text == "":
            continue
        reconstructed = row_values.get(name) if row_valid.get(name) else None
        if reconstructed is None:
            findings.append(
                AuditFinding(
                    session_id,
                    name,
                    VERDICT_EXPLAINED,
                    reference_text,
                    "null",
                    "no 体脂 record in this extract; the reference value is imputed or comes "
                    "from outside the provided records",
                )
            )
        else:
            findings.append(
                _compare(session_id, name, reference_text, reconstructed, note="module=body")
            )

    reference_mask = set(_split_keys(reference.get("label_masks", "")))
    contract = reconcile_reference_contract(labels, reference_mask)
    reconstructed_mask = {key for key in REFERENCE_LABEL_KEYS if contract.mask.get(key)}
    findings.append(
        _compare(
            session_id,
            "label_masks",
            ",".join(sorted(reference_mask)),
            ",".join(sorted(reconstructed_mask)),
            note="compared on the declared reference label contract after ABSTAIN reconciliation",
        )
    )
    if contract.abstained:
        findings.append(
            AuditFinding(
                session_id,
                "abstained_labels",
                VERDICT_EXPLAINED,
                ",".join(sorted(reference_mask)),
                ",".join(sorted(contract.abstained)),
                "the raw records can produce "
                + ",".join(sorted(contract.abstained))
                + " but the reference contract excludes them and no exclusion rule is declared, "
                "so the labels are not admitted (规则缺项一律 ABSTAIN)",
            )
        )

    for key in REFERENCE_LABEL_KEYS:
        reference_text = reference.get(key, "")
        if reference_text == "":
            findings.append(
                _not_recoverable(session_id, key, reference_text, "absent in reference")
            )
            continue
        reconstructed = labels.values.get(key)
        if reconstructed is None:
            verdict = VERDICT_MATCH if key not in reference_mask else VERDICT_MISMATCH
            findings.append(
                AuditFinding(
                    session_id,
                    key,
                    verdict,
                    reference_text,
                    "null",
                    f"rejected: {labels.reasons.get(key, 'no reason recorded')}",
                )
            )
            continue
        findings.append(
            AuditFinding(
                session_id,
                key,
                VERDICT_MATCH
                if _close(reconstructed, _number(reference_text))
                else VERDICT_MISMATCH,
                reference_text,
                str(reconstructed),
                "rebuilt from the raw Pre and Post endpoints",
            )
        )

    derived_outside = sorted(key for key in labels.values if key not in REFERENCE_LABEL_KEYS)
    if derived_outside:
        findings.append(
            AuditFinding(
                session_id,
                "derived_outside_contract",
                VERDICT_EXPLAINED,
                "not in reference label_masks",
                ",".join(derived_outside),
                "a delta is derivable but sits outside the declared reference label contract",
            )
        )

    coded_gender = session.coded_values.get("gender", "")
    findings.append(
        AuditFinding(
            session_id,
            "gender",
            VERDICT_MATCH
            if reference.get("gender", "") == coded_gender
            else VERDICT_MISMATCH,
            reference.get("gender", ""),
            coded_gender,
            "raw export code preserved verbatim; the code meaning is unverified",
        )
    )

    reference_constitution = _parse_list(reference.get("constitution_names", ""))
    raw_constitution = session.tcm_values.get("constitution")
    reconstructed_constitution = () if raw_constitution is None else (raw_constitution,)
    findings.append(
        _compare(
            session_id,
            "constitution_names",
            str(list(reference_constitution)),
            str(list(reconstructed_constitution)),
            note="module=tcm",
        )
    )

    reference_brain = reference.get("brain_record_id", "")
    if reference_brain == "":
        findings.append(
            _not_recoverable(
                session_id, "brain_record_id", reference_brain, "no brain record in this session"
            )
        )
    elif reference_brain == session.eeg_record_id:
        for name, key in (
            ("brain_overall", "brain_overall"),
            ("brain_avg_meditation", "brain_avg_meditation"),
            ("brain_duration_s", "brain_duration"),
            ("brain_sample_points", "sample_points"),
        ):
            findings.append(
                _compare(
                    session_id, name, reference.get(name, ""), session.eeg_values.get(key),
                    note="module=eeg",
                )
            )
    else:
        findings.append(
            AuditFinding(
                session_id,
                "brain_record_id",
                VERDICT_NOT_RECOVERABLE,
                reference_brain,
                session.eeg_record_id,
                "the reference selects a different 脑波 record; the selection rule is not "
                "recoverable from this extract",
            )
        )

    for column in EXTRACT_ABSENT_REFERENCE_COLUMNS:
        reference_text = reference.get(column, "")
        if reference_text != "":
            findings.append(
                _not_recoverable(
                    session_id,
                    column,
                    reference_text,
                    "column is not present in the provided de-identified raw extract",
                )
            )

    for column, expected in DECLARED_CONSTANT_COLUMNS.items():
        findings.append(
            _compare(
                session_id,
                column,
                reference.get(column, ""),
                expected,
                note="declared constant of the frozen export contract",
            )
        )

    for column in RESPONSE_HASH_COLUMNS:
        findings.append(
            _not_recoverable(
                session_id,
                column,
                reference.get(column, ""),
                "the reference row carries a response hash whose construction is not declared; "
                "the reference cannot be independently verified end to end",
            )
        )

    findings.append(
        AuditFinding(
            session_id,
            "qc_status",
            VERDICT_MATCH
            if reference.get("qc_status", "") == _expected_reference_qc_status(qc_row)
            else VERDICT_MISMATCH,
            reference.get("qc_status", ""),
            _expected_reference_qc_status(qc_row),
            "the reference qc_status is checked against its own 缺失字段 column",
        )
    )

    window_date = session.window_start.strftime("%Y-%m-%d")
    for column in DATE_COLUMNS:
        findings.append(
            _compare(
                session_id,
                column,
                reference.get(column, ""),
                window_date,
                note="session window start date",
            )
        )

    checks.extend(_qc_cross_checks(session, labels, qc_row, label_codes))
    return findings, checks


def _leading_int(text: str) -> int | None:
    """Read the leading integer of a reference cell such as ``0(仅快照)``."""
    digits = ""
    for character in str(text).strip():
        if character.isdigit():
            digits += character
        elif digits:
            break
        elif not character.isspace():
            return None
    return int(digits) if digits else None


def _qc_cross_checks(
    session: RealSession,
    labels: RealSessionLabels,
    qc_row: Mapping[str, str] | None,
    extra_codes: tuple[str, ...] = (),
) -> tuple[QCCrossCheck, ...]:
    session_id = session.session_id
    completed = sum(1 for record in session.records if record.value("is_complete") == "是")
    codes = session.issue_codes() + extra_codes
    qc_mask = set(_split_keys((qc_row or {}).get("label_masks", "")))
    contract_view = reconcile_reference_contract(labels, qc_mask)
    reference_admissible = {key for key in REFERENCE_LABEL_KEYS if contract_view.mask.get(key)}
    rr_usable = False
    for phase in (PHASE_PRE, PHASE_POST):
        endpoint = session.endpoints.get("health", {}).get(phase)
        measurement = endpoint.measurement("respiration_rate") if endpoint else None
        if measurement is not None and measurement.valid:
            rr_usable = True
    window = (qc_row or {}).get("窗口", "")
    reconstructed_window = (
        f"{session.endpoints['bp']['pre'].measured_at:%Y-%m-%d %H:%M:%S}~"
        f"{session.endpoints['bp']['post'].measured_at:%Y-%m-%d %H:%M:%S}"
        if "bp" in session.endpoints
        else ""
    )
    missing_fields = set(_split_keys((qc_row or {}).get("缺失字段", "")))
    has_body = session.module_present("body")
    has_tcm = session.module_present("tcm")
    has_eeg = session.module_present("eeg")

    def check(
        item: str,
        reference: str,
        reconstructed: str,
        verdict: str = VERDICT_MATCH,
        note: str = "",
    ) -> QCCrossCheck:
        return QCCrossCheck(session_id, item, verdict, reference, reconstructed, note)

    qc_cell = (qc_row or {}).get("健康完成态数", "")
    checks = [
        check(
            "health_completion_count",
            qc_cell,
            str(completed),
            VERDICT_MATCH if _leading_int(qc_cell) == completed else VERDICT_MISMATCH,
            "raw is_complete=是 records counted directly from the extract",
        ),
        check(
            "session_window",
            window,
            reconstructed_window,
            VERDICT_MATCH if window == reconstructed_window else VERDICT_MISMATCH,
            "reference window equals the Pre/Post blood-pressure span",
        ),
        check(
            "missing_field_body",
            "bodyfat" if "bodyfat" in missing_fields else "present",
            "missing" if not has_body else "present",
            VERDICT_MATCH if ("bodyfat" in missing_fields) == (not has_body) else VERDICT_MISMATCH,
            "体脂 record presence",
        ),
        check(
            "missing_field_tcm",
            "tongue" if "tongue" in missing_fields else "present",
            "missing" if not has_tcm else "present",
            VERDICT_MATCH if ("tongue" in missing_fields) == (not has_tcm) else VERDICT_MISMATCH,
            "舌面 record presence",
        ),
        check(
            "missing_field_eeg",
            "brain" if "brain" in missing_fields else "present",
            "missing" if not has_eeg else "present",
            VERDICT_MATCH if ("brain" in missing_fields) == (not has_eeg) else VERDICT_MISMATCH,
            "脑波 record presence",
        ),
        check(
            "missing_field_rr",
            "rr_pre/rr_post" if {"rr_pre", "rr_post"} & missing_fields else "present",
            "missing" if not rr_usable else "present",
            VERDICT_MATCH
            if bool({"rr_pre", "rr_post"} & missing_fields) == (not rr_usable)
            else VERDICT_MISMATCH,
            "respiration-rate endpoint usability",
        ),
        check(
            "bp_anomaly",
            (qc_row or {}).get("异常", ""),
            "血压异常" if QC_BP_DELTA_ANOMALY in codes else "无",
            VERDICT_MATCH
            if (QC_BP_DELTA_ANOMALY in codes) == ("异常" in ((qc_row or {}).get("异常", "")))
            else VERDICT_MISMATCH,
            "declared guardrail reproduces the reference anomaly verdict",
        ),
        check(
            "reference_label_mask",
            (qc_row or {}).get("label_masks", ""),
            ",".join(sorted(reference_admissible)),
            VERDICT_MATCH
            if qc_mask == reference_admissible
            else VERDICT_MISMATCH,
            "quality report mask versus the reconstructed admissible labels",
        ),
        check(
            "shared_account_identity",
            (qc_row or {}).get("拒绝原因", ""),
            "shared account flagged" if QC_IDENTITY_SHARED in codes else "not flagged",
            VERDICT_OBSERVATION,
            "the reference QC does not separate member and subject identity; the "
            "reconstruction flags it independently",
        ),
        check(
            "snapshot_endpoints",
            "labels derived from snapshot records",
            "flagged" if QC_SNAPSHOT_ENDPOINT in codes else "not flagged",
            VERDICT_OBSERVATION,
            "Pre/Post endpoints use is_complete=否 records",
        ),
        check(
            "point_stream_excluded",
            "point-level brain rows excluded",
            "excluded" if QC_POINT_STREAM_EXCLUDED in codes else "absent",
            VERDICT_OBSERVATION,
            "脑波逐点 rows are never features or labels",
        ),
        check(
            "hr_cross_module",
            "not checked by the reference QC",
            "flagged" if QC_HR_CROSS_MODULE_MISMATCH in codes else "agree",
            VERDICT_OBSERVATION,
            "the reconstruction adds a cross-module heart-rate check the reference omits",
        ),
        check(
            "module_family_coverage",
            "not reported by the reference QC",
            ",".join(
                module
                for module in ("bp", "health", "body", "tcm", "eeg")
                if session.module_present(module)
            ),
            VERDICT_OBSERVATION,
            "module families the provided extract actually carries",
        ),
    ]
    return tuple(checks)


def _open_items(findings: Sequence[AuditFinding]) -> tuple[str, ...]:
    """Every unresolved difference, whether a value mismatch or an ABSTAIN."""
    items: list[str] = []
    for finding in findings:
        if finding.verdict == VERDICT_MISMATCH:
            items.append(
                f"{finding.session_id or 'export'}: {finding.field} — reconstructed "
                f"{finding.reconstructed!r} but the reference is {finding.reference!r}"
                + (f" ({finding.note})" if finding.note else "")
            )
        elif finding.field == "abstained_labels":
            items.append(
                f"{finding.session_id or 'export'}: abstained_labels — {finding.note}"
            )
    return tuple(items)


def audit_real_export(
    sessions: Sequence[RealSession],
    reference_rows: Sequence[Mapping[str, str]],
    qc_rows: Sequence[Mapping[str, str]] = (),
    *,
    notes: Sequence[str] = (),
) -> RealDataAudit:
    """Compare every reference row against the reconstruction from raw records."""
    if not sessions:
        raise ValueError("audit requires at least one session")
    by_id = {row.get("session_id", ""): row for row in reference_rows}
    qc_by_id = {row.get("session_id", ""): row for row in qc_rows}
    findings: list[AuditFinding] = []
    checks: list[QCCrossCheck] = []
    for session in sessions:
        reference = by_id.get(session.session_id)
        if reference is None:
            findings.append(
                _not_recoverable(session.session_id, "reference_row", "", "no reference row")
            )
            continue
        session_findings, session_checks = _audit_session(
            session, reference, qc_by_id.get(session.session_id)
        )
        findings.extend(session_findings)
        checks.extend(session_checks)

    unknown = sorted(set(by_id) - {session.session_id for session in sessions})
    if unknown:
        findings.append(
            _not_recoverable("", "reference_only_sessions", ",".join(unknown), "no raw records")
        )

    mismatches = [finding for finding in findings if finding.verdict == VERDICT_MISMATCH]
    value_failures = [
        finding for finding in mismatches if finding.field not in CONTRACT_FIELDS
    ]
    abstained = [finding for finding in findings if finding.field == "abstained_labels"]
    if value_failures:
        status = STATUS_FAILED
    elif mismatches or abstained or _open_items(findings):
        status = STATUS_VERIFIED_WITH_EXCEPTIONS
    else:
        status = STATUS_VERIFIED

    return RealDataAudit(
        schema=AUDIT_SCHEMA_VERSION,
        status=status,
        sessions=len(sessions),
        records=sum(len(session.records) for session in sessions),
        findings=tuple(findings),
        qc_cross_checks=tuple(checks),
        issues=tuple(issue for session in sessions for issue in session.issues)
        + tuple(issue for session in sessions for issue in session_labels(session).issues),
        open_items=_open_items(findings),
        notes=tuple(notes),
    )


def summarize_findings(findings: Iterable[AuditFinding]) -> dict[str, tuple[str, ...]]:
    """Group ``session:field`` labels by verdict for a compact report."""
    grouped: dict[str, list[str]] = {}
    for finding in findings:
        grouped.setdefault(finding.verdict, []).append(f"{finding.session_id}:{finding.field}")
    return {verdict: tuple(items) for verdict, items in grouped.items()}
