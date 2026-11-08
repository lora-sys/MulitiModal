"""Unified engineering acceptance for the realtime physiology main line (#95).

This module checks the cross-cutting invariants the five sub-issues share. It
runs entirely on synthetic fixtures. A PASS here means the engineering
contracts hold; it is not evidence of real-device behaviour, model validity or
clinical effect.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .adapters import (
    NOT_CONNECTED,
    DeviceEventAdapter,
    OfflineSessionAdapter,
    ServiceEventAdapter,
)
from .artifacts import ModelArtifact
from .dataset import (
    IDENTITY_TRUSTED,
    FeatureRow,
    FoldImputer,
    admit_delta_targets,
    feature_row_from_window,
    make_subject_folds,
)
from .evaluation import STATUS_INSUFFICIENT, evaluate_regression
from .inference import (
    STATUS_ABSTAIN,
    STATUS_NOT_AVAILABLE,
    STATUS_OK,
    InferenceSession,
)
from .legacy_export import (
    ISSUE_PHASE_CONFLICT,
    ISSUE_SNAPSHOT_COMPLETED_MISMATCH,
    ExportFieldRegistry,
    ExportFieldSpec,
    import_legacy_rows,
)
from .metrics import (
    STATUS_INVALID_TYPE,
    STATUS_INVALID_VALUE,
    STATUS_OUT_OF_RANGE,
    STATUS_UNKNOWN_METRIC,
    STATUS_UNKNOWN_UNIT,
    STATUS_ZERO_UNKNOWN,
    route_measurement,
)
from .model import VALIDATION_VALIDATED
from .preprocessing import WindowScaler
from .programs import ProgramCapabilityCatalog
from .recommendation import (
    STATE_SAFE_STOP,
    MassageRecommendationService,
    UserPreference,
)
from .recommendation import (
    STATUS_ABSTAIN as REC_ABSTAIN,
)
from .recommendation import (
    STATUS_OK as REC_OK,
)
from .schema import (
    PHASE_DURING,
    PHASE_POST,
    PHASE_PRE,
    QUALITY_STALE,
    assemble_session,
)
from .stream import OnlineIngestor
from .tcm_exam import (
    STATUS_ABSTAIN as TCM_ABSTAIN,
)
from .tcm_exam import (
    STATUS_OK as TCM_OK,
)
from .tcm_exam import (
    STATUS_PARTIAL as TCM_PARTIAL,
)
from .tcm_exam import (
    KnowledgeRegistry,
    build_tcm_sidecar,
    parse_tcm_exam,
)
from .training import TrainingConfig, TrainingRefused, assert_trainable
from .windows import (
    WINDOW_FEATURE_NAMES,
    PhysioWindowBuilder,
    WindowConfig,
    WindowFeatures,
)

ACCEPTANCE_SCHEMA = "physio-acceptance-v1"
EVIDENCE_LEVEL = "LOCAL_EXPERIMENTAL"
ISOLATION_LEVEL = "AGENT_ISOLATION_UNVERIFIED"
SMOKE_ONLY = "SMOKE_ONLY"

BASE_TS = "2026-11-07T10:00:00+00:00"


@dataclass(frozen=True)
class CheckResult:
    name: str
    issue: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "issue": self.issue,
            "passed": self.passed,
            "detail": self.detail,
        }


def _failure_detail(exc: BaseException) -> str:
    """Name the failure precisely enough to tell a mapping error from a
    transient environment error, without dumping a whole traceback."""
    kind = f"{type(exc).__module__}.{type(exc).__qualname__}"
    frame = exc.__traceback__
    location = ""
    while frame is not None:
        location = f"{Path(frame.tb_frame.f_code.co_filename).name}:{frame.tb_lineno}"
        frame = frame.tb_next
    return f"{kind}: {exc} (raised at {location})"


def _ts(seconds: int) -> str:
    from datetime import UTC, datetime, timedelta

    return (datetime(2026, 11, 7, 10, 0, tzinfo=UTC) + timedelta(seconds=seconds)).isoformat()


def _device_payload(sequence: int, seconds: int, **measurements) -> dict[str, Any]:
    units = {
        "hr": "bpm",
        "spo2": "fraction",
        "skin_temp": "°C",
        "rr": "breaths/min",
        "sbp": "mmHg",
        "dbp": "mmHg",
    }
    values = measurements or {"hr": 70, "spo2": 0.97, "skin_temp": 33.2}
    return {
        "session_id": "sess-acceptance",
        "subject_ref": "pseudonym-acceptance",
        "device_ref": "chair-mock-01",
        "phase": PHASE_DURING,
        "measured_at": _ts(seconds),
        "received_at": _ts(seconds + 1),
        "sequence": sequence,
        "measurements": values,
        "units": units,
    }


def _catalog() -> ProgramCapabilityCatalog:
    return ProgramCapabilityCatalog.from_dict(
        {
            "catalog_version": "acceptance-catalog-v1",
            "reviewed": True,
            "programs": [
                {
                    "program_id": "M01", "name": "舒缓", "intensity_min": 1,
                    "intensity_max": 4, "duration_min_s": 300,
                    "duration_max_s": 1800, "actions": ["knead"],
                    "selectable": True, "reviewed": True,
                },
                {
                    "program_id": "M02", "name": "放松", "intensity_min": 2,
                    "intensity_max": 5, "duration_min_s": 600,
                    "duration_max_s": 1800, "actions": ["vibrate"],
                    "selectable": True, "reviewed": True,
                },
            ],
        }
    )


class _FakeValidatedModel:
    """SMOKE_ONLY stand-in that proves the OK path; not a clinical model."""

    state = hashlib.sha256(b"smoke-fake-v1").hexdigest()

    def state_sha256(self) -> str:
        return self.state

    def __call__(self, batch: Any, feature_mask: Any = None) -> Any:
        return SimpleNamespace(predictions={"fatigue_state": [42.0]})


def _artifact_for(
    model: Any,
    scaler: WindowScaler,
    *,
    validation_status: str = VALIDATION_VALIDATED,
) -> ModelArtifact:
    return ModelArtifact(
        model_version="smoke-fake-v1",
        scaler_sha256=scaler.digest(),
        state_sha256=model.state_sha256(),
        validation_status=validation_status,
    )


def _protocol_equivalence_check() -> CheckResult:
    try:
        payload = _device_payload(0, 0)
        device = DeviceEventAdapter().parse_event(payload)
        service = ServiceEventAdapter().parse_event(
            {**payload, "device_id": payload["device_ref"], "user_id": payload["subject_ref"]}
        )
        with tempfile.TemporaryDirectory(prefix="physio-acceptance-") as tmp:
            path = Path(tmp) / "session.csv"
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    ["sequence", "measured_at", "received_at", "phase", "metric", "value", "unit"]
                )
                for metric, value, unit in (
                    ("hr", "70", "bpm"),
                    ("spo2", "0.97", "fraction"),
                    ("skin_temp", "33.2", "°C"),
                ):
                    writer.writerow(
                        ["0", _ts(0), _ts(1), PHASE_DURING, metric, value, unit]
                    )
            offline = OfflineSessionAdapter().from_csv(
                path,
                session_id=payload["session_id"],
                subject_ref=payload["subject_ref"],
                device_ref=payload["device_ref"],
            ).events[0]
        signature = lambda e: {  # noqa: E731
            (m.metric, m.value, m.unit, m.valid, m.status) for m in e.measurements
        }
        passed = (
            signature(device) == signature(service) == signature(offline)
            and device.measured_at == offline.measured_at
        )
        return CheckResult(
            "protocol_equivalence", "#96", passed,
            "device/service/offline produce identical canonical measurements and timestamps",
        )
    except Exception as exc:  # noqa: BLE001 - acceptance reports, never crashes
        return CheckResult("protocol_equivalence", "#96", False, _failure_detail(exc))


def _replay_equivalence_check() -> CheckResult:
    try:
        payloads = [_device_payload(i, i * 10) for i in range(7)]
        adapter = DeviceEventAdapter()
        events = [adapter.parse_event(payload) for payload in payloads]
        offline = assemble_session(
            "sess-acceptance", "pseudonym-acceptance", "chair-mock-01", events
        )
        ingestor = OnlineIngestor("sess-acceptance", "pseudonym-acceptance", "chair-mock-01")
        for event in reversed(events):
            ingestor.push(event)
        online = ingestor.session()
        builder = PhysioWindowBuilder(WindowConfig(window_s=60, min_events=2))
        from datetime import UTC, datetime

        cutoff = datetime(2026, 11, 7, 10, 0, 55, tzinfo=UTC)
        first = builder.build(offline, cutoff)
        second = builder.build(online, cutoff)
        passed = first.to_dict() == second.to_dict() and first.usable
        return CheckResult(
            "replay_equivalence", "#97", passed,
            "same cutoff and events yield identical window features after out-of-order replay",
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult("replay_equivalence", "#97", False, _failure_detail(exc))


def _negative_path_check() -> CheckResult:
    try:
        cases = {
            "unknown_field": route_measurement("vendor_magic", 1, "score").status,
            "unknown_unit": route_measurement("heart_rate", 70, "furlong").status,
            "spo2_zero": route_measurement("spo2", 0, "%").status,
            "rr_zero": route_measurement("rr", 0, "breaths/min").status,
            "invalid_hr": route_measurement("heart_rate", -1, "bpm").status,
            "invalid_bp": route_measurement("sbp", 900, "mmHg").status,
            "invalid_number": route_measurement("heart_rate", "abc", "bpm").status,
        }
        passed = (
            cases["unknown_field"] == STATUS_UNKNOWN_METRIC
            and cases["unknown_unit"] == STATUS_UNKNOWN_UNIT
            and cases["spo2_zero"] == STATUS_INVALID_VALUE
            and cases["rr_zero"] == STATUS_ZERO_UNKNOWN
            and cases["invalid_hr"] == STATUS_OUT_OF_RANGE
            and cases["invalid_bp"] == STATUS_OUT_OF_RANGE
            and cases["invalid_number"] == STATUS_INVALID_TYPE
        )
        return CheckResult(
            "field_qc_negatives", "#96", passed,
            "unknown fields/units, SpO2 zero, rr zero, invalid HR/BP and non-numeric rejected",
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult("field_qc_negatives", "#96", False, _failure_detail(exc))


def _model_gate_check() -> CheckResult:
    try:
        events = [
            DeviceEventAdapter().parse_event(_device_payload(i, i * 10))
            for i in range(7)
        ]
        session = assemble_session(
            "sess-acceptance", "pseudonym-acceptance", "chair-mock-01", events
        )
        builder = PhysioWindowBuilder(WindowConfig(window_s=60, min_events=2))
        from datetime import UTC, datetime

        cutoff = datetime(2026, 11, 7, 10, 0, 55, tzinfo=UTC)
        unavailable = InferenceSession(session, builder).update(cutoff)
        window = builder.build(session, cutoff)
        scaler = WindowScaler.fit([window])
        model = _FakeValidatedModel()
        artifact = _artifact_for(model, scaler)
        validated = InferenceSession(
            session,
            builder,
            model=model,
            scaler=scaler,
            artifact=artifact,
        ).update(cutoff)
        wrong_names = WINDOW_FEATURE_NAMES[:-1]
        bad_scaler = InferenceSession(
            session,
            builder,
            model=_FakeValidatedModel(),
            scaler=WindowScaler(
                feature_names=wrong_names,
                mean=(0.0,) * len(wrong_names),
                std=(1.0,) * len(wrong_names),
                observed_counts=(1,) * len(wrong_names),
            ),
            artifact=artifact,
        ).update(cutoff)
        no_live = InferenceSession(session, builder).update(
            datetime(2026, 11, 7, 12, 0, tzinfo=UTC)
        )
        passed = (
            unavailable.status == STATUS_NOT_AVAILABLE
            and unavailable.reason == "model_not_available"
            and unavailable.prediction is None
            and validated.status == STATUS_OK
            and validated.prediction is not None
            and bad_scaler.status == STATUS_NOT_AVAILABLE
            and bad_scaler.prediction is None
            and no_live.status == STATUS_ABSTAIN
        )
        return CheckResult(
            "model_output_gate", "#97", passed,
            "unbound/tampered artifacts never score; stale input abstains",
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult("model_output_gate", "#97", False, _failure_detail(exc))


def _artifact_binding_check() -> CheckResult:
    try:
        events = [
            DeviceEventAdapter().parse_event(_device_payload(i, i * 10))
            for i in range(7)
        ]
        session = assemble_session(
            "sess-acceptance", "pseudonym-acceptance", "chair-mock-01", events
        )
        builder = PhysioWindowBuilder(WindowConfig(window_s=60, min_events=2))
        from datetime import UTC, datetime

        cutoff = datetime(2026, 11, 7, 10, 0, 55, tzinfo=UTC)
        scaler = WindowScaler.fit([builder.build(session, cutoff)])
        model = _FakeValidatedModel()
        good = _artifact_for(model, scaler)

        def status_for(artifact: ModelArtifact | None, runtime=model, runtime_scaler=scaler):
            return InferenceSession(
                session,
                builder,
                model=runtime,
                scaler=runtime_scaler,
                artifact=artifact,
            ).update(cutoff)

        tampered_model = ModelArtifact(
            model_version=good.model_version,
            scaler_sha256=good.scaler_sha256,
            state_sha256=hashlib.sha256(b"different-weights").hexdigest(),
            validation_status=VALIDATION_VALIDATED,
        )
        shifted = WindowScaler(
            feature_names=scaler.feature_names,
            mean=tuple(value + 1.0 for value in scaler.mean),
            std=scaler.std,
            observed_counts=scaler.observed_counts,
        )
        cases = {
            "ok": status_for(good).status,
            "missing": status_for(None).reason,
            "tampered_model": status_for(tampered_model).reason,
            "tampered_scaler": status_for(good, runtime_scaler=shifted).reason,
            "unvalidated": status_for(
                _artifact_for(model, scaler, validation_status="NOT_VALIDATED")
            ).reason,
        }
        passed = (
            cases["ok"] == STATUS_OK
            and cases["missing"] == "artifact_not_bound"
            and cases["tampered_model"].startswith("artifact_error")
            and cases["tampered_scaler"].startswith("artifact_error")
            and cases["unvalidated"] == "model_not_validated"
        )
        return CheckResult(
            "artifact_binding_gate", "#97", passed,
            "missing, tampered or unvalidated artifacts are refused before scoring",
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult("artifact_binding_gate", "#97", False, _failure_detail(exc))


def _recommendation_gate_check() -> CheckResult:
    try:
        live = SimpleNamespace(status=STATUS_NOT_AVAILABLE, quality="ok", prediction=None)
        stale = SimpleNamespace(status=STATUS_ABSTAIN, quality=QUALITY_STALE, prediction=None)
        service = MassageRecommendationService(_catalog())
        offered = service.update(
            live, user_preference=UserPreference(max_intensity=4, max_duration_s=900)
        )
        stopped = service.user_stop()
        blocked = MassageRecommendationService(_catalog()).update(stale)
        passed = (
            offered.status == REC_OK
            and bool(offered.candidates)
            and all(c.expected_effect is None for c in offered.candidates)
            and offered.control_command is None
            and stopped.state == STATE_SAFE_STOP
            and blocked.status == REC_ABSTAIN
            and blocked.candidates == []
        )
        return CheckResult(
            "recommendation_safety_gate", "#99", passed,
            "static candidates only; stale telemetry abstains; user stop is SAFE_STOP; no control",
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            "recommendation_safety_gate", "#99", False,
            _failure_detail(exc),
        )


def _response_label_check() -> CheckResult:
    try:
        from .response import build_delta_measures, pri_target

        device = DeviceEventAdapter()
        pre = device.parse_event(
            {
                **_device_payload(0, 0, hr=70),
                "phase": PHASE_PRE,
            }
        )
        post = device.parse_event(
            {
                **_device_payload(1, 300, hr=66),
                "phase": PHASE_POST,
            }
        )
        deltas = build_delta_measures(pre, post)
        hr = next(item for item in deltas if item.metric == "heart_rate")
        pri = pri_target()
        passed = (
            hr.valid
            and hr.delta == -4.0
            and pri.value is None
            and pri.available is False
        )
        return CheckResult(
            "response_label_gate", "#98", passed,
            "Post-Pre delta is signed observation only; y_pri stays null/unavailable",
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult("response_label_gate", "#98", False, _failure_detail(exc))


def _approved_registry() -> KnowledgeRegistry:
    return KnowledgeRegistry.from_json(
        {
            "entries": [
                {
                    "topic_id": "qi-general",
                    "title": "一般作息提示",
                    "text": "保持规律作息与适度活动。",
                    "source": "reviewed-wellness-fixture",
                    "license": "SYNTHETIC_TEST_ONLY",
                    "version": "v1",
                    "review_status": "APPROVED",
                    "reviewed_by": "synthetic-review",
                    "constitution_tags": ["气虚"],
                },
                {
                    "topic_id": "missing-license",
                    "title": "不得展示",
                    "text": "缺少许可证。",
                    "source": "fixture",
                    "version": "v1",
                    "review_status": "APPROVED",
                    "reviewed_by": "synthetic-review",
                    "constitution_tags": ["气虚"],
                },
            ]
        }
    )


def _tcm_sidecar_check() -> CheckResult:
    try:
        exam = parse_tcm_exam(
            {
                "session_id": "sess-acceptance",
                "subject_ref": "pseudonym-acceptance",
                "device_ref": "chair-mock-01",
                "measured_at": _ts(0),
                "received_at": _ts(1),
                "tongue_body_color": "淡红",
                "tongue_shape": "胖大、齿痕",
                "coating_color": "白",
                "coating_texture": "腻",
                "face_color": "萎黄",
                "complexion": "少华",
                "constitution_names": "气虚兼痰湿",
                "device_algorithm": "vendor-fixture",
                "vocabulary_version": "v1",
            }
        )
        sidecar = build_tcm_sidecar(exam, _approved_registry())
        no_registry = build_tcm_sidecar(exam)
        no_constitution = build_tcm_sidecar(
            parse_tcm_exam(
                {
                    "session_id": "sess-acceptance",
                    "subject_ref": "pseudonym-acceptance",
                    "device_ref": "chair-mock-01",
                    "measured_at": _ts(0),
                    "received_at": _ts(1),
                    "tongue_body_color": "淡红",
                }
            ),
            _approved_registry(),
        )
        passed = (
            sidecar.status in (TCM_OK, TCM_PARTIAL)
            and {item["topic_id"] for item in sidecar.knowledge} == {"qi-general"}
            and no_registry.status == TCM_ABSTAIN
            and no_constitution.status == "UNKNOWN"
            and sidecar.provenance["not_a_diagnosis"] is True
        )
        return CheckResult(
            "tcm_sidecar_gate", "#100", passed,
            "device labels map only to approved knowledge; missing review/license is excluded",
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult("tcm_sidecar_gate", "#100", False, _failure_detail(exc))


def _integration_boundary_check(repo_root: Path) -> CheckResult:
    try:
        source = (repo_root / "src" / "physio" / "inference.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
        offline_in_inference = any("adapters" in name or "Offline" in name for name in imported)
        demo = (repo_root / "scripts" / "physio_realtime_demo.py").read_text(
            encoding="utf-8"
        ) if (repo_root / "scripts" / "physio_realtime_demo.py").exists() else ""
        csv_in_demo = "from_csv" in demo or "import csv" in demo
        tcm_source = (repo_root / "src" / "physio" / "tcm_exam.py").read_text(
            encoding="utf-8"
        )
        research_import = "research." in tcm_source or "data.mendeley" in tcm_source
        passed = not offline_in_inference and not csv_in_demo and not research_import
        return CheckResult(
            "integration_boundaries", "#95", passed,
            "inference has no offline adapter import; demo has no CSV path; "
            "TCM has no Mendeley import",
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult("integration_boundaries", "#95", False, _failure_detail(exc))


def _legacy_import_gate() -> CheckResult:
    try:
        registry = ExportFieldRegistry(
            version="acceptance-legacy-v1",
            fields=(
                ExportFieldSpec("member_id", "identity"),
                ExportFieldSpec("session_key", "session"),
                ExportFieldSpec("record_id", "session"),
                ExportFieldSpec("record_kind", "session"),
                ExportFieldSpec("phase", "phase"),
                ExportFieldSpec("measured_at", "timestamp"),
                ExportFieldSpec("received_at", "timestamp"),
                ExportFieldSpec("sequence", "session"),
                ExportFieldSpec("heart_rate", "metric", canonical_metric="heart_rate",
                                phase=PHASE_DURING, unit="bpm"),
                ExportFieldSpec("heart_rate_post", "metric", canonical_metric="heart_rate",
                                phase=PHASE_POST, unit="bpm"),
            ),
        )
        base = {
            "member_id": "m1", "session_key": "s1", "record_id": "r1",
            "phase": PHASE_DURING, "measured_at": _ts(0),
            "received_at": _ts(2), "sequence": 0,
        }
        conflict = import_legacy_rows(
            [
                {**base, "record_kind": "snapshot", "heart_rate": 60},
                {**base, "record_kind": "completed", "heart_rate": 75,
                 "received_at": _ts(5)},
            ],
            registry, identity_trust=IDENTITY_TRUSTED,
        )
        masquerade = import_legacy_rows(
            [
                {**base, "record_kind": "completed", "phase": PHASE_PRE, "heart_rate": 70},
                {**base, "record_kind": "completed", "phase": PHASE_POST,
                 "heart_rate_post": 66, "sequence": 1, "measured_at": _ts(300),
                 "received_at": _ts(302)},
            ],
            registry, identity_trust=IDENTITY_TRUSTED,
        )
        passed = (
            len(conflict.sessions) == 1
            and conflict.sessions[0].events[0].measurement("heart_rate").value == 75
            and any(
                issue.status == ISSUE_SNAPSHOT_COMPLETED_MISMATCH
                for issue in conflict.issues
            )
            and masquerade.sessions == ()
            and any(issue.status == ISSUE_PHASE_CONFLICT for issue in masquerade.issues)
        )
        return CheckResult(
            "legacy_export_gate", "#96", passed,
            "snapshot/completed resolves deterministically; one record cannot be Pre and Post",
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult("legacy_export_gate", "#96", False, _failure_detail(exc))


def _dataset_label_gate() -> CheckResult:
    try:
        from datetime import UTC, datetime

        cutoff_dt = datetime(2026, 11, 7, 10, 1, tzinfo=UTC)
        train_window = WindowFeatures(
            usable=True,
            feature_names=("f1", "f2"),
            values=(10.0, 2.0),
            mask=(True, True),
            coverage=1.0,
            n_events=1,
            start=cutoff_dt,
            end=cutoff_dt,
            quality="ok",
            metrics_present=("heart_rate",),
        )
        target_window = WindowFeatures(
            usable=False,
            feature_names=("f1", "f2"),
            values=(0.0, 0.0),
            mask=(False, False),
            coverage=0.0,
            n_events=1,
            start=cutoff_dt,
            end=cutoff_dt,
            quality="ok",
            metrics_present=(),
            reason="acceptance-masked-row",
        )
        train_row = feature_row_from_window(
            train_window, subject_ref="s1", session_id="sess", cutoff=cutoff_dt
        )
        target_row = feature_row_from_window(
            target_window, subject_ref="s2", session_id="sess", cutoff=cutoff_dt
        )
        imputer = FoldImputer.fit([train_row], min_observed=1)
        transformed = imputer.transform(target_row)
        folds = make_subject_folds(["s1", "s2"], n_folds=2, seed=1)

        pre = DeviceEventAdapter().parse_event(
            {**_device_payload(0, 0, hr=70), "phase": PHASE_PRE}
        )
        post = DeviceEventAdapter().parse_event(
            {**_device_payload(1, 300, hr=66), "phase": PHASE_POST}
        )
        same = DeviceEventAdapter().parse_event(
            {**_device_payload(0, 300, hr=66), "phase": PHASE_POST}
        )
        good = admit_delta_targets(pre, post)
        bad = admit_delta_targets(pre, same)
        passed = (
            transformed.imputed_mask == (True, True)
            and transformed.observed_mask == (False, False)
            and transformed.valid_mask == (True, True)
            and folds.trusted
            and good.mask["y_delta_heart_rate"] is True
            and good.values["y_delta_heart_rate"] == -4.0
            and bad.mask["y_delta_heart_rate"] is False
            and bad.values["y_pri"] is None
        )
        return CheckResult(
            "dataset_label_gate", "#98", passed,
            "fold imputation is masked; y_delta requires independent valid endpoints; y_pri null",
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult("dataset_label_gate", "#98", False, _failure_detail(exc))


def _training_refusal_gate() -> CheckResult:
    try:
        from datetime import UTC, datetime

        row = FeatureRow(
            feature_names=("f1",),
            values=(1.0,),
            observed_mask=(True,),
            valid_mask=(True,),
            imputed_mask=(False,),
            imputation=("none",),
            subject_ref="s1",
            session_id="sess",
            cutoff=datetime(2026, 11, 7, 10, 1, tzinfo=UTC),
        )
        unlabeled = False
        try:
            assert_trainable(
                [row], [None], config=TrainingConfig(min_samples=1),
                identity_trust=IDENTITY_TRUSTED,
            )
        except TrainingRefused:
            unlabeled = True
        untrusted = False
        try:
            assert_trainable(
                [row], [10.0], config=TrainingConfig(min_samples=1),
                identity_trust="SHARED_ACCOUNT_UNKNOWN",
            )
        except TrainingRefused:
            untrusted = True
        return CheckResult(
            "training_refusal_gate", "#97", unlabeled and untrusted,
            "no label and untrusted identity both refuse training; y_pri never trains",
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            "training_refusal_gate", "#97", False, _failure_detail(exc)
        )


def _evaluation_gate() -> CheckResult:
    try:
        report = evaluate_regression(
            [1.0, 2.0], [1.0, None], [True, False], ["s1", "s2"], min_samples=30
        )
        passed = (
            report.status == STATUS_INSUFFICIENT
            and report.n_labeled == 1
            and report.metrics["mae"] is not None
            and "required 30" in report.reason
        )
        return CheckResult(
            "evaluation_evidence_gate", "#95", passed,
            "small sample set stays INSUFFICIENT_EVIDENCE; no generalization claim",
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            "evaluation_evidence_gate", "#95", False, _failure_detail(exc)
        )


def _real_export_mapping_gate(repo_root: Path) -> CheckResult:
    """#102: the declared real-export mapping rebuilds labels and refuses guesswork."""
    try:
        from physio.real_export import (
            IDENTITY_SHARED as REAL_IDENTITY_SHARED,
        )
        from physio.real_export import (
            RealExportError,
            import_real_export,
            session_labels,
        )

        raw = (
            repo_root
            / "tests"
            / "fixtures"
            / "physio-legacy-sessions-synthetic.csv"
        )
        sessions = import_real_export(raw)
        first = sessions[0]
        labels = session_labels(first)
        again = session_labels(first)

        assert len(sessions) == 5
        assert first.identity_trust == REAL_IDENTITY_SHARED
        assert labels.values.get("y_pri") is None
        assert "y_pri" not in labels.values
        assert labels.values["y_delta_sbp"] == -5.0
        assert labels.values["y_delta_vendor_health"] == -2.0
        assert labels.values == again.values
        assert session_labels(sessions[1]).mask["y_delta_sbp"] is False

        missing = repo_root / "tests" / "fixtures" / "absent-real-export.csv"
        refused = False
        try:
            import_real_export(missing)
        except RealExportError:
            refused = True
        assert refused

        return CheckResult(
            "real_export_mapping",
            "#102",
            True,
            "long-table import rebuilds y_delta labels deterministically; shared account "
            "identity is never trusted; an absent export is refused",
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            "real_export_mapping", "#102", False, _failure_detail(exc)
        )


def _real_data_audit_gate(repo_root: Path) -> CheckResult:
    """#102: every reference field is match, explained or unrecoverable."""
    try:
        from physio.real_data_audit import (
            STATUS_VERIFIED,
            STATUS_VERIFIED_WITH_EXCEPTIONS,
            VERDICT_MISMATCH,
            audit_real_export,
            load_reference_rows,
        )
        from physio.real_export import import_real_export

        fixtures = repo_root / "tests" / "fixtures"
        sessions = import_real_export(fixtures / "physio-legacy-sessions-synthetic.csv")
        audit = audit_real_export(
            sessions,
            load_reference_rows(fixtures / "physio-legacy-reference-synthetic.csv"),
            load_reference_rows(fixtures / "physio-legacy-qc-synthetic.csv"),
        )
        counts = audit.verdict_counts()
        assert audit.status in {STATUS_VERIFIED, STATUS_VERIFIED_WITH_EXCEPTIONS}
        assert audit.passed is True
        assert counts[VERDICT_MISMATCH] == 0
        assert counts["match"] > 0 and counts["explained"] > 0 and counts["not_recoverable"] > 0
        comparable = [c for c in audit.qc_cross_checks if c.verdict != "observation"]
        assert comparable
        assert all(check.verdict == "match" for check in comparable)

        tampered = [dict(row) for row in load_reference_rows(
            fixtures / "physio-legacy-reference-synthetic.csv"
        )]
        tampered[0]["y_delta_sbp"] = "-99"
        failed = audit_real_export(sessions, tampered)
        assert failed.status == "FAILED"
        assert failed.passed is False

        return CheckResult(
            "real_data_audit",
            "#102",
            True,
            "reference training set and quality report are rebuilt from raw records; a "
            "tampered reference value fails the audit instead of being repaired",
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult("real_data_audit", "#102", False, _failure_detail(exc))


def _real_training_gate(repo_root: Path) -> CheckResult:
    """#102: the real-data pipeline runs and never upgrades the evidence level."""
    try:
        from physio.real_export import import_real_export
        from physio.real_training import (
            STATUS_INSUFFICIENT,
            run_real_pre_experiment,
        )

        sessions = import_real_export(
            repo_root / "tests" / "fixtures" / "physio-legacy-sessions-synthetic.csv"
        )
        report = run_real_pre_experiment(sessions)
        assert report.evaluation_status == STATUS_INSUFFICIENT
        assert report.status == STATUS_INSUFFICIENT
        assert report.identity_trust == "SHARED_ACCOUNT_UNKNOWN"
        assert report.model_mae["train_fold_mean"] is not None
        assert report.model_mae["ridge"] is not None
        assert len(report.folds) == 5
        assert all(fold.n_train == 4 and fold.n_test == 1 for fold in report.folds)

        return CheckResult(
            "real_training_pipeline",
            "#102",
            True,
            "leave-one-subject-out folds, fold-internal imputation and scaling run on real "
            "rows; evaluation stays INSUFFICIENT_EVIDENCE and no performance claim is made",
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            "real_training_pipeline", "#102", False, _failure_detail(exc)
        )


def run_acceptance(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Run all synthetic engineering acceptance checks and return one report."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    checks = [
        _protocol_equivalence_check(),
        _replay_equivalence_check(),
        _negative_path_check(),
        _model_gate_check(),
        _artifact_binding_check(),
        _response_label_check(),
        _recommendation_gate_check(),
        _tcm_sidecar_check(),
        _integration_boundary_check(root),
        _legacy_import_gate(),
        _dataset_label_gate(),
        _training_refusal_gate(),
        _evaluation_gate(),
        _real_export_mapping_gate(root),
        _real_data_audit_gate(root),
        _real_training_gate(root),
    ]
    passed = sum(1 for check in checks if check.passed)
    report = {
        "schema_version": ACCEPTANCE_SCHEMA,
        "status": "PASS" if passed == len(checks) else "FAIL",
        "passed": passed,
        "total": len(checks),
        "evidence_level": EVIDENCE_LEVEL,
        "isolation": ISOLATION_LEVEL,
        "smoke_only": SMOKE_ONLY,
        "device_control": NOT_CONNECTED,
        "checks": [check.to_dict() for check in checks],
        "limitations": [
            "synthetic fixtures only; no real device, subject or paired label data",
            "the real 94-field export audit runs on the private data/local/ extract and is "
            "never committed",
            "no model performance, efficacy or clinical claim",
            "device control remains NOT_CONNECTED",
            "GO gates are independent and are not derived from these checks",
        ],
    }
    return report


__all__ = [
    "ACCEPTANCE_SCHEMA",
    "EVIDENCE_LEVEL",
    "ISOLATION_LEVEL",
    "SMOKE_ONLY",
    "CheckResult",
    "run_acceptance",
]
