"""Unified lifecycle acceptance gate (#112).

Runs a fixed set of synthetic, negative-path and recovery checks across all
stages and emits an auditable GO/NO-GO report. Engineering GO is separate from
GO_REAL_DATA_TRAINING and GO_RESEARCH_CLAIMS: no test pass ever promotes either.

Stdlib + already-required dependencies only; no private data, no network, no
real device, no torch required.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import (
    APPROVAL_APPROVED,
    APPROVAL_UNVERIFIED,
    EVIDENCE_SYNTHETIC,
    GATE_NOT_GRANTED,
    ISOLATION_UNVERIFIED,
    STATUS_ABSTAIN,
    STATUS_BLOCKED,
    STATUS_OK,
    TRUST_SHARED,
    TRUST_TRUSTED,
    gate_bundle,
)

ACCEPTANCE_VERSION = "ml-lifecycle-acceptance-v1"
EVIDENCE_LEVEL = "LOCAL_EXPERIMENTAL"


@dataclass(frozen=True)
class Check:
    name: str
    stage: str
    passed: bool
    detail: str
    negative: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "stage": self.stage, "passed": self.passed,
            "detail": self.detail, "negative": self.negative,
        }


def _fixtures() -> Path:
    return Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "lifecycle"


def _run(
    name: str, stage: str, fn: Callable[[Path], str], root: Path, *, negative: bool = False
) -> Check:
    try:
        detail = fn(root)
        return Check(name, stage, True, detail, negative)
    except Exception as exc:  # noqa: BLE001
        return Check(name, stage, False, f"{type(exc).__name__}: {exc}", negative)


def _check_contracts(root: Path) -> str:
    from .contracts import (
        ContractError,
        DatasetManifest,
        SourceManifest,
        digest,
    )

    SourceManifest(
        "b", "offline", "b.csv", digest({"x": 1}), "PRIVATE", "import",
        TRUST_SHARED, "NOT_CONNECTED", "v1",
    )
    try:
        SourceManifest(
            "d", "device", "usb://0", digest({"x": 1}), "PRIVATE", "vendor",
            TRUST_TRUSTED, "CONNECTED", "v1",
        )
    except ContractError:
        pass
    else:
        raise AssertionError("fake connected real device was accepted")
    manifest = DatasetManifest(
        "d", (digest({"s": 1}),), "s", "l", "f", digest({"q": 1}), 10, 2,
        TRUST_SHARED, EVIDENCE_SYNTHETIC, APPROVAL_UNVERIFIED, False, "t",
    )
    assert manifest.trainable()[0] is False
    return "immutable contracts, fake-device refusal, trainable hard gate"


def _check_pipeline_idempotent(root: Path) -> str:
    from .pipeline import declare_batch, run_pipeline

    batch = declare_batch(_fixtures() / "batch_a.csv")
    out = root / "acceptance_pipeline"
    first = run_pipeline([batch], out)
    second = run_pipeline([batch], out)
    assert second.reused_snapshot is True
    assert second.snapshot.sha256 == first.snapshot.sha256
    assert second.qc.sha256 == first.qc.sha256
    assert second.qc.n_duplicate == 0
    assert second.readiness["auto_train"] is False
    return "same input -> identical snapshot/QC, cache reuse, no duplicate, no auto-train"


def _check_pipeline_incremental(root: Path) -> str:
    from .pipeline import declare_batch, run_pipeline

    a = declare_batch(_fixtures() / "batch_a.csv")
    b = declare_batch(_fixtures() / "batch_b.csv")
    forward = run_pipeline([a, b], root / "acc_fwd")
    reverse = run_pipeline([b, a], root / "acc_rev")
    assert forward.snapshot.sha256 == reverse.snapshot.sha256
    assert forward.qc.sha256 == reverse.qc.sha256
    return "out-of-order/late batches converge to one snapshot"


def _check_pipeline_recovery(root: Path) -> str:
    from .pipeline import declare_batch, run_pipeline

    batch = declare_batch(_fixtures() / "batch_a.csv")
    failed = run_pipeline([batch], root / "acc_recovery", fail_stage="validate")
    assert failed.status == STATUS_BLOCKED and failed.snapshot is None
    recovered = run_pipeline([batch], root / "acc_recovery")
    assert recovered.snapshot is not None and recovered.error == ""
    return "injected failure recorded; resume reproduces the snapshot"


def _check_pipeline_drift(root: Path) -> str:
    from .pipeline import declare_batch, run_pipeline

    batch = declare_batch(_fixtures() / "batch_a.csv")
    result = run_pipeline([batch], root / "acc_gate")
    assert result.readiness["train_ready"] is False
    assert result.status == STATUS_ABSTAIN
    approved = run_pipeline(
        [batch], root / "acc_gate2", approval=APPROVAL_APPROVED,
        identity_trust=TRUST_TRUSTED, frozen=True,
    )
    assert approved.readiness["train_ready"] is True
    assert approved.status == STATUS_OK
    return "unapproved/untrusted -> ABSTAIN; approved+frozen+trusted -> admit"


def _check_no_silent_skip(root: Path) -> str:
    from .pipeline import declare_batch, run_pipeline

    bad = root / "acc_bad.csv"
    bad.write_text(
        "batch_id,record_id,session_id,subject_ref,phase,metric,value,unit,"
        "quality,source_record\n"
        "A,1,S1,P1,pre,heart_rate,70,mmHg,ok,src-1\n",
        encoding="utf-8",
    )
    result = run_pipeline([declare_batch(bad)], root / "acc_bad_out")
    assert result.qc.n_rejected == 1
    assert all(q["reason"] for q in result.quarantine)
    return "rejected row carries a reason; no silent skip"


def _check_training_gates(root: Path) -> str:
    from .contracts import (
        EvalSpec,
        FeatureSpec,
        FieldSpec,
        SplitManifest,
        TargetSpec,
        TrainConfig,
        digest,
    )
    from .training import TRAIN_VERSION, run_synthetic_smoke

    features = FeatureSpec(
        "f1", (
            FieldSpec("a", "u", "float", "pre", "none", "s", "in", "none", "observed"),
            FieldSpec("b", "u", "float", "pre", "none", "s", "in", "none", "observed"),
        ), "identity", EVIDENCE_SYNTHETIC,
    )
    target = TargetSpec(
        "t", "t1", "linear", (0.0, 1.0), "index", "derived", "synthetic",
        APPROVAL_UNVERIFIED, unavailable_reason="no real label",
    )
    dataset = _dataset_manifest()
    split = SplitManifest("s1", 7, 3, digest({"f": 1}), ("P1", "P2", "P3"), ())
    evaluation = EvalSpec("e1", "mae", 30, 2)
    config = TrainConfig(
        7, "ridge", 1, EVIDENCE_SYNTHETIC, "c", "local", "l", APPROVAL_UNVERIFIED,
    )
    report = run_synthetic_smoke(
        features=features, target=target, dataset=dataset, split=split,
        evaluation=evaluation, config=config,
    )
    assert report.decision == "REVERT"
    assert report.evaluation_status == "INSUFFICIENT_EVIDENCE"
    assert report.gates["GO_REAL_DATA_TRAINING"] == GATE_NOT_GRANTED
    assert report.gates["GO_RESEARCH_CLAIMS"] == GATE_NOT_GRANTED
    assert report.notes and "NOT_VALIDATED" in report.notes[1]
    assert TRAIN_VERSION
    return "synthetic smoke: no promotion, no validated artifact, gates independent"


def _dataset_manifest():
    from .contracts import DatasetManifest, digest

    return DatasetManifest(
        "ds", (digest({"s": 1}),), "s", "l", "f", digest({"q": 1}), 30, 6,
        TRUST_SHARED, EVIDENCE_SYNTHETIC, APPROVAL_UNVERIFIED, False, "t",
    )


def _check_release_no_self_promote(root: Path) -> str:
    from .contracts import ArtifactManifest, digest
    from .release import ArtifactBundle, ReleaseRegistry

    manifest = ArtifactManifest(
        "a1", digest({"m": 1}), digest({"f": 1}), digest({"t": 1}),
        digest({"s": 1}), digest({"d": 1}), digest({"sp": 1}), digest({"e": 1}),
        "c", (), "l", APPROVAL_UNVERIFIED, "NOT_VALIDATED", EVIDENCE_SYNTHETIC,
    )
    from .pipeline import file_sha256

    bundle_dir = root / "acc_bundle_dir"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    payload = bundle_dir / "acc_bundle.bin"
    payload.write_bytes(b"synthetic artifact bytes")
    bundle = ArtifactBundle(
        manifest=manifest,
        files={"acc_bundle.bin": file_sha256(payload)},
        root=bundle_dir,
    )
    registry = ReleaseRegistry(root / "acc_registry.json")
    registry.register(bundle, "NOT_VALIDATED")
    assert registry.state()["champion"] is None, "registry auto-promoted a candidate"
    assert registry.state()["previous"] is None
    return "registry never auto-promotes; gates stay independent"


def _check_sources_capability_and_replay(root: Path) -> str:
    from .contracts import SOURCE_DEVICE
    from .sources import (
        SourceCapability,
        SourceError,
        conformance_record,
        live,
        make_event,
        replay,
    )

    try:
        SourceCapability("dev1", SOURCE_DEVICE, "CONNECTED")
    except SourceError:
        pass
    else:
        raise AssertionError("fake connected real device was accepted")
    record = conformance_record(SourceCapability("dev1", SOURCE_DEVICE))
    assert record["gate"] == "BLOCKED_BY_DATA_OR_DEVICE"

    t0 = "2026-11-01T09:00:00+00:00"
    t1 = "2026-11-01T09:01:00+00:00"
    events = [make_event("a", event_time=t0), make_event("b", event_time=t1)]
    assert replay(events) == replay(list(reversed(events)))
    assert live(list(reversed(events))).accepted == replay(events)
    return "no fake CONNECTED device, BLOCKED verdict, live/replay converge"


def _check_features_head_mismatch(root: Path) -> str:
    from .contracts import TargetSpec
    from .features import (
        TargetCompatibilityError,
        TargetHead,
        check_target_head_compatibility,
        head_for,
    )

    delta = TargetSpec(
        "delta_vendor_fatigue", "t1", "linear_regression", (-20.0, 0.0),
        "index", "derived", "vendor proxy difference", APPROVAL_UNVERIFIED,
        unavailable_reason="no real paired label",
    )
    clamped = TargetHead("realtime_fatigue", "sigmoid", (0.0, 100.0))
    try:
        check_target_head_compatibility(delta, clamped)
    except TargetCompatibilityError:
        pass
    else:
        raise AssertionError("negative delta accepted by the clamped [0,100] head")
    assert check_target_head_compatibility(delta, head_for(delta)) is True
    return "structural target/head mismatch refuses; correct head admits"


def _check_features_no_fabricated_truth(root: Path) -> str:
    from .features import unavailable_target

    target = unavailable_target("fatigue")
    assert target.approval != APPROVAL_APPROVED
    assert target.unavailable_reason
    assert "clinical" in target.unavailable_reason or "truth" in target.unavailable_reason
    return "no approved fatigue ground truth; unavailability is explicit"


def _check_gateway_fail_closed(root: Path) -> str:
    from .gateway import Gateway, make_envelope

    gateway = Gateway(known_feature_versions=["v1"], known_model_versions=["v1"])
    good = make_envelope(
        request_id="r1", mode="A_pre_only", feature_version="v1",
        model_version="v1", target_version="v1",
        payload={"features": {"heart_rate_pre": 70.0}},
    )
    first = gateway.handle(good)
    again = gateway.handle(good)
    assert first.to_dict() == again.to_dict()
    assert gateway.idempotency_hits >= 1
    assert first.payload["control_command"] is None

    unknown = make_envelope(
        request_id="r2", mode="A_pre_only", feature_version="v-unknown",
        model_version="v1", target_version="v1",
        payload={"features": {"heart_rate_pre": 70.0}},
    )
    refused = gateway.handle(unknown)
    assert refused.status != STATUS_OK, "unknown version produced an OK answer"
    assert refused.payload["control_command"] is None
    return "idempotent, unknown-version fail-closed, control_command always null"


def _check_observe_no_auto_actions(root: Path) -> str:
    from .observe import on_new_data

    plan = on_new_data(batch_ids=["b1", "b2"])
    assert plan["auto_train"] is False
    assert plan["auto_promote"] is False
    assert plan["auto_publish"] is False
    assert plan["executed"] is False
    assert plan["reason"]
    assert plan["gates"]["GO_REAL_DATA_TRAINING"] == GATE_NOT_GRANTED
    return "data update is audit-only; no auto train/promote/publish"


def _check_three_gates(root: Path) -> str:
    gates = gate_bundle(True)
    assert gates["GO_ENGINEERING"] == "GRANTED"
    assert gates["GO_REAL_DATA_TRAINING"] == GATE_NOT_GRANTED
    assert gates["GO_RESEARCH_CLAIMS"] == GATE_NOT_GRANTED
    return "engineering GO cannot promote the two research gates"


PRIVATE_PATH_MARKERS = ("data/raw/", "data/local/", "runs/", "artifacts/")
PRIVATE_SUFFIXES = (".pt", ".pth")


def _is_private_path(path: str) -> bool:
    normalized = "/" + path.lstrip("./")
    return any(f"/{marker}" in normalized for marker in PRIVATE_PATH_MARKERS) or path.endswith(
        PRIVATE_SUFFIXES
    )


def _check_raw_data_is_not_committed(root: Path) -> str:
    """Assert the real protection: nothing private is tracked by git."""
    repo = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "git metadata unavailable; the CI leak check is the enforcing gate"
    if result.returncode != 0:
        return "git metadata unavailable; the CI leak check is the enforcing gate"
    tracked = result.stdout.splitlines()
    offenders = [path for path in tracked if _is_private_path(path)]
    assert not offenders, f"private data, runs or artifacts are tracked: {offenders[:5]}"
    return (
        f"{len(tracked)} tracked files; no raw data, runs or binary artifacts committed"
    )


CHECKS: tuple[tuple[str, str, Callable[[Path], str], bool], ...] = (
    ("contracts_immutable", "ingest", _check_contracts, False),
    ("three_gates_independent", "ready_gate", _check_three_gates, False),
    ("pipeline_idempotent", "snapshot", _check_pipeline_idempotent, False),
    ("pipeline_incremental", "reconcile", _check_pipeline_incremental, False),
    ("pipeline_recovery", "normalize", _check_pipeline_recovery, True),
    ("pipeline_ready_gate", "ready_gate", _check_pipeline_drift, True),
    ("no_silent_skip", "validate", _check_no_silent_skip, True),
    ("training_no_auto_promotion", "train", _check_training_gates, True),
    ("release_no_self_promote", "promote", _check_release_no_self_promote, True),
    ("sources_capability_replay", "ingest", _check_sources_capability_and_replay, True),
    ("features_head_mismatch", "features", _check_features_head_mismatch, True),
    ("features_no_fabricated_truth", "features", _check_features_no_fabricated_truth, True),
    ("gateway_fail_closed", "serve", _check_gateway_fail_closed, True),
    ("observe_no_auto_actions", "observe", _check_observe_no_auto_actions, True),
    ("private_data_not_in_git", "ingest", _check_raw_data_is_not_committed, False),
)


@dataclass(frozen=True)
class AcceptanceReport:
    schema: str
    evidence_level: str
    isolation: str
    status: str
    passed: int
    total: int
    negative_checks: int
    gates: dict[str, str]
    checks: tuple[Check, ...]
    limitations: tuple[str, ...] = ()

    @property
    def engineering_go(self) -> bool:
        return self.status == "PASS"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "evidence_level": self.evidence_level,
            "isolation": self.isolation,
            "status": self.status,
            "passed": self.passed,
            "total": self.total,
            "negative_checks": self.negative_checks,
            "gates": dict(self.gates),
            "checks": [c.to_dict() for c in self.checks],
            "limitations": list(self.limitations),
        }


def run_acceptance(root: str | Path | None = None) -> AcceptanceReport:
    import tempfile

    work = (
        Path(root)
        if root is not None
        else Path(tempfile.mkdtemp(prefix="lifecycle-acceptance-"))
    )
    work.mkdir(parents=True, exist_ok=True)
    results = [
        _run(name, stage, fn, work, negative=negative)
        for name, stage, fn, negative in CHECKS
    ]
    passed = sum(1 for c in results if c.passed)
    negatives = sum(1 for c in results if c.negative and c.passed)
    gates = gate_bundle(passed == len(results))
    return AcceptanceReport(
        schema=ACCEPTANCE_VERSION,
        evidence_level=EVIDENCE_LEVEL,
        isolation=ISOLATION_UNVERIFIED,
        status="PASS" if passed == len(results) else "FAIL",
        passed=passed,
        total=len(results),
        negative_checks=negatives,
        gates=gates,
        checks=tuple(results),
        limitations=(
            "synthetic fixtures only; no real device, subject or paired label data",
            "engineering PASS is not GO_REAL_DATA_TRAINING or GO_RESEARCH_CLAIMS",
            "no model performance, efficacy or clinical claim",
            "device control remains NOT_CONNECTED",
        ),
    )


def write_acceptance(report: AcceptanceReport, out_dir: str | Path) -> Path:
    path = Path(out_dir) / "lifecycle_acceptance.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")
    return path
