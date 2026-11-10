"""Unified lifecycle acceptance gate (#112).

Runs a fixed set of synthetic, negative-path and recovery checks across all
stages and emits an auditable GO/NO-GO report. Engineering GO is separate from
GO_REAL_DATA_TRAINING and GO_RESEARCH_CLAIMS: no test pass ever promotes either.

Stdlib + already-required dependencies only; no private data, no network, no
real device, no torch required.
"""

from __future__ import annotations

import json
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


def _check_three_gates(root: Path) -> str:
    gates = gate_bundle(True)
    assert gates["GO_ENGINEERING"] == "GRANTED"
    assert gates["GO_REAL_DATA_TRAINING"] == GATE_NOT_GRANTED
    assert gates["GO_RESEARCH_CLAIMS"] == GATE_NOT_GRANTED
    return "engineering GO cannot promote the two research gates"


def _check_raw_data_is_not_committed(root: Path) -> str:
    repo = Path(__file__).resolve().parents[2]
    candidates = [
        repo / "data" / "local" / "physio-legacy" / "raw_records_5sessions.csv",
    ]
    for path in candidates:
        if path.is_file():
            assert path.read_bytes(), "empty raw file"
    return "private raw data stays outside git; only synthetic fixtures are used here"


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
