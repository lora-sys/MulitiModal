"""#109 release packaging, versioned promotion, rollback and measured reports."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from lifecycle import release as rl
from lifecycle.contracts import (
    APPROVAL_APPROVED,
    APPROVAL_PENDING,
    CAP_NOT_CONNECTED,
    CONTRACTS_VERSION,
    EVIDENCE_LOCAL,
    EVIDENCE_SMOKE,
    EVIDENCE_SYNTHETIC,
    GATE_ENGINEERING,
    GATE_GRANTED,
    GATE_NOT_GRANTED,
    GATE_REAL_DATA_TRAINING,
    GATE_RESEARCH_CLAIMS,
    STATUS_BLOCKED,
    STATUS_NOT_AVAILABLE,
    STATUS_OK,
    ArtifactManifest,
    canonical_json,
    digest,
    file_sha256,
    gate_bundle,
)

pytestmark = pytest.mark.synthetic

HASH = digest({"synthetic": "hash"})
FILES = ("state.json", "weights.bin")


def _manifest(**kwargs: object) -> ArtifactManifest:
    base = dict(
        artifact_id="synthetic-a1",
        model_sha256=HASH,
        feature_sha256=HASH,
        target_sha256=HASH,
        scaler_sha256=HASH,
        dataset_sha256=HASH,
        split_sha256=HASH,
        eval_sha256=HASH,
        commit="synthetic-commit",
        dependencies=("python>=3.11",),
        license="SMOKE_ONLY",
        approval=APPROVAL_APPROVED,
        validation_status=rl.VALIDATION_VALIDATED,
        evidence_level=EVIDENCE_LOCAL,
    )
    base.update(kwargs)
    return ArtifactManifest(**base)


def _bundle(tmp_path: Path, *, artifact_id: str = "synthetic-a1", **manifest_kwargs: object):
    root = tmp_path / f"bundle-{artifact_id}"
    root.mkdir(parents=True, exist_ok=True)
    state = {"synthetic": True, "model": artifact_id}
    weights = b"synthetic-weights-v1"
    (root / "state.json").write_text(canonical_json(state) + "\n", encoding="utf-8")
    (root / "weights.bin").write_bytes(weights)
    manifest = _manifest(artifact_id=artifact_id, **manifest_kwargs)
    bundle = rl.ArtifactBundle.from_directory(
        root,
        manifest,
        files=FILES,
        environment="synthetic-local-cpu",
        compatibility={"hardware": "cpu", "format": "python-stdlib"},
        extra_evidence={"evidence_packet": "synthetic-review"},
    )
    return root, bundle


def _registry(tmp_path: Path, bundle: rl.ArtifactBundle, validation_status: str = "VALIDATED"):
    registry = rl.ReleaseRegistry(tmp_path / "release-registry.json")
    registry.register(bundle, validation_status)
    return registry


def _mutate(registry: rl.ReleaseRegistry, artifact_id: str, **changes: object) -> None:
    state = registry.state()
    entry = state["candidates"][artifact_id]
    for key, value in changes.items():
        if "." in key:
            parent, child = key.split(".", 1)
            entry[parent][child] = value
        else:
            entry[key] = value
    registry._write(state)


def _delays() -> tuple[float, ...]:
    return tuple(float(value) for value in range(1, 101))


def test_bundle_hashes_every_declared_file_and_rejects_mutation(tmp_path):
    root, bundle = _bundle(tmp_path)
    verified = bundle.verify()
    assert verified["status"] == "VERIFIED"
    assert verified["files"]["state.json"] == file_sha256(root / "state.json")
    assert verified["files"]["weights.bin"] == file_sha256(root / "weights.bin")
    assert verified["bundle_sha256"] == bundle.sha256 == bundle.bundle_sha256
    assert bundle.to_dict()["dependencies"] == ["python>=3.11"]
    assert bundle.to_dict()["evidence"]["evidence_packet"] == "synthetic-review"
    assert bundle.to_dict()["contracts_version"] == CONTRACTS_VERSION

    (root / "weights.bin").write_bytes(b"tampered-weights")
    with pytest.raises(rl.ReleaseError, match="changed.*weights.bin"):
        bundle.verify()

    (root / "weights.bin").write_bytes(b"synthetic-weights-v1")
    (root / "state.json").write_text(
        canonical_json({"synthetic": True, "model": "edited"}), encoding="utf-8"
    )
    with pytest.raises(rl.ReleaseError, match="changed.*state.json"):
        bundle.verify()


def test_bundle_rejects_missing_and_undeclared_files_without_silent_skip(tmp_path):
    root, bundle = _bundle(tmp_path)
    (root / "state.json").unlink()
    with pytest.raises(rl.ReleaseError, match="missing.*state.json"):
        bundle.verify()

    (root / "state.json").write_text(
        canonical_json({"synthetic": True, "model": "synthetic-a1"}) + "\n",
        encoding="utf-8",
    )
    (root / "extra.bin").write_bytes(b"undeclared")
    with pytest.raises(rl.ReleaseError, match="extra.*extra.bin"):
        bundle.verify()


def test_bundle_rejects_edited_hash_binding_and_unsafe_paths(tmp_path):
    _, bundle = _bundle(tmp_path)
    edited = bundle.to_dict(include_hash=True)
    edited["bundle_sha256"] = HASH
    with pytest.raises(rl.ReleaseError, match="bundle_sha256 mismatch"):
        rl.ArtifactBundle(
            root=Path(edited["root"]),
            manifest=ArtifactManifest(**edited["manifest"]),
            files=edited["files"],
            schema_version=edited["schema_version"],
            environment=edited["environment"],
            compatibility=edited["compatibility"],
            extra_evidence={"evidence_packet": edited["evidence"]["evidence_packet"]},
            bundle_sha256=edited["bundle_sha256"],
        )
    for unsafe_path in ("../outside.bin", r"nested\outside.bin"):
        with pytest.raises(rl.ReleaseError, match="unsafe file path"):
            rl.ArtifactBundle(
                root=tmp_path,
                manifest=bundle.manifest,
                files={unsafe_path: HASH},
            )
    with pytest.raises(rl.ReleaseError, match="unsupported release schema"):
        replace(bundle, schema_version="ml-lifecycle-release-v0")


def test_register_never_promotes_and_no_artifact_self_promotes(tmp_path):
    _, bundle = _bundle(tmp_path)
    registry = rl.ReleaseRegistry(tmp_path / "registry.json")
    entry = registry.register(bundle, rl.VALIDATION_VALIDATED)
    assert entry["artifact_id"] == bundle.artifact_id

    state = rl.ReleaseRegistry(registry.path).state()
    assert state["champion"] is None
    assert state["previous"] is None
    assert state["log"] == []
    assert set(state["candidates"]) == {bundle.artifact_id}
    assert not hasattr(bundle, "promote")
    assert not hasattr(bundle, "promote_to_champion")
    assert "promote" not in bundle.to_dict()


def test_promotion_refuses_each_independent_gate(tmp_path):
    pending = {
        "validation_status": "PENDING",
        "approval": APPROVAL_PENDING,
    }
    cases = [
        (pending, {}, "NOT_VALIDATED", "validation_status"),
        (pending, {"validation_status": "VALIDATED"}, "PENDING", "validation_status"),
        (
            {},
            {
                "manifest.evidence_level": EVIDENCE_SMOKE,
                "evidence.evidence_level": EVIDENCE_SMOKE,
            },
            rl.VALIDATION_VALIDATED,
            "evidence level",
        ),
        (
            {},
            {
                "manifest.evidence_level": EVIDENCE_SYNTHETIC,
                "evidence.evidence_level": EVIDENCE_SYNTHETIC,
            },
            rl.VALIDATION_VALIDATED,
            "evidence level",
        ),
    ]
    for index, (manifest_changes, mutations, registration_status, expected) in enumerate(
        cases
    ):
        case_path = tmp_path / str(index)
        case_path.mkdir()
        root, original = _bundle(case_path)
        bundle = original
        if manifest_changes:
            manifest = replace(original.manifest, **manifest_changes)
            bundle = rl.ArtifactBundle.from_directory(
                root,
                manifest,
                files=FILES,
                environment=original.environment,
                compatibility=original.compatibility,
                extra_evidence=original.extra_evidence,
            )
        registry = rl.ReleaseRegistry(case_path / "registry.json")
        registry.register(bundle, registration_status)
        if mutations:
            _mutate(registry, bundle.artifact_id, **mutations)
            registry = rl.ReleaseRegistry(registry.path)
        result = registry.promote(
            bundle.artifact_id, approver="synthetic-reviewer", reason="synthetic review"
        )
        assert result.accepted is False
        assert expected in result.reason
        assert result.state["champion"] is None
        assert result.state["log"] == []


def test_promotion_requires_approver_and_reason(tmp_path):
    _, bundle = _bundle(tmp_path)
    registry = _registry(tmp_path, bundle)
    for approver, reason, expected in (
        ("", "synthetic review", "approver"),
        ("  ", "synthetic review", "approver"),
        ("synthetic-reviewer", "", "reason"),
        ("synthetic-reviewer", " ", "reason"),
    ):
        result = registry.promote(bundle.artifact_id, approver=approver, reason=reason)
        assert result.accepted is False
        assert expected in result.reason
        assert result.state["champion"] is None


def test_promotion_is_atomic_logged_and_visible_after_cross_process_reload(tmp_path):
    _, first = _bundle(tmp_path, artifact_id="champion-a")
    _, second = _bundle(tmp_path, artifact_id="champion-b")
    registry = rl.ReleaseRegistry(tmp_path / "registry.json")
    registry.register(first, rl.VALIDATION_VALIDATED)
    registry.register(second, rl.VALIDATION_VALIDATED)

    first_result = registry.promote(
        first.artifact_id, approver="release-owner", reason="first reviewed release"
    )
    assert first_result.accepted and first_result.reason == "promoted"
    assert _artifact_id(first_result.state["champion"]) == first.artifact_id
    assert first_result.state["previous"] is None

    reread_first = rl.ReleaseRegistry(registry.path).state()
    assert _artifact_id(reread_first["champion"]) == first.artifact_id
    second_result = rl.ReleaseRegistry(registry.path).promote(
        second.artifact_id, approver="release-owner", reason="second reviewed release"
    )
    assert second_result.accepted
    assert _artifact_id(second_result.state["champion"]) == second.artifact_id
    assert _artifact_id(second_result.state["previous"]) == first.artifact_id
    assert second_result.state["log"][-1]["action"] == "promote"
    assert second_result.state["log"][-1]["approver"] == "release-owner"
    assert second_result.state["log"][-1]["reason"] == "second reviewed release"
    assert second_result.state["log"][-1]["timestamp"]
    assert second_result.state["log"][-1]["artifact_sha256"] == second.bundle_sha256
    assert len(second_result.state["log"][-1]["state_sha256"]) == 64
    assert second_result.state["log"][-1]["state_sha256"] == digest(
        {key: value for key, value in second_result.state.items() if key != "log"}
    )

    reread_second = rl.ReleaseRegistry(registry.path).state()
    assert _artifact_id(reread_second["champion"]) == second.artifact_id
    assert _artifact_id(reread_second["previous"]) == first.artifact_id
    assert reread_second["log"] == second_result.state["log"]
    assert list(registry.path.parent.glob(".*.tmp")) == []


def test_failed_atomic_write_leaves_persisted_champion_unchanged(tmp_path, monkeypatch):
    _, bundle = _bundle(tmp_path)
    registry = _registry(tmp_path, bundle)
    before = registry.state()

    def fail_replace(source: str, target: str) -> None:
        raise OSError("synthetic disk failure")

    monkeypatch.setattr("lifecycle.contracts.os.replace", fail_replace)
    result = registry.promote(
        bundle.artifact_id, approver="release-owner", reason="reviewed release"
    )
    assert result.accepted is False
    assert "atomic write failed" in result.reason
    assert result.state == before
    assert rl.ReleaseRegistry(registry.path).state() == before
    assert list(registry.path.parent.glob(".*.tmp")) == []


def test_rollback_restores_previous_champion_and_is_logged(tmp_path):
    _, first = _bundle(tmp_path, artifact_id="champion-a")
    _, second = _bundle(tmp_path, artifact_id="champion-b")
    registry = rl.ReleaseRegistry(tmp_path / "registry.json")
    registry.register(first, rl.VALIDATION_VALIDATED)
    registry.register(second, rl.VALIDATION_VALIDATED)
    registry.promote(first.artifact_id, approver="release-owner", reason="first release")
    registry.promote(second.artifact_id, approver="release-owner", reason="second release")

    result = rl.ReleaseRegistry(registry.path).rollback(
        "release-owner", "candidate regression observed"
    )
    assert result.accepted
    assert _artifact_id(result.state["champion"]) == first.artifact_id
    assert result.state["previous"] is None
    assert result.state["log"][-1]["action"] == "rollback"
    assert result.state["log"][-1]["approver"] == "release-owner"
    assert result.state["log"][-1]["reason"] == "candidate regression observed"
    assert result.state["log"][-1]["timestamp"]
    assert result.state["log"][-1]["artifact_sha256"] == first.bundle_sha256
    reread = rl.ReleaseRegistry(registry.path).state()
    assert _artifact_id(reread["champion"]) == first.artifact_id
    assert reread["log"] == result.state["log"]


def test_rollback_without_previous_is_refused_with_a_reason(tmp_path):
    _, bundle = _bundle(tmp_path)
    registry = _registry(tmp_path, bundle)
    result = registry.rollback("release-owner", "no previous champion")
    assert result.accepted is False
    assert "no previous champion" in result.reason
    assert result.state["champion"] is None
    assert result.state["log"] == []

    registry.promote(bundle.artifact_id, approver="release-owner", reason="first release")
    again = registry.rollback("release-owner", "already restored")
    assert again.accepted is False
    assert "no previous champion" in again.reason
    assert _artifact_id(again.state["champion"]) == bundle.artifact_id


def test_unknown_artifact_id_is_refused_with_a_reason(tmp_path):
    _, bundle = _bundle(tmp_path)
    registry = _registry(tmp_path, bundle)
    result = registry.promote(
        "does-not-exist", approver="release-owner", reason="unknown artifact"
    )
    assert result.accepted is False
    assert "unknown artifact id" in result.reason
    assert result.state["champion"] is None


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ({"schema_version": "ml-lifecycle-release-v0"}, "version-incompatible"),
        ({"contracts_version": "ml-lifecycle-contracts-v0"}, "contracts-version-incompatible"),
        ({"manifest.validation_status": "PENDING"}, "validation_status"),
        ({"manifest.approval": APPROVAL_PENDING}, "approval"),
        ({"manifest.evidence_level": EVIDENCE_SMOKE}, "inconsistent"),
        ({"evidence.approval": APPROVAL_PENDING}, "approval"),
        ({"evidence.evidence_level": EVIDENCE_SYNTHETIC}, "inconsistent"),
        ({"compatibility.format": "foreign-format-v2"}, "unsupported format"),
        ({"compatibility.hardware": "gpu"}, "unsupported hardware"),
        ({"manifest_sha256": HASH}, "bad hash"),
    ],
)
def test_version_and_manifest_tampering_are_refused_with_reasons(
    tmp_path, mutation, expected
):
    _, bundle = _bundle(tmp_path / expected)
    registry = _registry(tmp_path / expected, bundle)
    _mutate(registry, bundle.artifact_id, **mutation)
    result = rl.ReleaseRegistry(registry.path).promote(
        bundle.artifact_id, approver="release-owner", reason="reviewed release"
    )
    assert result.accepted is False
    assert expected in result.reason
    assert result.state["champion"] is None
    assert result.state["log"] == []


def test_registry_file_incompatible_version_is_refused(tmp_path):
    path = tmp_path / "registry.json"
    path.write_text(
        canonical_json({"schema_version": "old", "contracts_version": CONTRACTS_VERSION}),
        encoding="utf-8",
    )
    with pytest.raises(rl.ReleaseError, match="unknown registry version"):
        rl.ReleaseRegistry(path)


def test_bundle_manifest_corruption_after_registration_is_refused(tmp_path):
    root, bundle = _bundle(tmp_path)
    registry = _registry(tmp_path, bundle)
    (root / "state.json").write_text(
        canonical_json({"synthetic": True, "model": "tampered"}), encoding="utf-8"
    )
    result = registry.promote(
        bundle.artifact_id, approver="release-owner", reason="reviewed release"
    )
    assert result.accepted is False
    assert 'bundle file verification failed: {"changed":["state.json"]' in result.reason
    assert result.state["champion"] is None


def test_benchmark_percentiles_and_deltas_come_from_supplied_samples():
    baseline = rl.BenchmarkReport(
        latency_ms=_delays(),
        throughput_per_s=100.0,
        peak_memory_bytes=1024,
        environment="synthetic-cpu-baseline",
        evidence_level=EVIDENCE_SYNTHETIC,
        measurements={"runner": "synthetic"},
    )
    candidate = rl.BenchmarkReport(
        latency_ms=tuple(value * 2 for value in _delays()),
        throughput_per_s=250.0,
        peak_memory_bytes=2048,
        environment="synthetic-cpu-candidate",
        evidence_level=EVIDENCE_SYNTHETIC,
    )
    assert baseline.p50_ms == 50.5
    assert baseline.p95_ms == 95.05
    assert baseline.p99_ms == 99.01
    assert candidate.p50_ms == 101.0
    assert candidate.p95_ms == 190.1
    assert candidate.p99_ms == 198.02

    measured = baseline.to_dict()
    assert measured["measurement"] == "MEASURED"
    assert measured["latency_ms"]["samples"] == list(_delays())
    assert measured["throughput_per_s"] == 100.0
    assert measured["peak_memory_bytes"] == 1024
    assert measured["measurements"] == {"runner": "synthetic"}
    comparison = rl.BenchmarkReport.compare(baseline, candidate)
    assert comparison["measurement"] == "MEASURED_DELTA"
    assert comparison["p50_ms"] == {
        "baseline": 50.5,
        "candidate": 101.0,
        "delta": 50.5,
    }
    assert comparison["p95_ms"]["delta"] == pytest.approx(95.05)
    assert comparison["p99_ms"]["delta"] == pytest.approx(99.01)
    assert comparison["throughput_per_s"]["delta"] == 150.0
    assert comparison["peak_memory_bytes"]["delta"] == 1024
    assert comparison["verdict"] == "NOT_EVALUATED"
    assert not any("threshold" in str(value) for value in comparison["gates"].values())
    assert "sla" not in canonical_json(comparison).lower()


def test_benchmark_refuses_invented_or_negative_measurements():
    with pytest.raises(rl.ReleaseError, match="latency samples must not be empty"):
        rl.BenchmarkReport((), 1.0, 1, "synthetic-cpu")
    with pytest.raises(rl.ReleaseError, match="non-negative"):
        rl.BenchmarkReport((-1.0,), 1.0, 1, "synthetic-cpu")
    with pytest.raises(rl.ReleaseError, match="throughput"):
        rl.BenchmarkReport((1.0,), -1.0, 1, "synthetic-cpu")
    with pytest.raises(rl.ReleaseError, match="peak_memory"):
        rl.BenchmarkReport((1.0,), 1.0, -1, "synthetic-cpu")


def test_cpu_first_deployment_plan_has_no_guarantees_or_hardware_connection():
    cpu = rl.cpu_first_check()
    assert cpu["target"] == rl.CPU_TARGET
    assert cpu["capability"] == STATUS_OK
    assert cpu["status"] == STATUS_OK
    assert cpu["connected"] is True

    gpu = rl.cpu_first_check("gpu")
    assert gpu["capability"] == CAP_NOT_CONNECTED
    assert gpu["status"] == STATUS_BLOCKED
    assert gpu["connected"] is False

    benchmark = rl.BenchmarkReport(
        latency_ms=(1.0, 2.0, 3.0),
        throughput_per_s=3.0,
        peak_memory_bytes=64,
        environment="synthetic-cpu",
    )
    plan = rl.deployment_plan(benchmark=benchmark)
    assert plan["target"] == rl.CPU_TARGET
    assert plan["environment"] == "synthetic-cpu"
    assert plan["unsupported_hardware"] == {
        "capability": CAP_NOT_CONNECTED,
        "status": STATUS_BLOCKED,
    }
    assert plan["inference_status"] == STATUS_NOT_AVAILABLE
    assert plan["benchmark"] == benchmark.to_dict()
    assert "no clinical or performance guarantee is claimed" in plan["notes"]
    blocked = rl.deployment_plan(hardware="gpu")
    assert blocked["cpu_first"]["status"] == STATUS_BLOCKED
    assert blocked["inference_status"] == STATUS_NOT_AVAILABLE
    assert blocked["benchmark"] is None
    text = canonical_json(plan).lower()
    assert "guaranteed" not in text
    assert "sla" not in text


def test_gates_stay_independent_when_engineering_is_green(tmp_path):
    _, bundle = _bundle(tmp_path)
    registry = _registry(tmp_path, bundle)
    result = registry.promote(
        bundle.artifact_id, approver="release-owner", reason="reviewed release"
    )
    gates = gate_bundle(True)
    assert gates[GATE_ENGINEERING] == GATE_GRANTED
    assert gates[GATE_REAL_DATA_TRAINING] == GATE_NOT_GRANTED
    assert gates[GATE_RESEARCH_CLAIMS] == GATE_NOT_GRANTED
    assert result.state["log"][-1]["gates"] == gates
    assert rl.cpu_first_check()["gates"] == gates
    assert rl.deployment_plan()["gates"] == gates
    assert rl.BenchmarkReport.compare(
        rl.BenchmarkReport((1.0,), 1.0, 1, "a"),
        rl.BenchmarkReport((2.0,), 2.0, 2, "b"),
    )["gates"] == gates


def _artifact_id(entry: object) -> str | None:
    return entry.get("artifact_id") if isinstance(entry, dict) else None
