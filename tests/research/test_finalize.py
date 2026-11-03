"""Synthetic finalization order, manifest binding, and holdout release tests."""
from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path

import pytest

from data.mendeley import FEATURE_GROUP_SIZES, SubjectRecord
from data.real_split import build_real_manifest
from research.finalize import _sha256

pytestmark = pytest.mark.synthetic
DATASET_SHA = "f" * 64
CANDIDATE_SOURCE = (
    "def build_model(fold):\n"
    "    return None\n\n"
    "def train(model, fold, budget_seconds, seed):\n"
    "    return None\n"
)
WEIGHTS_SHA = "a" * 64
ARCHITECTURE_SHA = "b" * 64
IMAGE_ID = "sha256:" + "1" * 64


def fake_records(n=40, seed=3):
    rng = random.Random(seed)
    records = []
    for i in range(1, n + 1):
        scores = [None] + [float(rng.randint(10, 90)) for _ in range(8)]
        records.append(SubjectRecord(
            subject_index=i,
            features={
                name: [rng.random() for _ in range(size)]
                for name, size in FEATURE_GROUP_SIZES.items()
            },
            scores=scores,
            score_mask=[value is not None for value in scores],
            provenance={"source": "synthetic-test-fixture"},
        ))
    return records


def _git_init(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)


def _worktree(tmp_path, records):
    worktree = tmp_path / "wt"
    candidate = worktree / "src" / "research" / "candidate.py"
    candidate.parent.mkdir(parents=True)
    candidate.write_text(CANDIDATE_SOURCE)
    _git_init(worktree)
    subprocess.run(["git", "add", "-A"], cwd=worktree, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=worktree, check=True)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=worktree, check=True,
        text=True, capture_output=True,
    ).stdout.strip()
    frozen = build_real_manifest(records, dataset_sha256=DATASET_SHA)
    manifest_path = tmp_path / "frozen.json"
    manifest_path.write_text(json.dumps(frozen.data, ensure_ascii=False))
    return worktree, candidate, manifest_path, frozen, head


def _patch_preflight(monkeypatch):
    from research import finalize as finalize_mod
    from research import spawn as spawn_mod

    def sandbox_run(config, *, preflight=False, **kwargs):
        assert preflight is True
        assert config["time_budget_seconds"] > 0
        assert kwargs["timeout"] == spawn_mod.sandbox_worker_process_timeout(
            config["time_budget_seconds"]
        )
        return {
            "candidate_sha256": config["candidate_sha256"],
            "task": config["task"],
            "output_key": "category_probs",
            "output_shape": [8, 9],
            "weights_sha256": WEIGHTS_SHA,
            "architecture_sha256": ARCHITECTURE_SHA,
            "sandbox_image_id": IMAGE_ID,
        }

    monkeypatch.setattr(spawn_mod, "run_sandboxed_candidate", sandbox_run)
    monkeypatch.setattr(spawn_mod, "sandbox_image_identity", lambda: IMAGE_ID)
    monkeypatch.setattr(
        finalize_mod,
        "_validate_preflight_artifacts",
        lambda result, task, candidate_sha, config: {
            "output_key": "category_probs",
            "output_shape": [8, 9],
            "weights_sha256": WEIGHTS_SHA,
            "architecture_sha256": ARCHITECTURE_SHA,
            "sandbox_image_id": IMAGE_ID,
        },
    )


def _fit_result(config):
    pool = json.loads(Path(config["data"]["pool_manifest_path"]).read_text())
    return {
        "finalize": {
            "candidate_file": config["editable"],
            "candidate_sha256": config["candidate_sha256"],
            "git_commit": config["git_commit"],
            "pool_manifest_sha256": pool["manifest_sha256"],
            "selection_parent_manifest_sha256": pool[
                "selection_parent_manifest_sha256"
            ],
            "candidate_output": {
                "output_key": "category_probs",
                "output_shape": [32, 9],
            },
            "fit_seconds": 0.01,
        },
        "weights_path": str(Path(config["out_dir"]) / "champion" / "weights.pt"),
        "architecture_path": str(
            Path(config["out_dir"]) / "champion" / "weights-architecture.json"
        ),
        "weights_sha256": WEIGHTS_SHA,
        "architecture_sha256": ARCHITECTURE_SHA,
    }


def _patch_fitted_validator(monkeypatch, before_charge=None):
    from research import finalize as finalize_mod

    def validate(result, **kwargs):
        if before_charge is not None:
            before_charge()
        return {
            "output_key": "category_probs",
            "output_shape": [32, 9],
            "weights_sha256": WEIGHTS_SHA,
            "architecture_sha256": ARCHITECTURE_SHA,
        }

    monkeypatch.setattr(finalize_mod, "_validate_fitted_artifacts", validate)


def _refit_kwargs(worktree, manifest_path, head, records, out_dir):
    return {
        "worktree": worktree,
        "out_dir": out_dir,
        "manifest_path": manifest_path,
        "dataset_path": "synthetic-unused",
        "editable": "src/research/candidate.py",
        "commit": head,
        "records": records,
        "task": "constitution",
    }


@pytest.mark.parametrize("failure", ["task", "schema", "candidate_sha"])
def test_bad_task_schema_or_candidate_commit_fails_before_ledger(
    tmp_path, monkeypatch, failure,
):
    from research import finalize as finalize_mod

    records = fake_records()
    worktree, candidate, manifest_path, _frozen, head = _worktree(tmp_path, records)
    kwargs = _refit_kwargs(worktree, manifest_path, head, records, tmp_path / "out")
    if failure == "task":
        kwargs["task"] = "unknown"
        match = "unknown research task"
    elif failure == "schema":
        manifest_path.write_text(json.dumps({"task_type": "wrong"}))
        match = "manifest"
    else:
        candidate.write_text(CANDIDATE_SOURCE + "# changed after commit\n")
        match = "candidate file SHA"

    with pytest.raises((RuntimeError, ValueError), match=match):
        finalize_mod.refit_and_score(**kwargs)
    assert not (worktree / "runs" / "holdout-unlock-ledger.jsonl").exists()


def test_container_preflight_failure_does_not_charge_unlock(tmp_path, monkeypatch):
    from research import finalize as finalize_mod
    from research import spawn as spawn_mod

    records = fake_records()
    worktree, _candidate, manifest_path, _frozen, head = _worktree(tmp_path, records)
    ledger = finalize_mod.global_ledger_path(worktree)
    called = []

    def failed_preflight(config, *, preflight=False, **_kwargs):
        assert preflight is True
        called.append("container-preflight")
        raise RuntimeError("synthetic container syntax rejection")

    monkeypatch.setattr(spawn_mod, "run_sandboxed_candidate", failed_preflight)
    kwargs = _refit_kwargs(worktree, manifest_path, head, None, tmp_path / "out")
    kwargs["dataset_path"] = tmp_path / "unread-real-data"
    with pytest.raises(RuntimeError, match="container syntax rejection"):
        finalize_mod.refit_and_score(**kwargs)
    assert called == ["container-preflight"]
    assert not ledger.exists()


def test_fit_failure_does_not_charge_unlock(tmp_path, monkeypatch):
    from research import finalize as finalize_mod

    records = fake_records()
    worktree, _candidate, manifest_path, _frozen, head = _worktree(tmp_path, records)
    _patch_preflight(monkeypatch)
    kwargs = _refit_kwargs(worktree, manifest_path, head, records, tmp_path / "out")
    ledger = finalize_mod.global_ledger_path(worktree)

    def failed_fit(config, config_path):
        assert not ledger.exists()
        assert "ledger" not in json.dumps(config)
        assert "holdout" not in json.dumps(config)
        raise RuntimeError("synthetic fit failure")

    with pytest.raises(RuntimeError, match="synthetic fit failure"):
        finalize_mod.refit_and_score(**kwargs, spawn=failed_fit)
    assert not ledger.exists()


def test_fit_worker_timeout_covers_trusted_worker_startup_export_and_cleanup(
    tmp_path, monkeypatch,
):
    from research import finalize as finalize_mod
    from research import spawn as spawn_mod
    from research.tasks import adapter_for

    records = fake_records()
    worktree, candidate, manifest_path, frozen, head = _worktree(tmp_path, records)
    adapter = adapter_for("constitution")
    selection = adapter.pool_manifest(
        adapter.load_manifest(manifest_path), records, notes="synthetic timeout test",
    ).data
    pool_path = tmp_path / "pool-manifest.json"
    pool_path.write_text(json.dumps(selection, ensure_ascii=False))
    pool_rows = set(selection["subject_row_by_id"].values())
    pool_records = [record for record in records if record.subject_index in pool_rows]

    source_root = tmp_path / "sandbox-source"
    staged_candidate = source_root / "research" / "candidate.py"
    staged_candidate.parent.mkdir(parents=True)
    staged_candidate.write_bytes(candidate.read_bytes())
    output_dir = tmp_path / "out"
    (output_dir / "champion").mkdir(parents=True)
    budget = 20.0
    candidate_output = {"output_key": "category_probs", "output_shape": [32, 9]}
    observed = {}

    def sandbox_run(config, *, timeout, **kwargs):
        observed["timeout"] = timeout
        observed["candidate_output"] = config.get("candidate_output")
        return {
            "fit_seconds": 0.01,
            "input_spec_version": "synthetic-v1",
            "candidate_output": candidate_output,
            "sandbox_image_id": IMAGE_ID,
            "weights_sha256": WEIGHTS_SHA,
            "architecture_sha256": ARCHITECTURE_SHA,
            "weights_path": str(output_dir / "champion" / "weights.pt"),
            "architecture_path": str(
                output_dir / "champion" / "weights-architecture.json"
            ),
        }

    monkeypatch.setattr(spawn_mod, "run_sandboxed_candidate", sandbox_run)
    config = {
        "worktree_path": str(worktree),
        "sandbox_source_path": str(source_root),
        "editable": "src/research/candidate.py",
        "candidate_sha256": _sha256(candidate),
        "git_commit": head,
        "task": "constitution",
        "seed": 5,
        "time_budget_seconds": budget,
        "train_batch_size": 8,
        "out_dir": str(output_dir),
        "candidate_preflight": {"sandbox_image_id": IMAGE_ID},
        "data_parent_manifest_sha256": frozen.data["manifest_sha256"],
        "data": {
            "pool_manifest_path": str(pool_path),
            "pool_records_path": str(tmp_path / "pool-records.json"),
        },
    }

    result = finalize_mod._worker(config, records=pool_records)

    assert observed["timeout"] == spawn_mod.sandbox_worker_process_timeout(budget)
    assert observed["timeout"] > budget
    assert result["finalize"]["candidate_output"] == candidate_output


def test_invalid_trusted_artifact_does_not_charge_unlock(tmp_path, monkeypatch):
    from research import finalize as finalize_mod

    records = fake_records()
    worktree, _candidate, manifest_path, _frozen, head = _worktree(tmp_path, records)
    _patch_preflight(monkeypatch)
    kwargs = _refit_kwargs(worktree, manifest_path, head, records, tmp_path / "out")
    ledger = finalize_mod.global_ledger_path(worktree)

    def invalid_validator(*args, **kwargs):
        assert not ledger.exists()
        raise RuntimeError("synthetic artifact contract mismatch")

    monkeypatch.setattr(finalize_mod, "_validate_fitted_artifacts", invalid_validator)
    with pytest.raises(RuntimeError, match="artifact contract mismatch"):
        finalize_mod.refit_and_score(
            **kwargs, spawn=lambda config, _path: _fit_result(config),
        )
    assert not ledger.exists()


def test_budget_exhaustion_refuses_before_preflight_and_dataset_read(
    tmp_path, monkeypatch,
):
    from research import finalize as finalize_mod
    from research import spawn as spawn_mod

    records = fake_records()
    worktree, _candidate, manifest_path, _frozen, head = _worktree(tmp_path, records)
    ledger = finalize_mod.global_ledger_path(worktree)
    ledger.parent.mkdir(parents=True)
    ledger.write_text(
        json.dumps({"event_id": "1" * 32}) + "\n"
        + json.dumps({"event_id": "2" * 32}) + "\n"
    )
    called = []
    monkeypatch.setattr(
        spawn_mod, "run_sandboxed_candidate",
        lambda *args, **kwargs: called.append("candidate"),
    )
    kwargs = _refit_kwargs(worktree, manifest_path, head, None, tmp_path / "out")
    kwargs["dataset_path"] = tmp_path / "missing-real-dataset"
    with pytest.raises(RuntimeError, match="unlock budget"):
        finalize_mod.refit_and_score(**kwargs)
    assert called == []
    assert finalize_mod.unlock_count(ledger) == 2


def test_success_charges_after_fit_validation_before_holdout(
    tmp_path, monkeypatch,
):
    from research import finalize as finalize_mod
    from research import train_worker

    monkeypatch.setattr(
        train_worker,
        "validate_candidate_source",
        lambda _source: pytest.fail("candidate validation must run in the sandbox"),
    )
    records = fake_records()
    worktree, candidate, manifest_path, frozen, head = _worktree(tmp_path, records)
    _patch_preflight(monkeypatch)
    kwargs = _refit_kwargs(worktree, manifest_path, head, records, tmp_path / "out")
    ledger = finalize_mod.global_ledger_path(worktree)
    validation_seen = []

    def before_charge():
        validation_seen.append(True)
        assert not ledger.exists()

    _patch_fitted_validator(monkeypatch, before_charge=before_charge)

    def fit(config, _config_path):
        assert not ledger.exists()
        assert "ledger" not in json.dumps(config)
        assert "holdout" not in json.dumps(config)
        return _fit_result(config)

    def holdout(config, request):
        assert validation_seen == [True]
        assert finalize_mod.unlock_count(ledger) == 1
        event = json.loads(ledger.read_text().splitlines()[0])
        assert event["task"] == "constitution"
        assert event["candidate_sha256"] == _sha256(candidate)
        assert event["champion_commit"] == head
        assert event["pool_manifest_sha256"] == json.loads(
            Path(config["data"]["pool_manifest_path"]).read_text()
        )["manifest_sha256"]
        assert config["unlock"]["event_id"] == event["event_id"]
        assert "records_path" in config["data"]
        assert "pool_records_path" not in config["data"]
        return {"holdout": {"synthetic": True}, "baselines": {}}

    result = finalize_mod.refit_and_score(
        **kwargs, spawn=fit, spawn_holdout=holdout,
    )
    assert validation_seen == [True]
    assert result["holdout"]["synthetic"] is True
    assert result["finalize"]["holdout_rows_untouched"] == len(
        frozen.data["split"]["holdout"]
    )
    assert finalize_mod.unlock_count(ledger) == 1


def _holdout_fixture(tmp_path):
    from research import finalize as finalize_mod
    from research.tasks import adapter_for

    records = fake_records()
    worktree, candidate, frozen_path, _frozen_built, head = _worktree(tmp_path, records)
    adapter = adapter_for("constitution")
    frozen = adapter.load_manifest(frozen_path)
    pool = adapter.pool_manifest(frozen, records, notes="synthetic").data
    pool_path = tmp_path / "pool.json"
    pool_path.write_text(json.dumps(pool, ensure_ascii=False))
    ledger = finalize_mod.global_ledger_path(worktree, "constitution")
    event_id = "e" * 32
    preflight = {
        "output_key": "category_probs",
        "output_shape": [8, 9],
        "weights_sha256": "e" * 64,
        "architecture_sha256": "d" * 64,
        "sandbox_image_id": IMAGE_ID,
        "synthetic_manifest_sha256": "c" * 64,
    }
    fit_artifacts = {
        "output_key": "category_probs",
        "output_shape": [32, 9],
        "weights_sha256": WEIGHTS_SHA,
        "architecture_sha256": ARCHITECTURE_SHA,
    }
    unlock = {
        "ledger_path": str(ledger),
        "event_id": event_id,
        "task": "constitution",
        "candidate_sha256": _sha256(candidate),
        "champion_commit": head,
        "pool_manifest_sha256": pool["manifest_sha256"],
        "selection_parent_manifest_sha256": frozen["manifest_sha256"],
        "candidate_preflight": preflight,
        "fit_artifacts": fit_artifacts,
    }
    event = {
        "event": "holdout_unlock_intent",
        "event_id": event_id,
        "task": "constitution",
        "candidate_sha256": _sha256(candidate),
        "champion_commit": head,
        "candidate_preflight": preflight,
        "fit_artifacts": fit_artifacts,
        "pool_manifest_sha256": pool["manifest_sha256"],
        "selection_parent_manifest_sha256": frozen["manifest_sha256"],
    }
    config = {
        "worktree_path": str(worktree),
        "task": "constitution",
        "unlock": unlock,
        "data": {
            "pool_manifest_path": str(pool_path),
            "frozen_manifest_path": str(frozen_path),
        },
    }
    request = {
        "weights_path": "synthetic-weights.pt",
        "architecture_path": "synthetic-architecture.json",
        "weights_sha256": WEIGHTS_SHA,
        "architecture_sha256": ARCHITECTURE_SHA,
    }
    return finalize_mod, worktree, ledger, config, request, event


@pytest.mark.parametrize("ledger_mode", ["missing", "wrong_candidate", "wrong_pool"])
def test_holdout_refuses_before_loading_records(tmp_path, monkeypatch, ledger_mode):
    from research import evaluator as evaluator_mod
    from research.tasks import ConstitutionAdapter

    _finalize_mod, _worktree, ledger, config, request, event = _holdout_fixture(tmp_path)
    records_loaded = []
    monkeypatch.setattr(
        ConstitutionAdapter, "load_eval_records",
        lambda self, config: records_loaded.append(True),
    )
    if ledger_mode != "missing":
        if ledger_mode == "wrong_candidate":
            event["candidate_sha256"] = "0" * 64
        else:
            event["pool_manifest_sha256"] = "0" * 64
        ledger.parent.mkdir(parents=True, exist_ok=True)
        ledger.write_text(json.dumps(event) + "\n")
    with pytest.raises(RuntimeError, match="holdout unlock"):
        evaluator_mod.score_holdout(config, request)
    assert records_loaded == []


def test_valid_holdout_event_allows_synthetic_provider_after_validation(
    tmp_path, monkeypatch,
):
    from research import evaluator as evaluator_mod
    from research.tasks import ConstitutionAdapter

    _finalize_mod, _worktree, ledger, config, request, event = _holdout_fixture(tmp_path)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(json.dumps(event) + "\n")
    records_loaded = []
    monkeypatch.setattr(
        evaluator_mod, "_load_model", lambda _request, *, task: object()
    )
    monkeypatch.setattr(
        ConstitutionAdapter, "load_eval_records",
        lambda self, config: records_loaded.append(True) or ["synthetic"],
    )
    monkeypatch.setattr(
        ConstitutionAdapter, "score_holdout",
        lambda self, config, request, records, model: {
            "holdout": {"synthetic": True}, "baselines": {},
        },
    )
    result = evaluator_mod.score_holdout(config, request)
    assert records_loaded == [True]
    assert result["holdout"]["synthetic"] is True


def test_ledger_validation_fails_closed_on_corrupt_or_duplicate_events(tmp_path):
    from research import finalize as finalize_mod

    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text('{"event_id": "bad"}\n')
    with pytest.raises(RuntimeError, match="event_id"):
        finalize_mod.unlock_count(ledger)
    ledger.write_text(
        json.dumps({"event_id": "a" * 32}) + "\n"
        + json.dumps({"event_id": "a" * 32}) + "\n"
    )
    with pytest.raises(RuntimeError, match="duplicate"):
        finalize_mod.unlock_count(ledger)


def test_trusted_artifact_pair_caps_architecture_before_loading(tmp_path, monkeypatch):
    import hashlib

    from research import model_export
    from research.finalize import _trusted_artifact_pair

    output_dir = tmp_path / "trusted-output"
    output_dir.mkdir()
    weights = output_dir / "weights.pt"
    architecture = output_dir / "architecture.json"
    weights.write_bytes(b"synthetic weights")
    architecture.write_bytes(
        b" " * (model_export.MAX_ARCHITECTURE_BYTES + 1)
    )
    result = {
        "weights_path": str(weights),
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "architecture_path": str(architecture),
        "architecture_sha256": hashlib.sha256(
            architecture.read_bytes()
        ).hexdigest(),
    }

    with pytest.raises(RuntimeError, match="architecture.*limit"):
        _trusted_artifact_pair(
            result, output_dir, task="constitution"
        )


def test_preflight_uses_sandbox_validation_without_host_model_load(tmp_path, monkeypatch):
    import hashlib

    from research import model_export
    from research.finalize import _validate_preflight_artifacts
    output = tmp_path / "sandbox-output"
    output.mkdir()
    weights = output / "weights.pt"
    architecture = output / "weights-architecture.json"
    weights.write_bytes(b"opaque bounded weights")
    architecture.write_bytes(b"opaque bounded architecture")
    weights_sha = hashlib.sha256(weights.read_bytes()).hexdigest()
    architecture_sha = hashlib.sha256(architecture.read_bytes()).hexdigest()
    def host_load_forbidden(*_args, **_kwargs):
        pytest.fail("finalize must consume the sandbox attestation, not rebuild on host")
    monkeypatch.setattr(model_export, "load_eval_model", host_load_forbidden)
    result = {
        "candidate_sha256": "a" * 64,
        "task": "constitution",
        "output_key": "category_probs",
        "output_shape": [8, 9],
        "weights_path": str(weights),
        "weights_sha256": weights_sha,
        "architecture_path": str(architecture),
        "architecture_sha256": architecture_sha,
        "trusted_validation": {
            "task": "constitution",
            "output_key": "category_probs",
            "output_shape": [8, 9],
            "weights_sha256": weights_sha,
            "architecture_sha256": architecture_sha,
        },
    }
    validated = _validate_preflight_artifacts(
        result, "constitution", "a" * 64, {"output_dir": str(output)},
    )
    assert validated["output_key"] == "category_probs"
    assert validated["output_shape"] == [8, 9]
    assert validated["weights_sha256"] == weights_sha
