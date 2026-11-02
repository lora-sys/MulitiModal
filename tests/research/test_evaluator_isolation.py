"""Evaluator-process isolation tests (F-01 round, 2026-11-02).

The first round already moved scoring out of the trainer's process; the
second hardening round closes the remaining hole: `torch.load` of the full
pickled model executed candidate code inside the evaluator (module-level
code at unpickle import + a weights_only=False pickle could rebind
`sys.modules["runtime.tcm_eval"]._average_precision_from_tensors`). These
tests pin the closed design:

* the evaluator loads state_dict only (`weights_only=True`) into a model
  rebuilt from the frozen architecture whitelist — no candidate import, no
  executable pickle anywhere on the evaluation path;
* the saved model files are hash-checked against the parent's record;
* the fold is rebuilt from the frozen manifest, so in-place tampering with
  the trainer's objects cannot reach the score;
* the runner never imports the candidate either (F-02).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import pytest

from data.mendeley import FEATURE_GROUP_SIZES, SubjectRecord, records_to_json
from data.real_split import build_real_manifest, pool_manifest

pytestmark = pytest.mark.synthetic



def fake_records(n=30, seed=11):
    rng = random.Random(seed)
    records = []
    for i in range(1, n + 1):
        scores = [None] + [float(rng.randint(10, 90)) for _ in range(8)]
        records.append(
            SubjectRecord(
                subject_index=i,
                features={
                    name: [rng.random() for _ in range(size)]
                    for name, size in FEATURE_GROUP_SIZES.items()
                },
                scores=scores,
                score_mask=[s is not None for s in scores],
                provenance={"source": "test-fixture"},
            )
        )
    return records


def _config(tmp_path, records):
    full_payload = records_to_json(records).encode("utf-8")
    frozen = build_real_manifest(
        records, dataset_sha256=hashlib.sha256(full_payload).hexdigest()
    ).data
    selection = pool_manifest(frozen, records).data
    selection_rows = set(selection["subject_row_by_id"].values())
    selection_records = [
        record for record in records if record.subject_index in selection_rows
    ]
    dataset_payload = records_to_json(selection_records).encode("utf-8")
    selection["dataset_sha256"] = hashlib.sha256(dataset_payload).hexdigest()
    selection.pop("manifest_sha256", None)
    selection["manifest_sha256"] = hashlib.sha256(
        json.dumps(selection, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    manifest_path = tmp_path / "selection-manifest.json"
    manifest_path.write_text(json.dumps(selection, ensure_ascii=False))
    dataset_path = tmp_path / "selection-records.json"
    dataset_path.write_bytes(dataset_payload)
    config = {
        "data": {
            "manifest_path": str(manifest_path),
            "dataset_path": str(dataset_path),
        },
        "train_provider": {"batch_size": 8},
        "trainer": {"seed": 0},
    }
    return config, selection, manifest_path, selection_records


def _rehash_manifest(manifest):
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return manifest


def _saved_model(tmp_path, config, records, manifest, stem="m0"):
    from research.candidate import build_model
    from research.evaluator import integrity_snapshot
    from research.frozen import build_fold
    from research.model_export import save_eval_model

    fold_data, _ = build_fold(
        config, manifest, 0, 4, 0, records=records
    )
    model = build_model(fold_data)
    saved = save_eval_model(model, tmp_path, stem)
    request = {
        "fold": 0, "folds": 4, "repeat": 0,
        **saved,
        "integrity": integrity_snapshot(config),
    }
    return fold_data, model, request


def _fold_manifest(config, records, manifest):
    from research.frozen import build_fold

    _, data = build_fold(config, manifest, 0, 4, 0, records=records)
    return data


def test_the_evaluator_scores_state_dict_only(tmp_path):
    from research.evaluator import score_fold
    from runtime.real_data import real_eval_manifest
    from runtime.tcm_eval import TCMTrustedEvaluator

    records = fake_records()
    config, manifest, _manifest_path, selection_records = _config(tmp_path, records)
    config["worktree_path"] = str(Path(__file__).resolve().parents[2])
    config["editable"] = "src/research/candidate.py"
    fold_data, model, request = _saved_model(
        tmp_path, config, selection_records, manifest
    )
    scored = score_fold(config, request, records=selection_records)
    direct = TCMTrustedEvaluator(
        real_eval_manifest(_fold_manifest(config, selection_records, manifest))
    ).evaluate(model, fold_data.dev, partition="dev")
    assert scored["primary_value"] == pytest.approx(float(direct.primary_value))
    from research.runner import _spawn_evaluator
    isolated = _spawn_evaluator(config, request)
    assert isolated["primary_value"] == pytest.approx(float(direct.primary_value))


@pytest.mark.parametrize("entrypoint", ["runner", "evaluator"])
def test_selection_manifest_rejected_before_dataset_open(tmp_path, monkeypatch, entrypoint):
    config, manifest, manifest_path, _selection_records = _config(
        tmp_path, fake_records()
    )
    invalid = json.loads(json.dumps(manifest))
    holdout_id = "synthetic-holdout-row"
    invalid["subject_row_by_id"][holdout_id] = (
        max(invalid["subject_row_by_id"].values()) + 1
    )
    invalid["split"]["holdout"] = [holdout_id]
    _rehash_manifest(invalid)
    manifest_path.write_text(json.dumps(invalid, ensure_ascii=False))

    dataset_path = Path(config["data"]["dataset_path"]).absolute()
    dataset_opens = []
    original_open = os.open

    def recording_open(path, flags, *args, **kwargs):
        if Path(os.fspath(path)).absolute() == dataset_path:
            dataset_opens.append(path)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", recording_open)
    with pytest.raises(ValueError, match="empty holdout partition"):
        if entrypoint == "runner":
            import research.runner as runner_mod

            runner_mod.run(config, folds=2, repeats=1)
        else:
            from research.evaluator import score_fold

            score_fold(
                config,
                {"fold": 0, "folds": 2, "repeat": 0},
            )
    assert dataset_opens == []


@pytest.mark.parametrize("source_kind", ("oversized", "symlink"))
def test_integrity_snapshot_rejects_unbounded_or_symlink_candidate_source(
    tmp_path, source_kind
):
    from research.evaluator import integrity_snapshot
    from research.train_worker import MAX_CANDIDATE_SOURCE_BYTES

    worktree = tmp_path / "worktree"
    source_root = worktree / "src" / "research"
    source_root.mkdir(parents=True)
    candidate = source_root / "candidate.py"
    if source_kind == "oversized":
        candidate.write_bytes(b"x" * (MAX_CANDIDATE_SOURCE_BYTES + 1))
    else:
        target = tmp_path / "synthetic-target.py"
        target.write_text("synthetic only\n", encoding="utf-8")
        candidate.symlink_to(target)

    config = {
        "worktree_path": str(worktree),
        "editable": "src/research/candidate.py",
    }
    error = (
        "candidate source"
        if source_kind == "oversized"
        else "final component is a symlink"
    )
    with pytest.raises(RuntimeError, match=error):
        integrity_snapshot(
            config,
            input_hashes={
                "manifest_sha256": "a" * 64,
                "dataset_sha256": "b" * 64,
            },
        )


def test_runner_binds_task_specific_default_editable_once(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import research.evaluator as evaluator_module
    import research.frozen as frozen_module
    import research.runner as runner_module
    import research.tasks as tasks_module

    candidate = tmp_path / "src" / "research" / "candidate_sd.py"
    candidate.parent.mkdir(parents=True)
    candidate.write_text("syndrome candidate\n", encoding="utf-8")
    record = SimpleNamespace(subject_index=1)
    manifest = {"subject_row_by_id": {"syndrome-row": 1}}
    adapter = SimpleNamespace(
        default_editable="src/research/candidate_sd.py",
        fold_manifest=lambda *_args, **_kwargs: SimpleNamespace(data=manifest),
        records_to_json=lambda _records: "[]",
    )
    monkeypatch.setattr(tasks_module, "adapter_for", lambda _task: adapter)
    monkeypatch.setattr(
        evaluator_module,
        "load_selection_inputs",
        lambda *_args, **_kwargs: (
            manifest,
            [record],
            {"manifest_sha256": "a" * 64, "dataset_sha256": "b" * 64},
        ),
    )
    monkeypatch.setattr(
        frozen_module,
        "selection_fold_ids",
        lambda *_args, **_kwargs: (["syndrome-row"], []),
    )

    training_configs = []
    evaluator_configs = []

    def train(worker_config, _worker_config_path):
        training_configs.append(worker_config)
        return {
            "ok": True,
            "fit_timer": "trusted_candidate_phase_wall_v1",
            "fit_seconds": 0.1,
            "budget_seconds": 1.0,
            "weights_path": "synthetic-weights",
            "architecture_path": "synthetic-architecture",
            "weights_sha256": "c" * 64,
            "architecture_sha256": "d" * 64,
        }

    def score(config, _request):
        evaluator_configs.append(config)
        return {"primary_value": 0.5}

    config = {
        "task": "syndrome",
        "worktree_path": str(tmp_path),
        "time_budget_seconds": 1.0,
        "data": {"manifest_path": "synthetic", "dataset_path": "synthetic"},
        "trainer": {"seed": 3},
        "train_provider": {"batch_size": 2},
    }
    result = runner_module.run(
        config,
        folds=1,
        repeats=1,
        spawn_training=train,
        spawn_evaluator=score,
    )

    assert result["primary_value"] == 0.5
    assert config.get("editable") is None
    assert training_configs[0]["editable"] == "src/research/candidate_sd.py"
    assert training_configs[0]["candidate_sha256"] == hashlib.sha256(
        candidate.read_bytes()
    ).hexdigest()
    assert evaluator_configs[0]["editable"] == "src/research/candidate_sd.py"


@pytest.mark.parametrize("entrypoint", ["runner", "evaluator"])
def test_selection_manifest_seed_rejected_before_dataset_open(
    tmp_path, monkeypatch, entrypoint
):
    config, manifest, manifest_path, _selection_records = _config(
        tmp_path, fake_records()
    )
    invalid = json.loads(json.dumps(manifest))
    invalid["split"].pop("seed")
    _rehash_manifest(invalid)
    manifest_path.write_text(json.dumps(invalid, ensure_ascii=False))
    dataset_path = Path(config["data"]["dataset_path"]).absolute()
    dataset_opens = []
    original_open = os.open

    def recording_open(path, flags, *args, **kwargs):
        if Path(os.fspath(path)).absolute() == dataset_path:
            dataset_opens.append(path)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", recording_open)
    with pytest.raises(ValueError, match="selection manifest split seed"):
        if entrypoint == "runner":
            import research.runner as runner_mod

            runner_mod.run(config, folds=2, repeats=1)
        else:
            from research.evaluator import score_fold

            score_fold(config, {"fold": 0, "folds": 2, "repeat": 0})
    assert dataset_opens == []


@pytest.mark.parametrize("entrypoint", ["runner", "evaluator"])
def test_dataset_digest_checked_before_record_parsing(
    tmp_path, monkeypatch, entrypoint
):
    from research.tasks import ConstitutionAdapter

    config, _manifest, _manifest_path, _selection_records = _config(
        tmp_path, fake_records()
    )
    dataset_path = Path(config["data"]["dataset_path"])
    dataset_path.write_bytes(b"tampered synthetic selection bytes")
    parsed = []

    def forbidden_parse(_text):
        parsed.append(True)
        raise AssertionError("dataset parsing ran before its bytes hash matched")

    monkeypatch.setattr(ConstitutionAdapter, "records_from_json", forbidden_parse)
    with pytest.raises(ValueError, match="selection dataset SHA-256"):
        if entrypoint == "runner":
            import research.runner as runner_mod

            runner_mod.run(config, folds=2, repeats=1)
        else:
            from research.evaluator import score_fold

            score_fold(
                config,
                {"fold": 0, "folds": 2, "repeat": 0},
            )
    assert parsed == []


def test_the_evaluator_refuses_model_files_modified_after_training(tmp_path):
    from research.evaluator import score_fold

    records = fake_records()
    config, manifest, _manifest_path, selection_records = _config(tmp_path, records)
    _fold_data, _model, request = _saved_model(
        tmp_path, config, selection_records, manifest
    )
    Path(request["weights_path"]).write_bytes(
        Path(request["weights_path"]).read_bytes() + b"x"
    )
    with pytest.raises(RuntimeError, match="changed between training"):
        score_fold(config, request, records=selection_records)


def test_the_evaluator_refuses_tampered_training_inputs(tmp_path):
    from research.evaluator import integrity_snapshot, score_fold

    records = fake_records()
    config, manifest, manifest_path, selection_records = _config(tmp_path, records)
    snapshot = integrity_snapshot(config)
    manifest["notes"] = "tampered during training"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False))
    request = {
        "fold": 0, "folds": 4, "repeat": 0,
        "weights_path": "unused", "architecture_path": "unused",
        "integrity": snapshot,
    }
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        score_fold(config, request, records=selection_records)


def test_the_candidate_import_guard_blocks_frozen_machinery(tmp_path):
    import importlib.util

    from research.evaluator import install_candidate_import_guard

    evil = tmp_path / "candidate.py"
    # a module name that cannot already be in sys.modules: the audit event
    # only fires for imports that actually reach the import machinery, so a
    # cached module would silently bypass the hook
    evil.write_text("import runtime.nonexistent_frozen_probe\n")
    install_candidate_import_guard(evil)
    spec = importlib.util.spec_from_file_location("evil_candidate", evil)
    module = importlib.util.module_from_spec(spec)
    with pytest.raises(RuntimeError, match="may not import"):
        spec.loader.exec_module(module)


def test_the_evaluation_path_never_executes_candidate_code():
    src_root = Path(__file__).resolve().parents[2] / "src" / "research"
    evaluator_src = (src_root / "evaluator.py").read_text()
    # F-01: the evaluation path has no candidate import and no torch.load at
    # all — deserialization is state_dict + frozen architecture rebuild, in
    # model_export, behind the safe loader
    assert "import research.candidate" not in evaluator_src
    assert "torch.load" not in evaluator_src
    assert "weights_only" not in evaluator_src
    model_export = (src_root / "model_export.py").read_text()
    assert "torch.load(" in model_export
    assert "weights_only=True" in model_export
    assert "weights_only=False" not in model_export
    assert "import research.candidate" not in model_export


def test_runner_hands_the_evaluator_files_not_folds(tmp_path):
    import research.runner as runner_mod

    records = fake_records(40)
    config, base, _manifest_path, selection_records = _config(tmp_path, records)
    config["time_budget_seconds"] = 0.01
    config["worktree_path"] = str(tmp_path / "wt")
    config["editable"] = "src/research/candidate.py"
    worktree = tmp_path / "wt"
    (worktree / "src" / "research").mkdir(parents=True)
    (worktree / "src" / "research" / "candidate.py").write_text("WIDTH = 32\n")
    calls: list[tuple[dict, dict]] = []

    def spawn_training(worker_config, worker_config_path):
        # the training subprocess's config carries trimmed file paths only:
        # no dataset path, no frozen-manifest path, no holdout
        assert "dataset_path" not in worker_config
        assert worker_config["time_budget_seconds"] == 0.01
        assert "manifest_path" not in worker_config
        fit_manifest = json.loads(
            Path(worker_config["fit_manifest_path"]).read_text()
        )
        assert fit_manifest["split"]["holdout"] == []
        assert fit_manifest["split"]["dev"] == []
        fit_records = json.loads(
            Path(worker_config["fit_records_path"]).read_text()
        )
        fit_rows = set(fit_manifest["subject_row_by_id"].values())
        assert fit_records
        assert {r["subject_index"] for r in fit_records} == fit_rows
        return {
            "ok": True,
            "fit_seconds": 0.005,
            "fit_timer": "trusted_candidate_phase_wall_v1",
            "budget_seconds": 0.01,
            "weights_path": "unused.pt",
            "architecture_path": "unused.json",
            "weights_sha256": "0" * 64,
            "architecture_sha256": "0" * 64,
        }

    def spawn(cfg, req):
        # the request hands the evaluator saved model files, not the fold
        calls.append((cfg, req))
        return {"primary_value": 0.5}

    result = runner_mod.run(
        config, folds=2, repeats=1, records=selection_records,
        spawn_training=spawn_training, spawn_evaluator=spawn,
    )
    assert len(calls) == 2 and result["n_folds"] == 2
    assert result["primary_value"] == 0.5
    for _cfg, req in calls:
        assert set(req["integrity"]) == {"manifest_sha256", "dataset_sha256",
                                         "candidate_sha256"}
        assert req["fold"] in (0, 1) and req["folds"] == 2
        assert req["weights_path"] == "unused.pt"


def test_parent_training_timeout_includes_bounded_sandbox_lifecycle(
    tmp_path, monkeypatch
):
    import research.runner as runner_mod
    import research.spawn as spawn_mod

    captured = {}
    worker_config = tmp_path / "worker-config.json"
    config = {
        "worktree_path": str(tmp_path),
        "time_budget_seconds": 7.5,
    }
    monkeypatch.setattr(
        spawn_mod,
        "bootstrap_command",
        lambda *args: ["synthetic-worker-command"],
    )

    def fake_spawn(_config, _command, *, timeout):
        captured["timeout"] = timeout
        return {"ok": True}

    monkeypatch.setattr(runner_mod, "_spawn", fake_spawn)
    runner_mod._spawn_training(config, worker_config)
    expected = (
        runner_mod.PARENT_DOCKER_STARTUP_ALLOWANCE_SECONDS
        + spawn_mod.sandbox_worker_process_timeout(config["time_budget_seconds"])
        + spawn_mod.ARTIFACT_VALIDATION_TIMEOUT_SECONDS
        + spawn_mod.HOST_ARTIFACT_EXPORT_TIMEOUT_SECONDS
        * spawn_mod.HOST_ARTIFACT_EXPORT_FILE_COUNT
        + spawn_mod.HOST_HANDOFF_SECONDS
        + spawn_mod.HOST_CLEANUP_ALLOWANCE_SECONDS
        + runner_mod.PARENT_DOCKER_CLEANUP_ALLOWANCE_SECONDS
    )
    assert captured["timeout"] == expected
    assert captured["timeout"] > config["time_budget_seconds"]


def test_runner_never_touches_the_scoring_machinery_in_process():
    source = runner_source()
    # the in-process evaluator path is what made "the evaluator runs outside
    # the candidate's process" untrue: the runner must not import or call it,
    # and F-02 removes the candidate import entirely
    assert "TCMTrustedEvaluator" not in source
    assert "evaluate_model" not in source
    assert "research.candidate" not in source
    assert "research.evaluator" in source
    assert "research.train_worker" in source


def runner_source() -> str:
    import research.runner

    return Path(research.runner.__file__).read_text()

def test_score_fold_passes_task_to_safe_model_loader(monkeypatch):
    from types import SimpleNamespace

    from research import evaluator as evaluator_mod
    from research import model_export, tasks

    loaded_tasks = []
    model = object()

    class Adapter:
        default_editable = "src/research/candidate_sd.py"

        def build_fold(self, *_args, **_kwargs):
            return SimpleNamespace(dev="synthetic-dev"), "synthetic-fold"

        def evaluate_model(self, received_model, _fold, dev, partition):
            assert received_model is model
            assert dev == "synthetic-dev"
            assert partition == "dev"
            return SimpleNamespace(primary_value=0.625)

    adapter = Adapter()

    def get_adapter(task):
        assert task == "syndrome"
        return adapter

    def load_model(_weights, _architecture, *, task):
        loaded_tasks.append(task)
        return model

    monkeypatch.setattr(tasks, "adapter_for", get_adapter)
    monkeypatch.setattr(
        evaluator_mod,
        "load_selection_inputs",
        lambda *_args, **_kwargs: (
            {"synthetic": True}, ["synthetic-record"], {},
        ),
    )
    monkeypatch.setattr(evaluator_mod, "assert_integrity", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(model_export, "load_eval_model", load_model)
    monkeypatch.setattr(
        evaluator_mod, "install_candidate_import_guard", lambda *_args: None
    )

    result = evaluator_mod.score_fold(
        {"task": "syndrome"},
        {
            "fold": 0,
            "folds": 1,
            "repeat": 0,
            "weights_path": "synthetic-weights",
            "architecture_path": "synthetic-architecture",
        },
    )

    assert result["primary_value"] == pytest.approx(0.625)
    assert loaded_tasks == ["syndrome"]


def test_score_holdout_passes_task_to_safe_model_loader(monkeypatch):
    from research import evaluator as evaluator_mod
    from research import finalize, model_export, tasks

    loaded_tasks = []
    model = object()

    class Adapter:
        default_editable = "src/research/candidate_sd.py"

        def load_manifest(self, path):
            if path == "synthetic-frozen-manifest":
                return {"dataset_sha256": "a" * 64}
            return {"synthetic_path": path}

        def load_records_verified(self, path, expected_sha256):
            assert path == "synthetic-dataset"
            assert expected_sha256 == "a" * 64
            return ["synthetic-record"]

        def score_holdout(self, _config, _request, records, received_model):
            assert records == ["synthetic-record"]
            assert received_model is model
            return {"holdout": {"synthetic": True}, "baselines": {}}

    adapter = Adapter()

    def get_adapter(task):
        assert task == "syndrome"
        return adapter

    def load_model(_weights, _architecture, *, task):
        loaded_tasks.append(task)
        return model

    monkeypatch.setattr(tasks, "adapter_for", get_adapter)
    monkeypatch.setattr(tasks, "validate_manifest_contract", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(finalize, "verify_holdout_unlock", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(model_export, "load_eval_model", load_model)

    result = evaluator_mod.score_holdout(
        {
            "task": "syndrome",
            "data": {
                "frozen_manifest_path": "synthetic-frozen-manifest",
                "pool_manifest_path": "synthetic-pool-manifest",
                "dataset_path": "synthetic-dataset",
            },
        },
        {
            "weights_path": "synthetic-weights",
            "architecture_path": "synthetic-architecture",
        },
    )

    assert result["holdout"]["synthetic"] is True
    assert loaded_tasks == ["syndrome"]


def test_score_holdout_rejects_unverified_syndrome_records_path(monkeypatch):
    from research import evaluator as evaluator_mod
    from research import finalize, model_export, tasks

    class Adapter:
        default_editable = "src/research/candidate_sd.py"

        def load_manifest(self, path):
            if path == "synthetic-frozen-manifest":
                return {"dataset_sha256": "a" * 64}
            return {"synthetic_path": path}

        def load_records_verified(self, *_args):
            pytest.fail("records_path must be rejected before reading dataset bytes")

    monkeypatch.setattr(tasks, "adapter_for", lambda _task: Adapter())
    monkeypatch.setattr(tasks, "validate_manifest_contract", lambda *_a, **_k: None)
    monkeypatch.setattr(finalize, "verify_holdout_unlock", lambda *_a, **_k: None)
    monkeypatch.setattr(
        model_export,
        "load_eval_model",
        lambda *_a, **_k: pytest.fail("reject invalid data path before model load"),
    )

    with pytest.raises(ValueError, match="bound to the frozen manifest"):
        evaluator_mod.score_holdout(
            {
                "task": "syndrome",
                "data": {
                    "frozen_manifest_path": "synthetic-frozen-manifest",
                    "pool_manifest_path": "synthetic-pool-manifest",
                    "records_path": "/tmp/unverified-holdout-records.json",
                },
            },
            {
                "weights_path": "synthetic-weights",
                "architecture_path": "synthetic-architecture",
            },
        )


def _write_evaluator_stub(worktree, source):
    package = worktree / "src" / "research"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "evaluator.py").write_text(source)


@pytest.mark.parametrize("mode", ["fold", "holdout"])
def test_evaluator_subprocess_limits_apply_before_module_import(tmp_path, mode):
    from research import finalize, runner, spawn

    worktree = tmp_path / "worktree"
    probe_path = tmp_path / "fsize-probe"
    _write_evaluator_stub(
        worktree,
        f"""
import errno
import json
import mmap
import os
import resource
import signal

limits = {{
    "as": resource.getrlimit(resource.RLIMIT_AS),
    "cpu": resource.getrlimit(resource.RLIMIT_CPU),
    "fsize": resource.getrlimit(resource.RLIMIT_FSIZE),
    "nofile": resource.getrlimit(resource.RLIMIT_NOFILE),
    "core": resource.getrlimit(resource.RLIMIT_CORE),
}}
try:
    mapping = mmap.mmap(-1, limits["as"][0] + 4096)
except (MemoryError, OSError):
    memory_denied = True
else:
    mapping.close()
    memory_denied = False
signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
descriptor = os.open({str(probe_path)!r}, os.O_CREAT | os.O_WRONLY, 0o600)
try:
    os.write(descriptor, b"x")
except OSError as exc:
    file_denied = exc.errno == errno.EFBIG
else:
    file_denied = False
finally:
    os.close(descriptor)

def main():
    print(json.dumps({{
        "ok": True,
        "limits": limits,
        "memory_denied": memory_denied,
        "file_denied": file_denied,
    }}))
    return 0
""",
    )
    config = {"worktree_path": str(worktree), "evaluator_timeout_s": 15.0}

    if mode == "fold":
        result = runner._spawn_evaluator(config, {})
    else:
        result = finalize._spawn_holdout_evaluator(config, {})

    assert result["limits"]["as"] == [
        spawn.EVALUATOR_ADDRESS_SPACE_LIMIT_BYTES,
        spawn.EVALUATOR_ADDRESS_SPACE_LIMIT_BYTES,
    ]
    assert result["limits"]["cpu"] == [
        spawn.EVALUATOR_CPU_SOFT_LIMIT_SECONDS,
        spawn.EVALUATOR_CPU_HARD_LIMIT_SECONDS,
    ]
    assert result["limits"]["fsize"] == [0, 0]
    assert result["limits"]["nofile"][0] <= spawn.EVALUATOR_OPEN_FILES_LIMIT
    assert result["limits"]["nofile"][1] <= spawn.EVALUATOR_OPEN_FILES_LIMIT
    assert result["limits"]["core"] == [0, 0]
    assert result["memory_denied"] is True
    assert result["file_denied"] is True


def test_evaluator_cpu_rlimit_terminates_busy_child(tmp_path, monkeypatch):
    from research import runner, spawn

    monkeypatch.setattr(spawn, "EVALUATOR_CPU_SOFT_LIMIT_SECONDS", 1)
    monkeypatch.setattr(spawn, "EVALUATOR_CPU_HARD_LIMIT_SECONDS", 2)
    worktree = tmp_path / "worktree"
    _write_evaluator_stub(
        worktree,
        """def main():
    while True:
        pass
""",
    )

    with pytest.raises(RuntimeError, match="subprocess failed"):
        runner._spawn_evaluator(
            {"worktree_path": str(worktree), "evaluator_timeout_s": 10.0},
            {},
        )
