"""Synthetic regressions for trusted candidate timing and result isolation."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_train_worker import _fit_inputs, fake_records

from research import runner as runner_module
from research.artifact_validation import validate_artifacts
from research.runner import (
    PARENT_DOCKER_CLEANUP_ALLOWANCE_SECONDS,
    PARENT_DOCKER_STARTUP_ALLOWANCE_SECONDS,
    _training_parent_timeout,
    _trusted_fit_seconds,
)
from research.spawn import (
    CANDIDATE_SANDBOX_GID,
    CANDIDATE_SANDBOX_UID,
    TRUSTED_FIT_TIMER,
    sandbox_worker_process_timeout,
)
from research.train_worker import (
    CANDIDATE_POST_TRAIN_RESERVE_SECONDS,
    _candidate_output_paths,
    _trusted_candidate_phase_runner,
)
from research.worker_supervisor import (
    _require_trusted_fit_timing,
    _worker_process_timeout,
)

pytestmark = pytest.mark.synthetic

INPUT_DIR = Path("/sandbox/input")
OUTPUT_DIR = Path("/sandbox/output")


def _case_dir() -> Path:
    path = INPUT_DIR / f"trusted-timer-{uuid.uuid4().hex}"
    path.mkdir(mode=0o755)
    path.chmod(0o755)
    return path


def _worker_config(case_dir: Path, *, candidate_source: str | None,
                   budget: float, stem: str | None = None) -> dict:
    records = fake_records()
    _fit_manifest, manifest_path, records_path = _fit_inputs(case_dir, records)
    if candidate_source is None:
        worktree = Path("/sandbox")
        candidate_path = Path("/sandbox/src/research/candidate.py")
        source = candidate_path.read_bytes()
    else:
        worktree = case_dir / "worktree"
        candidate_path = worktree / "src" / "research" / "candidate.py"
        candidate_path.parent.mkdir(parents=True)
        candidate_path.write_text(candidate_source, encoding="utf-8")
        source = candidate_source.encode("utf-8")
    return {
        "worktree_path": str(worktree),
        "editable": "src/research/candidate.py",
        "candidate_sha256": hashlib.sha256(source).hexdigest(),
        "seed": 0,
        "time_budget_seconds": budget,
        "train_batch_size": 8,
        "fit_manifest_path": str(manifest_path),
        "fit_records_path": str(records_path),
        "output_dir": str(OUTPUT_DIR),
        "output_stem": stem or f"trusted_r0_f0_{uuid.uuid4().hex[:8]}",
    }


def _dispose_case(case_dir: Path, config: dict | None = None,
                  extra_paths: tuple[Path, ...] = ()) -> None:
    if config:
        stem = config["output_stem"]
        (OUTPUT_DIR / f"{stem}.pt").unlink(missing_ok=True)
        (OUTPUT_DIR / f"{stem}-architecture.json").unlink(missing_ok=True)
    for path in extra_paths:
        path.unlink(missing_ok=True)
    shutil.rmtree(case_dir, ignore_errors=True)


def test_candidate_monotonic_patch_cannot_forge_fit_seconds(
    monkeypatch, worker_sandbox,
):
    timer = _trusted_candidate_phase_runner()

    def candidate_operation(_started, _trusted_monotonic):
        time.monotonic = lambda: 0.0
        time.sleep(0.05)

    monkeypatch.setattr(time, "monotonic", lambda: 0.0)
    elapsed, _payload = timer(candidate_operation, 1.0)
    assert 0.03 < elapsed < 1.0


def test_candidate_overrun_is_killed_despite_clock_patch(
    monkeypatch, worker_sandbox,
):
    timer = _trusted_candidate_phase_runner()

    def candidate_operation(_started, _trusted_monotonic):
        time.monotonic = lambda: 0.0
        time.sleep(0.15)

    monkeypatch.setattr(time, "monotonic", lambda: 0.0)
    with pytest.raises(TimeoutError, match="strict 0.03s budget"):
        timer(candidate_operation, 0.03)


@pytest.mark.parametrize("module_name", ["research.candidate", "research.candidate_sd"])
def test_builtin_training_deadline_starts_before_optimizer_init(
    monkeypatch, module_name,
):
    import importlib

    candidate = importlib.import_module(module_name)
    events = []

    def monotonic():
        events.append("clock")
        return 0.0

    class FakeAdam:
        def __init__(self, _parameters, **_kwargs):
            events.append("optimizer")

    class FakeModel:
        def parameters(self):
            return []

        def train(self):
            pass

        def eval(self):
            pass

    monkeypatch.setattr(candidate.time, "monotonic", monotonic)
    monkeypatch.setattr(candidate.torch.optim, "Adam", FakeAdam)
    candidate.train(FakeModel(), SimpleNamespace(train=[]), budget_seconds=0.0, seed=0)
    assert events[0] == "clock"
    assert events.index("optimizer") > events.index("clock")


def test_candidate_build_model_is_inside_the_strict_worker_deadline(
    worker_sandbox,
):
    case_dir = _case_dir()
    marker = OUTPUT_DIR / f"build-finished-{uuid.uuid4().hex}"
    candidate = (
        "import time\n"
        "def build_model(fold):\n"
        "    batch = next(iter(fold.train)).batch\n"
        "    float(next(iter(batch.features.values()))[0, 0])\n"
        "    time.sleep(0.6)\n"
        f"    open({str(marker)!r}, 'w').write('finished')\n"
        "    return None\n"
        "def train(model, fold, budget_seconds, seed):\n"
        "    return None\n"
    )
    config = _worker_config(case_dir, candidate_source=candidate, budget=0.25)
    config_path = case_dir / "worker-config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    try:
        from research.worker_supervisor import _run_worker

        result = _run_worker(config_path, preflight=False)
        assert not result["ok"]
        assert "candidate phase exceeded its strict 0.25s budget" in result["error"]
        assert not marker.exists()
    finally:
        _dispose_case(case_dir, config, (marker,))


def test_candidate_train_receives_remaining_budget_with_export_reserve(
    worker_sandbox,
):
    case_dir = _case_dir()
    marker = OUTPUT_DIR / f"train-budget-{uuid.uuid4().hex}"
    baseline_source = Path("/sandbox/src/research/candidate.py").read_text(
        encoding="utf-8"
    )
    candidate_source = baseline_source.split("def train(", 1)[0] + (
        "def train(model, fold, budget_seconds, seed):" + chr(10)
        + f"    open({str(marker)!r}, 'w').write(str(budget_seconds))"
        + chr(10)
    )
    config = _worker_config(
        case_dir,
        candidate_source=candidate_source,
        budget=4.0,
        stem=f"remaining_r0_f0_{uuid.uuid4().hex[:8]}",
    )
    config_path = case_dir / "worker-config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    try:
        from research.worker_supervisor import _run_worker

        result = _run_worker(config_path, preflight=False)
        assert result["ok"], result
        received = float(marker.read_text(encoding="ascii"))
        assert 0 < received <= (
            config["time_budget_seconds"] - CANDIDATE_POST_TRAIN_RESERVE_SECONDS
        )
        assert result["fit_seconds"] <= config["time_budget_seconds"]
    finally:
        _dispose_case(case_dir, config, (marker,))


def test_builtin_baseline_finishes_inside_the_strict_worker_deadline(
    worker_sandbox,
):
    case_dir = _case_dir()
    config = _worker_config(
        case_dir,
        candidate_source=None,
        budget=6.0,
        stem=f"baseline_r0_f0_{uuid.uuid4().hex[:8]}",
    )
    config_path = case_dir / "worker-config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    try:
        from research.worker_supervisor import _run_worker

        result = _run_worker(config_path, preflight=False)
        assert result["ok"], result
        assert result["fit_timer"] == TRUSTED_FIT_TIMER
        assert 0.0 <= result["fit_seconds"] <= config["time_budget_seconds"]
        assert result["budget_seconds"] == config["time_budget_seconds"]
        assert result["trusted_validation"]["output_key"] == "category_probs"
        assert Path(result["weights_path"]).is_file()
        assert Path(result["architecture_path"]).is_file()
        assert CANDIDATE_POST_TRAIN_RESERVE_SECONDS == 2.0
    finally:
        _dispose_case(case_dir, config)


def test_detached_child_stdout_cannot_replace_trusted_worker_result(
    monkeypatch, worker_sandbox,
):
    from research import worker_supervisor

    case_dir = _case_dir()
    config_path = case_dir / "worker-config.json"
    config_path.write_text(
        json.dumps({
            "time_budget_seconds": 1.0,
            "output_dir": str(OUTPUT_DIR),
            "output_stem": f"spoof_r0_f0_{uuid.uuid4().hex[:8]}",
        }),
        encoding="utf-8",
    )
    pid_path = OUTPUT_DIR / f"candidate-pid-{uuid.uuid4().hex}"
    spoof_marker = OUTPUT_DIR / f"spoof-written-{uuid.uuid4().hex}"
    spoof_payload = (
        '{"ok":true,"fit_timer":"trusted_candidate_phase_wall_v1",'
        '"fit_seconds":0.0,"budget_seconds":1.0}'
        + chr(10)
    )
    worker_script = case_dir / "synthetic_worker.py"
    worker_script.write_text(
        "import json, os, sys, time\n"
        "fd = int(sys.argv[1])\n"
        "pid_path, marker_path, spoof = sys.argv[2:5]\n"
        "trusted_pid = os.getpid()\n"
        "os.write(fd, json.dumps({'ok': False, 'error': "
        "'synthetic trusted failure'}).encode())\n"
        "child = os.fork()\n"
        "if child == 0:\n"
        "    os.close(fd)\n"
        "    os.setsid()\n"
        "    os.setgroups([])\n"
        f"    os.setgid({CANDIDATE_SANDBOX_GID})\n"
        f"    os.setuid({CANDIDATE_SANDBOX_UID})\n"
        "    open(pid_path, 'w').write(str(os.getpid()))\n"
        "    while os.path.exists('/proc/' + str(trusted_pid)):\n"
        "        time.sleep(0.001)\n"
        "    os.write(1, spoof.encode())\n"
        "    open(marker_path, 'w').write('spoofed')\n"
        "    time.sleep(5.0)\n"
        "    os._exit(0)\n"
        "os._exit(1)\n",
        encoding="utf-8",
    )
    original_popen = worker_supervisor.subprocess.Popen

    def launch_synthetic_worker(_command, **kwargs):
        result_fd = kwargs["pass_fds"][0]
        return original_popen(
            [
                sys.executable,
                "-I",
                str(worker_script),
                str(result_fd),
                str(pid_path),
                str(spoof_marker),
                spoof_payload,
            ],
            **kwargs,
        )

    original_cleanup = worker_supervisor._kill_worker_uid_processes

    def allow_detached_child_to_write_then_clean():
        time.sleep(0.15)
        original_cleanup()

    monkeypatch.setattr(worker_supervisor.subprocess, "Popen", launch_synthetic_worker)
    monkeypatch.setattr(
        worker_supervisor,
        "_kill_worker_uid_processes",
        allow_detached_child_to_write_then_clean,
    )
    try:
        result = worker_supervisor._run_worker(config_path, preflight=False)
        assert not result["ok"]
        assert result["error"] == "synthetic trusted failure"
        assert spoof_marker.read_text(encoding="utf-8") == "spoofed"
        pid = int(pid_path.read_text(encoding="ascii"))
        status_path = Path("/proc") / str(pid) / "status"
        if status_path.exists():
            state_line = next(
                line for line in status_path.read_text(encoding="ascii").splitlines()
                if line.startswith("State:")
            )
            assert state_line.split()[1] in {"Z", "X"}
    finally:
        _dispose_case(case_dir, extra_paths=(pid_path, spoof_marker))


def test_worker_and_artifact_validator_reject_non_identifier_stems():
    valid = {
        "output_dir": "/sandbox/output",
        "output_stem": "model_r0_f3",
    }
    assert _candidate_output_paths(valid) == (
        "/sandbox/output",
        "model_r0_f3",
        "/sandbox/output/model_r0_f3.pt",
        "/sandbox/output/model_r0_f3-architecture.json",
    )
    with pytest.raises(RuntimeError, match="Python identifier"):
        _candidate_output_paths({
            "output_dir": "/sandbox/output",
            "output_stem": "model-r0-f3",
        })

    with pytest.raises(ValueError, match="stem is invalid"):
        validate_artifacts(
            {
                "task": "constitution",
                "candidate_sha256": "a" * 64,
                "output_dir": "/sandbox/output",
                "output_stem": "model-r0-f3",
            },
            {
                "candidate_sha256": "a" * 64,
                "task": "constitution",
            },
        )


def test_runner_generates_identifier_stems(monkeypatch, tmp_path):
    from research import evaluator, frozen, tasks

    class FakeAdapter:
        def fold_manifest(self, _manifest, _records, train_ids, _dev_ids, notes):
            row = int(train_ids[0].rsplit("-", 1)[1]) + 1
            return SimpleNamespace(data={
                "split": {"train": list(train_ids)},
                "subject_row_by_id": {train_ids[0]: row},
            })

        def records_to_json(self, records):
            return json.dumps([{"subject_index": item.subject_index} for item in records])

    records = [SimpleNamespace(subject_index=1)]
    adapter = FakeAdapter()
    monkeypatch.setattr(
        evaluator,
        "load_selection_inputs",
        lambda _config, adapter=None, records_override=None: (
            {"synthetic": True}, records_override, {},
        ),
    )
    monkeypatch.setattr(
        evaluator,
        "integrity_snapshot",
        lambda _config, **_kwargs: {
            "candidate_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
            "dataset_sha256": "c" * 64,
        },
    )
    monkeypatch.setattr(tasks, "adapter_for", lambda _task: adapter)
    monkeypatch.setattr(
        frozen,
        "selection_fold_ids",
        lambda _manifest, fold, _folds, _repeat: ([f"sid-{fold}"], []),
    )
    captured_stems = []

    def spawn_training(worker_config, _path):
        stem = worker_config["output_stem"]
        captured_stems.append(stem)
        return {
            "ok": True,
            "fit_timer": TRUSTED_FIT_TIMER,
            "fit_seconds": 0.1,
            "budget_seconds": 1.0,
            "weights_path": "weights",
            "architecture_path": "architecture",
            "weights_sha256": "d" * 64,
            "architecture_sha256": "e" * 64,
        }

    def spawn_evaluator(_config, _request):
        return {"primary_value": 0.5}

    result = runner_module.run(
        {
            "task": "constitution",
            "worktree_path": str(tmp_path),
            "editable": "src/research/candidate.py",
            "time_budget_seconds": 1.0,
            "train_provider": {"batch_size": 1},
            "trainer": {"seed": 0},
        },
        folds=2,
        repeats=1,
        records=records,
        spawn_training=spawn_training,
        spawn_evaluator=spawn_evaluator,
    )
    assert result["n_folds"] == 2
    assert captured_stems == ["model_r0_f0", "model_r0_f1"]
    assert all(stem.isascii() and stem.isidentifier() for stem in captured_stems)


def test_worker_deadlines_cover_startup_export_validation_and_host_handoff():
    budget = 7.5
    assert _worker_process_timeout(budget) == (
        budget + 30.0 + 30.0 + 5.0
    )
    assert _worker_process_timeout(budget) > budget
    expected = (
        PARENT_DOCKER_STARTUP_ALLOWANCE_SECONDS
        + sandbox_worker_process_timeout(budget)
        + 60.0
        + 20.0 * 3
        + 30.0
        + 25.0
        + PARENT_DOCKER_CLEANUP_ALLOWANCE_SECONDS
    )
    assert _training_parent_timeout(budget) == expected
    assert _training_parent_timeout(budget) > budget + 60.0


def test_missing_and_rounded_timing_fail_closed():
    with pytest.raises(RuntimeError, match="missing trusted fit timing"):
        _require_trusted_fit_timing({"ok": True}, 0.01)
    with pytest.raises(RuntimeError, match="did not provide trusted fit timing"):
        _trusted_fit_seconds({"ok": True}, 0.01)

    exact = {
        "fit_timer": TRUSTED_FIT_TIMER,
        "fit_seconds": 0.00999,
        "budget_seconds": 0.01,
    }
    assert _trusted_fit_seconds(exact, 0.01) == pytest.approx(0.00999)
    rounded_overrun = {
        "fit_timer": TRUSTED_FIT_TIMER,
        "fit_seconds": 0.01001,
        "budget_seconds": 0.01,
    }
    with pytest.raises(RuntimeError, match="budget overrun"):
        _trusted_fit_seconds(rounded_overrun, 0.01)
    with pytest.raises(RuntimeError, match="budget overrun"):
        _require_trusted_fit_timing(rounded_overrun, 0.01)
    mismatched_budget = {
        "fit_timer": TRUSTED_FIT_TIMER,
        "fit_seconds": 0.001,
        "budget_seconds": 0.02,
    }
    with pytest.raises(RuntimeError, match="does not match"):
        _trusted_fit_seconds(mismatched_budget, 0.01)
