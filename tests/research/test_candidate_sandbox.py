"""Candidate source hashes and the constrained Docker launch contract."""
from __future__ import annotations

import hashlib
import os
import sys
import types

import pytest

pytestmark = pytest.mark.synthetic

_requires_root = pytest.mark.skipif(
    os.geteuid() != 0,
    reason="models a rootful bind mount (chown to uid 12345); requires root euid",
)

CANDIDATE = (
    "def build_model(fold):\n"
    "    return None\n\n"
    "def train(model, fold, budget_seconds, seed):\n"
    "    return None\n"
)


def _worktree(tmp_path, source=CANDIDATE):
    root = tmp_path / "worktree"
    candidate = root / "src" / "research" / "candidate.py"
    candidate.parent.mkdir(parents=True)
    candidate.write_text(source)
    digest = hashlib.sha256(source.encode()).hexdigest()
    return root, candidate, digest


def test_candidate_hash_is_checked_before_import_execution(tmp_path, monkeypatch):
    from research.train_worker import import_candidate

    marker = tmp_path / "executed"
    candidate_source = (
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n"
        + CANDIDATE
    ).encode()
    worktree, _candidate, _digest = _worktree(tmp_path, candidate_source.decode())
    config = {
        "worktree_path": str(worktree),
        "editable": "src/research/candidate.py",
        "candidate_sha256": "0" * 64,
    }

    monkeypatch.setitem(
        sys.modules,
        "research.candidate",
        types.ModuleType("research.candidate"),
    )
    modules_before = sys.modules.copy()
    with pytest.raises(RuntimeError, match="do not match the recorded SHA"):
        import_candidate(config, source=candidate_source)
    assert not marker.exists()
    assert sys.modules == modules_before


def test_docker_args_use_only_scoped_mounts_and_restricted_runtime(
    tmp_path, monkeypatch
):
    from research import spawn
    from research.spawn import _docker_worker_command

    monkeypatch.setattr(
        spawn.shutil,
        "which",
        lambda name: "/usr/bin/docker" if name == "docker" else None,
    )
    source_root = tmp_path / "source"
    input_root = tmp_path / "input"
    source_root.mkdir()
    input_root.mkdir()
    image = "sha256:" + "d" * 64
    command = _docker_worker_command(
        image, "synthetic-container", source_root, input_root, True,
    )
    assert (
        command[:3] == ["/usr/bin/docker", "run", "--rm"]
        or command[1:2] == ["run"]
    )
    assert command[command.index("--network") + 1] == "none"
    assert {
        "OMP_NUM_THREADS=1",
        "MKL_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
        "NUMEXPR_NUM_THREADS=1",
    } <= set(command)
    assert "--read-only" in command

    def option_value(name):
        for index, argument in enumerate(command):
            if argument == name:
                return command[index + 1]
            if argument.startswith(name + "="):
                return argument[len(name) + 1:]
        raise AssertionError(f"missing Docker option {name}")

    assert option_value("--cap-drop") == "ALL"
    assert option_value("--security-opt") == "no-new-privileges"
    assert command[command.index("--pids-limit") + 1] == "32"
    assert command[command.index("--memory") + 1] == "4g"
    assert command[command.index("--memory-swap") + 1] == "4g"
    assert command[command.index("--cpus") + 1] == "2"
    assert {
        "--cap-add=SETUID",
        "--cap-add=SETGID",
        "--cap-add=KILL",
        "--cap-add=DAC_READ_SEARCH",
    } <= set(command)
    assert "--user" not in command
    assert "--log-driver=none" in command
    assert image in command
    assert "--privileged" not in command
    assert command[command.index("--network") + 1] != "host"

    tmpfs = [
        command[index + 1]
        for index, argument in enumerate(command[:-1])
        if argument == "--tmpfs"
    ]
    assert len(tmpfs) == 3
    temp_mount = next(mount for mount in tmpfs if mount.startswith("/tmp:"))
    assert "size=64m" in temp_mount
    assert f"nr_inodes={spawn.SANDBOX_TMP_INODES}" in temp_mount
    output_mount = next(
        mount for mount in tmpfs if mount.startswith("/sandbox/output:")
    )
    assert f"size={spawn.SANDBOX_OUTPUT_TMPFS}" in output_mount
    assert f"nr_inodes={spawn.SANDBOX_OUTPUT_INODES}" in output_mount
    control_mount = next(
        mount for mount in tmpfs if mount.startswith(f"{spawn.CONTROL_DIR}:")
    )
    assert "size=1m" in control_mount
    assert "nr_inodes=8" in control_mount

    mounts = [
        command[index + 1]
        for index, argument in enumerate(command[:-1])
        if argument == "--mount"
    ]
    assert len(mounts) == 2
    assert any(mount.endswith(",dst=/sandbox/src,readonly") for mount in mounts)
    assert any(mount.endswith(",dst=/sandbox/input,readonly") for mount in mounts)
    assert not any(
        token in mount.lower()
        for mount in mounts
        for token in ("/home", ".git", "ledger", "holdout", "docker.sock", "raw")
    )

def test_unavailable_docker_fails_closed_without_importing_candidate(
    tmp_path, monkeypatch,
):
    from research import spawn, train_worker

    marker = tmp_path / "candidate-imported"
    source = (
        f"open({str(marker)!r}, 'w').write('ran')\n"
        + CANDIDATE
    )
    worktree, _candidate, digest = _worktree(tmp_path, source)
    monkeypatch.setattr(train_worker, "validate_candidate_source", lambda source: None)
    monkeypatch.setattr(spawn.shutil, "which", lambda _name: None)
    config = {
        "worktree_path": str(worktree),
        "editable": "src/research/candidate.py",
        "candidate_sha256": digest,
        "task": "constitution",
        "time_budget_seconds": 1,
        "fit_manifest_path": str(tmp_path / "synthetic-manifest.json"),
        "fit_records_path": str(tmp_path / "synthetic-records.json"),
        "output_dir": str(tmp_path / "output"),
        "output_stem": "synthetic",
    }

    with pytest.raises(RuntimeError, match="Docker executable is unavailable"):
        spawn.run_sandboxed_candidate(config, preflight=True)
    assert not marker.exists()


@_requires_root
def test_staged_fit_inputs_are_private_from_candidate_and_host_users(
    tmp_path, monkeypatch,
):
    import json
    import os
    import stat
    import subprocess
    import sys
    from pathlib import Path

    from research import spawn

    class InputsStaged(Exception):
        pass

    worktree, _candidate, digest = _worktree(tmp_path)
    fit_manifest = tmp_path / "synthetic-manifest.json"
    fit_manifest.write_text('{"kind":"synthetic"}')
    original_text = "synthetic-original-input-text"
    fit_records = tmp_path / "synthetic-records.json"
    fit_records.write_text(json.dumps([
        {"user_id": "synthetic-user", "input_text": original_text},
    ]))
    observations = {}

    def probe(path, *, uid, gid, groups):
        script = (
            "from pathlib import Path; import sys\n"
            "try:\n"
            "    Path(sys.argv[1]).read_text()\n"
            "except PermissionError:\n"
            "    print('DENIED')\n"
            "else:\n"
            "    print('READ')\n"
        )

        def drop_identity():
            os.setgroups(groups)
            os.setgid(gid)
            os.setuid(uid)

        result = subprocess.run(
            [sys.executable, "-c", script, str(path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
            preexec_fn=drop_identity,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    def inspect_staged_mount(command, **_kwargs):
        def mounted_root(destination):
            mount = next(
                command[index + 1]
                for index, argument in enumerate(command[:-1])
                if argument == "--mount"
                and command[index + 1].endswith(f",dst={destination},readonly")
            )
            return Path(
                mount.removeprefix("type=bind,src=").split(",dst=", 1)[0]
            )

        input_root = mounted_root("/sandbox/input")
        source_root = mounted_root("/sandbox/src")
        staging_root = input_root.parent
        staged_records = input_root / "fit-records.json"
        observations["modes"] = {
            "temp_root": stat.S_IMODE(staging_root.stat().st_mode),
            "input_root": stat.S_IMODE(input_root.stat().st_mode),
            "files": {
                path.name: stat.S_IMODE(path.stat().st_mode)
                for path in input_root.iterdir()
            },
        }

        # Model a rootful bind mount whose UID 0 differs from the host owner.
        for path in input_root.iterdir():
            os.chown(path, 12345, 12345)
        os.chown(input_root, 12345, 12345)

        observations["host_user"] = probe(
            staged_records, uid=65534, gid=65534, groups=[],
        )
        os.chmod(staging_root, 0o755)
        candidate = {
            "uid": spawn.CANDIDATE_SANDBOX_UID,
            "gid": spawn.CANDIDATE_SANDBOX_GID,
            "groups": [],
        }
        observations["candidate"] = probe(staged_records, **candidate)
        observations["candidate_source"] = probe(
            source_root / "research" / "candidate.py", **candidate,
        )
        observations["container_other_user"] = probe(
            staged_records, uid=65534, gid=65534, groups=[],
        )
        observations["trusted_worker"] = probe(
            staged_records, uid=0, gid=0, groups=[],
        )

        for path in input_root.iterdir():
            os.chown(path, 0, 0)
        os.chown(input_root, 0, 0)
        os.chmod(staging_root, observations["modes"]["temp_root"])
        raise InputsStaged()

    monkeypatch.setattr(
        spawn.shutil,
        "which",
        lambda name: "/usr/bin/docker" if name == "docker" else None,
    )
    monkeypatch.setattr(spawn, "_image_id", lambda _image: "sha256:" + "d" * 64)
    monkeypatch.setattr(spawn, "_run_bounded_stdout", inspect_staged_mount)
    # This test stops before Docker is invoked; cleanup is exercised separately
    # by the startup-timeout regression below.
    monkeypatch.setattr(spawn, "_release_container", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(spawn, "_stop_container", lambda *_args, **_kwargs: None)

    with pytest.raises(InputsStaged):
        spawn.run_sandboxed_candidate({
            "worktree_path": str(worktree),
            "editable": "src/research/candidate.py",
            "candidate_sha256": digest,
            "task": "constitution",
            "seed": 0,
            "time_budget_seconds": 1,
            "fit_manifest_path": str(fit_manifest),
            "fit_records_path": str(fit_records),
            "output_dir": str(tmp_path / "output"),
            "output_stem": "synthetic",
        }, preflight=True)

    assert observations["host_user"] == "DENIED"
    assert observations["candidate"] == "DENIED"
    assert observations["candidate_source"] == "READ"
    assert observations["container_other_user"] == "DENIED"
    assert observations["trusted_worker"] == "READ"
    assert observations["modes"]["temp_root"] == 0o700
    assert observations["modes"]["input_root"] == 0o700
    assert observations["modes"]["files"] == {
        "fit-manifest.json": 0o600,
        "fit-records.json": 0o600,
        "worker-config.json": 0o600,
    }


def test_docker_run_timeout_still_cleans_pre_named_container(tmp_path, monkeypatch):
    import json
    import subprocess

    from research import spawn

    worktree, _candidate, digest = _worktree(tmp_path)
    fit_manifest = tmp_path / "synthetic-manifest.json"
    fit_manifest.write_text('{"kind":"synthetic"}')
    fit_records = tmp_path / "synthetic-records.json"
    fit_records.write_text(json.dumps([{"user_id": "synthetic-user"}]))
    run_commands = []
    cleanup_commands = []

    def timed_out_docker_run(command, *, timeout, **_kwargs):
        run_commands.append(command)
        raise subprocess.TimeoutExpired(command, timeout)

    def cleanup_run(command, **kwargs):
        cleanup_commands.append((command, kwargs["timeout"]))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(
        spawn.shutil,
        "which",
        lambda name: "/usr/bin/docker" if name == "docker" else None,
    )
    monkeypatch.setattr(spawn, "_image_id", lambda _image: "sha256:" + "d" * 64)
    monkeypatch.setattr(spawn, "_run_bounded_stdout", timed_out_docker_run)
    monkeypatch.setattr(spawn.subprocess, "run", cleanup_run)

    with pytest.raises(subprocess.TimeoutExpired):
        spawn.run_sandboxed_candidate({
            "worktree_path": str(worktree),
            "editable": "src/research/candidate.py",
            "candidate_sha256": digest,
            "task": "constitution",
            "seed": 0,
            "time_budget_seconds": 1,
            "fit_manifest_path": str(fit_manifest),
            "fit_records_path": str(fit_records),
            "output_dir": str(tmp_path / "output"),
            "output_stem": "synthetic",
        }, preflight=True)

    assert len(run_commands) == 1
    container_name = run_commands[0][run_commands[0].index("--name") + 1]
    assert container_name.startswith("research-candidate-")
    assert [command[1] for command, _timeout in cleanup_commands] == [
        "exec", "stop", "rm",
    ]
    assert cleanup_commands[0][0][2:] == [
        container_name, "touch", spawn.CONTROL_RELEASE_PATH,
    ]
    assert cleanup_commands[1][0][2:] == ["--time", "0", container_name]
    assert cleanup_commands[2][0][2:] == ["-f", container_name]
    assert all(
        0 < timeout <= spawn.CONTAINER_CLEANUP_ACTION_TIMEOUT_SECONDS
        for _command, timeout in cleanup_commands
    )


def test_worker_supervisor_exits_when_host_never_releases(tmp_path, monkeypatch):
    from research import worker_supervisor

    now = [0.0]

    def advance(seconds):
        now[0] += seconds

    monkeypatch.setattr(worker_supervisor.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(worker_supervisor.time, "sleep", advance)
    monkeypatch.setattr(
        worker_supervisor,
        "_run_worker",
        lambda _config_path, _preflight: {"ok": True},
    )
    monkeypatch.setattr(worker_supervisor, "_write_control_result", lambda _result: None)
    monkeypatch.setattr(worker_supervisor, "RELEASE_PATH", tmp_path / "missing-release")

    assert worker_supervisor.main(["--config", "synthetic-config.json"]) == 0
    assert now[0] == pytest.approx(worker_supervisor.HANDOFF_SECONDS, abs=0.1)


def test_container_cleanup_surfaces_docker_daemon_failures(monkeypatch):
    import subprocess

    from research import spawn

    def daemon_error(command, **_kwargs):
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr="Cannot connect to the Docker daemon",
        )

    monkeypatch.setattr(
        spawn.shutil,
        "which",
        lambda name: "/usr/bin/docker" if name == "docker" else None,
    )
    monkeypatch.setattr(spawn.subprocess, "run", daemon_error)

    with pytest.raises(RuntimeError, match="Docker exec cleanup failed"):
        spawn._release_container(
            "synthetic-container",
            deadline=spawn.time.monotonic() + 1,
        )
