"""Trusted-subprocess launch tests (H-02/H-05, 2026-11-03 round).

Every trusted child (train_worker, evaluator in both modes, finalize worker,
the runner itself) must start in isolated mode via a parent-written bootstrap
string — never `-m` with an inherited PYTHONPATH, which let a planted
sitecustomize.py run before any SHA check — and its environment must be a
minimal whitelist with no inherited variables at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.synthetic


def test_child_env_is_a_minimal_whitelist():
    # H-05: the child env was previously untested and kept PYTHONPATH; the
    # whitelist must not carry ANY inherited variable (TMPDIR, LD_PRELOAD,
    # PYTHONSTARTUP, ...) and must not point PYTHONPATH anywhere
    from research.spawn import child_env

    env = child_env()
    assert set(env) == {"PATH", "HOME", "PYTHONHASHSEED", "RESEARCH_SANDBOX_IMAGE"}
    for forbidden in ("PYTHONPATH", "TMPDIR", "LD_PRELOAD", "PYTHONSTARTUP",
                      "PYTHONHOME"):
        assert forbidden not in env
    assert env["PYTHONHASHSEED"] == "0"
    assert env["RESEARCH_SANDBOX_IMAGE"]
    # the path is the worktree's own src, taken from the parent's config —
    # never from the environment
    assert str(Path("/some/worktree") / "src")


def test_bootstrap_command_starts_isolated_without_module_resolution():
    # H-02: `-m` resolves modules through sys.path built from the environment;
    # the bootstrap hands the interpreter the src path itself, isolated mode
    # on, and no `research.<mod>` module-name resolution at startup
    from research.spawn import bootstrap_command

    command = bootstrap_command("/some/worktree", "research.train_worker",
                                "--config", "/tmp/cfg.json")
    assert command[1] == "-I"
    assert command[2] == "-c"
    bootstrap = command[3]
    assert "sys.path.insert(0" in bootstrap
    assert "from research.spawn import sandbox_main" in bootstrap
    assert "research.train_worker" not in " ".join(command[4:])
    # the src path comes from the parent's worktree argument, not the env
    assert command[4] == str(Path("/some/worktree") / "src")
    assert command[5:] == ["--config", "/tmp/cfg.json"]


def test_bootstrap_command_runs_the_real_entrypoint(tmp_path):
    # end-to-end: the bootstrap must actually run `main()` with the remaining
    # argv intact (sys.argv.pop(1) removes the injected src path)
    import subprocess
    import sys

    src = Path(__file__).resolve().parents[2] / "src"
    command_file = tmp_path / "argv-probe.json"
    command_file.write_text("{}")
    script = (
        "import sys, json; sys.path.insert(0, sys.argv.pop(1)); "
        "print(json.dumps({'argv': sys.argv[1:]}))"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(src), "--config",
         str(command_file)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    import json

    assert json.loads(result.stdout.strip().splitlines()[-1])["argv"] == [
        "--config", str(command_file)
    ]


def test_worker_container_args_enforce_output_and_memory_limits(tmp_path, monkeypatch):
    from research import spawn

    monkeypatch.setattr(spawn.shutil, "which", lambda _name: "/usr/bin/docker")
    image_id = "sha256:" + "a" * 64
    command = spawn._docker_worker_command(
        image_id, "synthetic-container", tmp_path / "src", tmp_path / "input", False
    )

    memory_index = command.index("--memory")
    swap_index = command.index("--memory-swap")
    assert command[memory_index + 1] == command[swap_index + 1] == "4g"
    assert "--log-driver=none" in command
    assert "--cap-add=SETUID" in command
    assert "--cap-add=SETGID" in command
    assert "--cap-add=KILL" in command
    tmpfs_mounts = [
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == "--tmpfs"
    ]
    assert any(
        mount.startswith("/sandbox/output:")
        and "size=128m" in mount
        and "nr_inodes=256" in mount
        for mount in tmpfs_mounts
    )
    assert not any(
        value.startswith("type=bind,") and "dst=/sandbox/output" in value
        for value in command
    )


def test_output_reader_refuses_fifo_without_blocking(tmp_path):
    import os

    from research.spawn import _open_regular_output

    fifo = tmp_path / "candidate-output"
    os.mkfifo(fifo)
    with pytest.raises(RuntimeError, match="regular file"):
        _open_regular_output(fifo, max_bytes=1024)


def test_streamed_output_rejects_aggregate_limit_before_writing(tmp_path):
    import sys

    from research.spawn import _copy_stream_to_file

    destination = tmp_path / "bounded-output.bin"
    script = (
        "import os, struct; "
        "os.write(1, struct.pack('>Q', 8)); "
        "os.write(1, b'12345678')"
    )
    with pytest.raises(RuntimeError, match="file or run limit"):
        _copy_stream_to_file(
            [sys.executable, "-c", script],
            destination,
            max_bytes=8,
            aggregate_remaining=7,
            timeout=3,
        )
    assert not destination.exists()


def test_bounded_docker_stdout_stops_after_limit():
    import sys

    from research.spawn import _run_bounded_stdout

    with pytest.raises(RuntimeError, match="stdout limit"):
        _run_bounded_stdout(
            [sys.executable, "-c", "print('x' * 1000000)"],
            timeout=3,
            max_bytes=64,
        )
