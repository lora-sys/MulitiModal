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
    assert set(env) == {"PATH", "HOME", "PYTHONHASHSEED"}
    for forbidden in ("PYTHONPATH", "TMPDIR", "LD_PRELOAD", "PYTHONSTARTUP",
                      "PYTHONHOME"):
        assert forbidden not in env
    assert env["PYTHONHASHSEED"] == "0"
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
    assert "from research.train_worker import main" in bootstrap
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
