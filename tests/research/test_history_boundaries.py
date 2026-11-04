"""Regression checks for experiment history and container ownership."""
import os
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from research.history import append_history, write_new_atomic
from research.sandbox import _arm_cleanup, _remove_owned_container


def append(path, n, **changes):
    fields = dict(commit=f"{n:040x}", metric="loss", value=float(n), direction="minimize",
                  decision="REVERT", description="trial", protocol="a" * 64,
                  evidence="SMOKE_ONLY")
    fields.update(changes)
    append_history(path, **fields)


def test_history_policy_and_crash_metadata(tmp_path):
    path = tmp_path / "results.tsv"
    append(path, 1, value=None, decision="CRASH", protocol="")
    with pytest.raises(ValueError, match="metadata"):
        append(path, 2, metric="accuracy")
    append(path, 2, decision="KEEP")
    with pytest.raises(ValueError, match="improvement"):
        append(path, 3, value=2.0, decision="KEEP")
    with pytest.raises(ValueError, match="metadata"):
        append(path, 3, promotion="external")
    other = tmp_path / "external.tsv"
    append(other, 1, decision="KEEP", promotion="external")
    append(other, 2, value=1.0, decision="KEEP", promotion="external")


def test_atomic_history_and_result_boundaries(tmp_path):
    path = tmp_path / "results.tsv"
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda n: append(path, n), range(1, 9)))
    assert len(path.read_text().splitlines()) == 9
    path.write_text(path.read_text() + "partial")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="incomplete"):
        append(path, 9)
    assert path.read_bytes() == before
    result = tmp_path / "result.json"
    write_new_atomic(result, b"first")
    with pytest.raises(FileExistsError):
        write_new_atomic(result, b"second")
    assert result.read_bytes() == b"first"
    link = tmp_path / "link"
    link.symlink_to(result)
    with pytest.raises(FileExistsError):
        write_new_atomic(link, b"third")
    assert result.read_bytes() == b"first"


def test_cleanup_never_removes_a_foreign_container(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="different-owner\n", stderr="")

    monkeypatch.setattr("research.sandbox.subprocess.run", run)
    _remove_owned_container("candidate", "our-run")
    assert len(calls) == 1
    assert calls[0][1] == "inspect"


def test_cleanup_inherits_docker_and_ssh_environment(monkeypatch):
    commands = []
    monkeypatch.setenv("DOCKER_HOST", "ssh://worker")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/test-agent.sock")
    monkeypatch.delenv("DOCKER_CONTEXT", raising=False)
    monkeypatch.setattr("research.sandbox.subprocess.run",
                        lambda command, **kwargs: commands.append(command))
    _arm_cleanup("id", "run", 10)
    assert "--setenv=DOCKER_HOST=ssh://worker" in commands[0]
    assert "--setenv=SSH_AUTH_SOCK=/tmp/test-agent.sock" in commands[0]
    assert "--setenv=DOCKER_CONTEXT=" in commands[0]


@pytest.mark.synthetic
def test_cleanup_timer_survives_parent_kill():
    run_id = uuid.uuid4().hex
    name = "autoresearch-" + run_id
    script = '''
import subprocess, sys, time
from research.sandbox import _arm_cleanup
name, run_id = sys.argv[1:]
_arm_cleanup(name, run_id, 5)
subprocess.run(["docker", "run", "-d", "--rm", "--name", name, "--label",
                "autoresearch.run_id=" + run_id, "multimodal-research-worker:2.7.0-cpu",
                "sleep", "60"], check=True, stdout=subprocess.DEVNULL)
print("ready", flush=True)
time.sleep(60)
'''
    parent = subprocess.Popen([sys.executable, "-c", script, name, run_id],
                              stdout=subprocess.PIPE, text=True, env=os.environ.copy())
    try:
        assert parent.stdout.readline().strip() == "ready"
        parent.kill()
        parent.wait(timeout=5)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            result = subprocess.run(["docker", "inspect", name], capture_output=True)
            if result.returncode:
                assert b"no such" in result.stderr.lower()
                break
            time.sleep(0.2)
        else:
            pytest.fail("watchdog did not remove the candidate after its parent was killed")
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)


@pytest.mark.synthetic
def test_expired_watchdog_prevents_late_container_start():
    run_id = uuid.uuid4().hex
    name = "autoresearch-" + run_id
    created = subprocess.run(
        ["docker", "create", "--rm", "--name", name, "--label", "autoresearch.run_id=" + run_id,
         "multimodal-research-worker:2.7.0-cpu", "sleep", "60"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    try:
        _arm_cleanup(created, run_id, 2)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            result = subprocess.run(["docker", "inspect", created], capture_output=True)
            if result.returncode:
                assert b"no such" in result.stderr.lower()
                break
            time.sleep(0.2)
        else:
            pytest.fail("watchdog did not remove the stopped candidate")
        assert subprocess.run(["docker", "start", created], capture_output=True).returncode != 0
    finally:
        subprocess.run(["docker", "rm", "-f", created], capture_output=True, check=False)
