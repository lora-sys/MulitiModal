"""Shared Docker integration availability checks for research worker tests."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def docker_daemon():
    """Skip end-to-end worker tests when Docker is unavailable in this environment."""
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker integration requires an available Docker CLI and daemon")
    try:
        result = subprocess.run(
            [docker, "info", "--format", "{{.ServerVersion}}"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        pytest.skip("Docker daemon did not respond to the bounded availability check")
    if result.returncode != 0:
        pytest.skip("Docker daemon is unavailable; integration test skipped")


@pytest.fixture
def worker_sandbox():
    """Skip tests that execute the trusted worker path in-place.

    The supervisor hard-requires the sandbox layout: the trusted phase runner
    needs a root euid to drop into the candidate UID, and its child and the
    artifact binding resolve /sandbox/src and /sandbox/output. These tests
    only run inside the research worker container, never on a host.
    """
    if os.geteuid() != 0 or not Path("/sandbox").is_dir():
        pytest.skip(
            "requires the worker sandbox layout (root euid and /sandbox); "
            "run inside the research worker container"
        )
