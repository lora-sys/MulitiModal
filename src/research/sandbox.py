"""Run an editable candidate with bounded input/output in an isolated container."""
from __future__ import annotations

import json
import os
import re
import selectors
import subprocess
import tempfile
import time
import uuid
from pathlib import Path


def _remove_owned_container(name: str, run_id: str) -> None:
    """Remove only the container created for this run, never a name collision."""
    result = subprocess.run(
        ["docker", "inspect", "--format", '{{ index .Config.Labels "autoresearch.run_id" }}',
         name], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=15,
    )
    if result.returncode:
        if any(message in result.stderr.lower() for message in
               ("no such object", "no such container")):
            return
        raise RuntimeError("could not verify candidate container cleanup")
    if result.stdout.strip() != run_id:
        return
    removed = subprocess.run(["docker", "rm", "-f", name], stdin=subprocess.DEVNULL,
                             capture_output=True, text=True, timeout=15)
    if removed.returncode and not any(message in removed.stderr.lower() for message in
                                      ("no such object", "no such container")):
        raise RuntimeError("could not remove candidate container")


def resolve_image(image: str) -> str:
    """Resolve a local image once so a run records and uses immutable identity."""
    result = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=15, check=True,
    )
    identity = result.stdout.strip()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", identity):
        raise ValueError("Docker image identity is unavailable")
    return identity


def run_candidate(candidate: bytes, train: object, inputs: object, *, image: str,
                  budget_seconds: int, seed: int) -> list:
    """Run a candidate without network; the caller must exclude reference labels."""
    if not candidate or len(candidate) > 2 * 1024 * 1024:
        raise ValueError("candidate source must be between 1 byte and 2 MiB")
    if (isinstance(budget_seconds, bool) or not isinstance(budget_seconds, int)
            or not 1 <= budget_seconds <= 3600):
        raise ValueError("budget must be between 1 and 3600 seconds")
    if not isinstance(image, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("candidate image must be pinned by digest")
    run_id = uuid.uuid4().hex
    container_name = "autoresearch-" + run_id
    with tempfile.TemporaryDirectory(prefix="autoresearch-") as directory:
        root = Path(directory)
        source = root / "train.py"
        source.write_bytes(candidate)
        for input_name, value in (("train.json", train), ("predict.json", inputs)):
            payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
            if len(payload) > 128 * 1024 * 1024:
                raise ValueError("candidate input exceeds the size limit")
            (root / input_name).write_bytes(payload)
        command = [
            "docker", "run", "--rm", "--name", container_name,
            "--label", f"autoresearch.run_id={run_id}", "--network", "none", "--read-only",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--pids-limit", "64", "--memory", "2g", "--cpus", "2",
            "--user", f"{os.getuid()}:{os.getgid()}", "--log-driver", "none",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
            "--mount", f"type=bind,src={root},dst=/input,readonly",
            image, "timeout", "--kill-after=2s", str(budget_seconds), "python", "/input/train.py",
            "--train", "/input/train.json", "--predict", "/input/predict.json",
            "--output", "/dev/stdout", "--seed", str(seed),
        ]
        process = None
        try:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            assert process.stdout is not None
            data = bytearray()
            deadline = time.monotonic() + budget_seconds + 30
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("candidate exceeded the container time limit")
                    if not selector.select(remaining):
                        raise TimeoutError("candidate exceeded the container time limit")
                    chunk = os.read(process.stdout.fileno(), 65536)
                    if not chunk:
                        break
                    data.extend(chunk)
                    if len(data) > 8 * 1024 * 1024:
                        raise ValueError("candidate output exceeds the limit")
            code = process.wait(timeout=max(1, deadline - time.monotonic()))
            if code:
                raise RuntimeError(f"candidate failed with exit code {code}")
            values = json.loads(data)
            if not isinstance(values, list):
                raise ValueError("candidate predictions must be a JSON list")
            return values
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()
            _remove_owned_container(container_name, run_id)

