"""Trusted in-container supervisor for isolated candidate execution."""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from research.spawn import (
    ARTIFACT_VALIDATION_TIMEOUT_SECONDS,
    CANDIDATE_SANDBOX_UID,
    MAX_WORKER_SECONDS,
    SANDBOX_WORKER_CLEANUP_ALLOWANCE_SECONDS,
    SANDBOX_WORKER_EXPORT_ALLOWANCE_SECONDS,
    SANDBOX_WORKER_STARTUP_ALLOWANCE_SECONDS,
    TRUSTED_FIT_TIMER,
)
from research.spawn import (
    HOST_HANDOFF_SECONDS as HANDOFF_SECONDS,
)

ARTIFACT_VALIDATION_CPU_SECONDS = 55
MAX_CAPTURE_BYTES = 65536
MAX_CONTROL_RESULT_BYTES = 128 * 1024
CANDIDATE_CLEANUP_ALLOWANCE_SECONDS = 5.0
CANDIDATE_CLEANUP_POLL_SECONDS = 0.05
CONTROL_DIR = Path("/run/worker-control")
RESULT_PATH = CONTROL_DIR / "result.json"
READY_PATH = CONTROL_DIR / "ready"
RELEASE_PATH = CONTROL_DIR / "release"
SANDBOX_UID = CANDIDATE_SANDBOX_UID


def _tail_append(tail: bytearray, chunk: bytes) -> None:
    tail.extend(chunk)
    if len(tail) > MAX_CAPTURE_BYTES:
        del tail[: len(tail) - MAX_CAPTURE_BYTES]


def _parse_worker_result(
    control: bytes, returncode: int, timed_out: bool, overflow: bool = False,
) -> dict:
    text = control.decode("utf-8", errors="replace")
    if overflow:
        parsed = {"ok": False, "error": "worker result exceeds the control pipe limit"}
    else:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if not isinstance(parsed, dict) or "ok" not in parsed:
            parsed = {"ok": False, "error": "worker produced no trusted result"}
    if timed_out:
        parsed = {"ok": False, "error": "worker exceeded its in-container deadline"}
    elif returncode != 0 and parsed.get("ok"):
        parsed = {
            "ok": False,
            "error": f"worker exited with status {returncode}",
            "worker_result": parsed,
        }
    parsed["worker_returncode"] = returncode
    parsed["worker_timed_out"] = timed_out
    if not parsed.get("ok"):
        parsed["worker_output_tail"] = text[-MAX_CAPTURE_BYTES:]
    return parsed


def _kill_worker_uid_processes() -> None:
    deadline = time.monotonic() + CANDIDATE_CLEANUP_ALLOWANCE_SECONDS
    while True:
        live_pids = []
        unreadable_pids = []
        for name in os.listdir("/proc"):
            if not name.isdigit():
                continue
            try:
                status = Path("/proc") / name / "status"
                lines = status.read_text(encoding="ascii").splitlines()
                uid_line = next(line for line in lines if line.startswith("Uid:"))
                state_line = next(line for line in lines if line.startswith("State:"))
                real_uid = int(uid_line.split()[1])
                state = state_line.split()[1]
                if real_uid == SANDBOX_UID and state not in {"Z", "X"}:
                    live_pids.append(int(name))
            except (FileNotFoundError, ProcessLookupError, StopIteration, ValueError):
                continue
            except PermissionError:
                unreadable_pids.append(int(name))
        for pid in live_pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if not live_pids and not unreadable_pids:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "candidate descendants did not stop before trusted result parsing"
            )
        time.sleep(CANDIDATE_CLEANUP_POLL_SECONDS)


def _write_control_result(result: dict) -> None:
    CONTROL_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = json.dumps(result, ensure_ascii=False).encode("utf-8")
    if len(payload) > MAX_CAPTURE_BYTES * 2:
        raise RuntimeError("worker result exceeds the control-file size limit")
    fd = os.open(
        RESULT_PATH,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(fd, "wb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    ready_fd = os.open(
        READY_PATH,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    os.close(ready_fd)


def _validate_worker_artifacts(config: dict, worker_result: dict) -> dict:
    bootstrap = (
        "import sys; sys.path.insert(0, '/sandbox/src'); "
        "from research.artifact_validation import main; "
        "raise SystemExit(main())"
    )
    request = json.dumps({
        "config": config,
        "worker_result": worker_result,
    }, separators=(",", ":"))
    environment = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": "/tmp",
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    def limit_cpu() -> None:
        import resource
        resource.setrlimit(
            resource.RLIMIT_CPU,
            (ARTIFACT_VALIDATION_CPU_SECONDS, ARTIFACT_VALIDATION_CPU_SECONDS),
        )
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-c", bootstrap],
            cwd="/sandbox",
            env=environment,
            input=request,
            text=True,
            capture_output=True,
            timeout=ARTIFACT_VALIDATION_TIMEOUT_SECONDS,
            preexec_fn=limit_cpu,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("trusted artifact validation exceeded its deadline") from exc
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        detail = (result.stderr or result.stdout)[-500:]
        raise RuntimeError(
            f"trusted artifact validation returned no result: {detail}"
        ) from exc
    if result.returncode != 0 or not payload.get("ok"):
        raise RuntimeError(
            f"trusted artifact validation failed: {payload.get('error', 'unknown error')}"
        )
    return payload

def _worker_process_timeout(budget: float) -> float:
    """Bound worker startup, strict fit supervision, export and cleanup."""
    return (
        budget
        + SANDBOX_WORKER_STARTUP_ALLOWANCE_SECONDS
        + SANDBOX_WORKER_EXPORT_ALLOWANCE_SECONDS
        + SANDBOX_WORKER_CLEANUP_ALLOWANCE_SECONDS
    )


def _require_trusted_fit_timing(result: dict, budget: float) -> float:
    if result.get("fit_timer") != TRUSTED_FIT_TIMER:
        raise RuntimeError("worker result is missing trusted fit timing")
    try:
        reported_budget = float(result.get("budget_seconds"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("worker result is missing its trusted fit budget") from exc
    if not math.isfinite(reported_budget) or reported_budget != budget:
        raise RuntimeError("worker result does not match the trusted fit budget")
    value = result.get("fit_seconds")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError("worker result contains invalid trusted fit timing")
    try:
        elapsed = float(value)
    except (OverflowError, ValueError) as exc:
        raise RuntimeError("worker result contains invalid trusted fit timing") from exc
    if not math.isfinite(elapsed) or elapsed < 0:
        raise RuntimeError("worker result contains invalid trusted fit timing")
    if elapsed > budget:
        raise RuntimeError("trusted fit timer reported a budget overrun")
    return elapsed


def _bind_fixed_artifacts(config: dict, result: dict) -> None:
    stem = config.get("output_stem")
    if not isinstance(stem, str) or not stem.isascii() or not stem.isidentifier():
        raise RuntimeError("worker output stem is invalid")
    output_dir = Path(config.get("output_dir", ""))
    if output_dir != Path("/sandbox/output"):
        raise RuntimeError("worker output directory is outside the sandbox mount")
    expected = {
        "weights_path": output_dir / f"{stem}.pt",
        "architecture_path": output_dir / f"{stem}-architecture.json",
    }
    for field, path in expected.items():
        if result.get(field) != str(path):
            raise RuntimeError(f"worker returned an unexpected {field}")
    from research.artifact_validation import _sha256_regular
    result["weights_sha256"] = _sha256_regular(expected["weights_path"])
    result["architecture_sha256"] = _sha256_regular(expected["architecture_path"])


def _run_worker(config_path: Path, preflight: bool) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    budget = float(config.get("time_budget_seconds", 20.0))
    if not math.isfinite(budget) or not 0.0 <= budget <= MAX_WORKER_SECONDS:
        raise ValueError("worker time budget is outside the supervisor limit")
    worker_timeout = _worker_process_timeout(budget)
    result_read_fd, result_write_fd = os.pipe()
    os.set_inheritable(result_write_fd, False)
    bootstrap = (
        "import sys; sys.path.insert(0, '/sandbox/src'); "
        "from research.train_worker import main; sys.exit(main())"
    )
    command = [
        sys.executable,
        "-I",
        "-c",
        bootstrap,
        "--config",
        str(config_path),
        "--result-fd",
        str(result_write_fd),
    ]
    if preflight:
        command.append("--preflight")
    worker_env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": "/tmp",
        "PYTHONHASHSEED": "0",
        "RESEARCH_CANDIDATE_SANDBOX": "docker",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    control = bytearray()
    overflow = False
    try:
        process = subprocess.Popen(
            command,
            cwd="/sandbox",
            env=worker_env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            pass_fds=(result_write_fd,),
            user=0,
            group=0,
            extra_groups=[],
        )
    except Exception:
        os.close(result_read_fd)
        os.close(result_write_fd)
        raise
    os.close(result_write_fd)

    def drain_control() -> None:
        nonlocal overflow
        while True:
            try:
                chunk = os.read(result_read_fd, 8192)
            except InterruptedError:
                continue
            if not chunk:
                return
            if len(control) + len(chunk) > MAX_CONTROL_RESULT_BYTES:
                overflow = True
            elif not overflow:
                control.extend(chunk)

    drainer = threading.Thread(target=drain_control, daemon=True)
    drainer.start()
    timed_out = False
    try:
        returncode = process.wait(timeout=worker_timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        returncode = process.wait(timeout=5.0)
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        _kill_worker_uid_processes()
    drainer.join(timeout=2.0)
    if drainer.is_alive():
        os.close(result_read_fd)
        drainer.join(timeout=1.0)
        raise RuntimeError("trusted worker result pipe did not close after cleanup")
    os.close(result_read_fd)
    result = _parse_worker_result(bytes(control), returncode, timed_out, overflow)
    if result.get("ok"):
        _require_trusted_fit_timing(result, budget)
        _bind_fixed_artifacts(config, result)
        result["trusted_validation"] = _validate_worker_artifacts(config, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="research.worker_supervisor")
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    result = {"ok": False, "error": "worker supervisor failed"}
    try:
        result = _run_worker(Path(args.config), args.preflight)
    except Exception as exc:
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        _write_control_result(result)
    except Exception as exc:
        print(
            f"could not publish worker result: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    # This bounds container lifetime when the host launcher dies before release.
    deadline = time.monotonic() + HANDOFF_SECONDS
    while not RELEASE_PATH.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
