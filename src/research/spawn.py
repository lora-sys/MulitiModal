"""Trusted subprocess launch helpers and the candidate Docker boundary.

Candidate code runs only in a one-shot, networkless container. The container
receives staged Python source, the worker's already-trimmed JSON inputs, and a
temporary output directory. It never receives the repository, Git metadata,
raw dataset paths, holdout inputs, ledger files, HOME, or the Docker socket.
"""
from __future__ import annotations

import hashlib
import json
import os
import selectors
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path, PurePosixPath

#: The image is built from Dockerfile.research-worker. Image ID is resolved
#: before launch and used instead of the mutable tag for each container.
DEFAULT_SANDBOX_IMAGE = "multimodal-research-worker:2.7.0-cpu"
CHILD_PATH = "/usr/bin:/bin:/usr/local/bin"
SANDBOX_WORKER_STARTUP_ALLOWANCE_SECONDS = 30.0
SANDBOX_WORKER_EXPORT_ALLOWANCE_SECONDS = 30.0
SANDBOX_WORKER_CLEANUP_ALLOWANCE_SECONDS = 5.0
SANDBOX_MEMORY = "4g"
SANDBOX_CPUS = "2"
SANDBOX_PIDS = "32"
SANDBOX_TMP_INODES = "512"
SANDBOX_OUTPUT_TMPFS = "128m"
SANDBOX_OUTPUT_INODES = "256"
MAX_OUTPUT_FILE_BYTES = 64 * 1024 * 1024
MAX_OUTPUT_TOTAL_BYTES = 96 * 1024 * 1024
MAX_WORKER_SECONDS = 120.0
HOST_HANDOFF_SECONDS = 30.0
ARTIFACT_VALIDATION_TIMEOUT_SECONDS = 60.0
HOST_ARTIFACT_EXPORT_TIMEOUT_SECONDS = 20.0
HOST_ARTIFACT_EXPORT_FILE_COUNT = 3
HOST_CLEANUP_ALLOWANCE_SECONDS = 25.0
CONTAINER_CLEANUP_ACTION_TIMEOUT_SECONDS = 5.0
MAX_DOCKER_STDOUT_BYTES = 4096
MAX_CONTROL_RESULT_BYTES = 128 * 1024
EVALUATOR_ADDRESS_SPACE_LIMIT_BYTES = 8 * 1024**3
EVALUATOR_CPU_SOFT_LIMIT_SECONDS = 540
EVALUATOR_CPU_HARD_LIMIT_SECONDS = 600
EVALUATOR_FILE_SIZE_LIMIT_BYTES = 0
EVALUATOR_OPEN_FILES_LIMIT = 128
CONTROL_DIR = "/run/worker-control"
CONTROL_RESULT_PATH = f"{CONTROL_DIR}/result.json"
CONTROL_READY_PATH = f"{CONTROL_DIR}/ready"
CONTROL_RELEASE_PATH = f"{CONTROL_DIR}/release"

TRUSTED_FIT_TIMER = "trusted_candidate_phase_wall_v1"
CANDIDATE_SANDBOX_UID = 65532
CANDIDATE_SANDBOX_GID = 65532


def sandbox_worker_process_timeout(compute_budget: float) -> float:
    """Total child allowance around the strict candidate fit deadline."""
    return (
        compute_budget
        + SANDBOX_WORKER_STARTUP_ALLOWANCE_SECONDS
        + SANDBOX_WORKER_EXPORT_ALLOWANCE_SECONDS
        + SANDBOX_WORKER_CLEANUP_ALLOWANCE_SECONDS
    )


def apply_evaluator_resource_limits() -> None:
    import resource

    requested_limits = (
        (
            resource.RLIMIT_AS,
            EVALUATOR_ADDRESS_SPACE_LIMIT_BYTES,
            EVALUATOR_ADDRESS_SPACE_LIMIT_BYTES,
        ),
        (
            resource.RLIMIT_CPU,
            EVALUATOR_CPU_SOFT_LIMIT_SECONDS,
            EVALUATOR_CPU_HARD_LIMIT_SECONDS,
        ),
        (
            resource.RLIMIT_FSIZE,
            EVALUATOR_FILE_SIZE_LIMIT_BYTES,
            EVALUATOR_FILE_SIZE_LIMIT_BYTES,
        ),
        (
            resource.RLIMIT_NOFILE,
            EVALUATOR_OPEN_FILES_LIMIT,
            EVALUATOR_OPEN_FILES_LIMIT,
        ),
        (resource.RLIMIT_CORE, 0, 0),
    )
    for resource_id, requested_soft, requested_hard in requested_limits:
        inherited_soft, inherited_hard = resource.getrlimit(resource_id)
        hard = (
            requested_hard
            if inherited_hard == resource.RLIM_INFINITY
            else min(requested_hard, inherited_hard)
        )
        soft = (
            requested_soft
            if inherited_soft == resource.RLIM_INFINITY
            else min(requested_soft, inherited_soft)
        )
        resource.setrlimit(resource_id, (min(soft, hard), hard))


def child_env() -> dict:
    """Whitelist the environment for trusted children."""
    return {
        "PATH": CHILD_PATH,
        "HOME": "",
        "PYTHONHASHSEED": "0",
        "RESEARCH_SANDBOX_IMAGE": os.environ.get(
            "RESEARCH_SANDBOX_IMAGE", DEFAULT_SANDBOX_IMAGE
        ),
    }


def bootstrap_command(worktree: str, module: str, *args: str) -> list[str]:
    """Start trusted child code through isolated startup.

    The train worker entry is a trusted Docker launcher on the host. It stages
    inputs and starts the candidate only inside the restricted container.
    Other trusted modules continue to use the isolated Python bootstrap.
    """
    if module == "research.train_worker":
        bootstrap = (
            "import sys; sys.path.insert(0, sys.argv.pop(1)); "
            "from research.spawn import sandbox_main; sys.exit(sandbox_main())"
        )
    else:
        bootstrap = (
            "import sys; sys.path.insert(0, sys.argv.pop(1)); "
            f"from {module} import main; sys.exit(main())"
        )
    return [
        sys.executable, "-I", "-c", bootstrap,
        str(Path(worktree) / "src"), *args,
    ]


def _editable_parts(editable: str) -> tuple[str, ...]:
    relative = PurePosixPath(editable)
    if (
        relative.is_absolute()
        or len(relative.parts) != 3
        or relative.parts[0] != "src"
        or any(part in ("", ".", "..") for part in relative.parts)
        or relative.suffix != ".py"
    ):
        raise RuntimeError(
            f"editable {editable!r} must be a src/<package>/<module>.py path"
        )
    return relative.parts


def _candidate_bytes(config: dict, source_root: Path | None = None) -> bytes:
    from research.train_worker import (
        MAX_CANDIDATE_SOURCE_BYTES,
        _read_regular_file_limited,
        read_candidate_source,
    )

    editable = config.get("editable", "src/research/candidate.py")
    _editable_parts(editable)
    if source_root is None:
        source, _ = read_candidate_source(config)
        return source
    candidate = source_root.joinpath(*_editable_parts(editable)[1:])
    if candidate.is_symlink() or not candidate.is_file():
        raise RuntimeError(f"staged candidate {candidate} is not a regular file")
    source = _read_regular_file_limited(
        candidate, max_bytes=MAX_CANDIDATE_SOURCE_BYTES, label="candidate source"
    )
    actual = hashlib.sha256(source).hexdigest()
    expected = config.get("candidate_sha256")
    if actual != expected:
        raise RuntimeError(
            f"staged candidate hashes to {actual[:12]}… but config binds "
            f"{str(expected)[:12]}…; refusing to run"
        )
    return source


def _allowed_source_roots(worktree: Path) -> tuple[Path, ...]:
    repo_root = worktree.resolve(strict=True)
    src_path = worktree / "src"
    if src_path.is_symlink():
        raise RuntimeError("worktree src root must not be a symlink")
    src_root = src_path.resolve(strict=True)
    if not src_root.is_relative_to(repo_root):
        raise RuntimeError("worktree src root escapes the worktree")
    trusted_src = Path(__file__).resolve().parents[1]
    return tuple({src_root, trusted_src})


def _stage_source_tree(worktree: Path, destination: Path,
                       candidate_editable: str, candidate_source: bytes) -> Path:
    """Copy only Python source files, following only trusted source symlinks."""
    editable_parts = _editable_parts(candidate_editable)
    src_root = worktree / "src"
    allowed_roots = _allowed_source_roots(worktree)
    src_resolved = src_root.resolve(strict=True)
    if not any(src_resolved == root for root in allowed_roots):
        raise RuntimeError(f"source root {src_root} is outside the trusted source trees")
    staged = destination / "src"
    staged.mkdir(parents=True, exist_ok=True)
    visited: set[Path] = set()

    def visit(source_dir: Path, relative_dir: Path) -> None:
        resolved_dir = source_dir.resolve(strict=True)
        if not any(resolved_dir.is_relative_to(root) for root in allowed_roots):
            raise RuntimeError(f"source symlink {source_dir} escapes trusted source trees")
        if resolved_dir in visited:
            return
        visited.add(resolved_dir)
        for child in sorted(source_dir.iterdir()):
            target_relative = relative_dir / child.name
            if child.is_dir():
                visit(child, target_relative)
                continue
            if child.suffix != ".py":
                continue
            resolved_file = child.resolve(strict=True)
            if not any(resolved_file.is_relative_to(root) for root in allowed_roots):
                raise RuntimeError(f"source file {child} escapes trusted source trees")
            target = staged / target_relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target_relative.parts == editable_parts[1:]:
                payload = candidate_source
            else:
                payload = resolved_file.read_bytes()
            if target.exists() and target.read_bytes() != payload:
                raise RuntimeError(f"conflicting source paths map to {target_relative}")
            target.write_bytes(payload)

    visit(src_root, Path())
    candidate_target = staged.joinpath(*editable_parts[1:])
    if not candidate_target.is_file():
        candidate_target.parent.mkdir(parents=True, exist_ok=True)
        candidate_target.write_bytes(candidate_source)
    if hashlib.sha256(candidate_target.read_bytes()).hexdigest() != config_sha(candidate_source):
        raise RuntimeError("staged candidate digest changed while preparing the container")
    _validate_source_stage(staged)
    return staged


def config_sha(source: bytes) -> str:
    return hashlib.sha256(source).hexdigest()


def stage_candidate_source(
    worktree: str | Path,
    editable: str,
    candidate_sha256: str,
    destination: str | Path,
) -> Path:
    """Make an immutable Python-only source snapshot for preflight and fit."""
    from research.train_worker import read_candidate_source

    config = {
        "worktree_path": str(Path(worktree)),
        "editable": editable,
        "candidate_sha256": candidate_sha256,
    }
    candidate_source, _ = read_candidate_source(config)
    return _stage_source_tree(
        Path(worktree).resolve(strict=True),
        Path(destination),
        editable,
        candidate_source,
    )


def _validate_source_stage(source_root: Path) -> None:
    if not source_root.is_dir():
        raise RuntimeError(f"source stage {source_root} does not exist")
    for path in source_root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"source stage contains a symlink: {path}")
        if path.is_file() and path.suffix != ".py":
            raise RuntimeError(f"source stage contains non-Python data: {path}")


def _stage_worker_inputs(config: dict, input_root: Path) -> dict:
    allowed = {
        "worktree_path", "editable", "task", "candidate_sha256", "seed",
        "time_budget_seconds", "train_batch_size", "fit_manifest_path",
        "fit_records_path", "output_dir", "output_stem", "preflight",
    }
    extra = sorted(set(config) - allowed)
    if extra:
        raise RuntimeError(f"worker config contains fields outside the sandbox API: {extra}")
    staged = {key: value for key, value in config.items() if key not in {
        "worktree_path", "fit_manifest_path", "fit_records_path", "output_dir",
    }}
    staged["worktree_path"] = "/sandbox"
    staged["output_dir"] = "/sandbox/output"
    for field, filename in (
        ("fit_manifest_path", "fit-manifest.json"),
        ("fit_records_path", "fit-records.json"),
    ):
        original = config.get(field)
        if not original:
            raise RuntimeError(f"worker config requires {field}")
        source = Path(original).resolve(strict=True)
        if not source.is_file():
            raise RuntimeError(f"worker input {field} is not a file")
        target = input_root / filename
        shutil.copyfile(source, target)
        staged[field] = f"/sandbox/input/{filename}"
    if not isinstance(staged.get("candidate_sha256"), str):
        raise RuntimeError("worker config requires candidate_sha256")
    _editable_parts(staged.get("editable", "src/research/candidate.py"))
    if staged.get("task", "constitution") not in ("constitution", "syndrome", "toy"):
        raise RuntimeError(f"unknown worker task {staged.get('task')!r}")
    stem = staged.get("output_stem")
    if not isinstance(stem, str) or not stem.isascii() or not stem.isidentifier():
        raise RuntimeError("worker output_stem must be a Python identifier")
    try:
        budget = float(staged.get("time_budget_seconds", 20.0))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("worker time budget must be numeric") from exc
    if not 0.0 <= budget <= MAX_WORKER_SECONDS:
        raise RuntimeError("worker time budget exceeds the sandbox limit")
    return staged


def sandbox_image_identity() -> str:
    """Return the immutable ID of the locally provisioned worker image."""
    image = os.environ.get("RESEARCH_SANDBOX_IMAGE", DEFAULT_SANDBOX_IMAGE)
    return _image_id(image)


def _run_bounded_stdout(
    command: list[str], *, timeout: float, max_bytes: int = MAX_DOCKER_STDOUT_BYTES
) -> tuple[int, str]:
    process = subprocess.Popen(
        command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, env=child_env(),
    )
    if process.stdout is None:
        raise RuntimeError("failed to open bounded Docker command output")
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    output = bytearray()
    deadline = time.monotonic() + timeout
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout)
            events = selector.select(min(remaining, 0.25))
            if not events:
                continue
            chunk = os.read(
                process.stdout.fileno(),
                min(4096, max_bytes + 1 - len(output)),
            )
            if not chunk:
                break
            output.extend(chunk)
            if len(output) > max_bytes:
                raise RuntimeError("Docker command exceeded its stdout limit")
        returncode = process.wait(timeout=max(0.1, deadline - time.monotonic()))
        return returncode, output.decode("utf-8", errors="replace")
    except (OSError, RuntimeError, subprocess.TimeoutExpired):
        process.kill()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        raise
    finally:
        selector.close()
        process.stdout.close()


def _image_id(image: str) -> str:
    docker = shutil.which("docker")
    if not docker:
        raise RuntimeError("Docker executable is unavailable; candidate execution refused")
    returncode, stdout = _run_bounded_stdout(
        [docker, "image", "inspect", "--format", "{{.Id}}", image],
        timeout=15,
        max_bytes=256,
    )
    if returncode != 0 or not stdout.strip().startswith("sha256:"):
        detail = stdout.strip()[:300] or "docker image inspect failed"
        raise RuntimeError(
            f"candidate sandbox image {image!r} is unavailable locally: {detail}"
        )
    return stdout.strip()


def _base_docker_args(image_id: str, container_name: str) -> list[str]:
    docker = shutil.which("docker")
    if not docker:
        raise RuntimeError("Docker executable is unavailable; candidate execution refused")
    return [
        docker, "run", "--detach", "--rm", "--pull=never", "--name", container_name,
        "--network", "none", "--read-only", "--cap-drop=ALL",
        "--cap-add=SETUID", "--cap-add=SETGID", "--cap-add=KILL",
        "--cap-add=DAC_READ_SEARCH",
        "--security-opt=no-new-privileges", "--pids-limit", SANDBOX_PIDS,
        "--memory", SANDBOX_MEMORY, "--memory-swap", SANDBOX_MEMORY,
        "--cpus", SANDBOX_CPUS, "--log-driver=none",
        "--tmpfs", f"/tmp:rw,nosuid,nodev,noexec,size=64m,nr_inodes={SANDBOX_TMP_INODES}",
        "--tmpfs", (
            f"/sandbox/output:rw,nosuid,nodev,noexec,"
            f"size={SANDBOX_OUTPUT_TMPFS},nr_inodes={SANDBOX_OUTPUT_INODES},mode=1777"
        ),
        "--tmpfs", (
            f"{CONTROL_DIR}:rw,nosuid,nodev,noexec,size=1m,"
            "nr_inodes=8,mode=0700"
        ),
        "--env", "HOME=/tmp", "--env", "PYTHONHASHSEED=0",
        "--env", "OMP_NUM_THREADS=1",
        "--env", "MKL_NUM_THREADS=1",
        "--env", "OPENBLAS_NUM_THREADS=1",
        "--env", "NUMEXPR_NUM_THREADS=1",
        "--env", "RESEARCH_CANDIDATE_SANDBOX=docker",
        "--workdir", "/sandbox", image_id,
    ]


def _docker_worker_command(
    image_id: str,
    container_name: str,
    source_root: Path,
    input_root: Path,
    preflight: bool,
) -> list[str]:
    mounts = [
        f"type=bind,src={source_root.resolve()},dst=/sandbox/src,readonly",
        f"type=bind,src={input_root.resolve()},dst=/sandbox/input,readonly",
    ]
    bootstrap = (
        "import sys; sys.path.insert(0, '/sandbox/src'); "
        "from research.worker_supervisor import main; sys.exit(main())"
    )
    command = _base_docker_args(image_id, container_name)
    image_index = command.index(image_id)
    mount_args = [argument for mount in mounts for argument in ("--mount", mount)]
    command[image_index:image_index] = mount_args
    command.extend([
        "python", "-I", "-c", bootstrap,
        "--config", "/sandbox/input/worker-config.json",
    ])
    if preflight:
        command.append("--preflight")
    return command


def _run_container_cleanup_command(
    command: list[str], *, deadline: float,
) -> None:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise RuntimeError("candidate container cleanup deadline expired")
    action = command[1]
    try:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=min(CONTAINER_CLEANUP_ACTION_TIMEOUT_SECONDS, remaining),
            env=child_env(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Docker {action} cleanup command failed") from exc
    if result.returncode == 0:
        return

    detail = (result.stderr or result.stdout).strip()
    normalized_detail = detail.casefold()
    if "no such container" in normalized_detail:
        return
    if action in {"exec", "stop"} and "not running" in normalized_detail:
        return
    # a --rm container's daemon-side removal may already be in flight after
    # `docker stop`; the removal goal is then already being met
    if action == "rm" and "removal of container" in normalized_detail and (
        "already in progress" in normalized_detail
    ):
        return
    summary = detail[:300] or f"exit status {result.returncode}"
    raise RuntimeError(f"Docker {action} cleanup failed: {summary}")


def _stop_container(container_name: str, *, deadline: float | None = None) -> None:
    docker = shutil.which("docker")
    if not docker:
        raise RuntimeError("Docker executable is unavailable during container cleanup")
    cleanup_deadline = (
        time.monotonic() + HOST_CLEANUP_ALLOWANCE_SECONDS
        if deadline is None
        else deadline
    )
    failures = []
    for command in (
        [docker, "stop", "--time", "0", container_name],
        [docker, "rm", "-f", container_name],
    ):
        try:
            _run_container_cleanup_command(command, deadline=cleanup_deadline)
        except RuntimeError as exc:
            failures.append(exc)
    if failures:
        details = "; ".join(str(failure) for failure in failures)
        raise RuntimeError(
            f"candidate container stop/remove failed: {details}"
        ) from failures[0]


def _container_is_running(container_name: str) -> bool:
    docker = shutil.which("docker")
    if not docker:
        return False
    try:
        returncode, stdout = _run_bounded_stdout(
            [docker, "inspect", "--format", "{{.State.Running}}", container_name],
            timeout=2,
            max_bytes=16,
        )
    except (OSError, RuntimeError, subprocess.TimeoutExpired):
        return False
    return returncode == 0 and stdout.strip() == "true"


def _container_ready(container_name: str) -> bool:
    docker = shutil.which("docker")
    if not docker:
        return False
    try:
        returncode, _stdout = _run_bounded_stdout(
            [docker, "exec", container_name, "test", "-f", CONTROL_READY_PATH],
            timeout=2,
            max_bytes=16,
        )
    except (OSError, RuntimeError, subprocess.TimeoutExpired):
        return False
    return returncode == 0


def _release_container(container_name: str, *, deadline: float | None = None) -> None:
    docker = shutil.which("docker")
    if not docker:
        raise RuntimeError("Docker executable is unavailable during container release")
    cleanup_deadline = (
        time.monotonic() + HOST_CLEANUP_ALLOWANCE_SECONDS
        if deadline is None
        else deadline
    )
    _run_container_cleanup_command(
        [docker, "exec", container_name, "touch", CONTROL_RELEASE_PATH],
        deadline=cleanup_deadline,
    )


def _wait_for_container_exit(container_name: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _container_is_running(container_name):
            return
        time.sleep(0.1)


def _open_regular_output(path: str | Path, max_bytes: int) -> tuple[int, int]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError("container artifact cannot be opened safely") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise RuntimeError("container artifact is not a regular file")
        if info.st_size > max_bytes:
            raise RuntimeError("container artifact exceeds its per-file limit")
        return fd, info.st_size
    except Exception:
        os.close(fd)
        raise


def _stream_container_output_file(path: str, max_bytes: int) -> None:
    fd, size = _open_regular_output(path, max_bytes)
    try:
        header = struct.pack(">Q", size)
        view = memoryview(header)
        while view:
            written = os.write(1, view)
            view = view[written:]
        remaining = size
        while remaining:
            chunk = os.read(fd, min(65536, remaining))
            if not chunk:
                raise RuntimeError("container artifact was truncated during export")
            view = memoryview(chunk)
            while view:
                written = os.write(1, view)
                view = view[written:]
            remaining -= len(chunk)
        if os.read(fd, 1):
            raise RuntimeError("container artifact grew during export")
    finally:
        os.close(fd)


_CONTAINER_COPY_SCRIPT = (
    "import sys; sys.path.insert(0, '/sandbox/src'); "
    "from research.spawn import _stream_container_output_file; "
    "_stream_container_output_file(sys.argv[1], int(sys.argv[2]))"
)


def _copy_stream_to_file(
    command: list[str],
    destination: Path,
    *,
    max_bytes: int,
    aggregate_remaining: int,
    timeout: float = 20.0,
) -> tuple[int, str]:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite output {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=child_env(),
    )
    if process.stdout is None:
        raise RuntimeError("failed to open streamed artifact output")
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    deadline = time.monotonic() + timeout
    digest = hashlib.sha256()
    created = False

    def read_exact(count: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < count:
            remaining_time = deadline - time.monotonic()
            if remaining_time <= 0:
                raise subprocess.TimeoutExpired(command, timeout)
            events = selector.select(min(remaining_time, 0.25))
            if not events:
                continue
            chunk = os.read(
                process.stdout.fileno(),
                min(65536, count - len(chunks)),
            )
            if not chunk:
                raise RuntimeError("container artifact stream ended early")
            chunks.extend(chunk)
        return bytes(chunks)

    try:
        size = struct.unpack(">Q", read_exact(8))[0]
        if size > max_bytes or size > aggregate_remaining:
            raise RuntimeError("container artifact exceeds its file or run limit")
        with destination.open("xb") as output:
            created = True
            remaining = size
            while remaining:
                chunk = read_exact(min(65536, remaining))
                output.write(chunk)
                digest.update(chunk)
                remaining -= len(chunk)
            output.flush()
            os.fsync(output.fileno())
        returncode = process.wait(timeout=max(0.1, deadline - time.monotonic()))
        extra = os.read(process.stdout.fileno(), 1)
        if extra:
            raise RuntimeError("container artifact stream exceeded its declared size")
        if returncode != 0:
            raise RuntimeError(
                f"container artifact export failed with status {returncode}"
            )
        return size, digest.hexdigest()
    except (OSError, RuntimeError, subprocess.TimeoutExpired):
        process.kill()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        if created:
            destination.unlink(missing_ok=True)
        raise
    finally:
        selector.close()
        process.stdout.close()


def _copy_container_file(
    container_name: str,
    source_path: str,
    destination: Path,
    *,
    max_bytes: int,
    aggregate_remaining: int,
    timeout: float = 20.0,
) -> tuple[int, str]:
    allowed_control = source_path == CONTROL_RESULT_PATH
    output_path = PurePosixPath(source_path)
    allowed_output = (
        output_path.is_absolute()
        and output_path.parent == PurePosixPath("/sandbox/output")
        and output_path.name not in ("", ".", "..")
    )
    if not (allowed_control or allowed_output):
        raise RuntimeError("container artifact path is outside the export allowlist")
    docker = shutil.which("docker")
    if not docker:
        raise RuntimeError("Docker executable is unavailable; artifact export refused")
    return _copy_stream_to_file(
        [
            docker, "exec", container_name, "python3", "-I", "-c",
            _CONTAINER_COPY_SCRIPT, source_path, str(max_bytes),
        ],
        destination,
        max_bytes=max_bytes,
        aggregate_remaining=aggregate_remaining,
        timeout=timeout,
    )


def run_sandboxed_candidate(
    config: dict,
    *,
    preflight: bool = False,
    timeout: float | None = None,
    source_root: Path | None = None,
    expected_image_id: str | None = None,
) -> dict:
    """Run the candidate worker in a bounded, networkless Docker container."""
    from research.train_worker import read_candidate_source

    editable = config.get("editable", "src/research/candidate.py")
    candidate_source = (
        read_candidate_source(config)[0]
        if source_root is None
        else _candidate_bytes(config, source_root)
    )
    image = os.environ.get("RESEARCH_SANDBOX_IMAGE", DEFAULT_SANDBOX_IMAGE)
    image_id = _image_id(image)
    if expected_image_id is not None and image_id != expected_image_id:
        raise RuntimeError("candidate sandbox image ID changed; refusing execution")

    budget = float(config.get("time_budget_seconds", 20.0))
    worker_timeout = (
        float(timeout)
        if timeout is not None
        else sandbox_worker_process_timeout(budget)
    )
    if worker_timeout < budget:
        raise ValueError("sandbox worker timeout cannot be shorter than the fit budget")
    hard_timeout = (
        worker_timeout
        + ARTIFACT_VALIDATION_TIMEOUT_SECONDS
        + HOST_ARTIFACT_EXPORT_TIMEOUT_SECONDS * HOST_ARTIFACT_EXPORT_FILE_COUNT
        + HOST_HANDOFF_SECONDS
        + HOST_CLEANUP_ALLOWANCE_SECONDS
    )
    container_name = f"research-candidate-{uuid.uuid4().hex[:12]}"
    created_outputs: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="research-candidate-sandbox-") as tmp:
        root = Path(tmp)
        os.chmod(root, 0o700)
        input_root = root / "input"
        input_root.mkdir(mode=0o700)
        os.chmod(input_root, 0o700)
        if source_root is None:
            staged_source = _stage_source_tree(
                Path(config["worktree_path"]).resolve(strict=True),
                root, editable, candidate_source,
            )
        else:
            staged_source = Path(source_root).resolve(strict=True)
            _validate_source_stage(staged_source)
        for path in (staged_source, *staged_source.rglob("*")):
            if path.is_dir():
                os.chmod(path, 0o755)
            elif path.is_file():
                os.chmod(path, 0o644)

        staged_config = {
            **config,
            "candidate_sha256": config["candidate_sha256"],
            "preflight": preflight,
        }
        staged_config = _stage_worker_inputs(staged_config, input_root)
        for path in input_root.iterdir():
            if path.is_file():
                os.chmod(path, 0o600)
        config_path = input_root / "worker-config.json"
        config_path.write_text(json.dumps(staged_config, ensure_ascii=False))
        os.chmod(config_path, 0o600)
        name = container_name
        command = _docker_worker_command(
            image_id, name, staged_source, input_root, preflight,
        )
        try:
            returncode, _container_id = _run_bounded_stdout(
                command, timeout=15, max_bytes=256
            )
            if returncode != 0:
                raise RuntimeError("Docker failed to start the candidate sandbox")
            hard_deadline = time.monotonic() + hard_timeout
            deadline = min(
                hard_deadline,
                time.monotonic()
                + worker_timeout
                + ARTIFACT_VALIDATION_TIMEOUT_SECONDS
                + SANDBOX_WORKER_CLEANUP_ALLOWANCE_SECONDS,
            )
            ready = False
            while time.monotonic() < deadline:
                if _container_ready(name):
                    ready = True
                    break
                if not _container_is_running(name):
                    raise RuntimeError(
                        "candidate sandbox exited before publishing its result"
                    )
                time.sleep(0.2)
            if not ready:
                raise TimeoutError(
                    f"isolated candidate worker exceeded {worker_timeout:.0f}s"
                )

            result_path = root / "worker-result.json"
            total_exported = 0
            result_size, _result_sha = _copy_container_file(
                name,
                CONTROL_RESULT_PATH,
                result_path,
                max_bytes=MAX_CONTROL_RESULT_BYTES,
                aggregate_remaining=MAX_OUTPUT_TOTAL_BYTES,
                timeout=min(
                    HOST_ARTIFACT_EXPORT_TIMEOUT_SECONDS,
                    hard_deadline - time.monotonic(),
                ),
            )
            total_exported += result_size
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            if not payload.get("ok"):
                raise RuntimeError(
                    f"isolated candidate worker failed: {payload.get('error', 'unknown error')}"
                )
            payload["sandbox_image_id"] = image_id

            output_dir = Path(config["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            stem = config["output_stem"]
            for field, filename in (
                ("weights_path", f"{stem}.pt"),
                ("architecture_path", f"{stem}-architecture.json"),
            ):
                container_path = payload.get(field)
                if container_path != f"/sandbox/output/{filename}":
                    raise RuntimeError(f"isolated worker returned an unexpected {field}")
                host_path = output_dir / filename
                size, digest = _copy_container_file(
                    name,
                    container_path,
                    host_path,
                    max_bytes=MAX_OUTPUT_FILE_BYTES,
                    aggregate_remaining=MAX_OUTPUT_TOTAL_BYTES - total_exported,
                    timeout=min(
                        HOST_ARTIFACT_EXPORT_TIMEOUT_SECONDS,
                        hard_deadline - time.monotonic(),
                    ),
                )
                created_outputs.append(host_path)
                total_exported += size
                payload[field] = str(host_path)
                digest_field = (
                    "weights_sha256"
                    if field == "weights_path"
                    else "architecture_sha256"
                )
                payload[digest_field] = digest

            _release_container(name)
            _wait_for_container_exit(name, timeout=5.0)
            return payload
        except OSError as exc:
            raise RuntimeError(
                f"could not start or collect the Docker candidate sandbox: {exc}"
            ) from exc
        finally:
            cleanup_errors = []
            cleanup_deadline = time.monotonic() + HOST_CLEANUP_ALLOWANCE_SECONDS
            try:
                for cleanup in (_release_container, _stop_container):
                    try:
                        cleanup(name, deadline=cleanup_deadline)
                    except Exception as exc:
                        cleanup_errors.append(exc)
            finally:
                # A partial artifact set is never returned as a successful run.
                if len(created_outputs) != 2:
                    for path in created_outputs:
                        path.unlink(missing_ok=True)
            if cleanup_errors:
                details = "; ".join(str(error) for error in cleanup_errors)
                raise RuntimeError(
                    f"candidate sandbox cleanup failed: {details}"
                ) from cleanup_errors[0]

def sandbox_main(argv: list[str] | None = None) -> int:
    """Host-side wrapper invoked by runner.py; it never imports candidate code."""
    import argparse
    import traceback

    parser = argparse.ArgumentParser(prog="research.sandbox")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    try:
        config = json.loads(Path(args.config).read_text())
        timeout = sandbox_worker_process_timeout(
            float(config.get("time_budget_seconds", 20.0))
        )
        result = run_sandboxed_candidate(config, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - report the worker failure to runner
        print(json.dumps({
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-1500:],
        }))
        return 1
    print(json.dumps({"ok": True, **result}))
    return 0
