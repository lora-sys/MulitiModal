"""Trusted one-shot worker for a single trimmed training fold.

The worker hashes and validates candidate source without executing it, loads
only the fold's training rows, then forks a candidate process under UID 65532.
Candidate import, build, train, forward validation, and state_dict export all
run in that child under one strict wall deadline. The trusted root parent
measures with its captured monotonic clock, kills the candidate process tree,
and sends the result through a supervisor-only pipe that the candidate cannot
inherit or reopen.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
import re
import signal
import stat
import sys
import time
import traceback
from pathlib import Path, PurePosixPath

from research.frozen import TIME_BUDGET_SECONDS
from research.spawn import (
    CANDIDATE_SANDBOX_GID,
    CANDIDATE_SANDBOX_UID,
    TRUSTED_FIT_TIMER,
)

MAX_CANDIDATE_SOURCE_BYTES = 1024 * 1024


CANDIDATE_POST_TRAIN_RESERVE_SECONDS = 2.0
CANDIDATE_CLEANUP_POLL_SECONDS = 0.001
# The validated output contract relayed from the forked candidate child is a
# tiny fixed-shape dict; bound the relay pipe read anyway.
MAX_CANDIDATE_PAYLOAD_BYTES = 65536


def _read_payload(fd: int) -> dict:
    """Drain the candidate child's one-shot payload pipe and parse it."""
    chunks: list[bytes] = []
    while True:
        chunk = os.read(fd, 4096)
        if not chunk:
            break
        chunks.append(chunk)
        if sum(len(item) for item in chunks) > MAX_CANDIDATE_PAYLOAD_BYTES:
            raise RuntimeError("candidate phase payload exceeds the relay limit")
    data = b"".join(chunks)
    if not data:
        return {}
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("candidate phase relayed an unparseable payload") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("candidate phase payload must be a JSON object")
    return payload


def _trusted_candidate_phase_runner(result_fd: int | None = None):
    """Time and reap the full candidate process tree in a trusted parent."""
    monotonic = time.monotonic
    sleep = time.sleep
    fork = os.fork
    pipe = os.pipe
    waitpid = os.waitpid
    kill = os.kill
    kill_group = os.killpg
    new_session = os.setsid
    close_fd = os.close
    set_groups = os.setgroups
    set_gid = os.setgid
    set_uid = os.setuid
    get_euid = os.geteuid
    list_processes = os.listdir
    read_text = Path.read_text
    exit_process = os._exit
    write_fd = os.write
    format_exception = traceback.format_exc
    wait_nohang = os.WNOHANG
    is_exited = os.WIFEXITED
    exit_status = os.WEXITSTATUS
    is_signaled = os.WIFSIGNALED
    term_signal = os.WTERMSIG
    kill_signal = signal.SIGKILL

    def kill_candidate_descendants(deadline: float) -> None:
        while True:
            candidate_pids = []
            for name in list_processes("/proc"):
                if not name.isdigit():
                    continue
                try:
                    lines = read_text(
                        Path("/proc") / name / "status", encoding="ascii"
                    ).splitlines()
                    uid_line = next(line for line in lines if line.startswith("Uid:"))
                    state_line = next(line for line in lines if line.startswith("State:"))
                    uid = int(uid_line.split()[1])
                    state = state_line.split()[1]
                except (
                    FileNotFoundError, PermissionError, ProcessLookupError,
                    StopIteration, ValueError,
                ):
                    continue
                if uid == CANDIDATE_SANDBOX_UID and state not in {"Z", "X"}:
                    candidate_pids.append(int(name))
            if not candidate_pids:
                return
            for candidate_pid in candidate_pids:
                try:
                    kill(candidate_pid, kill_signal)
                except ProcessLookupError:
                    pass
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "candidate descendants did not stop before the strict fit deadline"
                )
            sleep(min(CANDIDATE_CLEANUP_POLL_SECONDS, remaining))

    def run(operation, budget: float) -> tuple[float, dict]:
        """Run `operation` in a dropped-privilege child.

        Returns (elapsed, payload) where payload is the operation's return
        value, relayed from the child over a one-shot pipe so the validated
        output contract produced under the strict deadline survives the fork.
        """
        if get_euid() != 0:
            raise RuntimeError("trusted candidate worker must retain a separate root uid")
        started = monotonic()
        deadline = started + budget
        payload_read_fd, payload_write_fd = pipe()
        pid = fork()
        if pid == 0:
            try:
                new_session()
                if result_fd is not None:
                    close_fd(result_fd)
                close_fd(payload_read_fd)
                set_groups([])
                set_gid(CANDIDATE_SANDBOX_GID)
                set_uid(CANDIDATE_SANDBOX_UID)
                payload = operation(started, monotonic)
                if payload is not None:
                    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
                    if len(data) > MAX_CANDIDATE_PAYLOAD_BYTES:
                        raise RuntimeError(
                            "candidate phase payload exceeds the relay limit"
                        )
                    while data:
                        written = write_fd(payload_write_fd, data)
                        data = data[written:]
                close_fd(payload_write_fd)
            except BaseException:
                detail = format_exception().strip()
                if detail:
                    try:
                        write_fd(2, (detail + "\n").encode("utf-8", errors="replace"))
                    except OSError:
                        pass
                exit_process(1)
            exit_process(0)

        close_fd(payload_write_fd)
        try:
            while True:
                try:
                    waited, status = waitpid(pid, wait_nohang)
                except InterruptedError:
                    continue
                if waited == pid:
                    try:
                        kill_group(pid, kill_signal)
                    except ProcessLookupError:
                        pass
                    kill_candidate_descendants(deadline)
                    elapsed = monotonic() - started
                    if elapsed > budget:
                        raise TimeoutError(
                            f"candidate phase exceeded its strict {budget:g}s budget"
                        )
                    if not is_exited(status) or exit_status(status) != 0:
                        detail = (
                            f"signal {term_signal(status)}"
                            if is_signaled(status)
                            else f"exit status {exit_status(status)}"
                            if is_exited(status)
                            else "unknown process status"
                        )
                        raise RuntimeError(f"candidate phase failed: {detail}")
                    return elapsed, _read_payload(payload_read_fd)

                remaining = deadline - monotonic()
                if remaining <= 0:
                    try:
                        kill_group(pid, kill_signal)
                    except ProcessLookupError:
                        try:
                            kill(pid, kill_signal)
                        except ProcessLookupError:
                            pass
                    try:
                        waitpid(pid, 0)
                    except ChildProcessError:
                        pass
                    kill_candidate_descendants(deadline)
                    raise TimeoutError(
                        f"candidate phase exceeded its strict {budget:g}s budget"
                    )
                sleep(min(0.01, remaining))
        finally:
            close_fd(payload_read_fd)

    return run


def _read_regular_file_limited(
    path: Path, *, max_bytes: int, label: str
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError(f"{label} cannot be opened as a regular file") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise RuntimeError(f"{label} must be a regular file")
        if info.st_size > max_bytes:
            raise RuntimeError(f"{label} exceeds the {max_bytes}-byte limit")
        chunks = bytearray()
        while len(chunks) <= max_bytes:
            chunk = os.read(fd, min(65536, max_bytes + 1 - len(chunks)))
            if not chunk:
                break
            chunks.extend(chunk)
        if len(chunks) > max_bytes:
            raise RuntimeError(f"{label} exceeds the {max_bytes}-byte limit")
        if len(chunks) != info.st_size:
            raise RuntimeError(f"{label} changed while it was being read")
        if os.read(fd, 1):
            raise RuntimeError(f"{label} grew while it was being read")
        return bytes(chunks)
    finally:
        os.close(fd)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _candidate_path(config: dict) -> Path:
    editable = config.get("editable", "src/research/candidate.py")
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
    worktree = Path(config["worktree_path"]).resolve(strict=True)
    candidate_path = worktree.joinpath(*relative.parts)
    if candidate_path.is_symlink():
        raise RuntimeError(f"editable {editable!r} final component is a symlink")
    # Resolve only the parent. Resolving the final component before opening it
    # turns O_NOFOLLOW into a check on the symlink target if the path changes
    # between this check and the read.
    try:
        resolved_parent = candidate_path.parent.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"editable {editable!r} parent is unavailable") from exc
    source_roots = {
        (worktree / "src").resolve(strict=True),
        Path(__file__).resolve().parents[1],
    }
    if not any(resolved_parent.is_relative_to(root) for root in source_roots):
        raise RuntimeError(f"editable {editable!r} escapes the trusted src trees")
    # Keep the final name intact. _read_regular_file_limited opens it with
    # O_NOFOLLOW and fstat checks the descriptor, so a swapped symlink fails.
    return resolved_parent / candidate_path.name


def read_candidate_source(config: dict) -> tuple[bytes, str]:
    """Read once, verify the recorded digest, and return those exact bytes."""
    candidate_path = _candidate_path(config)
    source = _read_regular_file_limited(
        candidate_path,
        max_bytes=MAX_CANDIDATE_SOURCE_BYTES,
        label="candidate source",
    )
    actual = _sha256_bytes(source)
    expected = config.get("candidate_sha256")
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise RuntimeError("candidate_sha256 must be a lowercase SHA-256")
    if actual != expected:
        raise RuntimeError(
            f"candidate file {candidate_path} hashes to {actual[:12]}… but the "
            f"record binds {expected[:12]}…; refusing to run code that is not "
            "the recorded one"
        )
    return source, actual


def assert_candidate_binding(config: dict) -> str:
    """Refuse to import unless the file hashes to the recorded value."""
    _source, actual = read_candidate_source(config)
    return actual


def validate_candidate_source(source: str) -> None:
    """Compile and statically check the required candidate API without import."""
    from research.guard import (
        PatchRejected,
        validate_frozen_imports,
        validate_source_ast,
    )

    try:
        compile(source, "<candidate>", "exec")
        tree = ast.parse(source)
    except (SyntaxError, ValueError, TypeError) as exc:
        raise PatchRejected(f"candidate syntax is invalid: {exc}") from exc
    validate_frozen_imports(source)
    validate_source_ast(source)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    missing = sorted({"build_model", "train"} - functions.keys())
    if missing:
        raise PatchRejected(f"candidate contract is missing top-level functions: {missing}")
    build_args = functions["build_model"].args
    if not (build_args.posonlyargs or build_args.args or build_args.vararg):
        raise PatchRejected("candidate build_model must accept a fold")
    train_args = functions["train"].args
    if len(train_args.posonlyargs) + len(train_args.args) < 4 and not train_args.vararg:
        raise PatchRejected(
            "candidate train must accept model, fold, budget_seconds, and seed"
        )


def import_candidate(config: dict, source: bytes | None = None):
    """Execute only bytes that match the recorded digest."""
    if source is None:
        source, _actual = read_candidate_source(config)
    else:
        actual = _sha256_bytes(source)
        expected = config.get("candidate_sha256")
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise RuntimeError("candidate_sha256 must be a lowercase SHA-256")
        if actual != expected:
            raise RuntimeError(
                "candidate source bytes do not match the recorded SHA; refusing import"
            )
    candidate_path = _candidate_path(config)
    parts = PurePosixPath(config.get("editable", "src/research/candidate.py")).parts
    module_name = ".".join(parts[1:]).removesuffix(".py")
    try:
        code = compile(source, str(candidate_path), "exec")
    except SyntaxError as exc:
        raise RuntimeError(f"candidate syntax is invalid: {exc}") from exc
    spec = importlib.util.spec_from_file_location(module_name, candidate_path)
    if spec is None:
        raise RuntimeError(f"cannot create an import spec for {candidate_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        exec(code, module.__dict__)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _require_candidate_container() -> None:
    if (
        os.environ.get("RESEARCH_CANDIDATE_SANDBOX") != "docker"
        or not Path("/.dockerenv").is_file()
    ):
        raise RuntimeError(
            "candidate execution requires the restricted Docker sandbox "
            "(network=none, read-only root, scoped mounts)"
        )


def _load_fold(config: dict, adapter, records=None):
    from research.tasks import validate_manifest_contract

    fit_manifest = adapter.load_manifest(config["fit_manifest_path"])
    validate_manifest_contract(
        config.get("task", "constitution"), fit_manifest, selection_pool=True,
    )
    if fit_manifest["split"]["dev"] or fit_manifest["split"]["holdout"]:
        raise ValueError("fit manifest must have empty dev and holdout partitions")
    if records is None:
        records = adapter.records_from_json(
            Path(config["fit_records_path"]).read_text()
        )
    fit_rows = set(fit_manifest["subject_row_by_id"].values())
    leaked = sorted({record.subject_index for record in records} - fit_rows)
    if leaked:
        raise RuntimeError(
            f"fit records contain rows outside the fit manifest: {leaked[:3]} — "
            "refusing to train on data the fold contract did not trim"
        )
    fold = adapter.training_fold_data(fit_manifest, records, config)
    return fit_manifest, records, fold


def _validate_model_output(model, fold, task: str) -> dict:
    import torch

    from core.batch import ModelOutput

    if model.input_spec != fold.input_spec:
        raise RuntimeError("candidate model input_spec differs from the training fold")
    item = next(iter(fold.train), None)
    if item is None:
        raise RuntimeError("candidate preflight has no synthetic training batch")
    batch = item.batch
    model.eval()
    with torch.no_grad():
        output = model(batch)
    if not isinstance(output, ModelOutput):
        raise RuntimeError("candidate forward must return core.batch.ModelOutput")
    key = {"syndrome": "syndrome_probs",
           "toy": "toy_probs"}.get(task, "category_probs")
    predictions = output.predictions
    if not isinstance(predictions, dict) or key not in predictions:
        raise RuntimeError(f"candidate output is missing predictions[{key!r}]")
    values = predictions[key]
    batch_size = next(iter(batch.features.values())).shape[0]
    targets = int(fold.dims["targets"])
    if not torch.is_tensor(values) or tuple(values.shape) != (batch_size, targets):
        shape = tuple(values.shape) if torch.is_tensor(values) else type(values).__name__
        raise RuntimeError(
            f"candidate predictions[{key!r}] has shape {shape}, expected "
            f"({batch_size}, {targets})"
        )
    if not torch.isfinite(values).all():
        raise RuntimeError(f"candidate predictions[{key!r}] contain non-finite values")
    if task in ("syndrome", "toy"):
        if torch.any(values < 0.0) or torch.any(values > 1.0):
            raise RuntimeError(
                f"candidate predictions[{key!r}] probabilities must be within [0, 1]"
            )
        row_sums = values.sum(dim=-1)
        if not torch.allclose(
            row_sums, torch.ones_like(row_sums), rtol=1e-4, atol=1e-4
        ):
            raise RuntimeError(
                f"candidate predictions[{key!r}] probabilities must sum to one per row"
            )
    return {"output_key": key, "output_shape": [batch_size, targets]}


def _candidate_phase_budget(config: dict) -> float:
    try:
        budget = float(config.get("time_budget_seconds", TIME_BUDGET_SECONDS))
    except (TypeError, ValueError) as exc:
        raise ValueError("fit time budget must be numeric") from exc
    if not math.isfinite(budget) or budget < 0:
        raise ValueError("fit time budget must be finite and non-negative")
    return budget


def _candidate_output_paths(config: dict) -> tuple[str, str, str, str]:
    output_dir = str(config.get("output_dir", ""))
    stem = config.get("output_stem")
    if output_dir != "/sandbox/output":
        raise RuntimeError("candidate output directory is outside the sandbox mount")
    if not isinstance(stem, str) or not stem.isascii() or not stem.isidentifier():
        raise RuntimeError("candidate output stem must be a Python identifier")
    return (
        output_dir,
        stem,
        f"{output_dir}/{stem}.pt",
        f"{output_dir}/{stem}-architecture.json",
    )


def _run_candidate_phase(
    config: dict,
    source: bytes,
    actual_sha: str,
    fold,
    *,
    preflight: bool,
    trusted_result_fd: int | None = None,
) -> dict:
    import torch

    from research.model_export import save_eval_model

    task = config.get("task", "constitution")
    seed = int(config.get("seed", 0))
    budget = _candidate_phase_budget(config)
    output_dir, stem, weights_path, architecture_path = _candidate_output_paths(config)
    import_candidate_ref = import_candidate
    validate_output_ref = _validate_model_output
    save_eval_model_ref = save_eval_model
    phase_runner = _trusted_candidate_phase_runner(trusted_result_fd)

    def candidate_operation(started, monotonic) -> dict:
        candidate = import_candidate_ref(config, source=source)
        model = candidate.build_model(fold)
        if not isinstance(model, torch.nn.Module):
            raise RuntimeError("candidate build_model must return a torch.nn.Module")
        if preflight:
            train_budget = 0.0
        else:
            remaining = budget - (monotonic() - started)
            train_budget = remaining - CANDIDATE_POST_TRAIN_RESERVE_SECONDS
            if train_budget <= 0:
                raise TimeoutError(
                    "candidate import/build left no safe training and export budget"
                )
        candidate.train(model, fold, train_budget, seed)
        candidate_output = validate_output_ref(model, fold, task)
        save_eval_model_ref(model, output_dir, stem, task=task)
        # relayed to the trusted parent across the fork via the payload pipe
        return candidate_output

    elapsed, candidate_output = phase_runner(candidate_operation, budget)
    return {
        "candidate_sha256": actual_sha,
        "task": task,
        "fit_seconds": elapsed,
        "fit_timer": TRUSTED_FIT_TIMER,
        "seed": seed,
        "budget_seconds": budget,
        "input_spec_version": fold.input_spec.version,
        "candidate_output": candidate_output,
        "output_key": candidate_output["output_key"],
        "output_shape": candidate_output["output_shape"],
        "weights_path": weights_path,
        "architecture_path": architecture_path,
    }


def preflight_candidate(
    config: dict, *, trusted_result_fd: int | None = None
) -> dict:
    """Run candidate import/build/train/output/export under a strict deadline."""
    from research.tasks import adapter_for

    _require_candidate_container()
    source, actual_sha = read_candidate_source(config)
    validate_candidate_source(source.decode("utf-8"))
    adapter = adapter_for(config.get("task", "constitution"))
    _manifest, _records, fold = _load_fold(config, adapter)
    return _run_candidate_phase(
        config, source, actual_sha, fold, preflight=True,
        trusted_result_fd=trusted_result_fd,
    )


def fit(
    config: dict, records=None, *, trusted_result_fd: int | None = None
) -> dict:
    """Load the trimmed fold, then supervise the complete candidate phase."""
    from research.tasks import adapter_for

    source, actual_sha = read_candidate_source(config)
    _require_candidate_container()
    validate_candidate_source(source.decode("utf-8"))
    adapter = adapter_for(config.get("task", "constitution"))
    _fit_manifest, _records, fold = _load_fold(config, adapter, records)
    import torch
    torch.manual_seed(int(config.get("seed", 0)))
    return _run_candidate_phase(
        config, source, actual_sha, fold, preflight=False,
        trusted_result_fd=trusted_result_fd,
    )


def main(argv: list[str] | None = None) -> int:
    # Candidate code may register shutdown handlers or patch json/os globals.
    # Keep the final worker result bound to these pre-candidate callables and
    # exit without running candidate-installed atexit handlers.
    trusted_dumps = json.dumps
    trusted_write = os.write
    trusted_exit = os._exit
    parser = argparse.ArgumentParser(prog="research.train_worker")
    parser.add_argument("--config", required=True,
                        help="JSON file with data paths, batch size and seed")
    parser.add_argument("--preflight", action="store_true",
                        help="validate the model output contract on synthetic input")
    parser.add_argument("--result-fd", type=int, required=True,
                        help="trusted supervisor result pipe")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text())
    try:
        if not stat.S_ISFIFO(os.fstat(args.result_fd).st_mode):
            raise RuntimeError("trusted result fd must be an anonymous pipe")
        result = (
            preflight_candidate(config, trusted_result_fd=args.result_fd)
            if args.preflight
            else fit(config, trusted_result_fd=args.result_fd)
        )
        payload = {"ok": True, **result}
        returncode = 0
    except Exception as exc:  # noqa: BLE001 — the parent records the crash
        payload = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-1500:],
        }
        returncode = 1
    encoded = (trusted_dumps(payload) + "\n").encode("utf-8")
    view = memoryview(encoded)
    try:
        while view:
            written = trusted_write(args.result_fd, view)
            if written <= 0:
                trusted_exit(1)
            view = view[written:]
    except OSError:
        trusted_exit(1)
    trusted_exit(returncode)


if __name__ == "__main__":
    sys.exit(main())
