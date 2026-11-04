"""Record comparable experiment outcomes after task-specific validation."""
from __future__ import annotations

import csv
import fcntl
import io
import math
import os
import re
import stat
import subprocess
import uuid
from pathlib import Path, PurePosixPath

FIELDS = ("commit", "metric", "value", "direction", "decision", "description",
          "peak_vram_mb", "protocol_sha256", "evidence", "promotion")


def read_regular(path: Path, limit: int) -> bytes:
    """Read a bounded regular file without following a final symlink."""
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError("input is not a bounded regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            content = stream.read(limit + 1)
        if len(content) > limit:
            raise ValueError("input exceeds the size limit")
        return content
    finally:
        os.close(descriptor)


def open_regular(path: Path):
    """Open a private output file without following symbolic or extra hard links."""
    flags = os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("output must be a regular file without extra hard links")
        return os.fdopen(descriptor, "r+", encoding="utf-8", newline="")
    except BaseException:
        os.close(descriptor)
        raise


def committed_source(repo: Path, commit: str, editable: str) -> bytes:
    """Read one bounded committed file, without interpreting a revision argument."""
    if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit):
        raise ValueError("commit must be a full hexadecimal Git commit ID")
    path = PurePosixPath(editable)
    if path.is_absolute() or ".." in path.parts or not path.parts or str(path) != editable:
        raise ValueError("editable must be a relative Git path")
    obj = f"{commit}:{editable}"
    size = subprocess.run(["git", "cat-file", "-s", obj], cwd=repo,
                          stdin=subprocess.DEVNULL, capture_output=True,
                          text=True, check=True, timeout=15).stdout.strip()
    if not size.isdigit() or not 0 < int(size) <= 2 * 1024 * 1024:
        raise ValueError("commit does not contain a bounded training file")
    return subprocess.run(["git", "show", obj], cwd=repo, stdin=subprocess.DEVNULL,
                          capture_output=True, check=True, timeout=15).stdout


def write_new_atomic(path: Path, payload: bytes) -> None:
    """Create a complete file atomically without overwriting an existing result."""
    temporary = path.parent / (".result-" + uuid.uuid4().hex)
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
    finally:
        temporary.unlink(missing_ok=True)


def append_history(path: Path, *, commit: str, metric: str, value: float | None,
                   direction: str, decision: str, description: str,
                   protocol: str, evidence: str, peak_vram: str = "NOT_AVAILABLE",
                   promotion: str = "strict") -> None:
    """Append one decision; reject incomparable protocols and false improvements."""
    if (not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit)
            or direction not in ("minimize", "maximize")
            or decision not in ("KEEP", "REVERT", "CRASH")
            or promotion not in ("strict", "external")):
        raise ValueError("invalid direction or decision")
    if not metric or not evidence or not description.strip() or len(description) > 400:
        raise ValueError("metric, evidence and a short description are required")
    if decision == "CRASH":
        if value is not None:
            raise ValueError("a crash cannot carry a numeric score")
    elif (isinstance(value, bool) or not isinstance(value, (int, float))
          or not math.isfinite(value) or not re.fullmatch(r"[0-9a-f]{64}", protocol)):
        raise ValueError("a scored result needs a finite value and protocol hash")
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.parent / ("." + path.name + ".lock")
    with open_regular(lock_path) as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            contents = read_regular(path, 2 * 1024 * 1024).decode("utf-8")
        except FileNotFoundError:
            contents = ""
        if contents and not contents.endswith("\n"):
            raise ValueError("results file has an incomplete final row")
        reader = csv.DictReader(io.StringIO(contents), delimiter="\t", strict=True)
        existing = list(reader)
        if reader.fieldnames and reader.fieldnames != list(FIELDS):
            raise ValueError("results file has an incompatible schema")
        if any(None in row or any(row.get(field) is None for field in FIELDS) for row in existing):
            raise ValueError("results file contains an incomplete row")
        for row in existing:
            if row["decision"] not in ("KEEP", "REVERT", "CRASH"):
                raise ValueError("results file contains an invalid decision")
            if row["decision"] == "CRASH":
                if row["value"]:
                    raise ValueError("crash row contains a score")
            else:
                try:
                    prior = float(row["value"])
                except ValueError as exc:
                    raise ValueError("results file contains an invalid score") from exc
                if (not math.isfinite(prior)
                        or not re.fullmatch(r"[0-9a-f]{64}", row["protocol_sha256"])):
                    raise ValueError("results file contains an invalid score or protocol")
        if protocol and any(row.get("protocol_sha256") not in ("", protocol) for row in existing):
            raise ValueError("result protocol differs from this run's earlier results")
        if any((row.get("metric"), row.get("direction"), row.get("evidence"), row.get("promotion"))
               != (metric, direction, evidence, promotion) for row in existing):
            raise ValueError("result metadata differs from this run's earlier results")
        if any(row.get("commit") == commit for row in existing):
            raise ValueError("candidate commit is already recorded")
        if decision == "KEEP" and promotion == "strict":
            previous = [float(row["value"]) for row in existing
                        if row.get("decision") == "KEEP" and row.get("value")]
            if previous and ((direction == "maximize" and value <= max(previous))
                             or (direction == "minimize" and value >= min(previous))):
                raise ValueError("KEEP requires an improvement over the previous champion")
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(existing)
        writer.writerow(dict(zip(FIELDS, (commit, metric, "" if value is None else value, direction,
                                         decision, description, peak_vram, protocol, evidence,
                                         promotion), strict=True)))
        payload = output.getvalue().encode("utf-8")
        if len(payload) > 2 * 1024 * 1024:
            raise ValueError("results file exceeds the size limit")
        temporary = path.parent / (".history-" + uuid.uuid4().hex)
        try:
            with temporary.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
