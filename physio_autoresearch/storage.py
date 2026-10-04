"""OS-locked, hash-chained local records. This is integrity detection, not an OS sandbox."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path

from .contracts import ResearchError, digest


@contextmanager
def exclusive(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        # A Windows byte-range lock also denies reading the byte in another
        # descriptor; inspect length without reading locked content.
        if path.stat().st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ResearchError("ROUND_BUSY", "another process owns this research state") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def history(path: Path) -> list[dict]:
    previous, result = "0" * 64, []
    if not path.exists():
        return result
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            claimed = entry.pop("entry_sha256")
            if entry["previous_sha256"] != previous or digest(entry) != claimed:
                raise ResearchError("HISTORY_CORRUPTED", "journal hash chain mismatch")
            entry["entry_sha256"] = claimed
            previous = claimed
            result.append(entry)
    except (ValueError, KeyError) as exc:
        raise ResearchError("HISTORY_CORRUPTED", str(exc)) from exc
    return result


def append(path: Path, entry: dict):
    entries = history(path)
    value = {**entry, "previous_sha256": entries[-1]["entry_sha256"] if entries else "0" * 64}
    value["entry_sha256"] = digest(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
