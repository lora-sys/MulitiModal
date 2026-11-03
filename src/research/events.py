"""Trusted host-side progress events for a research run.

One writer, and it is the coordinator — never the candidate. The candidate
trains inside an isolated subprocess and has no handle on this module, so a
number appearing in `events.jsonl` was written by the host that also ran the
frozen evaluator. `status.json` is the same information in latest-wins form so
the dashboard can render without replaying the whole log.

Both files are append-only / atomically-replaced:

* `events.jsonl` is append-only; a torn final line (process killed mid-write)
  is DROPPED by the reader rather than raising, because a crash is exactly when
  someone wants to read the log;
* `status.json` is written to a temp file in the same directory and `os.replace`d,
  so a reader never sees a half-written JSON document.

A single writer is enforced with a lock file (`fcntl.flock`, non-blocking): two
coordinators appending to one run would otherwise interleave sequences and make
the dashboard's `after=<seq>` paging meaningless.

The event field set is the one the issue specifies, and nothing else may be
added to it casually — the dashboard and the tests both read it positionally
by name.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

EVENT_SCHEMA_VERSION = "autoresearch-events-v1"
STATUS_SCHEMA_VERSION = "autoresearch-status-v1"

#: stage of a round, in the order they occur
STAGES = (
    "DOCTOR",
    "PROPOSING",
    "TRAINING",
    "EVALUATING",
    "DECIDING",
    "IDLE",
    "DONE",
)

#: free-form status, kept small so the dashboard can colour it deterministically
STATUS_VALUES = (
    "started",
    "heartbeat",
    "finished",
    "failed",
    "blocked",
    "warning",
    "info",
    "stopped",
)

EVENTS_FILE = "events.jsonl"
STATUS_FILE = "status.json"
_LOCK_FILE = ".coordinator.lock"

#: cheap caps so one runaway round cannot make the log unreadable: a message is
#: a sentence, not a stack dump, and metrics are small scalars
MAX_MESSAGE_CHARS = 500
MAX_METRIC_KEYS = 24
MAX_EVENT_BYTES = 4096
HEARTBEAT_INTERVAL_SECONDS = 2.0
#: a reader treats a run with no heartbeat newer than this as STALE
STALE_AFTER_SECONDS = 15.0


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _truncate(value: str) -> str:
    text = str(value)
    return text if len(text) <= MAX_MESSAGE_CHARS else text[: MAX_MESSAGE_CHARS - 3] + "..."


def _clean_metrics(metrics: dict | None) -> dict | None:
    """Numbers and short strings only.

    The dashboard renders metrics into SVG text and a table; passing through
    arbitrary nested structures would let a candidate (or a future caller)
    smuggle a large object into every event. `None` means "no metrics", which is
    different from a metric that is missing — callers must say NOT_AVAILABLE in
    the message or omit the field entirely, never invent a number.
    """
    if not metrics:
        return None
    out: dict = {}
    for key, value in list(metrics.items())[:MAX_METRIC_KEYS]:
        if isinstance(value, bool) or value is None:
            out[str(key)[:40]] = value
        elif isinstance(value, (int, float)):
            out[str(key)[:40]] = value
        else:
            out[str(key)[:40]] = _truncate(value)
    return out


@dataclass
class EventWriter:
    """Append-only event log plus atomic latest-status for one run."""

    run_dir: Path
    run_id: str
    _lock_fd: int | None = field(default=None, repr=False)
    _seq: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._acquire_lock()
        self._seq = last_seq(self.events_path)

    # -- files --------------------------------------------------------------

    @property
    def events_path(self) -> Path:
        return self.run_dir / EVENTS_FILE

    @property
    def status_path(self) -> Path:
        return self.run_dir / STATUS_FILE

    # -- single writer ------------------------------------------------------

    def _acquire_lock(self) -> None:
        import fcntl

        fd = os.open(self.run_dir / _LOCK_FILE, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            raise RuntimeError(
                f"another coordinator already owns run {self.run_id!r} "
                f"({self.run_dir}); refusing to become a second writer"
            ) from exc
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        self._lock_fd = fd

    def close(self) -> None:
        if self._lock_fd is not None:
            import fcntl

            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(self._lock_fd)
                self._lock_fd = None

    def __enter__(self) -> EventWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- writing ------------------------------------------------------------

    def emit(
        self,
        stage: str,
        status: str,
        *,
        round_index: int | None = None,
        repeat: int | None = None,
        fold: int | None = None,
        folds_total: int | None = None,
        elapsed_seconds: float | None = None,
        budget_seconds: float | None = None,
        metrics: dict | None = None,
        message: str | None = None,
        extra: dict | None = None,
    ) -> dict:
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
        if status not in STATUS_VALUES:
            raise ValueError(f"unknown status {status!r}; expected one of {STATUS_VALUES}")
        self._seq += 1
        event = {
            "schema_version": EVENT_SCHEMA_VERSION,
            "seq": self._seq,
            "time": _now_iso(),
            "monotonic": time.monotonic(),
            "run_id": self.run_id,
            "stage": stage,
            "status": status,
            "round": round_index,
            "repeat": repeat,
            "fold": fold,
            "folds_total": folds_total,
            "elapsed_seconds": (
                None if elapsed_seconds is None else round(float(elapsed_seconds), 3)
            ),
            "budget_seconds": (
                None if budget_seconds is None else round(float(budget_seconds), 3)
            ),
            "metrics": _clean_metrics(metrics),
            "message": None if message is None else _truncate(message),
        }
        self._append(event)
        self._write_status(event, extra=extra)
        return event

    def _append(self, event: dict) -> None:
        line = json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
        if len(line.encode("utf-8")) > MAX_EVENT_BYTES:
            event = {**event, "metrics": None, "message": _truncate(event.get("message") or "")}
            line = json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
        # a torn tail is healed by appending on a fresh line rather than by
        # rewriting history: readers drop the incomplete line. The file is
        # opened "a+" so the last byte can be inspected before appending.
        with self.events_path.open("a+", encoding="utf-8") as handle:
            handle.seek(0, os.SEEK_END)
            end = handle.tell()
            if end > 0:
                handle.seek(end - 1)
                if handle.read(1) != "\n":
                    handle.write("\n")
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def _write_status(self, event: dict, extra: dict | None = None) -> None:
        payload = {
            "schema_version": STATUS_SCHEMA_VERSION,
            "run_id": self.run_id,
            "seq": event["seq"],
            "updated_at": event["time"],
            "updated_monotonic": event["monotonic"],
            "stage": event["stage"],
            "status": event["status"],
            "round": event["round"],
            "repeat": event["repeat"],
            "fold": event["fold"],
            "folds_total": event["folds_total"],
            "elapsed_seconds": event["elapsed_seconds"],
            "budget_seconds": event["budget_seconds"],
            "metrics": event["metrics"],
            "message": event["message"],
        }
        if extra:
            payload.update(extra)
        write_status_atomic(self.status_path, payload)


def write_status_atomic(path: Path, payload: dict) -> None:
    """Write `payload` to `path` so a concurrent reader sees old or new, never half."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_events(path: Path, after_seq: int = 0, limit: int = 500) -> list[dict]:
    """Events with `seq > after_seq`, oldest first.

    Malformed and torn lines are skipped, not fatal: the log is read most often
    right after something went wrong.
    """
    path = Path(path)
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict) or not isinstance(event.get("seq"), int):
                continue
            if event["seq"] <= after_seq:
                continue
            out.append(event)
            if len(out) >= limit:
                break
    return out


def last_seq(path: Path) -> int:
    seq = 0
    for event in read_events(path, after_seq=0, limit=100_000):
        seq = max(seq, int(event.get("seq", 0)))
    return seq


def read_status(path: Path, now_monotonic: float | None = None) -> dict | None:
    """Latest status, with `heartbeat_age_seconds` and a derived `stale` flag.

    A run that has not written anything for `STALE_AFTER_SECONDS` is STALE. The
    dashboard must show that rather than a frozen "TRAINING" that implies work
    is happening.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict):
        return None
    stamped = payload.get("updated_monotonic")
    now = time.monotonic() if now_monotonic is None else now_monotonic
    age = None if not isinstance(stamped, (int, float)) else max(0.0, now - stamped)
    payload["heartbeat_age_seconds"] = None if age is None else round(age, 1)
    payload["stale"] = bool(age is not None and age > STALE_AFTER_SECONDS)
    return payload