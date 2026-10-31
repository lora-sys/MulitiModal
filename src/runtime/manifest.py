"""Append-only RunManifest store for the #45 harness (#44 §6).

Entries carry a hash chain (`prev_entry_sha256` + `entry_sha256`) so naive
in-place edits, deletions, or reordering of recorded rounds are detected on
load. Honest scope of this guarantee (review round-5 R-01/R-02):

- Leading entries written before the chain existed pass through unverified;
  an UNCHAINED line after any chained entry is rejected (forgery/insertion).
- append() refuses entries that already carry a hash and repairs an
  interrupted (newline-less or truncated) trailing line before loading, so
  a killed append cannot wedge the store.
- Deleting the last entry, or rewriting the file with a RECOMPUTED chain,
  is undetectable: the hash is unkeyed and the store lives in the same
  agent-writable out_dir as the artifacts. This is tamper-EVIDENCE against
  accidental or naive corruption only — the control against a determined
  writer is the #45 AGENT_ISOLATION OS/container verification.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

MANIFEST_FILE = "manifest.jsonl"


def _entry_digest(entry: dict) -> str:
    payload = {k: v for k, v in entry.items() if k != "entry_sha256"}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


class RunManifestStore:
    """JSONL, one entry per candidate round; harness-owned, agent read-only."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> list[dict]:
        if not self.path.exists():
            return []
        entries = [
            json.loads(line)
            for line in self.path.read_text().splitlines() if line
        ]
        self._verify_chain(entries)
        return entries

    @staticmethod
    def _verify_chain(entries: list[dict]) -> None:
        """Chained entries must hash-verify and link; only LEADING legacy
        entries (written before the chain existed) pass unverified."""
        previous_digest: str | None = None
        seen_chained = False
        for index, entry in enumerate(entries):
            if "entry_sha256" not in entry:
                if seen_chained:
                    raise RuntimeError(
                        f"run manifest integrity check failed at line "
                        f"{index}: unchained entry after chained entries "
                        "(possible forged or inserted line)"
                    )
                continue  # leading legacy entry, no chain to verify
            if entry.get("prev_entry_sha256") != (previous_digest or ""):
                raise RuntimeError(
                    f"run manifest integrity check failed at line {index}: "
                    "entries were edited, deleted, or reordered after being "
                    "recorded (prev_entry_sha256 mismatch)"
                )
            if _entry_digest(entry) != entry["entry_sha256"]:
                raise RuntimeError(
                    f"run manifest integrity check failed at line {index}: "
                    "entry content does not match its recorded hash"
                )
            previous_digest = entry["entry_sha256"]
            seen_chained = True

    def append(self, entry: dict) -> None:
        if "entry_sha256" in entry:
            # a pre-hashed entry would be written verbatim, its
            # prev_entry_sha256 pointing at the wrong tail — the store would
            # fail verification on the very next load (review round-6)
            raise ValueError(
                "append() chains entries itself; refusing an entry that "
                "already carries entry_sha256"
            )
        tail = self._heal_torn_tail()
        previous = tail[-1] if tail else None
        entry["prev_entry_sha256"] = (
            previous["entry_sha256"]
            if previous and "entry_sha256" in previous else ""
        )
        entry["entry_sha256"] = _entry_digest(entry)
        with self.path.open("a") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _heal_torn_tail(self) -> list[dict]:
        """Load the store, first repairing an interrupted append: a completed
        append always ends with a newline, so a file that does not has a torn
        trailing line from a killed process. If that line is valid JSON it is
        only missing its newline (newline-terminate it); if it is truncated
        mid-object it never became a recorded entry and is dropped. Without
        this, load() would raise JSONDecodeError on the torn line and wedge
        append permanently (review round-6/round-7)."""
        if not (self.path.exists() and self.path.stat().st_size > 0):
            return self.load()
        if self.path.read_bytes()[-1:] == b"\n":
            return self.load()
        text = self.path.read_text()
        lines = text.splitlines()
        try:
            json.loads(lines[-1])
        except json.JSONDecodeError:
            self.path.write_text("".join(line + "\n" for line in lines[:-1]))
        else:
            self.path.write_text(text + "\n")
        return self.load()

    def champion(self) -> dict | None:
        """The last KEEP entry (its config, best metric and provenance).

        `artifact_dir` is deliberately NOT required. Under k-fold selection
        a round trains `folds` models and discards them, so no round owns a
        deliverable; the artifact comes from `AutoResearchHarness.finalize()`
        (one fit on the selection pool). Requiring `artifact_dir` here would
        make every fold-scored round invisible as a champion and silently
        turn the hill climb into "always keep the first candidate".
        """
        for entry in reversed(self.load()):
            # .get: leading legacy entries pass unverified and need not
            # carry any particular keys (review round-7 R-04)
            if entry.get("decision") == "KEEP":
                return entry
        return None

    def completed_rounds(self) -> int:
        return len(self.load())
