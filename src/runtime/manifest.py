"""Append-only RunManifest store for the #45 harness (#44 §6).

Entries carry a hash chain (`prev_entry_sha256` + `entry_sha256`) so naive
in-place edits, deletions, or reordering of recorded rounds are detected on
load. Honest scope of this guarantee (review round-5 R-01/R-02):

- Leading entries written before the chain existed pass through unverified;
  an UNCHAINED line after any chained entry is rejected (forgery/insertion).
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
            if entry["prev_entry_sha256"] != (previous_digest or ""):
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
        if "entry_sha256" not in entry:
            tail = self.load()
            previous = tail[-1] if tail else None
            entry["prev_entry_sha256"] = (
                previous["entry_sha256"]
                if previous and "entry_sha256" in previous else ""
            )
            entry["entry_sha256"] = _entry_digest(entry)
        with self.path.open("a") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def champion(self) -> dict | None:
        """Artifact dir + best metric of the last KEEP entry, or None."""
        for entry in reversed(self.load()):
            if entry["decision"] == "KEEP" and entry.get("artifact_dir"):
                return entry
        return None

    def completed_rounds(self) -> int:
        return len(self.load())
