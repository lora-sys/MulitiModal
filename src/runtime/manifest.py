"""Append-only RunManifest store for the #45 harness (#44 §6)."""

from __future__ import annotations

import json
from pathlib import Path

MANIFEST_FILE = "manifest.jsonl"


class RunManifestStore:
    """JSONL, one entry per candidate round; harness-owned, agent read-only."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line]

    def append(self, entry: dict) -> None:
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
