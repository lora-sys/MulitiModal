"""Bounded source checks for the host-side research loop."""
from __future__ import annotations

import json
import sys

from research.train_worker import MAX_CANDIDATE_SOURCE_BYTES, validate_candidate_source


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args not in (["validate"], ["complexity"]):
        print(json.dumps({"ok": False, "error": "invalid source preflight mode"}))
        return 2
    source_bytes = sys.stdin.buffer.read(MAX_CANDIDATE_SOURCE_BYTES + 1)
    if len(source_bytes) > MAX_CANDIDATE_SOURCE_BYTES:
        print(json.dumps({
            "ok": False,
            "error": "candidate source exceeds the 1048576-byte limit",
        }))
        return 2
    try:
        source = source_bytes.decode("utf-8")
        if args[0] == "validate":
            validate_candidate_source(source)
        from research.guard import editable_complexity_source
        complexity = editable_complexity_source(source)
    except Exception as exc:
        print(json.dumps({
            "ok": False,
            "error": f"{type(exc).__name__}: {str(exc)[:300]}",
        }))
        return 1
    print(json.dumps({"ok": True, "complexity": complexity}))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
