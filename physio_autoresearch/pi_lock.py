"""Hold an OS Pi-session lock until the parent's stdin closes (no stale-lock guessing)."""

import json
from pathlib import Path
import sys

from .harness import load_round, resolve_round
from .storage import exclusive


def main():
    try:
        round_path = resolve_round(Path(sys.argv[1]))
        with exclusive(round_path / "pi-session.lock"):
            load_round(round_path)
            print("LOCKED", flush=True)
            sys.stdin.read()
    except Exception as exc:
        print(json.dumps({"error": getattr(exc, "code", "LOCK_ERROR")}), flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
