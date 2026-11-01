"""Shared launch helpers for every trusted subprocess in the research loop.

This module exists so the runner, the finalize driver and the loop cannot
drift apart about HOW a trusted child is started. Two hard rules (H-02):

1. **Isolated startup.** A trusted child never starts as `python -m <mod>`:
   module resolution builds sys.path from the environment, so a planted
   `sitecustomize.py` on a PYTHONPATH directory would execute BEFORE any SHA
   check and could forge the child's result line. Children start as
   `python -I -c <bootstrap> <src> <args...>` — the bootstrap string is
   written here (trusted code), and the src path comes from the parent's
   config, never from the environment.
2. **Whitelist env.** The child environment carries no inherited variable at
   all: PATH is a minimal fixed set (the finalize worker needs `git`), HOME
   is kept for torch cache dirs, PYTHONHASHSEED=0 pins dict/set iteration
   for reproducibility. PYTHONPATH is absent — under `-I` it would be
   ignored anyway, and its absence is what makes a planted startup module
   unreachable even if a future launch path drops a flag.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: minimal PATH for trusted children: enough to find git and nothing user-
#: specific (no ~/.local/bin, no venv dirs — python itself is launched by
#: absolute path from sys.executable)
CHILD_PATH = "/usr/bin:/bin:/usr/local/bin"


def child_env(worktree: str) -> dict:
    """Whitelist environment for every child: no inherited secrets, no other
    checkout's modules, no attacker-influenced startup hooks."""
    return {
        "PATH": CHILD_PATH,
        "HOME": "",
        "PYTHONHASHSEED": "0",
    }


def bootstrap_command(worktree: str, module: str, *args: str) -> list[str]:
    """`python -I -c <bootstrap> <src> <args...>`: the child imports its
    entrypoint from THIS worktree's src, injected by the bootstrap string —
    isolated mode disables environment injection, user site-packages and the
    cwd on sys.path. The remaining argv reaches `main()` intact."""
    bootstrap = (
        "import sys; sys.path.insert(0, sys.argv.pop(1)); "
        f"from {module} import main; sys.exit(main())"
    )
    return [
        sys.executable, "-I", "-c", bootstrap,
        str(Path(worktree) / "src"), *args,
    ]
