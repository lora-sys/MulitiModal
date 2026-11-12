"""Single local/CI verification entry for the physiology main line (#95).

Default scope covers physiology tests and lint. ``--full`` runs the whole
repository test suite and lint scope.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _run(command: list[str], cwd: Path) -> dict[str, object]:
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    return {
        "command": command,
        "returncode": completed.returncode,
        "passed": completed.returncode == 0,
        "stdout_tail": completed.stdout.strip().splitlines()[-12:],
        "stderr_tail": completed.stderr.strip().splitlines()[-12:],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="run the whole repo")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from physio.acceptance import run_acceptance

    acceptance = run_acceptance(repo_root)
    python = sys.executable
    checks: list[dict[str, object]] = []

    if not args.skip_tests:
        test_target = ["-q"] if args.full else ["-q", "tests/physio"]
        checks.append(_run([python, "-m", "pytest", *test_target], repo_root))

    lint_target = (
        ["src", "tests", "scripts"]
        if args.full
        else ["src/physio", "tests/physio", "scripts/physio_realtime_demo.py",
              "scripts/physio_benchmark.py", "scripts/physio_verify.py"]
    )
    checks.append(_run([python, "-m", "ruff", "check", *lint_target], repo_root))

    report = {
        "schema": "physio-verify-v1",
        "evidence_level": "LOCAL_EXPERIMENTAL",
        "isolation": "AGENT_ISOLATION_UNVERIFIED",
        "scope": "full" if args.full else "physio",
        "acceptance": {
            "status": acceptance["status"],
            "passed": acceptance["passed"],
            "total": acceptance["total"],
        },
        "checks": checks,
    }
    report["passed"] = acceptance["status"] == "PASS" and all(
        bool(check["passed"]) for check in checks
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"acceptance: {acceptance['status']} "
              f"({acceptance['passed']}/{acceptance['total']})")
        for check in checks:
            print(f"{'PASS' if check['passed'] else 'FAIL'} {' '.join(check['command'])}")
        print("OVERALL:", "PASS" if report["passed"] else "FAIL")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
