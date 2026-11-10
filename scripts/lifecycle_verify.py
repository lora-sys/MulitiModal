"""Single local/CI verification entry for the ML lifecycle (#112).

Default scope: fast contracts, synthetic lifecycle suite, whole-repo lint and the
unified acceptance gate. ``--torch-free-smoke`` proves the stdlib contract layer
imports without torch; ``--acceptance-only`` runs just the acceptance gate.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# No [build-system] in pyproject: scripts must be runnable without the caller
# setting PYTHONPATH. Inject src/ for this process and for child processes.
SRC = str(ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
os.environ["PYTHONPATH"] = os.pathsep.join(
    part for part in (SRC, str(ROOT), os.environ.get("PYTHONPATH", "")) if part
)


def _run(command: list[str]) -> dict[str, object]:
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": "\n".join(completed.stdout.strip().splitlines()[-6:]),
        "stderr_tail": "\n".join(completed.stderr.strip().splitlines()[-6:]),
    }


def _torch_free_smoke() -> dict[str, object]:
    """Prove pure contracts import with no torch on sys.modules/meta_path."""
    spec = importlib.util.find_spec("torch")
    code = """import sys
class Block:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise ModuleNotFoundError('No module named torch')
sys.meta_path.insert(0, Block())
import lifecycle.contracts as c
import lifecycle.pipeline as p
from core.specs import make_physio_input_spec
print(c.CONTRACTS_VERSION, p.PIPELINE_VERSION,
      len(make_physio_input_spec(12, 7).modalities))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True
    )
    return {
        "torch_installed_on_this_host": spec is not None,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance-only", action="store_true")
    parser.add_argument("--torch-free-smoke", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.torch_free_smoke:
        result = _torch_free_smoke()
        ok = result["returncode"] == 0
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("TORCH_FREE_SMOKE:", "PASS" if ok else "FAIL")
        return 0 if ok else 1

    from lifecycle.acceptance import run_acceptance, write_acceptance

    report = run_acceptance()
    write_acceptance(report, ROOT / "runs" / "lifecycle")
    checks: list[dict[str, object]] = []
    if not args.acceptance_only:
        checks.append(_run([sys.executable, "-m", "pytest", "-m", "fast", "-q"]))
        checks.append(_run([sys.executable, "-m", "pytest", "tests/lifecycle", "-q"]))
        checks.append(_run([sys.executable, "-m", "ruff", "check", "src", "tests", "scripts"]))

    gates = report.gates
    passed = report.status == "PASS" and all(c["returncode"] == 0 for c in checks)
    payload = {
        "schema": "ml-lifecycle-verify-v1",
        "evidence_level": report.evidence_level,
        "isolation": report.isolation,
        "acceptance": f"{report.passed}/{report.total} (negatives: {report.negative_checks})",
        "gates": gates,
        "checks": checks,
        "passed": passed,
        "limitations": list(report.limitations),
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for check in checks:
            print(f"{'PASS' if check['returncode'] == 0 else 'FAIL'} {' '.join(check['command'])}")
        print(f"ACCEPTANCE: {report.status} {report.passed}/{report.total}")
        print("GATES:", ", ".join(f"{k}={v}" for k, v in gates.items()))
        print("EVIDENCE:", report.evidence_level, "/", report.isolation)
        print("OVERALL:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
