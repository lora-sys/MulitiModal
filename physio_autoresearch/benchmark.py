"""Run the fixed synthetic harness regression suite and save per-case evidence."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET

from .contracts import ResearchError, write_json
from .harness import ROOT, environment, now


def benchmark(output: Path) -> dict:
    output = output.resolve()
    if not output.is_relative_to(ROOT.resolve() / "runs"):
        raise ResearchError("UNSAFE_OUTPUT_PATH", "benchmark outputs must stay below ignored runs/")
    output.parent.mkdir(parents=True, exist_ok=True)
    xml = output.with_suffix(".xml")
    temporary = ROOT / "runs" / f"harness-benchmark-{uuid.uuid4().hex}"
    command = [sys.executable, "-m", "pytest", "tests/physio_autoresearch", "-m", "harness_bench", "-q",
               "--basetemp", str(temporary), f"--junitxml={xml}"]
    with output.with_suffix(".log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=300, check=False)
    cases = []
    if xml.exists():
        for case in ET.parse(xml).getroot().iter("testcase"):
            passed = not any(case.find(kind) is not None for kind in ("failure", "error", "skipped"))
            cases.append({"case": case.attrib["name"], "passed": passed, "seconds": float(case.attrib.get("time", 0))})
    result = {"created_at": now(), "source": "synthetic_failure_and_integrity_fixtures", "environment": environment(),
              "passed": sum(c["passed"] for c in cases), "total": len(cases), "cases": cases,
              "success": completed.returncode == 0 and bool(cases),
              "scope": "regression benchmark, not evidence of recursive improvement or physiological validity"}
    write_json(output, result)
    if not result["success"]:
        raise ResearchError("HARNESS_BENCHMARK_FAILED", str(output))
    return result
