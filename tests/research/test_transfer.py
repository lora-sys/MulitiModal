"""Run a second, task-independent synthetic research loop through the CLI."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from research.dashboard import read_run

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/linear_research"


@pytest.mark.synthetic
def test_second_task_baseline_improvement_and_monitor(tmp_path):
    repo = tmp_path / "worktree"
    repo.mkdir()
    candidate = repo / "examples/linear_research/train.py"
    candidate.parent.mkdir(parents=True)
    source = (EXAMPLE / "train.py").read_text()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "config", "user.name", "Synthetic"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "synthetic@example.invalid"],
                   cwd=repo, check=True)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    run = tmp_path / "run"

    def execute(index, code, description):
        candidate.write_text(code)
        subprocess.run(["git", "add", str(candidate.relative_to(repo))], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", description], cwd=repo, check=True)
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo,
                                         text=True).strip()
        result = run / f"round-{index}.json"
        subprocess.run([sys.executable, str(EXAMPLE / "prepare.py"), "--candidate", str(candidate),
                        "--out", str(result), "--budget", "20"], env=env, check=True,
                       capture_output=True, text=True, timeout=90)
        subprocess.run([sys.executable, str(EXAMPLE / "prepare.py"), "record", "--repo", str(repo),
                        "--commit", commit, "--results", str(run / "results.tsv"),
                        "--result", str(result), "--decision", "KEEP", "--description",
                        description],
                       env=env, check=True, capture_output=True, text=True, timeout=20)
        return json.loads(result.read_text())

    baseline = execute(0, source, "mean baseline")
    trial = execute(1, source.replace('MODEL = "mean"', 'MODEL = "linear"'), "linear fit")
    assert trial["value"] < baseline["value"]
    assert trial["value"] == pytest.approx(0)
    view = read_run(run)
    assert view["evidence"] == "SMOKE_ONLY"
    assert view["task"] == "synthetic-linear"
    assert [row["metric"] for row in view["results"]] == ["synthetic/mean_absolute_error"] * 2
    assert all(row["direction"] == "minimize" for row in view["results"])
