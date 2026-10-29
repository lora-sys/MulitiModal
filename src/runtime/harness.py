"""Bounded AutoResearch harness, TCM_ONLY milestone 1 (#45).

Implements the #44 design: propose -> static checks -> isolated training
(subprocess) -> trusted scoring (subprocess) -> KEEP/REVERT -> append-only
RunManifest, with resume-without-duplicates, per-run timeout, budget limits
and frozen-hash continuity checks.

Every run in this milestone is ENGINEERING_SMOKE with synthetic tensors:
SMOKE_ONLY output, agent AGENT_NOT_VERIFIED, isolation
AGENT_ISOLATION_UNVERIFIED. Physio profile is interface-only and refused.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from .agent import (
    AgentProvider,
    DeterministicStubProvider,
    Proposal,
    apply_config_patch,
    validate_config_patch,
    validate_source_patch,
)
from .manifest import MANIFEST_FILE, RunManifestStore

SRC_DIR = Path(__file__).resolve().parents[1]
PRIMARY_METRIC = "primary_value"  # TCMTrustedEvaluator report: subject_mae
TCM_ONLY_MESSAGE = (
    "physio profile is interface-only in #45 milestone 1 (TCM_ONLY); "
    "physio data/labels are NOT_AVAILABLE until #37/#42"
)


def default_tcm_runspec(rounds: int = 3, timeout_s: float = 300.0) -> dict:
    """Frozen TCM synthetic runspec per #44 section 4."""
    return {
        "task_id": "tcm_constitution",
        "profile": "TCM_V1",
        "task": "tcm",
        "frozen": {
            "dataset_hash": "synthetic-tensor-v0",
            "label_manifest_hash": "synthetic-labels-v0",
            "standard_version": "SYNTHETIC-CONTRACT-V0",
            "split_hash": "manifest:trusted-eval-v0",
            "evaluator": "runtime.tcm_eval.TCMTrustedEvaluator",
            "metric_version": "tcm-eval-v0",
            "primary_metric": "category_probs/subject_mae",
        },
        "seed": 0,
        "budget": {
            "max_candidates": rounds,
            "per_run_timeout_s": timeout_s,
            "max_wall_clock_s": timeout_s * (rounds + 1),
        },
        "editable_scope": [
            "model.hidden_dim",
            "model.fusion_dim",
            "model.include_score_head",
            "trainer.epochs",
            "trainer.learning_rate",
            "trainer.grad_clip",
            "trainer.batch_size",
        ],
        "base_config": {
            "task": "tcm",
            "model": {"hidden_dim": 32, "fusion_dim": 32},
            "trainer": {"epochs": 2, "seed": 0},
            "train_provider": {"n_batches": 2, "batch_size": 8, "seed": 0},
            "eval_provider": {"n_batches": 1, "batch_size": 8, "seed": 1},
        },
    }


def _worker_env() -> dict:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(SRC_DIR) + (os.pathsep + existing if existing else "")
    return env


def _worker(command: list[str], timeout_s: float) -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "runtime.worker", *command],
        env=_worker_env(), capture_output=True, text=True, timeout=timeout_s,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"worker {' '.join(command)} failed (rc={proc.returncode}): "
            f"{proc.stderr.strip()[-500:]}"
        )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _artifact_sha256(artifact_dir: str | Path) -> str:
    weights = Path(artifact_dir) / "weights.pt"
    return hashlib.sha256(weights.read_bytes()).hexdigest()


def _primary_value(score: dict) -> float | None:
    value = score.get("primary_value")
    if value is None or isinstance(value, str):
        return None
    value = float(value)
    return None if math.isnan(value) else value


class AutoResearchHarness:
    """Runs bounded candidate rounds against the frozen TCM evaluation set."""

    def __init__(
        self,
        runspec: dict,
        provider: AgentProvider | None = None,
        out_dir: str | Path = "runs/autoresearch",
    ) -> None:
        if runspec.get("task_id") != "tcm_constitution":
            raise ValueError(TCM_ONLY_MESSAGE)
        if runspec["frozen"].get("dataset_hash") == "NOT_AVAILABLE":
            raise ValueError(TCM_ONLY_MESSAGE)
        self.runspec = runspec
        self.provider = provider or DeterministicStubProvider()
        self.out_dir = Path(out_dir)
        self.store = RunManifestStore(self.out_dir / MANIFEST_FILE)
        self.mode = (
            "ENGINEERING_SMOKE"
            if isinstance(self.provider, DeterministicStubProvider)
            else "AGENT_INTEGRATED"
        )
        self.agent_status = (
            "AGENT_NOT_VERIFIED"
            if isinstance(self.provider, DeterministicStubProvider)
            else "AGENT_ISOLATION_UNVERIFIED"
        )

    # -- frozen-continuity ------------------------------------------------

    def _frozen_hashes(self) -> dict:
        frozen = dict(self.runspec["frozen"])
        frozen["seed"] = self.runspec["seed"]
        return frozen

    def _check_continuity(self, entries: list[dict]) -> None:
        if entries and entries[0].get("frozen_hashes") != self._frozen_hashes():
            raise RuntimeError(
                "frozen evaluation set changed after the run started "
                "(dataset/labels/split/evaluator/seed); start a new experiment"
            )

    # -- one candidate ----------------------------------------------------

    def _run_candidate(self, index: int, proposal: Proposal, timeout_s: float) -> dict:
        run_id = f"tcm-{index:04d}"
        started = time.monotonic()
        entry: dict = {
            "run_id": run_id,
            "round_index": index,
            "intent": proposal.intent,
            "patch": proposal.config_patch,
            "mode": self.mode,
            "smoke_only": True,
            "source_commit": _git_sha(),
            "frozen_hashes": self._frozen_hashes(),
        }

        def finish(decision: str, status: str, reason: str, **extra) -> dict:
            entry.update(
                decision=decision, status=status, reason=reason,
                elapsed_s=round(time.monotonic() - started, 2),
                best_so_far=None, **extra,
            )
            self.store.append(entry)
            return entry

        try:
            validate_config_patch(proposal.config_patch, self.runspec["editable_scope"])
            if proposal.source_patch is not None:
                validate_source_patch(proposal.source_patch)
        except ValueError as exc:
            return finish("REVERT", "rejected", f"static check: {exc}")

        candidate_dir = self.out_dir / run_id
        config = apply_config_patch(self.runspec["base_config"], proposal.config_patch)
        config.update({"task": "tcm", "out_dir": str(candidate_dir)})
        # Controlled comparison: every candidate uses the frozen runspec seed
        # and the same train/eval pools; only the editable patch differs.
        config["trainer"]["seed"] = self.runspec["seed"]
        config["train_provider"]["seed"] = self.runspec["seed"]
        # eval subjects must stay disjoint from the train pool (#27)
        config["eval_provider"]["subject_prefix"] = f"eval-{run_id}"

        try:
            train_report = _worker(
                ["train", "--config", self._dump_config(config, run_id)], timeout_s
            )
        except subprocess.TimeoutExpired:
            return finish("REVERT", "timeout", f"exceeded {timeout_s}s budget")
        except (RuntimeError, OSError) as exc:
            return finish("REVERT", "crash", f"training failed: {exc}")

        try:
            score = _worker(
                ["score", "--artifact", str(candidate_dir)], timeout_s
            )
        except subprocess.TimeoutExpired:
            return finish("REVERT", "timeout", f"scoring exceeded {timeout_s}s budget")
        except (RuntimeError, OSError, json.JSONDecodeError) as exc:
            return finish("REVERT", "score_failed", f"trusted scoring failed: {exc}")

        value = _primary_value(score)
        if value is None:
            return finish("REVERT", "score_failed", "primary metric missing or NaN")

        champion = self.store.champion()
        previous_best = champion.get("best_so_far") if champion else None
        keep = previous_best is None or value < previous_best
        entry.update(
            decision="KEEP" if keep else "REVERT",
            status="completed",
            reason=(
                f"{value:.6f} vs best {previous_best:.6f}"
                if previous_best is not None
                else f"baseline {value:.6f}"
            ),
            artifact_dir=str(candidate_dir),
            artifact_sha256=_artifact_sha256(candidate_dir),
            model_type=train_report.get("model_type"),
            metrics={"primary_metric": self.runspec["frozen"]["primary_metric"],
                     "primary_value": value, "report": score},
            best_so_far=value if keep else previous_best,
        )
        self.store.append(entry)
        return entry

    def _dump_config(self, config: dict, run_id: str) -> str:
        path = self.out_dir / f"{run_id}-config.json"
        path.write_text(json.dumps(config, indent=2))
        return str(path)

    # -- full loop ----------------------------------------------------------

    def run(self) -> dict:
        entries = self.store.load()
        self._check_continuity(entries)
        done = len(entries)
        max_candidates = self.runspec["budget"]["max_candidates"]
        timeout_s = self.runspec["budget"]["per_run_timeout_s"]
        deadline = time.monotonic() + self.runspec["budget"]["max_wall_clock_s"]

        decisions: list[dict] = []
        plan_exhausted = False
        for index in range(done, max_candidates):
            if time.monotonic() > deadline:
                break
            champion = self.store.champion()
            context = {
                "round_index": index,
                "best_metric": champion.get("best_so_far") if champion else None,
                "history": [
                    {"run_id": e["run_id"], "decision": e["decision"]}
                    for e in entries
                ],
            }
            proposal = self.provider.propose(index, context)
            if proposal is None:
                plan_exhausted = True
                break
            decisions.append(self._run_candidate(index, proposal, timeout_s))

        final = self.store.champion()
        return {
            "profile": self.runspec["profile"],
            "rounds_requested": max_candidates,
            "rounds_completed": len(self.store.load()),
            "decisions": [
                {"run_id": d["run_id"], "decision": d["decision"],
                 "status": d["status"], "reason": d["reason"]}
                for d in decisions
            ],
            "champion_artifact": final.get("artifact_dir") if final else None,
            "best_metric": final.get("best_so_far") if final else None,
            "plan_exhausted": plan_exhausted,
            "mode": self.mode,
            "agent_status": self.agent_status,
            "isolation": "AGENT_ISOLATION_UNVERIFIED",
            "smoke_only": True,
        }


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
            cwd=str(SRC_DIR),
        ).stdout.strip()
    except Exception:
        return "unknown"
