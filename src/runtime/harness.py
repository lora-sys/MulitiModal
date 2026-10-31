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
PRIMARY_METRIC = "primary_value"  # TCMTrustedEvaluator report: macro_f1
TCM_ONLY_MESSAGE = (
    "physio profile is interface-only in #45 milestone 1 (TCM_ONLY); "
    "physio data/labels are NOT_AVAILABLE until #37/#42"
)


def default_real_tcm_runspec(
    manifest_path: str,
    dataset_path: str,
    rounds: int = 5,
    timeout_s: float = 900.0,
) -> dict:
    """Frozen real-data TCM runspec (#50): the manifest carries the audited
    dataset hash, subject-disjoint split, train-fold scaler and RULE-derived
    label contract; AutoResearch may only optimize the editable scope."""
    from data.real_split import load_real_manifest

    manifest = load_real_manifest(manifest_path)
    return {
        "task_id": "tcm_constitution",
        "profile": "TCM_V1_REAL",
        "task": "tcm",
        "data_mode": "real",
        "frozen": {
            "dataset_hash": manifest["dataset_sha256"],
            "data_manifest_hash": manifest["manifest_sha256"],
            "label_manifest_hash": manifest["label_version"],
            "standard_version": manifest["standard_version"],
            "split_hash": f"manifest:{manifest['manifest_sha256'][:16]}",
            "evaluator": "runtime.tcm_eval.TCMTrustedEvaluator",
            "metric_version": "tcm-eval-v0",
            "primary_metric": "category_probs/macro_f1",
            "primary_direction": "max",
        },
        "data": {
            "manifest_path": str(manifest_path),
            "dataset_path": str(dataset_path),
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
            "trainer.epochs",
            "trainer.learning_rate",
            "trainer.grad_clip",
            "trainer.batch_size",
        ],
        "base_config": {
            "task": "tcm",
            "data_mode": "real",
            "data": {
                "manifest_path": str(manifest_path),
                "dataset_path": str(dataset_path),
                "task_type": manifest["task_type"],
            },
            "model": {"hidden_dim": 32, "fusion_dim": 32},
            "trainer": {"epochs": 40, "learning_rate": 3e-3, "seed": 0},
            "train_provider": {"batch_size": 32},
        },
    }


#: deterministic candidate plan for real-data rounds (#50: agent modifies
#: model/training within the editable scope; provider status stays
#: AGENT_NOT_VERIFIED until isolation is verified per #45)
REAL_TCM_PLAN: list[tuple[str, dict]] = [
    ("baseline: 32-wide trunk, 40 epochs, lr 3e-3", {}),
    ("wider trunk (64/64)", {"model.hidden_dim": 64, "model.fusion_dim": 64}),
    ("longer schedule (80 epochs)", {"trainer.epochs": 80}),
    ("higher LR with grad clip",
     {"trainer.learning_rate": 1e-2, "trainer.grad_clip": 1.0}),
    ("smaller batches (16), 64 epochs",
     {"trainer.batch_size": 16, "trainer.epochs": 64}),
    ("wide + high LR + long (combined best guess)",
     {"model.hidden_dim": 64, "model.fusion_dim": 64,
      "trainer.learning_rate": 1e-2, "trainer.grad_clip": 1.0,
      "trainer.epochs": 80}),
]


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
            "primary_metric": "category_probs/macro_f1",
            "primary_direction": "max",
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
    # sanitized whitelist: workers never inherit host credentials/proxy/auth
    # material from the harness environment (#45 audit finding); PYTHONPATH
    # is dropped, not preserved — an inherited host path could shadow
    # third-party imports inside the "isolated" worker (review R-04)
    from .worker import ALLOWED_ENV, sanitize_env

    env = sanitize_env(dict(os.environ))
    env["PYTHONPATH"] = str(SRC_DIR)
    assert "PATH" in ALLOWED_ENV
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
    lines = proc.stdout.strip().splitlines()
    if not lines:
        raise RuntimeError(f"worker {' '.join(command)} produced no output")
    return json.loads(lines[-1])


def _artifact_sha256(artifact_dir: str | Path) -> str:
    weights = Path(artifact_dir) / "weights.pt"
    return hashlib.sha256(weights.read_bytes()).hexdigest()


def context_digest(context: dict) -> str:
    """Canonical digest of a research state: lets the manifest prove which
    state an agent's proposal was based on (#45 agent provenance)."""
    canonical = json.dumps(context, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _primary_value(score: dict) -> float | None:
    value = score.get("primary_value")
    if value is None or isinstance(value, str) \
            or not isinstance(value, (int, float)):
        # a malformed report type routes to score_failed instead of raising
        # TypeError past every guard (review round-7 R-01)
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
        self.real_data = not self.runspec["frozen"]["dataset_hash"].startswith("synthetic")
        if self.real_data:
            # the frozen manifest must exist and hash-verify before any run
            from data.real_split import load_real_manifest

            load_real_manifest(self.runspec["data"]["manifest_path"])
        self._external_provider = not isinstance(
            self.provider, DeterministicStubProvider
        )
        self.smoke_only = not self.real_data

    @property
    def mode(self) -> str:
        if self.real_data:
            return "REAL_DATA_TRAINING"
        return "AGENT_INTEGRATED" if self._external_provider else "ENGINEERING_SMOKE"

    @property
    def agent_status(self) -> str:
        return (
            "AGENT_ISOLATION_UNVERIFIED" if self._external_provider
            else "AGENT_NOT_VERIFIED"
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

    def _champion_config(self) -> dict:
        """Effective config the next candidate starts from: the current
        champion's, so KEEPs accumulate (hill climbing). Starting every
        candidate from base_config instead would make rounds independent
        probes, not research (2026-10-31 audit finding: the v4 champion was
        32-wide while the report described it as 64-wide)."""
        for entry in reversed(self.store.load()):
            cfg = entry.get("effective_config")
            if entry.get("decision") == "KEEP" and cfg:
                return {
                    "model": dict(cfg["model"]),
                    "trainer": dict(cfg["trainer"]),
                    "train_provider": dict(cfg.get("train_provider", {})),
                }
        base = self.runspec["base_config"]
        return {
            "model": dict(base.get("model", {})),
            "trainer": {
                k: v for k, v in base.get("trainer", {}).items() if k != "seed"
            },
            "train_provider": dict(base.get("train_provider", {})),
        }

    def _run_candidate(
        self, index: int, proposal: Proposal, timeout_s: float,
        provenance: dict | None = None,
    ) -> dict:
        run_id = f"tcm-{index:04d}"
        started = time.monotonic()
        entry: dict = {
            "run_id": run_id,
            "round_index": index,
            "intent": proposal.intent,
            "patch": proposal.config_patch,
            "mode": self.mode,
            "agent_status": self.agent_status,
            "smoke_only": self.smoke_only,
            "source_commit": _git_sha(),
            "frozen_hashes": self._frozen_hashes(),
        }
        if provenance:
            entry.update(provenance)

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
            # inside the same static-check guard: a scope key outside the
            # known sections raises here, not as an uncaught loop crash
            start = {**self.runspec["base_config"], **self._champion_config()}
            config = apply_config_patch(start, proposal.config_patch)
        except ValueError as exc:
            return finish("REVERT", "rejected", f"static check: {exc}")

        entry["effective_config"] = {
            "model": dict(config["model"]),
            "trainer": {
                k: v for k, v in config["trainer"].items() if k != "seed"
            },
            "train_provider": {"batch_size": config["train_provider"].get("batch_size")},
        }

        candidate_dir = self.out_dir / run_id
        config.update({"task": "tcm", "out_dir": str(candidate_dir)})
        # Controlled comparison: every candidate uses the frozen runspec seed
        # and the same train/eval pools; only the editable patch differs.
        config["trainer"]["seed"] = self.runspec["seed"]
        if self.real_data:
            config["data_mode"] = "real"
            config["data"] = {
                **self.runspec["data"],
                "task_type": self.runspec["base_config"]["data"]["task_type"],
            }
        else:
            config["train_provider"]["seed"] = self.runspec["seed"]
            # eval subjects must stay disjoint from the train pool (#27)
            config["eval_provider"]["subject_prefix"] = f"eval-{run_id}"

        try:
            train_report = _worker(
                ["train", "--config", self._dump_config(config, run_id)], timeout_s
            )
        except subprocess.TimeoutExpired:
            return finish("REVERT", "timeout", f"exceeded {timeout_s}s budget")
        except (RuntimeError, OSError, json.JSONDecodeError) as exc:
            return finish("REVERT", "crash", f"training failed: {exc}")

        score_cmd = ["score", "--artifact", str(candidate_dir)]
        if self.real_data:
            score_cmd += [
                "--manifest", str(self.runspec["data"]["manifest_path"]),
                "--dataset", str(self.runspec["data"]["dataset_path"]),
            ]
        try:
            score = _worker(score_cmd, timeout_s)
        except subprocess.TimeoutExpired:
            return finish("REVERT", "timeout", f"scoring exceeded {timeout_s}s budget")
        except (RuntimeError, OSError, json.JSONDecodeError) as exc:
            return finish("REVERT", "score_failed", f"trusted scoring failed: {exc}")

        value = _primary_value(score)
        if value is None:
            return finish("REVERT", "score_failed", "primary metric missing or NaN")

        try:
            artifact_sha = _artifact_sha256(candidate_dir)
        except OSError as exc:
            return finish(
                "REVERT", "score_failed", f"artifact hashing failed: {exc}"
            )

        champion = self.store.champion()
        previous_best = champion.get("best_so_far") if champion else None
        # the frozen runspec declares whether the primary metric is minimised
        # (a loss) or maximised (a score); KEEP requires a STRICT improvement
        direction = self.runspec["frozen"].get("primary_direction", "min")
        if previous_best is None:
            keep = True
        elif direction == "max":
            keep = value > previous_best
        else:
            keep = value < previous_best
        entry.update(
            decision="KEEP" if keep else "REVERT",
            status="completed",
            reason=(
                f"{value:.6f} vs best {previous_best:.6f}"
                if previous_best is not None
                else f"baseline {value:.6f}"
            ),
            artifact_dir=str(candidate_dir),
            artifact_sha256=artifact_sha,
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

    # -- agent-facing research state ---------------------------------------

    def research_context(self) -> dict:
        """Full research state for an agent proposal: best metric, editable
        scope, base config and per-round history WITH metrics/patches (a
        real agent must see what was tried and what it scored, #45)."""
        entries = self.store.load()
        self._check_continuity(entries)
        champion = self.store.champion()
        return {
            "round_index": len(entries),
            "rounds_remaining": max(
                0, self.runspec["budget"]["max_candidates"] - len(entries)
            ),
            "best_metric": champion.get("best_so_far") if champion else None,
            "editable_scope": self.runspec["editable_scope"],
            "current_config": self._champion_config(),
            "base_config": self.runspec["base_config"],
            "primary_metric": self.runspec["frozen"]["primary_metric"],
            "history": [
                {
                    "run_id": e["run_id"],
                    "intent": e.get("intent"),
                    "patch": e.get("patch"),
                    "decision": e["decision"],
                    "status": e.get("status"),
                    "reason": e.get("reason"),
                    "primary_value": (e.get("metrics") or {}).get("primary_value"),
                }
                for e in entries
            ],
        }

    def step(self, proposal: Proposal, provenance: dict | None = None) -> dict:
        """Run exactly ONE candidate round with an externally supplied
        proposal (research-step driver). Budget and continuity checks apply.

        `provenance` records WHO proposed the round and WHAT it was shown
        (proposer id, context digest, proposal digest) so a third party can
        verify agent-driven research from the manifest alone (#45)."""
        entries = self.store.load()
        self._check_continuity(entries)
        index = len(entries)
        if index >= self.runspec["budget"]["max_candidates"]:
            raise RuntimeError(
                f"candidate budget exhausted ({index} rounds already recorded)"
            )
        if provenance:
            expected = context_digest(self.research_context())
            seen = provenance.get("context_sha256")
            if seen and seen != expected:
                raise RuntimeError(
                    "the agent's research context is stale: it was printed "
                    "before the manifest changed, so its proposal was not "
                    "based on the current research state"
                )
            if not provenance.get("proposer"):
                raise RuntimeError("provenance requires a non-empty proposer id")
        # proposals arriving through step() come from an external agent
        self._external_provider = True
        return self._run_candidate(
            index, proposal, self.runspec["budget"]["per_run_timeout_s"],
            provenance=provenance,
        )

    # -- full loop ----------------------------------------------------------

    def run(self) -> dict:
        max_candidates = self.runspec["budget"]["max_candidates"]
        timeout_s = self.runspec["budget"]["per_run_timeout_s"]
        deadline = time.monotonic() + self.runspec["budget"]["max_wall_clock_s"]

        decisions: list[dict] = []
        plan_exhausted = False
        for _ in range(max_candidates):
            if time.monotonic() > deadline:
                break
            context = self.research_context()
            if context["round_index"] >= max_candidates:
                break  # budget already consumed by a previous run (resume)
            round_started = time.monotonic()
            try:
                proposal = self.provider.propose(
                    context["round_index"], context
                )
            except Exception as exc:  # noqa: BLE001 — agent failures are rounds
                # an agent failure is a recorded round, not a crashed loop;
                # it consumes one budget round so the manifest stays honest
                index = context["round_index"]
                failed = {
                    "run_id": f"tcm-{index:04d}", "round_index": index,
                    "intent": None, "patch": None,
                    "mode": self.mode, "agent_status": self.agent_status,
                    "smoke_only": self.smoke_only,
                    "source_commit": _git_sha(),
                    "frozen_hashes": self._frozen_hashes(),
                    "decision": "REVERT", "status": "agent_error",
                    "reason": f"agent proposal failed: "
                              f"{type(exc).__name__}: {exc}",
                    "elapsed_s": round(time.monotonic() - round_started, 2),
                    "best_so_far": None,
                }
                self.store.append(failed)
                decisions.append(failed)
                continue
            if proposal is None:
                plan_exhausted = True
                break
            decisions.append(self._run_candidate(
                context["round_index"], proposal, timeout_s,
                provenance={
                    # every round names its proposer, even the offline stub:
                    # otherwise a batch run is indistinguishable from agent
                    # research in the manifest (audit A-01/A-03)
                    "proposer": getattr(
                        self.provider, "proposer_id",
                        f"provider:{type(self.provider).__name__}",
                    ),
                    "context_sha256": context_digest(context),
                },
            ))

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
            "smoke_only": self.smoke_only,
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
