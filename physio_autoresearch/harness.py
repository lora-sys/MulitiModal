"""Data-driven, bounded offline rounds and KEEP/REVERT without Git mutation."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

from .contracts import ResearchError, candidate_id, digest, file_hash, load_contract, read_json, validate_candidate, write_json
from .data import parse_data, summary
from .storage import append, exclusive, history

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STORE = ROOT / "runs" / "physio-autoresearch"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def core_hashes() -> dict:
    package = Path(__file__).resolve().parent
    return {str(p.relative_to(package)): file_hash(p) for p in sorted(package.rglob("*"))
            if p.is_file() and p.suffix in {".py", ".mjs"} and not p.name.endswith(".test.mjs")}


def archive_code(round_path: Path, manifest: dict):
    # A zip retains exact source without multiplying Windows path depth.
    version_path = round_path / "code-versions" / (digest(manifest["environment"])[:24] + ".zip")
    if version_path.exists():
        with zipfile.ZipFile(version_path) as archive:
            saved = json.loads(archive.read("manifest.json"))
            if saved["environment"] != manifest["environment"]:
                raise ResearchError("CODE_ARCHIVE_COLLISION", "source archive ID collision")
        return
    version_path.parent.mkdir(parents=True, exist_ok=True)
    package = Path(__file__).resolve().parent
    with zipfile.ZipFile(version_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, sort_keys=True))
        for filename in manifest["core_hashes"]:
            archive.write(package / filename, "physio_autoresearch/" + filename)


def environment() -> dict:
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "xgboost", "PyYAML", "joblib")}
    versions["python"] = sys.version.split()[0]
    for package in ("@earendil-works/pi-coding-agent", "@earendil-works/pi-ai", "pi-autoresearch"):
        manifest = ROOT / "node_modules" / package / "package.json"
        if manifest.exists():
            versions[package] = read_json(manifest)["version"]
    if (ROOT / "package-lock.json").exists():
        versions["npm_lock_sha256"] = file_hash(ROOT / "package-lock.json")
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    return {"versions": versions, "base_commit": revision.stdout.strip(), "source_sha256": digest(core_hashes())}


def safe_store(store: Path) -> Path:
    store = store.resolve()
    # Runtime outputs remain in ignored runs/. Private data must never enter tracked paths.
    if not store.is_relative_to(ROOT.resolve() / "runs"):
        raise ResearchError("UNSAFE_OUTPUT_PATH", "runtime state must be below this repository's ignored runs/")
    return store


def ingest(data: Path, contract_path: Path, store: Path = DEFAULT_STORE, review_code_upgrade: bool = False) -> Path:
    store = safe_store(store)
    contract = load_contract(contract_path)
    resolved_input = data.resolve()
    if contract["source"] == "company" and resolved_input.is_relative_to(ROOT.resolve()) and not resolved_input.is_relative_to(ROOT.resolve() / "runs"):
        raise ResearchError("PRIVATE_DATA_IN_TRACKED_AREA", "company inputs must be external or below ignored runs/")
    raw = data.read_bytes()
    frame = parse_data(raw, contract)
    import hashlib
    data_sha = hashlib.sha256(raw).hexdigest()
    version = digest({"data": data_sha, "contract": digest(contract)})[:24]
    round_path = store / "rounds" / version
    with exclusive(store / "ingest.lock"):
        if round_path.exists():
            if review_code_upgrade:
                with exclusive(round_path / "round.lock"):
                    old = read_json(round_path / "manifest.json")
                    if old["data_sha256"] != file_hash(round_path / "data.csv") or old["contract_sha256"] != digest(read_json(round_path / "contract.json")):
                        raise ResearchError("SNAPSHOT_CHANGED", "upgrades cannot repair altered raw data/contracts")
                    entries = history(round_path / "experiments.jsonl")
                    ensure_budget(contract, entries, model_usage(round_path)["elapsed_seconds"])
                    current = {**old, "environment": environment(), "core_hashes": core_hashes()}
                    if current["environment"] != old["environment"]:
                        archive_code(round_path, current)
                        append(round_path / "experiments.jsonl", {"kind": "code_upgrade", "at": now(),
                               "old_environment": old["environment"], "new_environment": current["environment"],
                               "remaining_attempts": contract["budget"]["max_experiments"] - len([e for e in entries if e["kind"] == "start"])})
                        write_json(round_path / "manifest.json", current)
            load_round(round_path)
            return round_path
        import tempfile
        round_path.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".new-", dir=round_path.parent))
        (staging / "data.csv").write_bytes(raw)
        write_json(staging / "contract.json", contract)
        manifest = {"version": version, "data_sha256": data_sha,
                   "contract_sha256": digest(contract), "source": contract["source"], "qc": summary(frame, contract),
                   "created_at": now(), "environment": environment(), "core_hashes": core_hashes()}
        write_json(staging / "manifest.json", manifest)
        archive_code(staging, manifest)
        staging.replace(round_path)
    return round_path


def resolve_round(round_path: Path) -> Path:
    path = round_path.resolve()
    if not path.is_relative_to(ROOT.resolve() / "runs"):
        raise ResearchError("FORBIDDEN_TEST_ACCESS", "only explicitly ingested development rounds can be accessed")
    return path


def load_round(round_path: Path):
    round_path = resolve_round(round_path)
    manifest = read_json(round_path / "manifest.json")
    contract = read_json(round_path / "contract.json")
    if digest(contract) != manifest["contract_sha256"]:
        raise ResearchError("CONTRACT_CHANGED", "new protocols require explicit human ingestion")
    if file_hash(round_path / "data.csv") != manifest["data_sha256"]:
        raise ResearchError("DATA_CHANGED", "raw snapshot changed")
    if core_hashes() != manifest["core_hashes"] or environment()["versions"] != manifest["environment"]["versions"]:
        raise ResearchError("EVALUATOR_CHANGED", "code/dependencies changed since this round was frozen")
    return manifest, contract, history(round_path / "experiments.jsonl")


def results(entries: list[dict]) -> list[dict]:
    return [entry for entry in entries if entry["kind"] == "result"]


def model_usage(round_path: Path) -> dict:
    path = round_path / "pi-usage.jsonl"
    entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    return {"requests": len([e for e in entries if e["kind"] == "request"]),
            "elapsed_seconds": sum(e.get("elapsed_seconds", 0) for e in entries if e["kind"] == "usage"),
            "tokens": sum(e.get("usage", {}).get("totalTokens", 0) for e in entries if e["kind"] == "usage"),
            "reported_cost": sum(e.get("usage", {}).get("cost", {}).get("total", 0) for e in entries if e["kind"] == "usage")}


def champion_from(entries: list[dict]):
    kept = [entry for entry in current_results(entries) if entry["decision"] == "KEEP"]
    return kept[-1] if kept else None


def current_results(entries: list[dict]) -> list[dict]:
    boundary = next((i for i, entry in reversed(list(enumerate(entries))) if entry["kind"] == "code_upgrade"), -1)
    return results(entries[boundary + 1:])


def round_status(round_path: Path) -> dict:
    manifest, contract, entries = load_round(round_path)
    started = [entry for entry in entries if entry["kind"] == "start"]
    finished = results(entries)
    pending = sorted({e["attempt"] for e in started} - {e["attempt"] for e in finished})
    remote = model_usage(round_path)
    stop_reason = None
    try:
        ensure_budget(contract, entries, remote["elapsed_seconds"])
    except ResearchError as exc:
        stop_reason = exc.code
    return {"round": str(round_path.resolve()), "data_version": manifest["version"], "source": manifest["source"],
            "qc": manifest["qc"], "attempts": len(started), "budget": contract["budget"],
            "elapsed_seconds": sum(e["elapsed_seconds"] for e in finished) + remote["elapsed_seconds"], "pending_attempts": pending,
            "model_usage": remote,
            "stop_reason": stop_reason,
            "champion": champion_from(entries), "recent_results": finished[-5:],
            "model_status": "SMOKE_MODEL_NOT_VALIDATED" if contract["status"] == "smoke" else "DEVELOPMENT_MODEL_NOT_FINAL_TEST"}


def ensure_budget(contract: dict, entries: list[dict], remote_seconds: float = 0):
    started = [entry for entry in entries if entry["kind"] == "start"]
    finished = results(entries)
    if {e["attempt"] for e in started} - {e["attempt"] for e in finished}:
        raise ResearchError("INTERRUPTED_ROUND", "unfinished reservation; inspect it before authorizing recovery")
    if len(started) >= contract["budget"]["max_experiments"]:
        raise ResearchError("OVER_BUDGET", "data-version experiment budget exhausted")
    elapsed = sum(e["elapsed_seconds"] for e in finished) + remote_seconds
    if elapsed >= contract["budget"]["max_seconds"]:
        raise ResearchError("OVER_BUDGET", "data-version runtime budget exhausted")
    current = current_results(entries)
    last_keep = next((i for i, e in reversed(list(enumerate(current))) if e["decision"] == "KEEP"), -1)
    if len(current) - last_keep - 1 >= contract["budget"]["patience"]:
        raise ResearchError("NO_PROGRESS_STOP", "consecutive attempts without improvement reached patience")


def run_worker(operation: str, candidate: Path, timeout: float):
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    command = [sys.executable, "-m", "physio_autoresearch.trusted.worker", operation, str(candidate)]
    try:
        with (candidate / f"{operation}.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       timeout=max(0.01, timeout), check=False)
    except subprocess.TimeoutExpired as exc:
        raise ResearchError("TRAIN_TIMEOUT", f"{operation} exceeded its remaining runtime") from exc
    if completed.returncode:
        error_path = candidate / f"{operation}-error.json"
        error = read_json(error_path) if error_path.exists() else {"code": "TRAIN_CRASH", "detail": f"{operation} exit {completed.returncode}"}
        raise ResearchError(error["code"], error["detail"])


def attempt(round_path: Path, proposal: dict) -> dict:
    round_path = resolve_round(round_path)
    with exclusive(round_path / "round.lock"):
        manifest, contract, entries = load_round(round_path)
        remote_seconds = model_usage(round_path)["elapsed_seconds"]
        ensure_budget(contract, entries, remote_seconds)
        number = len([e for e in entries if e["kind"] == "start"]) + 1
        start = time.monotonic()
        previous = champion_from(entries)
        append(round_path / "experiments.jsonl", {"kind": "start", "attempt": number, "started_at": now(), "proposal": proposal})
        candidate = round_path / "candidates" / f"{number:03d}"
        candidate.mkdir(parents=True)
        decision, reason, spec, evaluation = "REVERT", "", None, None
        candidate_hash = None
        try:
            spec = validate_candidate(proposal, contract)
            candidate_hash = candidate_id(spec)
            same = [entry for entry in current_results(entries) if entry.get("candidate_sha256") == candidate_hash
                    and entry["reason"] != "BASELINE_REQUIRED"]
            if any(entry["evaluation"] is not None for entry in same) or len(same) >= 2:
                raise ResearchError("DUPLICATE_EXPERIMENT", "identical semantic candidate already attempted")
            baseline = any(entry["candidate"] and entry["candidate"]["model"] == "naive" and entry["evaluation"] is not None
                           for entry in current_results(entries))
            if not baseline and spec["model"] != "naive":
                raise ResearchError("BASELINE_REQUIRED", "the first attempt must be the naive baseline")
            write_json(candidate / "request.json", {"contract": contract, "candidate": spec,
                       "data_path": str(round_path / "data.csv"), "data_sha256": manifest["data_sha256"]})
            already_spent = sum(e["elapsed_seconds"] for e in results(entries)) + remote_seconds
            allowance = min(contract["budget"]["candidate_timeout_seconds"], contract["budget"]["max_seconds"] - already_spent)
            run_worker("train", candidate, allowance - (time.monotonic() - start))
            run_worker("evaluate", candidate, allowance - (time.monotonic() - start))
            load_round(round_path)  # Recheck core, raw data and frozen contract before promotion.
            evaluation = read_json(candidate / "evaluation.json")
            metric = evaluation["metric"]
            if metric is None:
                reason = "FIT_ONLY_NOT_VALIDATED"
            elif previous is None or metric < previous["evaluation"]["metric"] - contract["min_improvement"]:
                decision, reason = "KEEP", "first valid baseline" if previous is None else "lower development metric"
            else:
                reason = "NO_IMPROVEMENT"
        except Exception as exc:
            reason = getattr(exc, "code", "TRAIN_CRASH")
            write_json(candidate / "failure.json", {"code": reason, "detail": str(exc)})
        record = {"kind": "result", "attempt": number, "finished_at": now(), "decision": decision, "reason": reason,
                  "candidate_sha256": candidate_hash, "candidate": spec, "evaluation": evaluation,
                  "artifact_directory": str(candidate), "elapsed_seconds": time.monotonic() - start,
                  "data_version": manifest["version"], "contract_sha256": manifest["contract_sha256"],
                  "source_sha256": manifest["environment"]["source_sha256"], "parent_attempt": previous["attempt"] if previous else None}
        append(round_path / "experiments.jsonl", record)
        # A failed attempt changes only its own directory/journal. Champion is derived
        # from the immutable accepted history; no reset/checkout touches the dev tree.
        return record


def fixed_search(contract: dict) -> list[dict]:
    base = [{"model": "naive"}]
    base += [{"model": "elastic_net", "params": {"alpha": alpha, "l1_ratio": ratio}}
             for alpha, ratio in [(0.1, 0.2), (1.0, 0.5), (5.0, 0.8)]]
    base += [{"model": "random_forest", "params": {"n_estimators": 30, "max_depth": depth, "min_samples_leaf": leaf}}
             for depth, leaf in [(1, 1), (2, 1), (2, 2)]]
    base += [{"model": "xgboost", "params": {"n_estimators": 20, "max_depth": depth, "learning_rate": rate}}
             for depth, rate in [(1, 0.1), (1, 0.2), (2, 0.1)]]
    for group in sorted({f["group"] for f in contract["features"]}):
        selected = [f["name"] for f in contract["features"] if f["group"] == group]
        if len(selected) < len(contract["features"]):
            base.append({"model": "elastic_net", "params": {"alpha": 1.0, "l1_ratio": 0.5}, "features": selected})
    return base


def run_round(round_path: Path, proposals: list[dict] | None = None) -> dict:
    _, contract, entries = load_round(round_path)
    known = {e.get("candidate_sha256") for e in current_results(entries) if e["evaluation"] is not None}
    for proposal in fixed_search(contract) if proposals is None else proposals:
        try:
            try:
                spec = validate_candidate(proposal, contract)
                if candidate_id(spec) in known:
                    continue
            except ResearchError:
                pass  # Invalid proposals still consume a reservation and enter memory.
            record = attempt(round_path, proposal)
            if record["evaluation"] is not None:
                known.add(record["candidate_sha256"])
        except ResearchError as exc:
            if exc.code in {"OVER_BUDGET", "NO_PROGRESS_STOP"}:
                break
            raise
    return round_status(round_path)
