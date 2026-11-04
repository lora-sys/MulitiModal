"""Trusted, in-container validation of candidate-produced model artifacts."""
from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from pathlib import Path

MAX_INPUT_BYTES = 256 * 1024
MAX_VALIDATION_FILE_BYTES = 64 * 1024 * 1024

def _sha256_regular(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ValueError("artifact cannot be opened as a regular file") from exc
    digest = hashlib.sha256()
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("artifact must be a regular file")
        if info.st_size > MAX_VALIDATION_FILE_BYTES:
            raise ValueError("artifact exceeds the validation file limit")
        remaining = info.st_size
        while remaining:
            chunk = os.read(fd, min(65_536, remaining))
            if not chunk:
                raise ValueError("artifact was truncated during validation")
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            raise ValueError("artifact grew during validation")
        return digest.hexdigest()
    finally:
        os.close(fd)

def validate_artifacts(config: dict, worker_result: dict) -> dict:
    task = config.get("task", "constitution")
    if task not in ("constitution", "syndrome", "toy"):
        raise ValueError("artifact validation task is invalid")
    expected_sha = config.get("candidate_sha256")
    if worker_result.get("candidate_sha256") != expected_sha:
        raise ValueError("worker result does not bind the candidate SHA")
    if worker_result.get("task", task) != task:
        raise ValueError("worker result does not bind the task")
    output_dir = Path(config.get("output_dir", ""))
    if output_dir != Path("/sandbox/output"):
        raise ValueError("artifact output directory is outside the sandbox output mount")
    stem = config.get("output_stem")
    if not isinstance(stem, str) or not stem.isascii() or not stem.isidentifier():
        raise ValueError("artifact output stem is invalid")
    weights = output_dir / f"{stem}.pt"
    architecture = output_dir / f"{stem}-architecture.json"
    if worker_result.get("weights_path") != str(weights):
        raise ValueError("worker returned an unexpected weights path")
    if worker_result.get("architecture_path") != str(architecture):
        raise ValueError("worker returned an unexpected architecture path")
    from research.model_export import load_eval_model
    weights_sha = _sha256_regular(weights)
    architecture_sha = _sha256_regular(architecture)
    if worker_result.get("weights_sha256") != weights_sha:
        raise ValueError("worker weights SHA does not match the artifact")
    if worker_result.get("architecture_sha256") != architecture_sha:
        raise ValueError("worker architecture SHA does not match the artifact")
    model = load_eval_model(weights, architecture, task=task)
    from research.tasks import adapter_for, validate_manifest_contract
    from research.train_worker import _validate_model_output
    adapter = adapter_for(task)
    manifest_path = Path(config["fit_manifest_path"])
    records_path = Path(config["fit_records_path"])
    manifest = adapter.load_manifest(manifest_path)
    validate_manifest_contract(task, manifest, selection_pool=True)
    if manifest["split"]["dev"] or manifest["split"]["holdout"]:
        raise ValueError("artifact validation manifest must contain train rows only")
    records = adapter.records_from_json(records_path.read_text(encoding="utf-8"))
    fold = adapter.training_fold_data(manifest, records, config)
    actual = _validate_model_output(model, fold, task)
    reported = (
        {
            "output_key": worker_result.get("output_key"),
            "output_shape": worker_result.get("output_shape"),
        }
        if config.get("preflight")
        else worker_result.get("candidate_output")
    )
    if (
        not isinstance(reported, dict)
        or reported.get("output_key") != actual["output_key"]
        or reported.get("output_shape") != actual["output_shape"]
    ):
        raise ValueError("worker output record differs from trusted first-batch validation")
    return {
        "task": task,
        "weights_sha256": weights_sha,
        "architecture_sha256": architecture_sha,
        **actual,
    }

def main() -> int:
    payload = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
    if len(payload) > MAX_INPUT_BYTES:
        print(json.dumps({"ok": False, "error": "artifact validation input exceeds the limit"}))
        return 2
    try:
        request = json.loads(payload)
        result = validate_artifacts(request["config"], request["worker_result"])
    except Exception as exc:
        print(json.dumps({
            "ok": False,
            "error": f"{type(exc).__name__}: {str(exc)[:500]}",
        }))
        return 1
    print(json.dumps({"ok": True, **result}))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
