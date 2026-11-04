"""THE EVALUATOR PROCESS: score a saved model, outside any training code.

Run as `python -m research.evaluator --mode {fold,holdout} --config <json>
--request <json>`. Prints a JSON line; fold mode returns
{"ok": true, "primary_value": float}, holdout mode returns the holdout report
and the baselines (the single holdout read of a finalize run).

Evaluation executes NO candidate code, in any form:

* the model is loaded from a state_dict-only weights file (safe
  deserialization: a file of tensors cannot carry pickled code) into a model
  rebuilt from the frozen architecture whitelist
  (`research/model_export.py`). Before the 2026-11-02 hardening round this
  process deserialized a full pickled model, and unpickling executed the
  candidate's module-level code, which could rebind the scoring machinery in
  `runtime.tcm_eval` and forge its own score (F-01);
* the fold (or the holdout view) is rebuilt here from the frozen manifest(s)
  — deterministic, scaler included — so nothing the training subprocess held
  in memory can reach the score;
* the request carries an integrity snapshot of the manifest, the dataset and
  the candidate file taken BEFORE training; this process re-hashes them with
  its own pristine hashlib and refuses to score if any changed;
* the audit hook (`install_candidate_import_guard`) stays installed as
  defense in depth even though nothing on the evaluation path imports the
  candidate any more.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path


def _read_regular_file_no_follow(path: str | Path, label: str) -> bytes:
    """Read a regular file through a checked, no-follow descriptor."""
    import os
    import stat

    if not hasattr(os, "O_NOFOLLOW"):
        raise OSError("O_NOFOLLOW is required for selection artifact reads")
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"{label} must be a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            return stream.read()
    finally:
        os.close(descriptor)


def read_verified_dataset_bytes(
    path: str | Path, expected_sha256: str, label: str = "dataset",
) -> bytes:
    """Read the exact regular-file bytes bound by a frozen manifest."""
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(char not in "0123456789abcdef" for char in expected_sha256)
    ):
        raise ValueError(f"{label} manifest SHA-256 is malformed")
    payload = _read_regular_file_no_follow(path, label)
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 does not match frozen manifest dataset_sha256"
        )
    return payload


def _validate_selection_manifest(task: str, manifest: dict) -> None:
    from research.tasks import validate_manifest_contract

    validate_manifest_contract(task, manifest, selection_pool=True)
    if manifest["split"]["holdout"] != []:
        raise ValueError("selection manifest must have an empty holdout partition")
    if task != "syndrome":
        return

    row_map = manifest["subject_row_by_id"]
    subject_users = manifest.get("subject_user_by_id")
    user_partitions = manifest.get("user_partition_by_id")
    user_strata = manifest.get("user_stratum_by_id")
    if not all(
        isinstance(value, dict)
        for value in (subject_users, user_partitions, user_strata)
    ):
        raise ValueError("selection manifest lacks frozen pool user metadata")
    if set(subject_users) != set(row_map):
        raise ValueError("selection subject-user mapping does not match selection rows")
    if set(user_partitions) != set(user_strata):
        raise ValueError("selection user partitions and strata have different users")
    frozen_users = set(user_partitions)
    present_users = set(subject_users.values())
    if not present_users <= frozen_users:
        raise ValueError("selection rows reference an unfrozen user")
    if any(
        not isinstance(user_id, str) or not user_id for user_id in frozen_users
    ):
        raise ValueError("selection manifest contains an invalid frozen user")
    if any(
        partition not in {"train", "dev"}
        for partition in user_partitions.values()
    ):
        raise ValueError("selection user metadata must not contain holdout users")
    if any(
        not isinstance(stratum, str) or not stratum
        for stratum in user_strata.values()
    ):
        raise ValueError("selection manifest contains an invalid frozen user stratum")
    for partition in ("train", "dev"):
        if any(
            user_partitions[subject_users[subject_id]] != partition
            for subject_id in manifest["split"][partition]
        ):
            raise ValueError(
                f"selection user partitions disagree with split {partition!r}"
            )
    for name in (
        "selection_parent_manifest_sha256",
        "selection_parent_dataset_sha256",
    ):
        value = manifest.get(name)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
        ):
            raise ValueError(f"selection manifest lacks a valid {name}")


def _decode_selection_manifest(payload: bytes, task: str) -> dict:
    manifest = json.loads(payload.decode("utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("selection manifest must be a JSON object")
    expected = manifest.pop("manifest_sha256", None)
    actual = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    if expected != actual:
        raise ValueError(
            f"manifest hash mismatch: expected {expected}, computed {actual}; refusing"
        )
    manifest["manifest_sha256"] = expected
    _validate_selection_manifest(task, manifest)
    return manifest


def _validate_selection_records(task: str, manifest: dict, records: list) -> None:
    expected_rows = set(manifest["subject_row_by_id"].values())
    actual_rows = [record.subject_index for record in records]
    if len(actual_rows) != len(set(actual_rows)):
        raise ValueError("selection dataset contains duplicate subject rows")
    if set(actual_rows) != expected_rows:
        raise ValueError("selection dataset rows do not match the selection manifest")
    if task == "syndrome":
        subject_by_row = {
            row: subject_id
            for subject_id, row in manifest["subject_row_by_id"].items()
        }
        user_by_subject = manifest["subject_user_by_id"]
        for record in records:
            subject_id = subject_by_row[record.subject_index]
            if record.user_id != user_by_subject[subject_id]:
                raise ValueError("selection record user does not match manifest")


def load_selection_inputs(
    config: dict,
    *,
    adapter=None,
    records_override=None,
    parse_records: bool = True,
    manifest_validator=None,
) -> tuple[dict, list | None, dict]:
    """Validate selection manifest bytes before reading exact selection data bytes."""
    from research.tasks import adapter_for

    task = config.get("task", "constitution")
    adapter = adapter or adapter_for(task)
    data = config["data"]
    manifest_bytes = _read_regular_file_no_follow(
        data["manifest_path"], "selection manifest"
    )
    manifest = _decode_selection_manifest(manifest_bytes, task)
    if manifest_validator is not None:
        manifest_validator(manifest)

    dataset_bytes = read_verified_dataset_bytes(
        data["dataset_path"], manifest["dataset_sha256"], "selection dataset"
    )
    dataset_sha256 = hashlib.sha256(dataset_bytes).hexdigest()

    records = None
    if parse_records:
        if records_override is None:
            records = adapter.records_from_json(dataset_bytes.decode("utf-8"))
        else:
            records = records_override
            serialized = adapter.records_to_json(records).encode("utf-8")
            if hashlib.sha256(serialized).hexdigest() != dataset_sha256:
                raise ValueError("provided records do not match selection dataset bytes")
        _validate_selection_records(task, manifest, records)

    hashes = {
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "dataset_sha256": dataset_sha256,
    }
    return manifest, records, hashes


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(_read_regular_file_no_follow(path, str(path))).hexdigest()


def integrity_snapshot(config: dict, *, input_hashes: dict | None = None) -> dict:
    """Hash validated selection artifacts and the editable candidate source."""
    if input_hashes is None:
        _manifest, _records, input_hashes = load_selection_inputs(
            config, parse_records=False
        )
    snapshot = dict(input_hashes)
    worktree = config.get("worktree_path")
    if worktree:
        editable = config.get("editable")
        if editable is None:
            from research.tasks import adapter_for

            editable = adapter_for(config.get("task", "constitution")).default_editable
            config = {**config, "editable": editable}
        from research.train_worker import (
            MAX_CANDIDATE_SOURCE_BYTES,
            _candidate_path,
            _read_regular_file_limited,
        )

        source = _read_regular_file_limited(
            _candidate_path(config),
            max_bytes=MAX_CANDIDATE_SOURCE_BYTES,
            label="candidate source",
        )
        snapshot["candidate_sha256"] = hashlib.sha256(source).hexdigest()
    return snapshot


def assert_integrity(
    config: dict, request: dict, *, input_hashes: dict | None = None
) -> None:
    """Refuse to score if validated training inputs changed after training."""
    recorded = request.get("integrity") or {}
    actual = integrity_snapshot(config, input_hashes=input_hashes)
    changed = sorted(
        k for k in set(recorded) | set(actual)
        if recorded.get(k) != actual.get(k)
    )
    if changed:
        raise RuntimeError(
            f"training inputs were modified during training: {changed} — "
            "refusing to score a fold whose data contract moved underneath it"
        )


def assert_weights_unmodified(request: dict) -> None:
    """Verify the saved model files against the hashes the parent recorded."""
    for sha_key, path_key in (
        ("weights_sha256", "weights_path"),
        ("architecture_sha256", "architecture_path"),
    ):
        expected = request.get(sha_key)
        if expected and _sha256(request[path_key]) != expected:
            raise RuntimeError(
                f"{path_key} does not match the recorded {sha_key}; refusing to "
                "score a model that changed between training and evaluation"
            )


def install_candidate_import_guard(candidate_path: Path | None = None) -> None:
    """Defense in depth: block frozen-machinery imports from candidate code.

    Nothing on the evaluation path executes candidate code any more (the
    model arrives as state_dict + a frozen architecture description), so this
    hook has nothing to catch today. It stays because its absence would make
    "evaluation never imports the candidate" a promise instead of a checked
    invariant: any future code path that does import the candidate here gets
    its frozen-machinery imports blocked at the candidate's stack frames.
    """
    if candidate_path is None:
        candidate_path = Path(__file__).resolve().parent / "candidate.py"
    candidate_file = str(candidate_path)

    def _hook(event, args):
        if event != "import":
            return
        name = args[0]
        if not isinstance(name, str):
            return
        root = name.split(".")[0]
        # "research" is blocked too (R1 F4): the syndrome editable lives under
        # src/research/, so its frozen machinery (task_sd, evaluator, frozen)
        # is importable by root name from candidate code.
        if root not in ("runtime", "data", "research") and name != "core.registry":
            return
        frame = sys._getframe(1)
        while frame is not None:
            if frame.f_code.co_filename == candidate_file:
                raise RuntimeError(
                    f"candidate code may not import {name!r}: the scoring "
                    "machinery is frozen and not reachable from this file"
                )
            frame = frame.f_back

    sys.addaudithook(_hook)


def _load_model(request: dict, *, task: str):
    from research.model_export import load_eval_model

    assert_weights_unmodified(request)
    return load_eval_model(
        request["weights_path"], request["architecture_path"], task=task
    )


def _editable_path(config: dict) -> Path | None:
    """The task-specific candidate file bound by this run's config."""
    worktree = config.get("worktree_path")
    editable = config.get("editable")
    if worktree and editable is None:
        from research.tasks import adapter_for

        editable = adapter_for(config.get("task", "constitution")).default_editable
    if worktree and editable:
        return Path(worktree) / editable
    return None


def extract_diagnostics(report) -> dict:
    """Extract actionable diagnostic slices (worst performing classes, zero F1)
    to provide reflection context for the research loop proposer."""
    diagnostics = {}
    per_class = getattr(report, "per_class", None)
    if per_class is None:
        per_class = getattr(report, "per_category", None)
    if isinstance(per_class, dict) and per_class:
        zero_f1 = []
        low_f1 = []
        for cls_id, stats in per_class.items():
            if not isinstance(stats, dict):
                continue
            support = stats.get("support", 0)
            f1 = stats.get("f1", 0.0)
            if support > 0:
                if f1 == 0.0:
                    zero_f1.append({"class": cls_id, "support": support})
                else:
                    low_f1.append({"class": cls_id, "f1": round(float(f1), 4), "support": support})
        low_f1.sort(key=lambda x: x["f1"])
        diagnostics["zero_f1_count"] = len(zero_f1)
        diagnostics["worst_classes"] = (zero_f1 + low_f1)[:5]
        diagnostics["total_evaluated_classes"] = len(per_class)
    return diagnostics


def score_fold(config: dict, request: dict, records=None) -> dict:
    """Rebuild and score one fold from validated selection-only inputs."""
    from research.tasks import adapter_for

    task = config.get("task", "constitution")
    adapter = adapter_for(task)
    if config.get("editable") is None:
        config = {**config, "editable": adapter.default_editable}
    manifest, loaded_records, input_hashes = load_selection_inputs(
        config,
        adapter=adapter,
        records_override=records,
    )
    assert_integrity(config, request, input_hashes=input_hashes)
    records = loaded_records
    fold_data, fold_manifest = adapter.build_fold(
        config,
        manifest,
        request["fold"],
        request["folds"],
        request["repeat"],
        records=records,
    )
    model = _load_model(request, task=task)
    install_candidate_import_guard(_editable_path(config))
    report = adapter.evaluate_model(model, fold_manifest, fold_data.dev, "dev")
    res = {"primary_value": float(report.primary_value)}
    diag = extract_diagnostics(report)
    if diag:
        res["diagnostics"] = diag
    return res
def score_holdout(config: dict, request: dict, records=None) -> dict:
    """Validate the charged event before materializing any holdout records."""
    from research.finalize import verify_holdout_unlock
    from research.tasks import adapter_for, validate_manifest_contract

    task = config.get("task", "constitution")
    adapter = adapter_for(task)
    if config.get("editable") is None:
        config = {**config, "editable": adapter.default_editable}
    frozen = adapter.load_manifest(config["data"]["frozen_manifest_path"])
    pool = adapter.load_manifest(config["data"]["pool_manifest_path"])
    validate_manifest_contract(task, frozen)
    validate_manifest_contract(task, pool, selection_pool=True)
    verify_holdout_unlock(config, request, pool, frozen)
    if task == "syndrome":
        if records is not None or config["data"].get("records_path"):
            raise ValueError(
                "syndrome holdout requires records loaded from the dataset bytes "
                "bound to the frozen manifest"
            )
        records = adapter.load_records_verified(
            config["data"]["dataset_path"], frozen["dataset_sha256"]
        )
    elif records is None:
        records = adapter.load_eval_records(config)
    model = _load_model(request, task=task)
    install_candidate_import_guard(_editable_path(config))
    return adapter.score_holdout(config, request, records, model)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="research.evaluator")
    parser.add_argument("--mode", choices=("fold", "holdout"), default="fold")
    parser.add_argument("--config", required=True,
                        help="JSON file with the data paths and worktree binding")
    parser.add_argument("--request", required=True,
                        help="JSON file with fold/repeat or the model file "
                             "paths and the trainer's integrity snapshot")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text())
    request = json.loads(Path(args.request).read_text())
    try:
        result = (
            score_holdout(config, request) if args.mode == "holdout"
            else score_fold(config, request)
        )
    except Exception as exc:  # noqa: BLE001 — the parent records the refusal
        print(json.dumps({
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-1500:],
        }))
        return 1
    print(json.dumps({"ok": True, **result}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
