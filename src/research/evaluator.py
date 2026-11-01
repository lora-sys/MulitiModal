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


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def integrity_snapshot(config: dict) -> dict:
    """Hash every input the trainer could tamper with during train()."""
    snapshot = {
        "manifest_sha256": _sha256(config["data"]["manifest_path"]),
        "dataset_sha256": _sha256(config["data"]["dataset_path"]),
    }
    worktree = config.get("worktree_path")
    if worktree:
        editable = config.get("editable", "src/research/candidate.py")
        candidate_path = Path(worktree) / editable
        if candidate_path.exists():
            snapshot["candidate_sha256"] = _sha256(candidate_path)
    return snapshot


def assert_integrity(config: dict, request: dict) -> None:
    """Refuse to score if the training inputs changed after training began."""
    recorded = request.get("integrity") or {}
    actual = integrity_snapshot(config)
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


def _load_model(request: dict):
    from research.model_export import load_eval_model

    assert_weights_unmodified(request)
    return load_eval_model(request["weights_path"], request["architecture_path"])


def _editable_path(config: dict) -> Path | None:
    """The candidate file this run's config binds, when the config carries it."""
    worktree = config.get("worktree_path")
    editable = config.get("editable")
    if worktree and editable:
        return Path(worktree) / editable
    return None


def score_fold(config: dict, request: dict, records=None) -> dict:
    """Rebuild the fold from the frozen manifest and score the saved model.

    The training subprocess's in-memory objects are never consulted: the fold
    is rebuilt (deterministic — identical config yields an identical fold
    manifest, scaler and batches), and the model is rebuilt from state_dict +
    the frozen architecture description. `records` exists so tests can inject
    a parsed fixture. The task's adapter (config["task"]) owns the fold and
    scoring seams."""
    from research.tasks import adapter_for

    adapter = adapter_for(config.get("task", "constitution"))
    assert_integrity(config, request)
    if records is None:
        records = adapter.load_eval_records(config)
    fold_data, fold_manifest = adapter.build_fold(
        config, config["data"]["manifest_path"], config["data"]["dataset_path"],
        request["fold"], request["folds"], request["repeat"], records=records,
    )
    model = _load_model(request)
    install_candidate_import_guard(_editable_path(config))
    report = adapter.evaluate_model(model, fold_manifest, fold_data.dev, "dev")
    return {"primary_value": float(report.primary_value)}


def score_holdout(config: dict, request: dict, records=None) -> dict:
    """The single holdout read of a finalize run, in this isolated process.

    The model is loaded state_dict-only and handed to the task's adapter,
    which rebuilds the holdout view from the frozen parent manifest and
    computes the task's report and baselines under that view. Baselines are
    computed in the same isolated process, under the same view."""
    from research.tasks import adapter_for

    adapter = adapter_for(config.get("task", "constitution"))
    if records is None:
        records = adapter.load_eval_records(config)
    model = _load_model(request)
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
