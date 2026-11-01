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
        if root not in ("runtime", "data") and name != "core.registry":
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


def score_fold(config: dict, request: dict, records=None) -> dict:
    """Rebuild the fold from the frozen manifest and score the saved model.

    The training subprocess's in-memory objects are never consulted: the fold
    is rebuilt (deterministic — identical config yields an identical fold
    manifest, scaler and batches), and the model is rebuilt from state_dict +
    the frozen architecture description. `records` exists so tests can inject
    a parsed fixture.
    """
    from research.frozen import build_fold
    from runtime.real_data import real_eval_manifest
    from runtime.tcm_eval import TCMTrustedEvaluator

    assert_integrity(config, request)
    if records is None:
        records = _load_records(config)
    fold_data, fold_manifest = build_fold(
        config, config["data"]["manifest_path"], config["data"]["dataset_path"],
        request["fold"], request["folds"], request["repeat"], records=records,
    )
    evaluator = TCMTrustedEvaluator(real_eval_manifest(fold_manifest))
    model = _load_model(request)
    install_candidate_import_guard()
    report = evaluator.evaluate(model, fold_data.dev, partition="dev")
    return {"primary_value": float(report.primary_value)}


def score_holdout(config: dict, request: dict, records=None) -> dict:
    """The single holdout read of a finalize run, in this isolated process.

    The holdout view is rebuilt from the frozen parent manifest (holdout ids
    + row mapping) under the pool manifest's own refit scaler; the pool
    manifest on disk carries no holdout at all, so nothing the fit worker
    could have read points here. Baselines are computed in the same isolated
    process, under the same view.
    """
    from data.real_split import holdout_view, load_real_manifest
    from runtime.baselines import evaluate_baselines
    from runtime.real_data import RealBatchProvider, real_eval_manifest
    from runtime.tcm_eval import TCMTrustedEvaluator

    if records is None:
        records = _load_records(config)
    pool = load_real_manifest(config["data"]["pool_manifest_path"])
    frozen = load_real_manifest(config["data"]["frozen_manifest_path"])
    view = holdout_view(pool, frozen)
    model = _load_model(request)
    install_candidate_import_guard()
    holdout = RealBatchProvider(view, records, "holdout", batch_size=64)
    report = TCMTrustedEvaluator(real_eval_manifest(view)).evaluate(
        model, holdout, partition="holdout"
    )
    baselines = evaluate_baselines(view, records)
    return {
        "holdout": report.to_dict(),
        "baselines": {
            name: {
                "macro_ap": vals["holdout"]["macro_ap"],
                "macro_f1": vals["holdout"]["macro_f1"],
                "subject_mae": vals["holdout"]["subject_mae"],
            }
            for name, vals in baselines.items()
            if isinstance(vals, dict) and "holdout" in vals
        },
    }


def _load_records(config: dict):
    """Trusted-side records: the workbook, or the JSON the driver wrote."""
    from data.mendeley import load_mendeley_scores, records_from_json

    data = config["data"]
    if data.get("records_path"):
        return records_from_json(Path(data["records_path"]).read_text())
    return load_mendeley_scores(data["dataset_path"])


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
