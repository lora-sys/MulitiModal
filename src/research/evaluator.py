"""THE EVALUATOR PROCESS: score one trained fold model, outside the trainer.

Run as `python -m research.evaluator --config <json> --request <json>`.
Prints a JSON line: {"ok": true, "primary_value": float}.

This module exists because the first implementation broke its own contract:
`evaluate_model` ran in the same interpreter that had just executed
`candidate.train()`, so a candidate could rebind
`runtime.tcm_eval.TCMTrustedEvaluator` for the in-process late import, or
poison in place the FoldData the evaluator was about to score. Now:

- the runner (a subprocess of the loop) trains the candidate, saves the model
  to a file and DISCARDS the fold data together with the in-process score
  path — nothing the candidate trained against is trusted afterwards;
- this process REBUILDS the fold from the frozen manifest (deterministic:
  identical config -> identical fold, scaler included) and loads the model
  here. The candidate cannot tamper with objects it was never given, and its
  code is not running in this interpreter — except at import time, because
  torch.load pickles the model by reference and unpickling imports
  `research.candidate`. That window is closed two ways: the scoring
  machinery is grabbed BEFORE the unpickler runs, and an audit hook blocks
  any runtime.*/data.*/core.registry import whose stack originates in the
  candidate file (see `install_candidate_import_guard`);
- the request carries an integrity snapshot of the manifest, the dataset and
  the candidate file taken BEFORE training; this process re-hashes them with
  its own pristine hashlib and refuses to score if any changed (the runner
  does the same comparison in-process, so tampering is caught on both sides).
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


def install_candidate_import_guard(candidate_path: Path | None = None) -> None:
    """Block frozen-machinery imports from candidate code in this process.

    torch.load pickles the model by reference, so unpickling imports
    research.candidate — and module-level code in that file executes HERE. An
    audit hook makes any runtime.*/data.*/core.registry import whose stack
    originates in the candidate file raise, so the candidate cannot rebind
    the evaluator, the split or the label derivation at import time. Callers
    grab their own references to the scoring machinery BEFORE installing the
    hook (see score_fold), so even a later successful patch cannot reach them.
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


def score_fold(config: dict, request: dict, records=None) -> dict:
    """Rebuild the fold from the frozen manifest and score the saved model.

    The trainer's in-memory FoldData is never consulted: the fold is rebuilt
    (deterministic — identical config yields an identical fold manifest,
    scaler and batches), so in-place tampering with the trainer's copy cannot
    reach the score. `records` exists so tests can inject a parsed fixture.
    """
    from data.mendeley import load_mendeley_scores
    from research.frozen import build_fold
    from runtime.real_data import real_eval_manifest
    from runtime.tcm_eval import TCMTrustedEvaluator

    assert_integrity(config, request)
    if records is None:
        records = load_mendeley_scores(config["data"]["dataset_path"])
    fold_data, fold_manifest = build_fold(
        config, config["data"]["manifest_path"], config["data"]["dataset_path"],
        request["fold"], request["folds"], request["repeat"], records=records,
    )
    # grab the scoring machinery BEFORE the unpickler imports the candidate:
    # module-level code in candidate.py runs inside THIS process and must not
    # be able to rebind what the score is computed with
    evaluator = TCMTrustedEvaluator(real_eval_manifest(fold_manifest))
    eval_fn = TCMTrustedEvaluator.evaluate
    install_candidate_import_guard()
    import torch

    model = torch.load(request["model_path"], weights_only=False)
    report = eval_fn(evaluator, model, fold_data.dev, partition="dev")
    return {"primary_value": float(report.primary_value)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="research.evaluator")
    parser.add_argument("--config", required=True,
                        help="JSON file with the data paths and worktree binding")
    parser.add_argument("--request", required=True,
                        help="JSON file with fold/repeat, model path and the "
                             "trainer's integrity snapshot")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text())
    request = json.loads(Path(args.request).read_text())
    try:
        result = score_fold(config, request)
    except Exception as exc:  # noqa: BLE001 — the runner records the refusal
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
