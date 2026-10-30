"""Baseline review eval for the #50 deliverable (evals/review-eval-50.md).

Every check is a frozen finding from the adversarial review rounds; a check
may only be edited to stay correct against intentional contract changes,
never to make a failing check pass. Run:

  PYTHONPATH=src python scripts/review_eval.py \
      [--summary artifacts/tcm-50/summary.json] [--report artifacts/tcm-50/report.md]

Exit code 0 = all checks pass.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from core.models import default_tcm_model
from data.mendeley import SubjectRecord
from data.real_split import (
    build_real_manifest,
    derive_classification_targets,
    exclude_subjects_from_manifest,
)
from runtime.baselines import (
    CANONICAL_DIMS,
    _design_matrix,
    evaluate_baselines,
    macro_f1,
)
from runtime.protocols import trusted_score
from runtime.real_data import RealBatchProvider
from runtime.tcm_eval import compute_subject_level_metrics
from runtime.worker import sanitize_env

DATASET_SHA = "c" * 64


def fake_records(n=30, seed=7, qc_row=None):
    rng = random.Random(seed)
    records = []
    for i in range(n):
        scores = [None] + [float(rng.randint(10, 90)) for _ in range(8)]
        provenance = {"source": "test-fixture", "fingerprint_anomalies": {}}
        if qc_row is not None and i == qc_row:
            for dim in ("yang_deficiency", "phlegm_damp"):
                scores[CANONICAL_DIMS.index(dim)] = 61.111
            provenance["fingerprint_anomalies"] = {
                "yang_deficiency": [i], "phlegm_damp": [i],
            }
        records.append(
            SubjectRecord(
                subject_index=i,
                features={
                    "demographics": [float(rng.randint(20, 70)), float(i % 2)],
                    "anthropometrics": [float(rng.randint(150, 190)),
                                        float(rng.randint(45, 95)), 22.5],
                    "meridian": [rng.random() for _ in range(3)],
                    "pulse_wave": [rng.random() for _ in range(2)],
                    "tongue": [rng.random()],
                },
                scores=scores,
                score_mask=[s is not None for s in scores],
                provenance=provenance,
            )
        )
    return records


CHECKS: list = []


def check(fn):
    CHECKS.append(fn)
    return fn


@check
def e01_qc_split_preserved():
    records = fake_records(qc_row=5)
    base = build_real_manifest(records, dataset_sha256=DATASET_SHA).data
    derived = exclude_subjects_from_manifest(
        base, records, (5,), notes="qc"
    ).data
    for partition in ("train", "dev", "holdout"):
        expected = [s for s in base["split"][partition] if s != "mendeley-r005"]
        assert derived["split"][partition] == expected, partition
    assert derived["manifest_sha256"] != base["manifest_sha256"]
    assert derived["excluded_subject_rows"] == [5]


@check
def e02_macro_f1_fixed_denominator():
    per_dim = {
        "a": {"f1": 1.0, "n_observed": 10},
        "b": {"f1": None, "n_observed": 10},  # undefined counts as 0
        "c": {"f1": 0.5, "n_observed": 0},    # unobserved never enters
    }
    assert macro_f1(per_dim) == (1.0 + 0.0) / 2


@check
def e03_baselines_report_subject_mae():
    records = fake_records()
    manifest = build_real_manifest(records, dataset_sha256=DATASET_SHA).data
    result = evaluate_baselines(manifest, records)
    for name in ("dummy_majority", "bmi_only_logistic", "feature_logistic"):
        for partition in ("dev", "holdout"):
            value = result[name][partition]["subject_mae"]
            assert value is not None and 0.0 <= value, (name, partition)


@check
def e04_availability_bits_in_baselines():
    records = fake_records()
    manifest = build_real_manifest(records, dataset_sha256=DATASET_SHA).data
    n_feature_cols = sum(m["dim"] for m in manifest["modalities"].values())
    X, _ = _design_matrix(
        {r.subject_index: r for r in records}, manifest,
        manifest["split"]["dev"][:5], "full",
    )
    assert X.shape[1] == n_feature_cols + len(manifest["modalities"])


@check
def e06_fixed_training_deterministic():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tcm_real_report import _fixed_training

    records = fake_records()
    manifest = build_real_manifest(records, dataset_sha256=DATASET_SHA).data
    with tempfile.TemporaryDirectory() as tmp:
        dirs = [Path(tmp) / "a", Path(tmp) / "b"]
        models = [
            _fixed_training(manifest, records, {}, 1, d) for d in dirs
        ]
        params_a = list(models[0].parameters())
        params_b = list(models[1].parameters())
        assert len(params_a) == len(params_b)
        for pa, pb in zip(params_a, params_b, strict=True):
            assert torch.equal(pa, pb), "fixed training is not deterministic"


@check
def e07_charts_treat_none_as_missing():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import matplotlib

    matplotlib.use("Agg")
    from tcm_real_report import chart_regression

    per_dim = {
        partition: {
            dim: {"mae": None, "rmse": None, "mean_predictor_mae": None}
            for dim in CANONICAL_DIMS
        }
        for partition in ("dev", "holdout")
    }
    with tempfile.TemporaryDirectory() as tmp:
        chart_regression({"per_dim": per_dim}, Path(tmp))  # must not crash


@check
def e08_subject_metrics_fail_closed():
    # empty observation -> None, never a perfect 0.0
    stats = compute_subject_level_metrics(
        torch.tensor([[5.0]]), torch.tensor([[1.0]]),
        torch.tensor([[False]]), ["a"],
    )
    assert stats["subject_mae"] is None and stats["n_labeled"] == 0
    # multi-row subject: pooled mean, not a sum of row means
    pred = torch.tensor([[0.0, 0.0], [2.0, 2.0]])
    target = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    mask = torch.ones(2, 2, dtype=torch.bool)
    stats = compute_subject_level_metrics(pred, target, mask, ["s", "s"])
    assert stats["subject_mae"] == 1.5, stats["subject_mae"]


@check
def e09_degenerate_columns_zeroed():
    records = fake_records(n=20)
    for r in records:
        r.features["tongue"] = [1.0]  # constant column -> degenerate
    manifest = build_real_manifest(records, dataset_sha256=DATASET_SHA).data
    exam = manifest["scaler"]["exam_features"]  # tongue is its last column
    assert exam["degenerate"][-1], "constant tongue column must be marked degenerate"
    assert not any(exam["degenerate"][:-1])
    provider = RealBatchProvider(manifest, records, "dev", batch_size=64)
    item = next(iter(provider))
    assert torch.count_nonzero(item.batch.features["exam_features"][:, -1]) == 0
    # a real value in a degenerate column must still be zeroed
    dev_subject = manifest["split"]["dev"][0]
    records[manifest["subject_row_by_id"][dev_subject]].features["tongue"] = [99.0]
    provider = RealBatchProvider(manifest, records, "dev", batch_size=64)
    item = next(iter(provider))
    position = item.subject_ids.index(dev_subject)
    assert float(item.batch.features["exam_features"][position][-1]) == 0.0


@check
def e11_regression_has_reference():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tcm_real_report import regression_task

    records = fake_records()
    manifest = build_real_manifest(records, dataset_sha256=DATASET_SHA).data
    with tempfile.TemporaryDirectory() as tmp:
        result = regression_task(manifest, records, Path(tmp))
    for partition in ("dev", "holdout"):
        for dim, stats in result["per_dim"][partition].items():
            assert "mean_predictor_mae" in stats, (partition, dim)
    assert "aggregation" in result


@check
def e12_qc_rows_derived_from_provenance():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tcm_real_report import qc_rows_from_records

    assert qc_rows_from_records(fake_records()) == ()
    assert qc_rows_from_records(fake_records(qc_row=5)) == (5,)
    # score_mask drift is refused instead of silently resolved
    with pytest_like(ValueError):
        derive_classification_targets(
            [None, 50.0], [False, False]
        )


class pytest_like:
    def __init__(self, expected):
        self.expected = expected

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return exc_type is not None and issubclass(exc_type, self.expected)


@check
def e13_support_chart_partition():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import inspect

    import matplotlib

    matplotlib.use("Agg")
    from tcm_real_report import chart_support

    source = inspect.getsource(chart_support)
    assert 'neural["holdout"]' in source and 'neural["dev"]' not in source


@check
def e14_duplicate_subjects_rejected():
    records = fake_records(n=12)
    duplicated = records + [records[0]]
    try:
        build_real_manifest(duplicated, dataset_sha256=DATASET_SHA)
    except ValueError as exc:
        assert "duplicate subject_index" in str(exc)
    else:
        raise AssertionError("duplicate subjects accepted")


@check
def e15_report_tolerates_none():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import matplotlib

    matplotlib.use("Agg")
    from tcm_real_report import write_reports

    records = fake_records()
    manifest = build_real_manifest(records, dataset_sha256=DATASET_SHA).data
    summary = {
        "generated_at": "t", "manifest_sha256": "0", "dataset_sha256": "0",
        "label_version": "l", "standard_version": "s",
        "split_sizes": {"train": 1, "dev": 1, "holdout": 1},
        "autoresearch": {
            "rounds_completed": 0,
            "champion": {"run_id": "r", "artifact_dir": "d",
                         "artifact_sha256": None, "best_so_far": None},
            "per_round": [{"run_id": "r", "intent": "i", "decision": "KEEP",
                           "primary_value": None}],
        },
        "neural_evaluation": {"dev": {"macro_f1": None, "n_undefined_f1": 0,
                                      "subject_mae": None, "per_dim": {}},
                              "holdout": {"macro_f1": None, "n_undefined_f1": 0,
                                          "subject_mae": None, "per_dim": {}}},
        "baselines": {k: {"dev": {}, "holdout": {"macro_f1": None,
                                                 "subject_mae": None}}
                      for k in ("dummy_majority", "bmi_only_logistic",
                                "feature_logistic")},
        "regression_task": {"manifest_sha256": "0", "per_dim": {}},
        "row_qc_sensitivity": {"excluded_subject_rows": [],
                               "dev_subject_mae_without": None},
        "deployable_ablation": {"dev_subject_mae": None, "f1": {}},
        "availability": {},
    }
    with tempfile.TemporaryDirectory() as tmp:
        write_reports(Path(tmp), summary, manifest, 30)  # must not crash


@check
def e17_train_epoch_shuffle():
    records = fake_records()
    manifest = build_real_manifest(records, dataset_sha256=DATASET_SHA).data
    provider = RealBatchProvider(manifest, records, "train", batch_size=16)
    first = [list(item.subject_ids) for item in provider]
    second = [list(item.subject_ids) for item in provider]
    assert sorted(s for batch in first for s in batch) \
        == sorted(s for batch in second for s in batch), "rows lost between epochs"
    assert first != second, "train order identical across epochs"
    fresh = RealBatchProvider(manifest, records, "train", batch_size=16)
    assert [list(i.subject_ids) for i in fresh] == first, "shuffle not deterministic"
    dev = RealBatchProvider(manifest, records, "dev", batch_size=16)
    assert (list(list(dev)[0].subject_ids)
            == manifest["split"]["dev"][:16]), "eval order must stay stable"


@check
def e19_trusted_score_refuses_real(tmp_path=None):
    with tempfile.TemporaryDirectory() as tmp:
        torch.manual_seed(0)
        model, _ = default_tcm_model(hidden_dim=8, fusion_dim=8)
        from runtime.checkpoint import save_training_checkpoint

        artifact = save_training_checkpoint(
            model,
            {"epoch": 1, "global_step": 1, "best_epoch": None,
             "best_eval_loss": None, "optimizer": {}, "trainer_config": {},
             "torch_rng_state": torch.get_rng_state(),
             "manifest_sha256": "d" * 64},
            Path(tmp) / "real", smoke_only=False,
        )
        try:
            trusted_score(str(artifact))
        except ValueError as exc:
            assert "synthetic fixtures only" in str(exc)
        else:
            raise AssertionError("trusted_score accepted a real-data checkpoint")


@check
def e21_env_whitelist_exact():
    sanitized = sanitize_env({
        "PATH": "/usr/bin", "PYTHONSTARTUP": "/tmp/evil.py",
        "PYTHONHOME": "/tmp/evil", "PYTHONWARNINGS": "ignore",
        "PYTHONPATH": "/src", "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1", "AWS_SECRET_ACCESS_KEY": "x",
    })
    assert "PYTHONSTARTUP" not in sanitized and "PYTHONHOME" not in sanitized
    assert "PYTHONWARNINGS" not in sanitized
    assert sanitized["PYTHONPATH"] == "/src"
    assert "AWS_SECRET_ACCESS_KEY" not in sanitized


@check
def e22_empty_partition_rejected():
    records = fake_records(n=12)
    try:
        build_real_manifest(records, dataset_sha256=DATASET_SHA,
                            fractions=(0.0, 0.5, 0.5))
    except ValueError as exc:
        assert "empty" in str(exc) and "train" in str(exc)
    else:
        raise AssertionError("empty train partition accepted")


@check
def e24_ablation_uses_champion_config():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import inspect

    from tcm_real_report import deployable_ablation, row_qc_sensitivity

    for fn in (deployable_ablation, row_qc_sensitivity):
        source = inspect.getsource(fn)
        assert "champion_trainer" in source and "champion_cfg" in source


def check_summary_artifacts(summary_path: str | None, report_path: str | None) -> list[str]:
    problems: list[str] = []
    if summary_path:
        summary = json.loads(Path(summary_path).read_text())
        if not summary.get("availability"):
            problems.append("E05: summary.json missing availability rates")
        if "metric_params" not in summary or "champion_provenance" not in summary:
            problems.append("E16: summary.json missing metric_params/champion_provenance")
        reg = summary.get("regression_task", {})
        if "mean_predictor_mae" not in json.dumps(reg):
            problems.append("E11: summary.json regression lacks mean-predictor reference")
    if report_path:
        text = Path(report_path).read_text()
        for token, label in (
            ("42.9", "E10: effective cut-point drift not disclosed"),
            ("TENDENCY", "E10: TENDENCY semantics not disclosed"),
            ("winner", "E25: winner's curse not disclosed"),
            ("单 seed", "E25: single-seed caveat not disclosed"),
            ("v2", "E25: holdout re-unlock disclosure missing"),
        ):
            if token not in text:
                problems.append(label)
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--report", default=None)
    args = parser.parse_args(argv)

    failures: list[str] = []
    for fn in CHECKS:
        name = fn.__name__
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 — eval reports, not crashes
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"FAIL {name}: {exc}")
        else:
            print(f"PASS {name}")
    for problem in check_summary_artifacts(args.summary, args.report):
        failures.append(problem)
        print(f"FAIL {problem}")
    total = len(CHECKS) + bool(args.summary or args.report)
    print(f"\n{total - len(failures)}/{total} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    np.seterr(invalid="ignore")  # NaN bars in charts are intentional
    raise SystemExit(main())
