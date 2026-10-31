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
from labels.standards import CANONICAL_ORDER
from runtime.baselines import (
    CANONICAL_DIMS,
    _design_matrix,
    evaluate_baselines,
    macro_f1,
)
from runtime.protocols import trusted_score
from runtime.real_data import RealBatchProvider, real_input_spec
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
                                      "subject_mae": None, "subject_rmse": None,
                                      "per_dim": {}},
                              "holdout": {"macro_f1": None, "n_undefined_f1": 0,
                                          "subject_mae": None,
                                          "subject_rmse": None, "per_dim": {}}},
        "baselines": {k: {"dev": {}, "holdout": {"macro_f1": None,
                                                 "subject_mae": None,
                                                 "subject_rmse": None}}
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


# --- second review round (Pi R-01..R-12), frozen as E26+ -------------------


@check
def e26_availability_doc_polarity():
    # R-05: the docstring said "unusable" while True means available
    import inspect

    from core import batch

    source = inspect.getsource(batch.ModelBatch)
    assert "unusable" not in source
    assert "True = the row's features" in source


@check
def e27_trainer_restore_semantics():
    # R-06, amended (intentional contract change, see E27 note): the default
    # checkpoint carries LAST-epoch weights declared via weight_epoch; the
    # restore mechanism is opt-in and actually restores the best epoch.
    from core.batch import ModelOutput
    from runtime.batches import SyntheticBatchProvider
    from runtime.trainer import Trainer, TrainerConfig

    class DivergingStub(torch.nn.Module):
        model_type = "stub_diverging"

        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, 2)
            self._calls = 0

        @property
        def model_config(self):
            return {"stub": "diverging"}

        def forward(self, batch):
            return ModelOutput(predictions={})

        def compute_loss(self, batch):
            self._calls += 1
            probe = self.lin(torch.ones(4))
            return float(self._calls) * (probe**2).mean(), {}

    def run(epochs: int, **cfg) -> Trainer:
        torch.manual_seed(0)  # seed BEFORE stub construction: identical init
        trainer = Trainer(DivergingStub(), TrainerConfig(seed=0, epochs=epochs, **cfg))
        trainer.fit(
            SyntheticBatchProvider("tcm", n_batches=2, batch_size=8, seed=0),
            SyntheticBatchProvider("tcm", n_batches=1, batch_size=8, seed=5,
                                   subject_prefix="eval"),
        )
        return trainer

    one, three = run(1), run(3)
    assert three.best_epoch == 0
    assert any(
        not torch.equal(pa, pb)
        for pa, pb in zip(one.model.parameters(), three.model.parameters(), strict=True)
    ), "default weights are not last-epoch"
    three_best = run(3, restore_best=True)
    for pa, pb in zip(one.model.parameters(), three_best.model.parameters(), strict=True):
        assert torch.equal(pa, pb), "restore_best did not restore best-epoch weights"


@check
def e28_report_binds_champion_to_manifest():
    # R-01: the report path must refuse a champion trained on another manifest
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import inspect

    import tcm_real_report
    from tcm_real_report import assert_champion_matches_manifest

    manifest = {"manifest_sha256": "a" * 64, "dataset_sha256": "b" * 64}
    assert_champion_matches_manifest(
        {"smoke_only": False, "manifest_sha256": "a" * 64,
         "dataset_sha256": "b" * 64}, manifest,
    )
    with pytest_like(ValueError):
        assert_champion_matches_manifest(
            {"smoke_only": False, "manifest_sha256": "c" * 64,
             "dataset_sha256": "b" * 64}, manifest,
        )
    with pytest_like(ValueError):
        assert_champion_matches_manifest(
            {"smoke_only": False, "manifest_sha256": "a" * 64,
             "dataset_sha256": "z" * 64}, manifest,
        )
    with pytest_like(ValueError):
        assert_champion_matches_manifest({"smoke_only": True}, manifest)
    assert "assert_champion_matches_manifest" in inspect.getsource(
        tcm_real_report.main
    )


@check
def e29_derived_manifests_mirror_frozen_split():
    # R-02/R-11: task-variant manifests must reuse the frozen fractions and
    # exclusions AND verify partition membership instead of relying on
    # default-argument coincidence
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tcm_real_report import derived_manifest

    records = fake_records()
    base = build_real_manifest(records, dataset_sha256=DATASET_SHA).data
    for variant in ({"task_type": "score_regression"},
                    {"modality_set": "deployable"}):
        derived = derived_manifest(base, records, notes="n", **variant)
        for partition in ("train", "dev", "holdout"):
            assert set(derived["split"][partition]) == set(base["split"][partition]), partition
    tampered = json.loads(json.dumps(base))
    tampered["split"]["train"][0] = "mendeley-r999"
    with pytest_like(ValueError):
        derived_manifest(tampered, records)


@check
def e30_input_spec_version_discriminates():
    # R-09: a full-set and a deployable-set manifest must never share an
    # input spec version
    records = fake_records()
    full = build_real_manifest(records, dataset_sha256=DATASET_SHA).data
    dep = build_real_manifest(records, dataset_sha256=DATASET_SHA,
                              modality_set="deployable").data
    v_full, v_dep = real_input_spec(full).version, real_input_spec(dep).version
    assert v_full != v_dep
    assert ":full:" in v_full and ":deployable:" in v_dep
    assert "body_features=" in v_full and "exam_features=" in v_full


@check
def e31_report_interpolates_no_hardcoded_metrics():
    # R-04: report numbers must come from the summary, never be literals
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import inspect

    from tcm_real_report import write_reports

    source = inspect.getsource(write_reports)
    for literal in ("0.345", "0.275", "0.2978", "≈0.29", "<50", "6 个候选"):
        assert literal not in source, literal


@check
def e32_canonical_dims_single_source():
    # R-10: baselines must import the frozen canonical order, not redefine it
    assert CANONICAL_DIMS is CANONICAL_ORDER


@check
def e33_qc_sensitivity_same_denominator():
    # R-03: the WITH-QC champion must also be evaluated on the reduced dev
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import inspect

    from tcm_real_report import row_qc_sensitivity

    source = inspect.getsource(row_qc_sensitivity)
    assert "champion_model" in source
    assert "dev_subject_mae_champion_on_reduced" in source


@check
def e34_scaler_semantics_recorded():
    # R-08: the manifest states what the scaler was fit on
    manifest = build_real_manifest(fake_records(), dataset_sha256=DATASET_SHA).data
    assert "TRAIN-fold" in manifest["scaler_policy"]
    assert manifest["scaler_fit_rows"] == len(manifest["split"]["train"])


@check
def e35_chart_rounds_real_axis():
    # R-12: the rounds chart x-axis uses real round indices, not 1..n
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import inspect

    from tcm_real_report import chart_rounds

    source = inspect.getsource(chart_rounds)
    assert "run_id" in source and "rsplit" in source
    assert "range(1, len" not in source


@check
def e36_subject_identity_declared():
    # R-07: row-order subject identity is an assumption and must be labelled
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import inspect

    import tcm_real_report

    assert "subject_identity" in inspect.getsource(tcm_real_report.main)
    assert "UNVERIFIED" in inspect.getsource(tcm_real_report.write_reports)


@check
def e37_champion_evaluated_under_own_scaler():
    # GLM round-3: the champion-on-reduced-dev contrast must not feed the
    # champion inputs standardized by the QC manifest's refit scaler — each
    # side of the contrast runs under its own manifest, same subject set.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import inspect

    from tcm_real_report import row_qc_sensitivity

    source = inspect.getsource(row_qc_sensitivity)
    assert "real_eval_manifest(manifest)" in source
    assert 'RealBatchProvider(\n            manifest, records, "dev", subject_ids=reduced_dev' \
        in source
    # the provider must reject subjects from other partitions
    records = fake_records()
    manifest = build_real_manifest(records, dataset_sha256=DATASET_SHA).data
    foreign = manifest["split"]["train"][0]
    try:
        RealBatchProvider(manifest, records, "dev", subject_ids=(foreign,))
    except ValueError as exc:
        assert "outside partition" in str(exc)
    else:
        raise AssertionError("cross-partition subject_ids accepted")
    subset = manifest["split"]["dev"][:3]
    provider = RealBatchProvider(manifest, records, "dev", subject_ids=subset)
    assert list(provider.subject_ids) == list(subset)


@check
def e57_agent_provenance_recorded():
    # 2026-10-31 audit: the manifest recorded an intent string but NOT who
    # proposed it or what state it saw — "the model really did AutoResearch"
    # was unverifiable from the artifacts. step() now requires a proposer
    # and binds the proposal to the exact research state it was based on.
    from runtime.agent import Proposal
    from runtime.harness import AutoResearchHarness, context_digest, default_tcm_runspec

    with tempfile.TemporaryDirectory() as tmp:
        harness = AutoResearchHarness(
            default_tcm_runspec(rounds=2, timeout_s=120.0),
            out_dir=Path(tmp) / "ar",
        )
        context = harness.research_context()
        stale = {"proposer": "test:agent", "context_sha256": "0" * 64}
        try:
            harness.step(Proposal(intent="x", config_patch={}), provenance=stale)
        except RuntimeError as exc:
            assert "stale" in str(exc), exc
        else:
            raise AssertionError("stale agent context accepted")

        try:
            harness.step(Proposal(intent="x", config_patch={}),
                         provenance={"proposer": ""})
        except RuntimeError as exc:
            assert "proposer" in str(exc), exc
        else:
            raise AssertionError("empty proposer accepted")

        entry = harness.step(
            Proposal(intent="baseline", config_patch={}),
            provenance={"proposer": "test:agent",
                        "context_sha256": context_digest(context),
                        "proposal_sha256": "a" * 64},
        )
        assert entry["proposer"] == "test:agent"
        assert entry["context_sha256"] == context_digest(context)
        assert entry["proposal_sha256"] == "a" * 64

        # round 2's digest must describe the state AFTER round 1's result,
        # i.e. the agent is shown the history it is asked to improve on
        second = harness.research_context()
        assert second["history"] and second["round_index"] == 1
        assert context_digest(second) != context_digest(context)


@check
def e58_cli_requires_proposer_for_attribution_args():
    # --context-file/--proposal-raw are attribution evidence; using them
    # without --proposer would record hashes with no claimed author
    from runtime.cli import main

    with tempfile.TemporaryDirectory() as tmp:
        ctx = Path(tmp) / "ctx.json"
        assert main(["research-step", "--profile", "tcm", "--rounds", "1",
                     "--out", str(Path(tmp) / "runs"), "--print-context",
                     "--context-out", str(ctx)]) == 0
        assert json.loads(ctx.read_text())["round_index"] == 0

        proposal = Path(tmp) / "p.json"
        proposal.write_text(json.dumps({"intent": "x", "config_patch": {}}))
        # no --proposer, but an attribution file is supplied -> refused
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            rc = main(["research-step", "--profile", "tcm", "--rounds", "1",
                       "--out", str(Path(tmp) / "runs2"),
                       "--proposal-file", str(proposal),
                       "--context-file", str(ctx)])
        assert rc == 2, buffer.getvalue()
        assert "--proposer" in buffer.getvalue()


@check
def e60_patches_accumulate_over_champion():
    # 2026-10-31 audit: every candidate was trained from base_config, so a
    # KEEP never became the starting point and the report described a
    # 32-wide champion as 64-wide. KEEPs must now accumulate (hill climbing).
    from runtime.agent import DeterministicStubProvider
    from runtime.harness import AutoResearchHarness, default_tcm_runspec
    from runtime.manifest import RunManifestStore

    with tempfile.TemporaryDirectory() as tmp:
        provider = DeterministicStubProvider(plan=[
            ("widen", {"model.hidden_dim": 64, "model.fusion_dim": 64}),
            ("raise lr only", {"trainer.learning_rate": 0.01}),
        ])
        harness = AutoResearchHarness(
            default_tcm_runspec(rounds=2, timeout_s=120.0),
            provider=provider, out_dir=Path(tmp) / "ar",
        )
        summary = harness.run()
        entries = RunManifestStore(Path(tmp) / "ar" / "manifest.jsonl").load()
        assert summary["rounds_completed"] == 2
        first = entries[0]["effective_config"]
        assert first["model"]["hidden_dim"] == 64
        # the second candidate changes only lr, but must still run at 64-wide
        second = entries[1]["effective_config"]
        assert second["model"]["hidden_dim"] == 64, second
        assert second["trainer"]["learning_rate"] == 0.01, second
        # the round-2 config the agent is shown reflects the champion
        context = harness.research_context()
        assert context["current_config"]["model"]["hidden_dim"] == 64


@check
def e61_report_champion_config_matches_checkpoint():
    # the delivered report must describe the champion's ACTUAL config: the
    # audit found "64-wide" written for a 32-wide champion
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import inspect

    import tcm_real_report

    source = inspect.getsource(tcm_real_report.main)
    assert "champion_cfg" in source and "state" in source
    # champion_provenance must carry the checkpoint's own trainer config
    write_src = inspect.getsource(tcm_real_report)
    assert '"trainer_config": champion_trainer' in write_src
    assert '"model_config": champion_cfg' in write_src


@check
def e38_manifest_chain_tamper_evidence():
    # Pi round-4 R-01 + round-5 R-01: the append-only manifest must detect
    # edits, deletions, reordering AND unchained forged lines (after any
    # chained entry) instead of trusting entry 0 only. Known undetectable:
    # last-entry deletion and whole-file rewrite with a recomputed chain
    # (unkeyed hash, same writable dir) — the real control is #45 isolation.
    from runtime.manifest import RunManifestStore

    with tempfile.TemporaryDirectory() as tmp:
        store = RunManifestStore(Path(tmp) / "m.jsonl")
        store.append({"run_id": "tcm-0000", "decision": "KEEP", "status": "done",
                      "artifact_dir": "runs/ar/tcm-0000"})
        store.append({"run_id": "tcm-0001", "decision": "REVERT", "status": "done"})
        store.append({"run_id": "tcm-0002", "decision": "KEEP", "status": "done",
                      "artifact_dir": "runs/ar/tcm-0002"})
        assert len(store.load()) == 3

        path = Path(tmp) / "m.jsonl"
        lines = path.read_text().splitlines()

        def expect_integrity_failure(content: str, why: str) -> None:
            path.write_text(content)
            try:
                RunManifestStore(path).load()
            except RuntimeError as exc:
                assert "integrity check failed" in str(exc), why
            else:
                raise AssertionError(f"loaded without error: {why}")

        # hand-edit of one KEEP entry -> detected
        tampered = json.loads(lines[0])
        tampered["best_so_far"] = 0.0
        expect_integrity_failure(
            "\n".join([json.dumps(tampered, ensure_ascii=False),
                       *lines[1:]]) + "\n",
            "edited entry",
        )
        # deletion of a middle entry (breaks the next link) -> detected
        expect_integrity_failure(
            "\n".join([lines[0], lines[2]]) + "\n", "deleted entry"
        )
        # reordering -> detected
        expect_integrity_failure(
            "\n".join([lines[1], lines[0], lines[2]]) + "\n", "reordered"
        )
        # forged UNCHAINED line after chained entries (round-5 R-01) -> detected
        forged = {"run_id": "tcm-9999", "decision": "KEEP",
                  "best_so_far": 0.0001, "artifact_dir": "runs/ar/evil"}
        expect_integrity_failure(
            "\n".join([*lines, json.dumps(forged)]) + "\n", "forged line"
        )
        # intact store loads and reports the honest champion
        path.write_text("\n".join(lines) + "\n")
        champion = RunManifestStore(path).champion()
        assert champion is not None and champion["run_id"] == "tcm-0002"


@check
def e39_resume_plan_alignment():
    # Pi round-4 R-02: the stub plan must be indexed by round_index so a
    # resumed run continues the plan instead of replaying tried candidates.
    from runtime.agent import DeterministicStubProvider

    plan = [("baseline", {}), ("wider", {"model.hidden_dim": 64}),
            ("third", {"trainer.epochs": 3})]
    first = DeterministicStubProvider(plan=plan)
    assert first.propose(0, {}).intent == "baseline"
    # a fresh provider on resume must propose plan[1], not replay plan[0]
    resumed = DeterministicStubProvider(plan=plan)
    assert resumed.propose(1, {}).intent == "wider"
    assert resumed.propose(2, {}).intent == "third"
    assert resumed.propose(3, {}) is None  # exhausted past the plan


@check
def e40_fail_to_revert_round_paths():
    # Pi round-4 R-03 + round-5 R-03/R-05: every in-loop failure records a
    # REVERT entry instead of crashing the loop without a manifest record.
    from runtime.agent import DeterministicStubProvider, Proposal
    from runtime.harness import AutoResearchHarness, default_tcm_runspec

    with tempfile.TemporaryDirectory() as tmp:
        # a scope key outside the known sections passes patch validation but
        # fails in apply_config_patch -> recorded rejected round
        runspec = default_tcm_runspec(rounds=2, timeout_s=120.0)
        runspec["editable_scope"] = ["model.hidden_dim", "eval_provider.n"]
        harness = AutoResearchHarness(
            runspec,
            provider=DeterministicStubProvider(plan=[
                ("baseline", {}), ("illegal section", {"eval_provider.n": 3}),
            ]),
            out_dir=Path(tmp) / "ar",
        )
        summary = harness.run()
        rejected = summary["decisions"][1]
        assert rejected["decision"] == "REVERT" and rejected["status"] == "rejected"
        assert "not editable" in rejected["reason"]

    with tempfile.TemporaryDirectory() as tmp:
        # ANY exception from the provider (not just ValueError) is recorded
        # as an agent_error round and appears in decisions (round-5 R-03/R-06)
        class Broken:
            def propose(self, round_index, context):
                raise TimeoutError("LLM backend timed out")

        harness = AutoResearchHarness(
            default_tcm_runspec(rounds=2, timeout_s=120.0),
            provider=Broken(), out_dir=Path(tmp) / "ar2",
        )
        summary = harness.run()
        assert summary["rounds_completed"] == 2
        assert len(summary["decisions"]) == 2
        for decision in summary["decisions"]:
            assert decision["decision"] == "REVERT"
            assert decision["status"] == "agent_error"
            assert "TimeoutError" in decision["reason"]

    with tempfile.TemporaryDirectory() as tmp:
        # a successful score followed by an unreadable artifact is recorded
        # too (round-5 R-05: artifact hashing is a guarded path)
        import runtime.harness as harness_module

        harness = AutoResearchHarness(
            default_tcm_runspec(rounds=1, timeout_s=120.0),
            provider=DeterministicStubProvider(plan=[("baseline", {})]),
            out_dir=Path(tmp) / "ar3",
        )
        original_hasher = harness_module._artifact_sha256

        def broken_hasher(_artifact_dir):
            raise OSError("weights.pt vanished")

        harness_module._artifact_sha256 = broken_hasher
        try:
            entry = harness.step(Proposal(intent="baseline", config_patch={}))
        finally:
            harness_module._artifact_sha256 = original_hasher
        assert entry["decision"] == "REVERT", entry
        assert entry["status"] == "score_failed", entry
        assert "artifact hashing failed" in entry["reason"]


@check
def e41_worker_env_drops_inherited_pythonpath():
    # Pi round-4 R-04: workers must not inherit the host PYTHONPATH —
    # behavioral: poison the real env and inspect what _worker_env returns.
    import os

    from runtime.harness import SRC_DIR, _worker_env

    poisoned = "/host/junk:/more/junk"
    old = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = poisoned
    try:
        env = _worker_env()
    finally:
        if old is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = old
    assert env["PYTHONPATH"] == str(SRC_DIR), env["PYTHONPATH"]
    assert poisoned not in env["PYTHONPATH"]


@check
def e42_real_training_forbids_dev_steering():
    # Pi round-4 R-05: dev is the Trainer's eval provider in real mode for
    # reporting only; early stop / best-epoch restore must fail closed —
    # behavioral: _train_real refuses the config before touching the dataset.
    from runtime.protocols import train_candidate

    records = fake_records()
    manifest = build_real_manifest(records, dataset_sha256=DATASET_SHA)
    with tempfile.TemporaryDirectory() as tmp:
        manifest_path = Path(tmp) / "manifest.json"
        manifest.save(manifest_path)
        base = {
            "task": "tcm", "data_mode": "real",
            "data": {"manifest_path": str(manifest_path),
                     "dataset_path": "unused.xls"},
            "out_dir": str(Path(tmp) / "runs"),
        }
        for key, value in (("early_stop_patience", 2), ("restore_best", True)):
            config = {**base, "trainer": {"epochs": 1, key: value}}
            try:
                train_candidate(config)
            except ValueError as exc:
                assert f"trainer.{key}" in str(exc), (key, exc)
            else:
                raise AssertionError(
                    f"real training accepted trainer.{key} (dev steering)"
                )


def _flatten_numbers(obj) -> list[float]:
    out: list[float] = []
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for value in obj.values():
            out.extend(_flatten_numbers(value))
    elif isinstance(obj, list):
        for value in obj:
            out.extend(_flatten_numbers(value))
    return out


@check
def e63_selection_metric_tracks_the_deliverable():
    # Audits A-06 + the v6 run together: the AutoResearch objective must rank
    # models the way the deliverable ranks them. subject_mae (L1) is minimised
    # by the median — 0 for a binary label under 50% prevalence — so a hill
    # climb on it cannot beat an all-zero predictor; subject_rmse (Brier) is
    # proper but its optimum for a signal-free model is the base-rate constant
    # (measured: that champion scored dev macro-F1 0.021). The selection metric
    # is now the reported classifier metric, maximised.
    import numpy as np

    from runtime.baselines import _per_dim_f1, macro_f1
    from runtime.tcm_eval import PRIMARY_DIRECTION, PRIMARY_METRIC, TCMReport

    assert PRIMARY_METRIC == "category_probs/macro_f1", PRIMARY_METRIC
    assert PRIMARY_DIRECTION == "max", PRIMARY_DIRECTION
    report = TCMReport(
        metric_version="t", primary_metric=PRIMARY_METRIC, partition="dev",
        standard_version="s", label_tier="RULE", smoke_only=False,
        n_subjects=8, n_labeled=8, subject_mae=0.3, subject_rmse=0.5,
        macro_f1=0.42,
    )
    assert report.to_dict()["primary_value"] == 0.42, "primary_value is not F1"

    y = np.zeros((4, 9))                        # nine canonical dims
    y[:, 0] = [1.0, 0.0, 1.0, 0.0]
    mask = np.ones_like(y, dtype=bool)
    never_positive = np.zeros_like(y)          # the degenerate classifier
    real = y.copy()                            # a classifier with actual skill
    f1_degenerate = macro_f1(_per_dim_f1(y, never_positive, mask))
    f1_real = macro_f1(_per_dim_f1(y, real, mask))
    assert f1_degenerate == 0.0, f1_degenerate
    assert f1_real is not None and f1_real > f1_degenerate, (f1_real, f1_degenerate)


@check
def e65_the_two_f1_implementations_agree():
    # The selection metric is computed by the trusted evaluator (tensors), the
    # published metric by the report script (numpy). If they ever diverge, the
    # loop optimises something other than what the report prints.
    import numpy as np
    import torch

    from runtime.baselines import _per_dim_f1, macro_f1
    from runtime.tcm_eval import _macro_f1_from_tensors

    rng = np.random.default_rng(0)
    for _ in range(5):
        y = (rng.random((12, 9)) < 0.3).astype(float)
        pred = (rng.random((12, 9)) < 0.35).astype(float)
        mask = rng.random((12, 9)) < 0.9
        mask[0, :] = True  # every dim observed at least once
        per_dim = _per_dim_f1(y, pred, mask)
        expected = macro_f1(per_dim)
        got, undefined = _macro_f1_from_tensors(
            torch.tensor(pred), torch.tensor(y), torch.tensor(mask)
        )
        assert got is not None and abs(expected - got) < 1e-12, (expected, got)
        assert undefined == sum(
            1 for v in per_dim.values() if v["n_observed"] > 0 and v["f1"] is None
        ), (undefined, per_dim)


@check
def e64_batch_real_data_run_refuses_the_deterministic_stub():
    # Audit A-03: `research --data-manifest` silently replayed a human-written
    # plan from DeterministicStubProvider, so a batch run was indistinguishable
    # from agent-driven research. It must refuse, and when explicitly allowed
    # every round must name the stub as proposer.
    import contextlib
    import io

    import runtime.harness as harness_mod
    from runtime.cli import main

    original = harness_mod.default_real_tcm_runspec
    harness_mod.default_real_tcm_runspec = lambda *a, **k: _fake_real_runspec()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                rc = main(["research", "--profile", "tcm", "--rounds", "1",
                           "--data-manifest", "m.json", "--dataset", "d.xls",
                           "--out", tmp])
            assert rc == 2, buffer.getvalue()
            assert "research-step" in buffer.getvalue(), buffer.getvalue()
            assert not (Path(tmp) / "manifest.jsonl").exists(), \
                "a refused run still wrote a manifest"
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                rc = main(["research", "--profile", "tcm", "--rounds", "1",
                           "--data-manifest", "m.json", "--dataset", "d.xls",
                           "--out", tmp, "--allow-deterministic-stub"])
            assert rc == 0, buffer.getvalue()
            entries = [
                json.loads(line)
                for line in (Path(tmp) / "manifest.jsonl").read_text().splitlines()
                if line.strip()
            ]
            assert entries, "no manifest written for the allowed stub run"
            for entry in entries:
                assert entry.get("proposer") == "deterministic-stub", entry
                assert entry.get("context_sha256"), entry
    finally:
        harness_mod.default_real_tcm_runspec = original


def _fake_real_runspec() -> dict:
    """Synthetic runspec so the guard can be exercised without a dataset; the
    CLI refuses on --data-manifest regardless of the runspec's data mode."""
    from runtime.harness import default_tcm_runspec

    runspec = default_tcm_runspec(rounds=1, timeout_s=120.0)
    runspec["budget"]["max_candidates"] = 1
    return runspec


@check
def e62_committed_docs_check_bites_and_does_not_false_positive():
    # The first cut of E60 scanned every "N 宽" in the doc, so the sentence
    # narrating the v4 defect ("called a 32-wide champion 64-wide") tripped
    # it. Narrowing it to the champion's own config line must keep both
    # properties: stale/changed attribution fails, narrative prose passes.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        summary = tmp / "summary.json"
        summary.write_text(json.dumps({
            "manifest_sha256": "a" * 64,
            "dataset_sha256": "b" * 64,
            "primary_value": 0.25,
            "champion_provenance": {"model_config": {"hidden_dim": 32}},
        }))
        base = (
            "## 重跑修复\n\n旧实现把 32 宽的冠军写成了 64 宽。\n\n"
            "## 冠军\n\n冠军 **tcm-0003**：{width} 宽 trunk、60 epochs。\n"
        )
        doc = tmp / "doc.md"

        doc.write_text(base.format(width=32))
        assert check_committed_docs_numbers(str(summary), doc) == [], \
            "historical narrative about the old defect false-positived"

        doc.write_text(base.format(width=64))
        problems = check_committed_docs_numbers(str(summary), doc)
        assert any("64-wide" in p and "32-wide" in p for p in problems), problems

        doc.write_text("## 冠军\n\n冠军 **tcm-0003** 跑了 60 epochs。\n")
        problems = check_committed_docs_numbers(str(summary), doc)
        assert any("never states the champion's width" in p for p in problems), problems

        doc.write_text(base.format(width=32) + "\n部署 macro-F1 0.349。\n")
        problems = check_committed_docs_numbers(str(summary), doc)
        assert any("0.349" in p for p in problems), problems


def check_committed_docs_numbers(
    summary_path: str | None, docs_path: str | Path | None = None
) -> list[str]:
    """E59: every metric-looking number in the committed report must be
    backed by summary.json. The 2026-10-31 audit found a stale deployable
    macro-F1 (0.349 vs artifact 0.2191), a wrong holdout positive rate
    (0.32 vs 0.1994) and a mismatched regression manifest hash."""
    problems: list[str] = []
    if not summary_path:
        return problems
    docs = Path(docs_path) if docs_path else (
        Path(__file__).resolve().parent.parent / "docs/reports/tcm-50-real-training.md"
    )
    if not docs.exists():
        return problems
    summary = json.loads(Path(summary_path).read_text())
    supported = _flatten_numbers(summary)
    # a cross-run narrative legitimately cites earlier delivered runs; those
    # numbers must still be checkable, so every sibling artifact summary counts
    root = Path(__file__).resolve().parent.parent / "artifacts"
    if root.exists():
        for other in sorted(root.glob("*/summary.json")):
            try:
                supported.extend(_flatten_numbers(json.loads(other.read_text())))
            except (OSError, json.JSONDecodeError):
                continue
    text = docs.read_text()
    import re

    for token in re.findall(r"(?<![\d.])0\.\d{3,4}(?![\d])", text):
        value = float(token)
        decimals = len(token.split(".")[1])
        tol = max(0.5 * 10 ** -decimals, 2e-4)  # rounding + prose approximations
        if not any(abs(value - s) <= tol for s in supported):
            problems.append(
                f"E59: committed report states {token}, which no summary.json "
                "value supports (stale or fabricated metric)"
            )
    for match in re.findall(r"manifest `([0-9a-f]{16})…`", text):
        known = [summary.get(k, "") for k in ("manifest_sha256", "dataset_sha256")]
        known += _flatten_strings(summary)
        if not any(str(v).startswith(match) for v in known):
            problems.append(
                f"E59: committed report cites manifest {match}…, which does "
                "not match any hash in summary.json"
            )
    # E60: the width the report ATTRIBUTES to the champion must equal the
    # checkpoint's real width. Only the champion's own config sentence counts —
    # narrative about a past defect ("v4 called a 32-wide champion 64-wide")
    # is not an attribution and must not trip the check.
    hidden = (
        summary.get("champion_provenance", {}).get("model_config", {}) or {}
    ).get("hidden_dim")
    if hidden is not None:
        claims = [
            int(w)
            for line in re.findall(r"冠军\s*\*\*tcm-\d+\*\*：([^\n]*)", text)
            for w in re.findall(r"(\d+)\s*宽", line)
        ]
        if not claims:
            problems.append(
                "E60: committed report never states the champion's width on "
                "the champion config line (cannot verify against checkpoint)"
            )
        for width in claims:
            if width != int(hidden):
                problems.append(
                    f"E60: committed report calls the champion {width}-wide, "
                    f"but the checkpoint is {hidden}-wide"
                )
    return problems


def check_committed_doc_availability(
    summary_path: str | None, docs_path: str | Path | None = None
) -> list[str]:
    """E69: the committed report's modality-availability fractions must match
    summary.availability. The v5 audit found hand-edited numbers (179/191,
    38/41, 39/42) that disagreed with the artifact (175/191, 40/41, 41/42);
    E59 only guarded 0.x metrics, so nothing caught it."""
    problems: list[str] = []
    if not summary_path:
        return problems
    docs = Path(docs_path) if docs_path else (
        Path(__file__).resolve().parent.parent / "docs/reports/tcm-50-real-training.md"
    )
    if not docs.exists():
        return problems
    avail = json.loads(Path(summary_path).read_text()).get("availability") or {}
    known = {(int(v["n_available"]), int(v["n_rows"])) for v in avail.values()}
    if not known:
        return problems
    import re

    # only modality lines carry availability fractions; "191 / 41 / 42" split
    # sizes and other a/b prose must not be read as availability
    text = "\n".join(
        line for line in docs.read_text().splitlines()
        if "body_features" in line or "exam_features" in line
    )
    for n_available, n_rows in re.findall(r"(\d+)\s*/\s*(\d+)", text):
        pair = (int(n_available), int(n_rows))
        if pair not in known:
            problems.append(
                f"E69: committed report states availability {n_available}/{n_rows},"
                " which matches no summary.availability entry (stale or invented)"
            )
    return problems


def _flatten_strings(obj) -> list[str]:
    out: list[str] = []
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for value in obj.values():
            out.extend(_flatten_strings(value))
    elif isinstance(obj, list):
        for value in obj:
            out.extend(_flatten_strings(value))
    return out


def check_delivered_run_bundle(summary_path: str | None) -> list[str]:
    """E70: when a delivery ships `run-bundle/`, its champion weights must hash
    to the champion artifact recorded in summary.json — otherwise the artifact
    a third party can re-run is not the artifact that was scored. Absent bundle
    is not an error (only v7+ ships one)."""
    problems: list[str] = []
    if not summary_path:
        return problems
    summary_dir = Path(summary_path).resolve().parent
    bundle = summary_dir / "run-bundle"
    if not bundle.exists():
        return problems
    summary = json.loads(Path(summary_path).read_text())
    expected = (summary.get("autoresearch", {}).get("champion") or {}).get(
        "artifact_sha256"
    )
    if not expected:
        return problems
    weights = bundle / "champion" / "weights.pt"
    if not weights.exists():
        problems.append("E70: run-bundle has no champion/weights.pt")
        return problems
    import hashlib

    actual = hashlib.sha256(weights.read_bytes()).hexdigest()
    if actual != expected:
        problems.append(
            f"E70: bundled champion weights hash to {actual[:16]}… but the "
            f"scored champion is {expected[:16]}…"
        )
    manifest = bundle / "manifest.jsonl"
    if not manifest.exists():
        problems.append("E70: run-bundle has no manifest.jsonl")
    return problems


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
        for field in ("records_total", "subjects_in_manifest", "subject_identity"):
            if field not in summary:
                problems.append(f"E31/E36: summary.json missing {field}")
        if "manifest_sha256" not in summary.get("champion_provenance", {}):
            problems.append("E28: champion_provenance lacks manifest binding")
        # E57 audit follow-up: every round must be attributable to an agent
        for entry in summary.get("autoresearch", {}).get("per_round", []):
            missing = [
                k for k in ("proposer", "context_sha256", "proposal_sha256")
                if not entry.get(k)
            ]
            if missing:
                problems.append(
                    f"E57: round {entry.get('run_id')} lacks agent provenance "
                    f"{missing} (who proposed it / what state it saw)"
                )
        problems.extend(check_committed_docs_numbers(summary_path))
        problems.extend(check_committed_doc_availability(summary_path))
        problems.extend(check_delivered_run_bundle(summary_path))
    if report_path:
        text = Path(report_path).read_text()
        for token, label in (
            ("42.9", "E10: effective cut-point drift not disclosed"),
            ("TENDENCY", "E10: TENDENCY semantics not disclosed"),
            ("winner", "E25: winner's curse not disclosed"),
            ("单 seed", "E25: single-seed caveat not disclosed"),
            ("v2", "E25: holdout re-unlock disclosure missing"),
            ("统计强度", "E66: small-sample significance caveat missing"),
            ("自报归属", "E67: provenance strength (operator-reported) not disclosed"),
            ("逐维失败维度", "E68: per-dimension total misses not named"),
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
