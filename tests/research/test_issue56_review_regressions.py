from __future__ import annotations

import hashlib

import pytest

from research.task_sd import (
    FEATURE_GROUP,
    SyndromeBatchProvider,
    SyndromeRecord,
    SyndromeTrustedEvaluator,
    build_syndrome_manifest,
    records_to_json,
    syndrome_input_spec,
)
from research.tasks import adapter_for


@pytest.mark.synthetic
def test_finalize_loader_rejects_dataset_bytes_not_bound_to_frozen_manifest(tmp_path):
    records = [
        SyndromeRecord(
            subject_index=i,
            user_id=f"synthetic-user-{i}",
            norm_syndrome=f"synthetic-{i % 3}",
            features={FEATURE_GROUP: [float(i), float(i % 5)]},
            label=i % 5,
            provenance={"source": "SYNTHETIC"},
        )
        for i in range(1, 41)
    ]
    payload = records_to_json(records)
    records_path = tmp_path / "synthetic-records.json"
    records_path.write_text(payload, encoding="utf-8")
    frozen = build_syndrome_manifest(
        records,
        dataset_sha256=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        n_classes=5,
        label_taxonomy_version="synthetic-taxonomy-v1",
        label_tier="SMOKE_ONLY",
    ).data

    adapter = adapter_for("syndrome")
    loaded = adapter.load_records_verified(
        records_path, frozen["dataset_sha256"]
    )
    assert len(loaded) == len(records)

    records_path.write_text(payload + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="dataset SHA-256"):
        adapter.load_records_verified(records_path, frozen["dataset_sha256"])


@pytest.mark.synthetic
def test_syndrome_eval_loader_rejects_unverified_records_path(tmp_path, monkeypatch):
    adapter = adapter_for("syndrome")
    records_path = tmp_path / "unverified-records.json"
    records_path.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(
        adapter,
        "load_records_verified",
        lambda *_args: pytest.fail("unverified records_path must not be opened"),
    )

    with pytest.raises(ValueError, match="bound to the frozen manifest"):
        adapter.load_eval_records({"data": {"records_path": str(records_path)}})


@pytest.mark.synthetic
def test_syndrome_finalize_rejects_in_memory_records_before_any_run(tmp_path):
    from research.finalize import refit_and_score

    with pytest.raises(ValueError, match="dataset bytes verified"):
        refit_and_score(
            worktree=tmp_path,
            out_dir=tmp_path / "out",
            manifest_path=tmp_path / "manifest.json",
            dataset_path=tmp_path / "records.json",
            editable="src/research/candidate_sd.py",
            commit="synthetic",
            records=[{"synthetic": True}],
            task="syndrome",
        )


@pytest.mark.synthetic
@pytest.mark.parametrize(
    "malformation",
    ("wrong_shape", "nan_on_second_batch", "negative", "over_one", "not_normalized"),
)
def test_trusted_evaluator_rejects_invalid_probabilities_in_every_dev_batch(
    malformation,
):
    import torch

    from core.batch import ModelOutput

    records = [
        SyndromeRecord(
            subject_index=i,
            user_id=f"synthetic-user-{i}",
            norm_syndrome=f"synthetic-{i % 3}",
            features={FEATURE_GROUP: [float(i), float(i % 5)]},
            label=i % 5,
            provenance={"source": "SYNTHETIC"},
        )
        for i in range(1, 41)
    ]
    frozen = build_syndrome_manifest(
        records,
        dataset_sha256="d" * 64,
        n_classes=5,
        label_taxonomy_version="synthetic-taxonomy-v1",
        label_tier="SMOKE_ONLY",
    ).data

    class MalformedModel:
        input_spec = syndrome_input_spec(frozen)

        def __init__(self):
            self.calls = 0

        def __call__(self, batch):
            self.calls += 1
            batch_size = next(iter(batch.features.values())).shape[0]
            probs = torch.full((batch_size, 5), 0.2)
            if malformation == "wrong_shape":
                probs = torch.full((batch_size, 4), 0.25)
            elif malformation == "nan_on_second_batch" and self.calls == 2:
                probs[0, 0] = float("nan")
            elif malformation == "negative":
                probs[0, 0] = -0.01
            elif malformation == "over_one":
                probs[0, 0] = 1.01
            elif malformation == "not_normalized":
                probs.fill_(0.1)
            return ModelOutput(predictions={"syndrome_probs": probs})

    provider = SyndromeBatchProvider(
        frozen, records, "dev", batch_size=1
    )
    with pytest.raises(ValueError, match="probabilit|finite|shape|syndrome_probs"):
        SyndromeTrustedEvaluator(frozen).evaluate(
            MalformedModel(), provider, partition="dev"
        )


@pytest.mark.synthetic
def test_repeated_cv_scores_keep_descriptive_fold_dispersion_constant():
    from research.runner import summarize_cv_scores

    scores = [0.2, 0.4, 0.6, 0.8]
    once = summarize_cv_scores(scores)
    repeated = summarize_cv_scores(scores * 3)

    assert once["primary_std"] == pytest.approx(repeated["primary_std"])
    assert "descriptive" in once["uncertainty_semantics"].lower()
    assert "independent" in once["uncertainty_semantics"].lower()
    assert "primary_sem" not in once

    from research.loop import SourceResearchLoop

    decision_once = SourceResearchLoop.decide(
        0.26, once["primary_std"], 0, 0.20
    )
    decision_repeated = SourceResearchLoop.decide(
        0.26, repeated["primary_std"], 0, 0.20
    )
    assert decision_once == decision_repeated
    assert decision_once[0] is False
