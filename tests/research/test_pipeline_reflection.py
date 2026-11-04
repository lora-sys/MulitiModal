"""Tests for the general Feature+Model Pipeline and Diagnostic Reflection."""

import pytest
import torch
from dataclasses import dataclass

from core.batch import ModelOutput
from core.specs import InputSpecSet, ModalitySpec
from research.candidate import (
    ConstitutionPipeline,
    FusionMLP,
    build_model as build_constitution_model,
)
from research.candidate_sd import (
    SyndromePipeline,
    SyndromeMLP,
    build_model as build_syndrome_model,
)
from research.evaluator import extract_diagnostics
from research.frozen import FoldData
from research.model_export import export_architecture


@dataclass
class _DummyReport:
    per_class: dict


def test_extract_diagnostics_identifies_zero_and_worst_f1():
    report = _DummyReport(
        per_class={
            0: {"precision": 0.8, "recall": 0.8, "f1": 0.8, "support": 10},
            1: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 5},
            2: {"precision": 0.2, "recall": 0.5, "f1": 0.2857, "support": 8},
            3: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0},  # unobserved
            4: {"precision": 0.1, "recall": 0.2, "f1": 0.1333, "support": 4},
        }
    )
    diag = extract_diagnostics(report)
    assert diag["total_evaluated_classes"] == 5
    assert diag["zero_f1_count"] == 1  # class 1 has support 5 and f1 0.0; class 3 has support 0
    worst = [item["class"] for item in diag["worst_classes"]]
    assert 1 in worst
    assert 4 in worst
    assert 2 in worst


def test_extract_diagnostics_handles_empty():
    assert extract_diagnostics(None) == {}
    assert extract_diagnostics(_DummyReport(per_class={})) == {}


def test_syndrome_pipeline_traces_cleanly():
    spec = InputSpecSet(
        version="v1",
        modalities=(ModalitySpec("hashed_text", "tabular", 64),),
    )
    fold = FoldData(
        train=[],
        dev=[],
        input_spec=spec,
        dims={"modalities": {"hashed_text": 64}, "targets": 10},
        n_train_subjects=10,
        n_dev_subjects=5,
        seed=42,
    )
    model = build_syndrome_model(fold)
    assert isinstance(model, SyndromePipeline)
    assert isinstance(model, SyndromeMLP)

    # Test tracing with torch.fx under the frozen model_export whitelist
    arch = export_architecture(model, task="syndrome", expected_n_classes=10)
    assert arch is not None
    assert arch["task"] == "syndrome"


def test_constitution_pipeline_traces_cleanly():
    spec = InputSpecSet(
        version="v1",
        modalities=(ModalitySpec("features", "tabular", 32),),
    )
    fold = FoldData(
        train=[],
        dev=[],
        input_spec=spec,
        dims={"modalities": {"features": 32}, "targets": 9},
        n_train_subjects=10,
        n_dev_subjects=5,
        seed=42,
    )
    model = build_constitution_model(fold)
    assert isinstance(model, ConstitutionPipeline)
    assert isinstance(model, FusionMLP)

    # Test tracing with torch.fx under the frozen model_export whitelist
    arch = export_architecture(model, task="constitution", expected_n_classes=9)
    assert arch is not None
    assert arch["task"] == "constitution"
