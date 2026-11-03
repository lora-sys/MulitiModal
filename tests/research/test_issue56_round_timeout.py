"""Synthetic regression coverage for bounded source-loop round deadlines."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from research.loop import SourceResearchLoop
from research.runner import (
    DEFAULT_EVALUATOR_TIMEOUT_SECONDS,
    ROUND_FOLD_ORCHESTRATION_ALLOWANCE_SECONDS,
    ROUND_LAUNCH_HANDOFF_ALLOWANCE_SECONDS,
    _training_parent_timeout,
    source_loop_round_timeout_seconds,
)

pytestmark = pytest.mark.synthetic


def test_round_deadline_covers_all_worker_and_evaluator_limits():
    folds = 4
    repeats = 2
    compute_budget = 20.0
    per_fold = (
        _training_parent_timeout(compute_budget)
        + DEFAULT_EVALUATOR_TIMEOUT_SECONDS
        + ROUND_FOLD_ORCHESTRATION_ALLOWANCE_SECONDS
    )
    expected = (
        folds * repeats * per_fold
        + ROUND_LAUNCH_HANDOFF_ALLOWANCE_SECONDS
    )

    timeout = source_loop_round_timeout_seconds(folds, repeats, compute_budget)

    assert timeout == expected
    assert timeout > folds * repeats * (compute_budget + 20.0) + 120.0


def test_loop_uses_stage_derived_timeout_without_expanding_fit_budget(tmp_path):
    loop = SourceResearchLoop(
        worktree=SimpleNamespace(),
        out_dir=tmp_path / "out",
        manifest_path="synthetic-manifest.json",
        dataset_path="synthetic-records.json",
        folds=4,
        repeats=2,
        budget_s=20.0,
    )

    assert loop.timeout_s == source_loop_round_timeout_seconds(4, 2, 20.0)
    assert loop.timeout_source == "derived_from_stage_limits"
    assert loop.budget == 20.0
    assert loop.evaluator_timeout_s == DEFAULT_EVALUATOR_TIMEOUT_SECONDS


def test_loop_rejects_an_outer_deadline_shorter_than_legal_stages(tmp_path):
    with pytest.raises(ValueError, match="shorter than the stage-derived"):
        SourceResearchLoop(
            worktree=SimpleNamespace(),
            out_dir=tmp_path / "out-short",
            manifest_path="synthetic-manifest.json",
            dataset_path="synthetic-records.json",
            folds=4,
            repeats=2,
            budget_s=20.0,
            timeout_s=440.0,
        )
