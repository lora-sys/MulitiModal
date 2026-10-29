"""Batch provider protocol and synthetic fixture adapter (#49).

No real dataset file is ever read here (#49 out of scope): all batches are
generated torch tensors with explicit SYNTHETIC/SMOKE_ONLY semantics.
Subject IDs are simulated so the subject-disjoint split assertion (#27) can
be exercised without real data.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, Protocol

import torch

from core.batch import ModelBatch
from core.specs import InputSpecSet, make_physio_input_spec, make_tcm_input_spec

BODY_DIM = 12
PHYSIO_DIM = 7
NUM_SCORES = 9

TaskName = Literal["tcm", "physio"]


@dataclass
class BatchItem:
    """One training/eval item: a ModelBatch plus the subject IDs it covers."""

    batch: ModelBatch
    subject_ids: list[str]


class BatchProvider(Protocol):
    """Iterable of BatchItem with a full subject manifest for leak checks."""

    subject_ids: tuple[str, ...]

    def __iter__(self) -> Iterator[BatchItem]:
        ...


def assert_subject_disjoint(a: BatchProvider, b: BatchProvider) -> None:
    """#27 contract: the same subject must never appear in two partitions."""
    overlap = set(a.subject_ids) & set(b.subject_ids)
    if overlap:
        sample = sorted(overlap)[:3]
        raise ValueError(
            "subject-level split violation (#27): "
            f"{len(overlap)} subjects appear in both partitions, e.g. {sample}"
        )


class SyntheticBatchProvider:
    """Seeded synthetic provider for one task ('tcm' or 'physio').

    The full subject manifest is exposed via `subject_ids`; each batch covers
    a disjoint consecutive slice. Distinct providers must use distinct seeds
    or subject prefixes or the split assertion will (correctly) fail.
    """

    def __init__(
        self,
        task: str,
        n_batches: int = 4,
        batch_size: int = 8,
        seed: int = 0,
        *,
        subject_prefix: str | None = None,
        with_targets: bool = True,
    ) -> None:
        if task not in ("tcm", "physio"):
            raise ValueError(f"unknown task {task!r}; expected 'tcm' or 'physio'")
        if n_batches < 1 or batch_size < 1:
            raise ValueError("n_batches and batch_size must be >= 1")
        self.task = task
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.seed = seed
        self.with_targets = with_targets
        prefix = subject_prefix or f"synthetic-{task}-s{seed}"
        self.subject_ids = tuple(
            f"{prefix}-{i}" for i in range(n_batches * batch_size)
        )

    def input_spec(self) -> InputSpecSet:
        if self.task == "tcm":
            return make_tcm_input_spec(BODY_DIM, PHYSIO_DIM)
        return make_physio_input_spec(BODY_DIM, PHYSIO_DIM)

    def __iter__(self) -> Iterator[BatchItem]:
        g = torch.Generator().manual_seed(self.seed)
        for idx in range(self.n_batches):
            lo, hi = idx * self.batch_size, (idx + 1) * self.batch_size
            n = self.batch_size
            features = self._features(n, g)
            targets = target_mask = None
            if self.with_targets:
                targets, target_mask = self._targets(features, g)
            yield BatchItem(
                batch=ModelBatch(
                    features=features,
                    targets=targets,
                    target_mask=target_mask,
                ),
                subject_ids=list(self.subject_ids[lo:hi]),
            )

    def _features(self, n: int, g: torch.Generator) -> dict[str, torch.Tensor]:
        if self.task == "tcm":
            return {
                "body_features": torch.rand(n, BODY_DIM, generator=g),
                "physio_features": torch.rand(n, PHYSIO_DIM, generator=g),
            }
        return {
            "body_pre": torch.rand(n, BODY_DIM, generator=g),
            "physio_pre": torch.rand(n, PHYSIO_DIM, generator=g),
        }

    def _targets(
        self, features: dict[str, torch.Tensor], g: torch.Generator
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if self.task == "tcm":
            # Deterministic, learnable from body features (NOT real labels;
            # exists so harness candidates differ measurably on smoke runs).
            body = features["body_features"]
            idx = torch.arange(NUM_SCORES) % body.shape[1]
            labels = (body[:, idx] > 0.5).float()
            n = labels.shape[0]
            return (
                {"category_labels": labels},
                {"category_labels": torch.rand(n, NUM_SCORES, generator=g) > 0.3},
            )
        n = features["body_pre"].shape[0]
        return (
            {
                "efri": torch.randn(n, 1, generator=g),
                "bpsi": torch.rand(n, 1, generator=g) * 100.0,
            },
            {
                "efri": torch.ones(n, 1, dtype=torch.bool),
                "bpsi": torch.rand(n, 1, generator=g) > 0.5,
            },
        )
