"""ML lifecycle engineering layer (#104-#112).

Contracts, incremental data pipeline, source capability, features/targets,
train-ready preparation, artifact release, gateway and observability. Pure
engineering: no real training, no auto-promotion, no clinical claims.

Importing this package must not require torch, sklearn or the private dataset.
"""

from __future__ import annotations

__all__ = [
    "acceptance",
    "contracts",
    "features",
    "gateway",
    "observe",
    "pipeline",
    "release",
    "sources",
    "training",
]
