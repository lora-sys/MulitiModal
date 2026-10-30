"""Wire the frozen baseline review eval (evals/review-eval-50.md) into pytest.

Every check in scripts/review_eval.py runs here as well, so CI regresses the
review findings even when the eval script is not invoked manually.
"""

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import review_eval  # noqa: E402


@pytest.mark.synthetic
@pytest.mark.parametrize(
    "check_fn",
    review_eval.CHECKS,
    ids=[fn.__name__ for fn in review_eval.CHECKS],
)
def test_review_eval_check(check_fn):
    check_fn()
