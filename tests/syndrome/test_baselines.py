"""基线契约：多数类 Dummy 与 TF-IDF char 1–3 + LR，合成数据上的可复现性。"""

from __future__ import annotations

import numpy as np
import pytest

from syndrome.baselines import (
    MAJORITY_BASELINE_NAME,
    NGRAM_BASELINE_NAME,
    MostFrequentBaseline,
    TfidfNgramBaseline,
    TfidfNgramConfig,
    environment_record,
)

pytestmark = pytest.mark.synthetic


@pytest.fixture(scope="module")
def small_corpus(make_corpus):
    return make_corpus(n_per_class=8, seed=3)


def test_dummy_majority(small_corpus):
    texts, labels = small_corpus
    dummy = MostFrequentBaseline().fit(texts, labels)
    preds = dummy.predict(["任意输入", ""])
    assert len(preds) == 2 and len(set(preds)) == 1
    probs = dummy.predict_proba(["x"])
    assert probs.shape == (1, max(labels) + 1)
    assert probs[0].sum() == pytest.approx(1.0)


def test_dummy_unfitted_raises():
    with pytest.raises(RuntimeError):
        MostFrequentBaseline().predict(["x"])


def test_tfidf_learns_synthetic_markers(small_corpus):
    """合成语料的类别标记词应可被 TF-IDF 基线完全学会（下限保护）。"""
    texts, labels = small_corpus
    clf = TfidfNgramBaseline(TfidfNgramConfig(seed=7)).fit(texts, labels)
    preds = clf.predict(texts)
    acc = sum(p == y for p, y in zip(preds, labels, strict=False)) / len(labels)
    assert acc > 0.9


def test_tfidf_reproducible_with_fixed_seed(small_corpus):
    texts, labels = small_corpus
    a = TfidfNgramBaseline(TfidfNgramConfig(seed=11)).fit(texts, labels)
    b = TfidfNgramBaseline(TfidfNgramConfig(seed=11)).fit(texts, labels)
    assert a.predict(texts) == b.predict(texts)
    assert np.allclose(a.predict_proba(texts), b.predict_proba(texts))


def test_tfidf_probability_rows_sum_to_one(small_corpus):
    texts, labels = small_corpus
    clf = TfidfNgramBaseline().fit(texts, labels)
    probs = clf.predict_proba(texts[:5])
    assert probs.shape == (5, 5)
    assert all(abs(row.sum() - 1.0) < 1e-6 for row in probs)


def test_environment_record_documents_versions():
    rec = environment_record(seed=20261101)
    assert rec["seed"] == 20261101
    assert rec["sklearn_version"]
    assert rec["python_version"]


def test_baseline_names_stable():
    assert MAJORITY_BASELINE_NAME == "dummy-most-frequent"
    assert NGRAM_BASELINE_NAME == "tfidf-char-1-3-logreg"
