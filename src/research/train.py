"""Editable TCM-SD candidate: fit on labelled texts and write predictions.

The fixed runner supplies only training labels and validation texts. It owns
the time limit and evaluates the returned class IDs outside the candidate.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from sklearn import set_config
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--predict", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    training = json.loads(Path(args.train).read_text(encoding="utf-8"))
    texts = json.loads(Path(args.predict).read_text(encoding="utf-8"))
    labels = [row["label"] for row in training]

    # Half-strength inverse-frequency weights: between class_weight=None and
    # "balanced", normalized so the mean weight stays 1 (comparable C).
    counts = Counter(labels)
    n, k = len(labels), len(counts)
    raw = [(n / (k * counts[label])) ** 0.5 for label in labels]
    scale = n / sum(raw)
    sample_weight = [value * scale for value in raw]

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), min_df=1,
                                 binary=True, sublinear_tf=True)
    train_x = vectorizer.fit_transform([row["text"] for row in training])
    predict_x = vectorizer.transform(texts)
    set_config(enable_metadata_routing=True)
    classifier = OneVsRestClassifier(LogisticRegression(
        solver="liblinear", C=10.0, max_iter=2000,
        random_state=args.seed,
    ).set_fit_request(sample_weight=True))
    classifier.fit(train_x, labels, sample_weight=sample_weight)
    predictions = classifier.predict(predict_x).tolist()
    Path(args.output).write_text(json.dumps(predictions), encoding="utf-8")


if __name__ == "__main__":
    main()
