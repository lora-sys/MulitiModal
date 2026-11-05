"""Editable TCM-SD candidate: fit on labelled texts and write predictions.

The fixed runner supplies only training labels and validation texts. It owns
the time limit and evaluates the returned class IDs outside the candidate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--predict", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    training = json.loads(Path(args.train).read_text(encoding="utf-8"))
    texts = json.loads(Path(args.predict).read_text(encoding="utf-8"))
    model = make_pipeline(
        TfidfVectorizer(analyzer="char", ngram_range=(1, 4), min_df=1,
                        binary=True, sublinear_tf=True),
        RidgeClassifier(alpha=1.0, class_weight="balanced",
                        random_state=args.seed),
    )
    model.fit([row["text"] for row in training], [row["label"] for row in training])
    predictions = model.predict(texts).tolist()
    Path(args.output).write_text(json.dumps(predictions), encoding="utf-8")


if __name__ == "__main__":
    main()
