"""Editable synthetic example: predict a number from one input number."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MODEL = "mean"


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("train", "predict", "output", "seed"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    training = json.loads(Path(args.train).read_text(encoding="utf-8"))
    inputs = json.loads(Path(args.predict).read_text(encoding="utf-8"))
    x_mean = sum(item["x"] for item in training) / len(training)
    y_mean = sum(item["y"] for item in training) / len(training)
    if MODEL == "mean":
        slope = 0.0
    elif MODEL == "linear":
        denominator = sum((item["x"] - x_mean) ** 2 for item in training)
        slope = sum((item["x"] - x_mean) * (item["y"] - y_mean)
                    for item in training) / denominator
    else:
        raise ValueError("unknown model")
    predictions = [y_mean + slope * (value - x_mean) for value in inputs]
    Path(args.output).write_text(json.dumps(predictions), encoding="utf-8")


if __name__ == "__main__":
    main()
