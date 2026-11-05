"""GPU candidate: hashed character n-gram embedding bag on CUDA.

The fixed runner supplies only training labels and validation texts. It owns
the time limit and evaluates the returned class IDs outside the candidate.
This candidate sees no repository files: the vocabulary, label mapping and
class weights are all derived from the supplied training fold alone.
"""
from __future__ import annotations

import argparse
import json
import time
import zlib
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

BUCKETS = 1 << 18
MAX_NGRAM = 4
DIM = 128
BATCH = 256
EPOCHS = 12
TIME_LIMIT = 210.0


def encode(text: str) -> list[int]:
    """Stable hashed 1-4 character n-grams (crc32 is not randomized)."""
    ids = []
    for n in range(1, MAX_NGRAM + 1):
        for i in range(len(text) - n + 1):
            gram = text[i:i + n].encode("utf-8")
            ids.append(zlib.crc32(gram) & (BUCKETS - 1))
    return ids or [0]


def pack(texts: list[str]):
    """Flatten all n-gram IDs and return GPU tensors plus CPU boundaries."""
    flat: list[int] = []
    offsets = [0]
    for text in texts:
        flat.extend(encode(text))
        offsets.append(len(flat))
    ids = torch.tensor(flat, dtype=torch.long, device="cuda")
    offs = torch.tensor(offsets[:-1], dtype=torch.long, device="cuda")
    return ids, offs, offsets


class BagClassifier(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.embedding = nn.EmbeddingBag(BUCKETS, DIM, mode="mean")
        self.head = nn.Linear(DIM, n_classes)

    def forward(self, ids: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return self.head(self.embedding(ids, offsets))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--predict", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    started = time.monotonic()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    training = json.loads(Path(args.train).read_text(encoding="utf-8"))
    texts = json.loads(Path(args.predict).read_text(encoding="utf-8"))

    # Shuffle once, then pack contiguous rows: each batch slice and its
    # offsets always refer to exactly the same documents.
    order = torch.randperm(len(training), generator=torch.Generator().manual_seed(args.seed))
    training = [training[i] for i in order.tolist()]
    classes = sorted({row["label"] for row in training})
    class_to_index = {label: index for index, label in enumerate(classes)}
    labels = [class_to_index[row["label"]] for row in training]
    counts = Counter(labels)
    weights = torch.tensor(
        [min(len(labels) / (len(classes) * counts[i]), 100.0)
         for i in range(len(classes))],
        dtype=torch.float32, device="cuda",
    )

    train_ids, train_offs, boundaries = pack([row["text"] for row in training])
    y = torch.tensor(labels, dtype=torch.long, device="cuda")
    model = BagClassifier(len(classes)).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    batch_ranges = [(start, min(start + BATCH, len(labels)))
                    for start in range(0, len(labels), BATCH)]
    epoch_generator = torch.Generator().manual_seed(args.seed + 1)

    model.train()
    for _ in range(EPOCHS):
        if time.monotonic() - started > TIME_LIMIT:
            break
        for batch in torch.randperm(len(batch_ranges), generator=epoch_generator).tolist():
            start, end = batch_ranges[batch]
            lo, hi = boundaries[start], boundaries[end]
            batch_ids = train_ids[lo:hi]
            batch_offsets = train_offs[start:end] - lo
            logits = model(batch_ids, batch_offsets)
            loss = F.cross_entropy(logits, y[start:end], weight=weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    del train_ids, train_offs, y
    torch.cuda.empty_cache()
    model.eval()
    predictions: list[int] = []
    predict_ids, predict_offs, predict_boundaries = pack(texts)
    with torch.no_grad():
        for start in range(0, len(texts), BATCH):
            end = min(start + BATCH, len(texts))
            lo, hi = predict_boundaries[start], predict_boundaries[end]
            batch_ids = predict_ids[lo:hi]
            batch_offsets = predict_offs[start:end] - lo
            scores = model(batch_ids, batch_offsets)
            predictions.extend(classes[i] for i in scores.argmax(dim=1).tolist())
    Path(args.output).write_text(json.dumps(predictions), encoding="utf-8")


if __name__ == "__main__":
    main()
