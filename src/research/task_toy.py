"""A TCM-free example task: the engine must run without MulitiModal's data.

Why this exists: the research engine was written against two real tasks whose
data lives on a dev host. That makes "does the engine work?" and "does the
project's data work?" indistinguishable. This adapter generates its own small,
deterministic, clearly-synthetic dataset — short complaint-like strings plus a
few numeric fields, three classes — and implements every seam the engine needs,
so `autoresearch run --task toy` proves the loop, the Doctor, the isolation
chain, the dashboard and the Skill with no TCM data, no network and no GPU.

What it deliberately reuses: nothing task-specific from the two real adapters.
It brings its own tiny record format, its own user-level split, its own scaler,
its own providers and its own trusted evaluator, because the point is to show
that the engine's task seams are real seams rather than a thin alias of the
constitution task. The shared code it does use is the generic engine itself
(`frozen.selection_fold_ids`, the isolation/spawn machinery, the manifest).

Honesty about this task: the labels come from a scripted generator, so a rising
metric here says the plumbing works. It is `SMOKE_ONLY` and must never be
reported as evidence about any medical task.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

from research.tasks import TaskAdapter

TOY_TASK = "toy"
TOY_TASK_TYPE = "toy_classification"
TOY_MANIFEST_VERSION = "toy-manifest-v1"
TOY_LABEL_TIER = "SMOKE_ONLY"
PRIMARY_METRIC_TOY = "toy/macro_f1"
N_CLASSES = 3
BUCKETS = 64

#: modality -> dim; "toy_bucket" is the hashed short text, "toy_numeric" the
#: measured fields. Two modalities on purpose: the fusion path is exercised.
MODALITY_DIMS = {"toy_bucket": BUCKETS, "toy_numeric": 4}
SCALED_MODALITIES = ("toy_numeric",)

#: the generator's vocabulary. Class meaning lives in the first token, so the
#: task is learnable from the hashed text and not from row order.
MEANING_TOKENS = ("ache", "dizzy", "fatigue")
NOISE_TOKENS = ("slight", "again", "worse", "morning", "evening", "after", "meal")


@dataclass
class ToyRecord:
    subject_index: int
    user_id: str
    text: str
    numeric: list[float]
    label: int | None
    label_mask: bool

    @property
    def features(self) -> dict[str, list[float]]:
        return {"toy_bucket": _bucketise(self.text), "toy_numeric": list(self.numeric)}


def _bucketise(text: str) -> list[float]:
    """Deterministic bag-of-tokens, L2-normalized: no state, no fitting."""
    vector = [0.0] * BUCKETS
    tokens = [t for t in text.lower().split() if t]
    for token in tokens or ["<empty>"]:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        vector[int.from_bytes(digest[:4], "big") % BUCKETS] += 1.0
    norm = sum(v * v for v in vector) ** 0.5 or 1.0
    return [v / norm for v in vector]


def generate_records(n_subjects: int = 90, seed: int = 20261105) -> list[ToyRecord]:
    """A deterministic, learnable, deliberately imperfect toy dataset.

    Class 0/1 dominate and follow the meaning token; class 2 is rare (~10%) and
    its token is ambiguous, so macro-F1 has a tail class to be honest about.
    """
    rng = random.Random(seed)
    records: list[ToyRecord] = []
    for i in range(n_subjects):
        roll = rng.random()
        if roll < 0.45:
            meaning = MEANING_TOKENS[0]
            label = 0
        elif roll < 0.90:
            meaning = MEANING_TOKENS[1]
            label = 1
        else:
            meaning = MEANING_TOKENS[2]
            label = 2
        # 12% label noise: a text whose meaning says one thing and whose label
        # says another, which is what a real annotation set looks like
        if rng.random() < 0.12:
            label = rng.randrange(N_CLASSES)
        # a per-case token keeps every text unique: without it the small
        # vocabulary made 4 cases collide across train and dev, which the Doctor
        # (correctly) refused. The token carries no signal — dev ids are unseen —
        # so it only removes accidental duplicates.
        words = [meaning, f"case{i:03d}"] + [
            rng.choice(NOISE_TOKENS) for _ in range(rng.randint(1, 3))
        ]
        rng.shuffle(words)
        numeric = [
            round(rng.uniform(0.0, 1.0), 4) + (0.4 if label == 1 else 0.0),
            round(rng.uniform(0.0, 1.0), 4),
            round(rng.uniform(0.0, 1.0), 4) + (0.4 if label == 2 else 0.0),
            round(rng.uniform(0.0, 1.0), 4),
        ]
        records.append(ToyRecord(
            subject_index=i, user_id=f"u{i:03d}", text=" ".join(words),
            numeric=numeric, label=label, label_mask=True,
        ))
    return records


# -- serialisation --------------------------------------------------------------


def records_to_json(records: list[ToyRecord]) -> str:
    return json.dumps([
        {"subject_index": r.subject_index, "user_id": r.user_id, "text": r.text,
         "numeric": list(r.numeric), "label": r.label, "label_mask": r.label_mask}
        for r in records
    ], ensure_ascii=False)


def records_from_json(text: str) -> list[ToyRecord]:
    return [
        ToyRecord(subject_index=int(row["subject_index"]), user_id=str(row["user_id"]),
                  text=str(row["text"]), numeric=[float(v) for v in row["numeric"]],
                  label=None if row["label"] is None else int(row["label"]),
                  label_mask=bool(row["label_mask"]))
        for row in json.loads(text)
    ]


# -- manifest -------------------------------------------------------------------


def build_manifest(records: list[ToyRecord], *, seed: int,
                   fractions: tuple[float, float, float] = (0.6, 0.2, 0.2)) -> dict:
    """User-level split: every record of one user lands in exactly one partition."""
    users = sorted({r.user_id for r in records})
    rng = random.Random(seed)
    rng.shuffle(users)
    n_train = int(len(users) * fractions[0])
    n_dev = int(len(users) * fractions[1])
    partitions = {
        "train": users[:n_train],
        "dev": users[n_train:n_train + n_dev],
        "holdout": users[n_train + n_dev:],
    }
    by_user: dict[str, list[ToyRecord]] = {}
    for record in records:
        by_user.setdefault(record.user_id, []).append(record)
    split: dict[str, list[str]] = {}
    row_by_id: dict[str, int] = {}
    for name, ids in partitions.items():
        subject_ids: list[str] = []
        for user in ids:
            for record in by_user[user]:
                sid = f"{TOY_TASK}-r{record.subject_index:04d}"
                subject_ids.append(sid)
                row_by_id[sid] = record.subject_index
        split[name] = sorted(subject_ids)
    data = {
        "version": TOY_MANIFEST_VERSION,
        "dataset_id": f"{TOY_TASK}-synthetic",
        "dataset_sha256": hashlib.sha256(records_to_json(records).encode()).hexdigest(),
        "task": TOY_TASK,
        "task_type": TOY_TASK_TYPE,
        "label_tier": TOY_LABEL_TIER,
        "label_version": f"{TOY_TASK}-scripted-v1",
        "standard_version": "SYNTHETIC-CONTRACT-V0",
        "n_classes": N_CLASSES,
        "split": {**split, "seed": int(seed),
                  "fractions": list(fractions),
                  "user_ids": {name: partitions[name] for name in partitions}},
        "subject_row_by_id": row_by_id,
        "modalities": {
            name: {"dim": dim, "feature_groups": [name]}
            for name, dim in MODALITY_DIMS.items()
        },
        "modality_set": "toy",
        "scaler": _fit_scaler(records, _rows_for(
            {"subject_row_by_id": row_by_id}, split["train"])),
        "scaler_fit_rows": len(split["train"]),
        "excluded_subject_rows": [],
        "notes": "synthetic example task for issue #61; SMOKE_ONLY",
    }
    data["manifest_sha256"] = _digest(data)
    return data


def _digest(data: dict) -> str:
    body = {k: v for k, v in data.items() if k != "manifest_sha256"}
    blob = json.dumps(body, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def _fit_scaler(records: list[ToyRecord], train_rows: list[int]) -> dict:
    """Per-modality statistics from the named TRAIN rows only (#29).

    A hashed-bucket modality is already L2-normalized and has no fitted state,
    so it is declared degenerate-by-design (transform emits the value as-is);
    only the measured fields are standardized. A column with no train value or
    no variance is marked degenerate so the transform emits 0.0 rather than
    passing a raw magnitude through.
    """
    by_row = {r.subject_index: r for r in records}
    scaler: dict[str, dict] = {}
    for name, dim in MODALITY_DIMS.items():
        if name not in SCALED_MODALITIES:
            scaler[name] = {"mean": [0.0] * dim, "std": [1.0] * dim,
                            "degenerate": [True] * dim}
            continue
        means: list[float] = []
        stds: list[float] = []
        degenerate: list[bool] = []
        for j in range(dim):
            column = [by_row[row].numeric[j] for row in train_rows if row in by_row]
            if not column:
                means.append(0.0)
                stds.append(1.0)
                degenerate.append(True)
                continue
            mean = sum(column) / len(column)
            std = (sum((v - mean) ** 2 for v in column) / len(column)) ** 0.5
            degenerate.append(std < 1e-9)
            means.append(mean)
            stds.append(std if std >= 1e-9 else 1.0)
        scaler[name] = {"mean": means, "std": stds, "degenerate": degenerate}
    return scaler


def _rows_for(manifest: dict, subject_ids: list[str]) -> list[int]:
    mapping = manifest["subject_row_by_id"]
    return [mapping[sid] for sid in subject_ids if sid in mapping]


def fold_manifest(manifest: dict, records: list[ToyRecord], train_ids: list[str],
                  dev_ids: list[str], notes: str = "") -> ToyManifest:
    """One selection fold: scaler refit on this fold's train rows, holdout EMPTY."""
    data = json.loads(json.dumps(manifest))
    data.pop("manifest_sha256", None)
    data["split"] = {"train": list(train_ids), "dev": list(dev_ids), "holdout": [],
                     "seed": manifest["split"]["seed"],
                     "fractions": manifest["split"]["fractions"]}
    keep = set(train_ids) | set(dev_ids)
    data["subject_row_by_id"] = {k: v for k, v in manifest["subject_row_by_id"].items()
                                 if k in keep}
    # the fold must be subject-disjoint by construction, not by the caller's
    # good intentions: a leak here would make the fold incomprehensible rather
    # than merely wrong, so it is refused before any training happens
    user_by_row = {r.subject_index: r.user_id for r in records}
    train_users = {user_by_row[row] for row in _rows_for(manifest, train_ids)}
    dev_users = {user_by_row[row] for row in _rows_for(manifest, dev_ids)}
    crossing = sorted(train_users & dev_users)
    if crossing:
        raise ValueError(
            f"fold train/dev share subject(s) {crossing[:3]}; the toy task's "
            "folds are user-level, so this fold would score people it trained on"
        )
    data["scaler"] = _fit_scaler(records, _rows_for(manifest, train_ids))
    data["scaler_fit_rows"] = len(train_ids)
    data["selection_parent_manifest_sha256"] = manifest["manifest_sha256"]
    if notes:
        data["notes"] = notes
    data["manifest_sha256"] = _digest(data)
    return ToyManifest(data=data, sha256=data["manifest_sha256"])


def pool_manifest(manifest: dict, records: list[ToyRecord], notes: str = "") -> ToyManifest:
    """The selection pool (train + dev) for the single finalize refit."""
    data = json.loads(json.dumps(manifest))
    data.pop("manifest_sha256", None)
    data["split"] = {
        "train": sorted(list(manifest["split"]["train"]) + list(manifest["split"]["dev"])),
        "dev": [], "holdout": list(manifest["split"]["holdout"]),
        "seed": manifest["split"]["seed"], "fractions": manifest["split"]["fractions"],
    }
    data["scaler"] = _fit_scaler(records, _rows_for(data, data["split"]["train"]))
    data["scaler_fit_rows"] = len(data["split"]["train"])
    data["selection_parent_manifest_sha256"] = manifest["manifest_sha256"]
    if notes:
        data["notes"] = notes
    data["manifest_sha256"] = _digest(data)
    return ToyManifest(data=data, sha256=data["manifest_sha256"])


@dataclass
class ToyManifest:
    data: dict
    sha256: str

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
        return path


def load_manifest(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = data.pop("manifest_sha256", None)
    digest = _digest(data)
    if expected is not None and digest != expected:
        raise ValueError(
            f"manifest {path} failed its own digest check "
            f"({expected[:12]} != {digest[:12]}); refusing to train on it"
        )
    data["manifest_sha256"] = digest
    return data


# -- providers ------------------------------------------------------------------


class _ToyProvider:
    """Batches over one partition, scaling by the manifest's own statistics."""

    def __init__(self, manifest: dict, records: list[ToyRecord], partition: str,
                 batch_size: int = 32, seed: int = 0, shuffle: bool = False) -> None:
        self.manifest = manifest
        self.partition = partition
        mapping = manifest["subject_row_by_id"]
        self.subject_ids = tuple(manifest["split"][partition])
        by_row = {r.subject_index: r for r in records}
        rows = [mapping[sid] for sid in self.subject_ids]
        missing = [row for row in rows if row not in by_row]
        if missing:
            raise ValueError(f"manifest rows missing from records: {missing[:3]}")
        if not rows:
            raise ValueError(f"partition {partition!r} has no rows")
        self._rows = rows
        self._records = by_row
        self.batch_size = int(batch_size)
        self._seed = int(seed)
        self._shuffle = bool(shuffle)
        self._tensors = self._build_tensors()
        self._epoch = 0

    def _build_tensors(self) -> dict:
        import torch

        features: dict[str, list] = {name: [] for name in MODALITY_DIMS}
        availability: dict[str, list] = {name: [] for name in MODALITY_DIMS}
        labels: list[int] = []
        masks: list[bool] = []
        for row in self._rows:
            record = self._records[row]
            for name, dim in MODALITY_DIMS.items():
                raw = record.features[name]
                params = self.manifest["scaler"][name]
                scaled = [
                    0.0 if params["degenerate"][j]
                    else (raw[j] - params["mean"][j]) / params["std"][j]
                    for j in range(dim)
                ]
                features[name].append(scaled)
                availability[name].append(True)
            labels.append(-1 if record.label is None else int(record.label))
            masks.append(bool(record.label_mask))
        return {
            "features": {name: torch.tensor(v, dtype=torch.float32)
                         for name, v in features.items()},
            "availability": {name: torch.tensor(v, dtype=torch.bool)
                             for name, v in availability.items()},
            "labels": torch.tensor(labels, dtype=torch.long),
            "mask": torch.tensor(masks, dtype=torch.bool),
        }

    def __len__(self) -> int:
        return (len(self._rows) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        import torch

        from core.batch import ModelBatch
        from runtime.batches import BatchItem

        order = list(range(len(self._rows)))
        if self._shuffle:
            random.Random(self._seed + self._epoch).shuffle(order)
        self._epoch += 1
        tensors = self._tensors
        for start in range(0, len(order), self.batch_size):
            idx = order[start:start + self.batch_size]
            index = torch.tensor(idx, dtype=torch.long)
            yield BatchItem(
                batch=ModelBatch(
                    features={name: tensors["features"][name][index] for name in MODALITY_DIMS},
                    availability={name: tensors["availability"][name][index]
                                  for name in MODALITY_DIMS},
                    targets={"toy_labels": tensors["labels"][index]},
                    target_mask={"toy_labels": tensors["mask"][index]},
                ),
                subject_ids=[self.subject_ids[i] for i in idx],
            )


def input_spec(manifest: dict):
    from core.specs import InputSpecSet, ModalitySpec

    dims = ",".join(
        f"{name}={params['dim']}" for name, params in manifest["modalities"].items()
    )
    return InputSpecSet(
        version=f"toy-input-v1:{manifest['modality_set']}:{dims}",
        modalities=tuple(
            ModalitySpec(name=name, kind="tabular", dim=params["dim"],
                         units="standardized_schema",
                         description="synthetic example task (#61)")
            for name, params in manifest["modalities"].items()
        ),
    )


def training_fold_data(fit_manifest: dict, records: list[ToyRecord], config: dict):
    from research.frozen import FoldData

    train = _ToyProvider(fit_manifest, records, "train",
                         batch_size=int(config.get("train_batch_size", 32)),
                         seed=int(config.get("seed", 0)), shuffle=True)
    return FoldData(
        train=train, dev=None, input_spec=input_spec(fit_manifest),
        dims={"modalities": {n: p["dim"] for n, p in fit_manifest["modalities"].items()},
              "targets": int(fit_manifest.get("n_classes", N_CLASSES))},
        n_train_subjects=len(train.subject_ids), n_dev_subjects=0,
        seed=int(config.get("seed", 0)),
    )


def build_fold(config: dict, manifest_path, dataset_path, fold: int, folds: int,
               repeat: int, records=None):
    from research.frozen import FoldData, selection_fold_ids

    manifest = load_manifest(manifest_path)
    if records is None:
        records = records_from_json(Path(dataset_path).read_text(encoding="utf-8"))
    train_ids, dev_ids = selection_fold_ids(manifest, fold, folds, repeat)
    saved = fold_manifest(manifest, records, train_ids, dev_ids,
                          notes=f"selection fold {fold} repeat {repeat}")
    data = saved.data
    train = _ToyProvider(data, records, "train",
                         batch_size=int(config.get("train_batch_size", 32)),
                         seed=int(config.get("seed", 0)), shuffle=True)
    dev = _ToyProvider(data, records, "dev", batch_size=max(1, len(dev_ids)))
    fold_data = FoldData(
        train=train, dev=dev, input_spec=input_spec(data),
        dims={"modalities": {n: p["dim"] for n, p in data["modalities"].items()},
              "targets": int(data.get("n_classes", N_CLASSES))},
        n_train_subjects=len(train_ids), n_dev_subjects=len(dev_ids),
        seed=int(config.get("seed", 0)),
    )
    return fold_data, data


class ToyReport:
    """The trusted report for the toy task: macro-F1 over all three classes.

    A class with no observed rows is excluded (never invented as 0), and a class
    with no positive predictions has F1 0.0 rather than being dropped, so the
    denominator cannot drift between candidates.
    """

    def __init__(self, primary_value: float | None, *, macro_f1: float | None,
                 per_class: dict, n_subjects: int, n_labeled: int,
                 primary_metric: str = PRIMARY_METRIC_TOY) -> None:
        self.primary_metric = primary_metric
        self.macro_f1 = macro_f1
        self.per_class = per_class
        self.n_subjects = n_subjects
        self.n_labeled = n_labeled
        self.primary_value = primary_value

    def to_dict(self) -> dict:
        return {"primary_metric": self.primary_metric,
                "primary_value": self.primary_value,
                "macro_f1": self.macro_f1, "per_class": self.per_class,
                "n_subjects": self.n_subjects, "n_labeled": self.n_labeled,
                "label_tier": TOY_LABEL_TIER}


def score_model(model, provider) -> ToyReport:
    """Masked macro-F1 over classes with support; runs inside the evaluator child."""
    import torch

    preds: list[int] = []
    targets: list[int] = []
    model.eval()
    with torch.no_grad():
        for item in provider:
            out = model(item.batch)
            probs = out.predictions["toy_probs"]
            mask = item.batch.target_mask["toy_labels"]
            preds.extend(int(v) for v in torch.argmax(probs, dim=-1)[mask].tolist())
            targets.extend(int(v) for v in item.batch.targets["toy_labels"][mask].tolist())
    per_class: dict = {}
    f1s: list[float] = []
    for cls in range(N_CLASSES):
        tp = sum(1 for p, t in zip(preds, targets, strict=True)
                 if p == cls and t == cls)
        fp = sum(1 for p, t in zip(preds, targets, strict=True)
                 if p == cls and t != cls)
        fn = sum(1 for p, t in zip(preds, targets, strict=True)
                 if p != cls and t == cls)
        support = tp + fn
        if support == 0 and (tp + fp) == 0:
            per_class[str(cls)] = {"support": 0, "f1": None, "precision": None,
                                   "recall": None, "n_observed": 0}
            continue
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1s.append(f1)
        per_class[str(cls)] = {"support": support, "f1": f1,
                               "precision": precision, "recall": recall,
                               "n_observed": support}
    macro = sum(f1s) / len(f1s) if f1s else None
    return ToyReport(macro, macro_f1=macro, per_class=per_class,
                     n_subjects=len(set(provider.subject_ids)),
                     n_labeled=len(targets))


class ToyAdapter(TaskAdapter):
    """The example task's seams; every method is task-local by design."""

    name = TOY_TASK
    default_editable = "src/research/candidate_toy.py"
    has_text = True
    single_label = True

    @property
    def primary_metric(self) -> str:
        return PRIMARY_METRIC_TOY

    # -- doctor hooks -------------------------------------------------------

    def record_user_key(self, record):
        return getattr(record, "user_id", None)

    def record_text_key(self, record):
        return getattr(record, "text", None)

    def record_label_key(self, record):
        return getattr(record, "label", None)

    def record_has_label(self, record):
        return bool(getattr(record, "label_mask", False))

    def label_space(self, manifest):
        n = int((manifest or {}).get("n_classes", N_CLASSES))
        return [str(i) for i in range(n)]

    def metric_names(self):
        return (PRIMARY_METRIC_TOY,)

    # -- records -------------------------------------------------------------

    def load_manifest(self, path):
        return load_manifest(path)

    def load_records(self, dataset_path):
        return records_from_json(Path(dataset_path).read_text(encoding="utf-8"))

    def load_eval_records(self, config):
        data = config["data"]
        path = data.get("records_path") or data["dataset_path"]
        return records_from_json(Path(path).read_text(encoding="utf-8"))

    def records_from_json(self, text):
        return records_from_json(text)

    def records_to_json(self, records):
        return records_to_json(records)

    # -- manifests -----------------------------------------------------------

    def fold_manifest(self, manifest, records, train_ids, dev_ids, notes=""):
        return fold_manifest(manifest, records, train_ids, dev_ids, notes=notes)

    def pool_manifest(self, frozen, records, notes=""):
        return pool_manifest(frozen, records, notes=notes)

    # -- folds ---------------------------------------------------------------

    def build_fold(self, config, manifest_path, dataset_path, fold, folds,
                   repeat, records=None):
        return build_fold(config, manifest_path, dataset_path, fold, folds,
                          repeat, records=records)

    def training_fold_data(self, fit_manifest, records, config):
        return training_fold_data(fit_manifest, records, config)

    # -- scoring -------------------------------------------------------------

    def evaluate_model(self, model, fold_manifest, provider, partition):
        return score_model(model, provider)

    def score_holdout(self, config, request, records, model) -> dict:
        pool = load_manifest(config["data"]["pool_manifest_path"])
        frozen = load_manifest(config["data"]["frozen_manifest_path"])
        view = json.loads(json.dumps(pool))
        view["split"] = {**pool["split"], "holdout": list(frozen["split"]["holdout"])}
        view["subject_row_by_id"] = dict(frozen["subject_row_by_id"])
        view["scaler"] = dict(pool["scaler"])
        view.pop("manifest_sha256", None)
        view["manifest_sha256"] = _digest(view)
        provider = _ToyProvider(view, records, "holdout",
                                batch_size=max(1, len(view["split"]["holdout"])))
        report = score_model(model, provider)
        # a majority-class reference on the same view, so the number has a floor
        from collections import Counter

        train_rows = [view["subject_row_by_id"][sid] for sid in view["split"]["train"]]
        by_row = {r.subject_index: r for r in records}
        counts = Counter(by_row[r].label for r in train_rows
                         if by_row[r].label is not None)
        majority = counts.most_common(1)[0][0] if counts else 0
        holdout_labels = [by_row[view["subject_row_by_id"][sid]].label
                          for sid in view["split"]["holdout"]]
        hit = sum(1 for label in holdout_labels if label == majority)
        return {
            "holdout": report.to_dict(),
            "baselines": {
                "majority_class": {
                    "macro_f1": None,
                    "accuracy": (hit / len(holdout_labels)) if holdout_labels else None,
                    "majority_label": majority,
                }
            },
        }


def write_inputs(out_dir: str | Path, *, n_subjects: int = 90,
                 seed: int = 20261105) -> tuple[Path, Path]:
    """Materialize the toy dataset and its frozen manifest.

    The engine's subprocesses reload the data from a path, so the generated
    records are written once and then treated as read-only, exactly like a real
    dataset file would be. Nothing is downloaded and no TCM data is involved.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = generate_records(n_subjects=n_subjects, seed=seed)
    data_path = out / "toy_records.json"
    data_path.write_text(records_to_json(records), encoding="utf-8")
    manifest = build_manifest(records, seed=seed)
    manifest_path = out / "toy_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    return manifest_path, data_path