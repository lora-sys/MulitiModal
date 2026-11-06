"""Fixed selection and scoring for the single-file AutoResearch engine.

The candidate (`src/research/train.py`) is executed by the same interpreter on
the development host. It receives only training labels and validation texts
through files; the metric is computed here, outside the candidate process.
Local results are development experiments (`LOCAL_EXPERIMENTAL`), not
independently attested evaluations.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
import os
import random
import re
import stat
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

SELECTION_FOLD_VERSION = "tcm-sd-stratified-user-fold-v2"
SELECTION_DATASET_SHA256 = "e63682908e6bed7488ada3cd68cf50b4a1817a4bfbfa28c0a00b74fbc0af4d55"
SELECTION_MANIFEST_SHA256 = "f6ac971a9224542551163d692d40ec095b78465dcdd4ea0a67e0bd168b98f04e"
EVIDENCE_LOCAL = "LOCAL_EXPERIMENTAL"
EVIDENCE_SMOKE = "SMOKE_ONLY"

#: results.tsv columns, in order. A run's rows must all agree on these.
FIELDS = ("commit", "metric", "value", "direction", "decision", "description",
          "peak_vram_mb", "protocol_sha256", "evidence", "promotion")

#: results.tsv stays the frozen decision history. Crashes and repair retries are
#: additionally recorded as immutable, append-only events in events.jsonl.
EVENT_SCHEMA = "autoresearch-event-v1"
EVENT_LOG_NAME = "events.jsonl"
EVENT_DECISIONS = ("KEEP", "REVERT", "CRASH")
MAX_EVENT_LOG_BYTES = 4 * 1024 * 1024
EVENT_KEYS = ("schema_version", "event_id", "candidate_id", "attempt", "commit",
              "decision", "evidence", "metric", "value", "protocol_sha256",
              "description", "repair_of", "result_path", "log_path", "results_path")
EVENT_IDENTITY_KEYS = ("schema_version", "event_id", "candidate_id", "attempt",
                       "commit", "decision", "evidence", "metric", "value",
                       "description", "repair_of")


def score_predictions(labels: list[int], predictions: list[int], n_classes: int) -> float:
    """Score class IDs outside the candidate process with the frozen metric."""
    from syndrome.metrics import macro_f1

    if (isinstance(n_classes, bool) or not isinstance(n_classes, int)
            or not 2 <= n_classes <= 148):
        raise ValueError("n_classes must be an integer from 2 to 148")
    if not labels or len(labels) != len(predictions):
        raise ValueError("labels and predictions must have the same nonzero length")
    for value in (*labels, *predictions):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < n_classes:
            raise ValueError("class ID is outside the frozen taxonomy")
    return macro_f1(labels, predictions, n_classes)


def fixed_code_hashes() -> dict[str, str]:
    """Identify the evaluator and metric implementation this run scores with."""
    import syndrome.metrics as metric_module

    paths = {"evaluator": __file__, "metric": metric_module.__file__}
    return {name: hashlib.sha256(read_regular(Path(path), 4 * 1024 * 1024)).hexdigest()
            for name, path in paths.items()}


def local_device() -> dict[str, str]:
    """Record the device the candidate actually trained on."""
    device = "cpu"
    gpu = "NOT_AVAILABLE"
    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
            gpu = torch.cuda.get_device_name(0)
    except Exception:  # pragma: no cover - torch is optional for linear candidates
        pass
    return {"device": device, "gpu": gpu}


def run_candidate(candidate: bytes, train: object, inputs: object, *,
                  budget_seconds: int, seed: int, repo_root: Path | None = None) -> list:
    """Run the candidate locally with a hard time limit; return class IDs.

    The candidate sees only the two JSON files it is handed: it runs in a
    private temporary directory, so it cannot reach the repository, the frozen
    manifest or the validation labels. Network use is not blocked, and the
    result is therefore a development experiment, not an attested one.
    """
    if not candidate or len(candidate) > 2 * 1024 * 1024:
        raise ValueError("candidate source must be between 1 byte and 2 MiB")
    if (isinstance(budget_seconds, bool) or not isinstance(budget_seconds, int)
            or not 1 <= budget_seconds <= 3600):
        raise ValueError("budget must be between 1 and 3600 seconds")
    with tempfile.TemporaryDirectory(prefix="autoresearch-candidate-") as directory:
        root = Path(directory)
        source = root / "train.py"
        source.write_bytes(candidate)
        for name, value in (("train.json", train), ("predict.json", inputs)):
            payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
            if len(payload) > 128 * 1024 * 1024:
                raise ValueError("candidate input exceeds the size limit")
            (root / name).write_bytes(payload)
        output = root / "output.json"
        command = [
            sys.executable, "-I", str(source),
            "--train", str(root / "train.json"),
            "--predict", str(root / "predict.json"),
            "--output", str(output),
            "--seed", str(seed),
        ]
        try:
            completed = subprocess.run(
                command, cwd=root, stdin=subprocess.DEVNULL,
                capture_output=True, text=True, timeout=budget_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"candidate exceeded the {budget_seconds}s fold budget"
            ) from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()[-400:]
            raise RuntimeError(
                f"candidate failed with exit code {completed.returncode}: {detail}"
            )
        predictions = json.loads(output.read_text(encoding="utf-8"))
    if not isinstance(predictions, list):
        raise ValueError("candidate predictions must be a JSON list")
    return predictions


# -- results.tsv and committed-source helpers --------------------------------


def validate_manifest_contract(
    task: str, manifest: dict, *, selection_pool: bool = False,
) -> None:
    """Validate the task-specific, hash-checked frozen manifest shape.

    The adapter loader verifies the canonical digest. This second check makes
    the task routing contract explicit before any unlock event can be charged.
    """
    import re

    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    for name in ("manifest_sha256", "dataset_sha256"):
        value = manifest.get(name)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"manifest {name} must be a lowercase SHA-256")
    modalities = manifest.get("modalities")
    if not isinstance(modalities, dict) or not modalities:
        raise ValueError("manifest modalities must be a non-empty object")
    split = manifest.get("split")
    row_map = manifest.get("subject_row_by_id")
    if not isinstance(split, dict) or not isinstance(row_map, dict) or not row_map:
        raise ValueError("manifest must contain split and subject_row_by_id objects")
    if selection_pool:
        split_seed = split.get("seed")
        if (
            isinstance(split_seed, bool)
            or not isinstance(split_seed, int)
            or split_seed < 0
        ):
            raise ValueError(
                "selection manifest split seed must be a non-negative integer"
            )
    if any(
        not isinstance(subject_id, str) or not subject_id
        or isinstance(row, bool) or not isinstance(row, int) or row < 1
        for subject_id, row in row_map.items()
    ):
        raise ValueError("manifest subject_row_by_id has an invalid subject or row")
    if len(set(row_map.values())) != len(row_map):
        raise ValueError("manifest subject_row_by_id rows must be unique")
    partitions = []
    for name in ("train", "dev", "holdout"):
        values = split.get(name)
        empty_pool_partition = selection_pool and name in ("dev", "holdout")
        if (
            not isinstance(values, list)
            or (not values and not empty_pool_partition)
        ):
            qualifier = "a list" if empty_pool_partition else "a non-empty list"
            raise ValueError(f"manifest split {name} must be {qualifier}")
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError(f"manifest split {name} contains an invalid subject id")
        if len(values) != len(set(values)):
            raise ValueError(f"manifest split {name} contains duplicate subjects")
        partitions.append(set(values))
    if any(partitions[i] & partitions[j]
           for i in range(len(partitions)) for j in range(i + 1, len(partitions))):
        raise ValueError("manifest subject partitions overlap")
    if set.union(*partitions) != set(row_map):
        raise ValueError("manifest partitions do not match subject_row_by_id")
    for name, modality in modalities.items():
        if not isinstance(name, str) or not name or not isinstance(modality, dict):
            raise ValueError("manifest modality entries must be named objects")
        dim = modality.get("dim")
        if isinstance(dim, bool) or not isinstance(dim, int) or dim < 1:
            raise ValueError(f"manifest modality {name!r} has an invalid dim")
    if task == "syndrome":
        if (
            manifest.get("task") != "syndrome"
            or manifest.get("task_type") != "syndrome_classification"
        ):
            raise ValueError("syndrome task requires a syndrome classification manifest")
        n_classes = manifest.get("n_classes")
        if isinstance(n_classes, bool) or not isinstance(n_classes, int) or n_classes < 2:
            raise ValueError("syndrome manifest n_classes must be an integer >= 2")
        taxonomy_version = manifest.get("label_taxonomy_version")
        if not isinstance(taxonomy_version, str) or not taxonomy_version:
            raise ValueError("syndrome manifest requires label_taxonomy_version")
        label_tier = manifest.get("label_tier")
        if "text" not in modalities or not isinstance(label_tier, str):
            raise ValueError("syndrome manifest requires the text modality and label_tier")
        if label_tier == "PUBLISHED" and not selection_pool:
            subject_users = manifest.get("subject_user_by_id")
            user_partitions = manifest.get("user_partition_by_id")
            user_strata = manifest.get("user_stratum_by_id")
            if not all(
                isinstance(value, dict)
                for value in (subject_users, user_partitions, user_strata)
            ):
                raise ValueError(
                    "PUBLISHED syndrome manifest requires frozen user strata "
                    "and partition metadata"
                )
            if set(subject_users) != set(row_map):
                raise ValueError(
                    "PUBLISHED syndrome manifest requires complete subject-to-user mapping"
                )
            if any(
                not isinstance(user_id, str) or not user_id
                for user_id in subject_users.values()
            ):
                raise ValueError("PUBLISHED syndrome manifest has an invalid user mapping")
            if set(user_strata) != set(user_partitions):
                raise ValueError(
                    "PUBLISHED syndrome manifest frozen user strata do not "
                    "cover frozen partitions"
                )
            if any(
                not isinstance(user_id, str) or not user_id
                for user_id in user_partitions
            ) or any(
                not isinstance(stratum, str) or not stratum
                for stratum in user_strata.values()
            ):
                raise ValueError(
                    "PUBLISHED syndrome manifest has invalid user strata metadata"
                )
            if not set(subject_users.values()) <= set(user_partitions):
                raise ValueError(
                    "PUBLISHED syndrome subject mapping references an unfrozen user"
                )
            expected_parts = {"train": "train", "dev": "dev", "holdout": "holdout"}
            for part, user_part in expected_parts.items():
                if any(
                    user_partitions[subject_users[subject_id]] != user_part
                    for subject_id in split[part]
                ):
                    raise ValueError(
                        f"PUBLISHED syndrome user partitions disagree with split {part!r}"
                    )
    else:
        raise ValueError(f"unknown research task {task!r}")



def _read_regular_file_no_follow(path: str | Path, label: str) -> bytes:
    """Read a regular file through a checked, no-follow descriptor."""
    import os
    import stat

    if not hasattr(os, "O_NOFOLLOW"):
        raise OSError("O_NOFOLLOW is required for selection artifact reads")
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        limit = 256 * 1024 * 1024
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError(f"{label} must be a bounded regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            content = stream.read(limit + 1)
        if len(content) > limit:
            raise ValueError(f"{label} exceeds the size limit")
        return content
    finally:
        os.close(descriptor)


def read_verified_dataset_bytes(
    path: str | Path, expected_sha256: str, label: str = "dataset",
) -> bytes:
    """Read the exact regular-file bytes bound by a frozen manifest."""
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(char not in "0123456789abcdef" for char in expected_sha256)
    ):
        raise ValueError(f"{label} manifest SHA-256 is malformed")
    payload = _read_regular_file_no_follow(path, label)
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 does not match frozen manifest dataset_sha256"
        )
    return payload


def _validate_selection_manifest(task: str, manifest: dict) -> None:
    validate_manifest_contract(task, manifest, selection_pool=True)
    if manifest["split"]["holdout"] != []:
        raise ValueError("selection manifest must have an empty holdout partition")
    if task != "syndrome":
        return

    row_map = manifest["subject_row_by_id"]
    subject_users = manifest.get("subject_user_by_id")
    user_partitions = manifest.get("user_partition_by_id")
    user_strata = manifest.get("user_stratum_by_id")
    if not all(
        isinstance(value, dict)
        for value in (subject_users, user_partitions, user_strata)
    ):
        raise ValueError("selection manifest lacks frozen pool user metadata")
    if set(subject_users) != set(row_map):
        raise ValueError("selection subject-user mapping does not match selection rows")
    if set(user_partitions) != set(user_strata):
        raise ValueError("selection user partitions and strata have different users")
    frozen_users = set(user_partitions)
    present_users = set(subject_users.values())
    if not present_users <= frozen_users:
        raise ValueError("selection rows reference an unfrozen user")
    if any(
        not isinstance(user_id, str) or not user_id for user_id in frozen_users
    ):
        raise ValueError("selection manifest contains an invalid frozen user")
    if any(
        partition not in {"train", "dev"}
        for partition in user_partitions.values()
    ):
        raise ValueError("selection user metadata must not contain holdout users")
    if any(
        not isinstance(stratum, str) or not stratum
        for stratum in user_strata.values()
    ):
        raise ValueError("selection manifest contains an invalid frozen user stratum")
    for partition in ("train", "dev"):
        if any(
            user_partitions[subject_users[subject_id]] != partition
            for subject_id in manifest["split"][partition]
        ):
            raise ValueError(
                f"selection user partitions disagree with split {partition!r}"
            )
    for name in (
        "selection_parent_manifest_sha256",
        "selection_parent_dataset_sha256",
    ):
        value = manifest.get(name)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
        ):
            raise ValueError(f"selection manifest lacks a valid {name}")


def _decode_selection_manifest(payload: bytes, task: str) -> dict:
    manifest = json.loads(payload.decode("utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("selection manifest must be a JSON object")
    expected = manifest.pop("manifest_sha256", None)
    actual = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    if expected != actual:
        raise ValueError(
            f"manifest hash mismatch: expected {expected}, computed {actual}; refusing"
        )
    manifest["manifest_sha256"] = expected
    _validate_selection_manifest(task, manifest)
    return manifest


def _validate_selection_records(task: str, manifest: dict, records: list) -> None:
    expected_rows = set(manifest["subject_row_by_id"].values())
    actual_rows = [record.subject_index for record in records]
    if len(actual_rows) != len(set(actual_rows)):
        raise ValueError("selection dataset contains duplicate subject rows")
    if set(actual_rows) != expected_rows:
        raise ValueError("selection dataset rows do not match the selection manifest")
    if task == "syndrome":
        subject_by_row = {
            row: subject_id
            for subject_id, row in manifest["subject_row_by_id"].items()
        }
        user_by_subject = manifest["subject_user_by_id"]
        for record in records:
            subject_id = subject_by_row[record.subject_index]
            if record.user_id != user_by_subject[subject_id]:
                raise ValueError("selection record user does not match manifest")



def selection_fold_ids(
    manifest: dict, fold: int, folds: int, repeat: int
) -> tuple[list[str], list[str]]:
    """Derive one selection fold's (train, dev) ids from the pool.

    The single derivation both the runner (which prepares the training
    subprocess's trimmed inputs) and the evaluator (which rebuilds the fold
    for scoring) must agree on: identical (manifest, fold, folds, repeat)
    yield identical partitions.

    Syndrome manifests carry the complete frozen user partition and a
    subject-to-user map. Their folds are assigned over that complete train +
    dev user set, then mapped to the rows still present in the manifest. Fold
    membership never changes when scenario filtering removes a user's last
    row. If that leaves a validation fold without records, selection fails
    closed instead of moving another user between folds.
    """
    if folds < 2 or fold < 0 or fold >= folds or repeat < 0:
        raise ValueError(
            f"invalid selection fold coordinates: fold={fold}, folds={folds}, "
            f"repeat={repeat}"
        )
    if manifest.get("task") == "syndrome":
        subject_rows = manifest.get("subject_row_by_id")
        subject_users = manifest.get("subject_user_by_id")
        user_partitions = manifest.get("user_partition_by_id")
        user_strata = manifest.get("user_stratum_by_id")
        if not all(
            isinstance(value, dict)
            for value in (
                subject_rows,
                subject_users,
                user_partitions,
                user_strata,
            )
        ):
            raise ValueError(
                "syndrome manifest lacks frozen user partition metadata"
            )
        subject_ids = set(subject_rows)
        if set(subject_users) != subject_ids:
            raise ValueError(
                "syndrome subject-to-user mapping does not match subject rows"
            )
        if set(user_strata) != set(user_partitions):
            raise ValueError(
                "syndrome frozen user strata do not cover frozen partitions"
            )
        if any(
            not isinstance(stratum, str) or not stratum
            for stratum in user_strata.values()
        ):
            raise ValueError("syndrome manifest has an invalid frozen user stratum")
        valid_parts = {"train", "dev", "holdout"}
        if not set(user_partitions.values()) <= valid_parts:
            raise ValueError("syndrome manifest has an unknown user partition")
        if not set(subject_users.values()) <= set(user_partitions):
            raise ValueError(
                "syndrome subject-to-user mapping references an unfrozen user"
            )
        expected_parts = {
            "train": "train",
            "dev": "dev",
            "holdout": "holdout",
        }
        for split_name, user_part in expected_parts.items():
            split_ids = set(manifest["split"][split_name])
            if not split_ids <= subject_ids:
                raise ValueError(
                    f"syndrome split {split_name!r} has unmapped subjects"
                )
            mismatched = [
                sid
                for sid in split_ids
                if user_partitions[subject_users[sid]] != user_part
            ]
            if mismatched:
                raise ValueError(
                    f"syndrome user partition disagrees with split "
                    f"{split_name!r}: {mismatched[:3]}"
                )

        pool_by_stratum: dict[str, list[str]] = {}
        for user_id, user_part in user_partitions.items():
            if user_part in {"train", "dev"}:
                pool_by_stratum.setdefault(user_strata[user_id], []).append(user_id)
        pool_users = [
            user_id for users in pool_by_stratum.values() for user_id in users
        ]
        pool_ids = sorted(
            set(manifest["split"]["train"]) | set(manifest["split"]["dev"])
        )
        pool_user_ids = {subject_users[subject_id] for subject_id in pool_ids}
        if len(pool_users) < folds:
            raise ValueError(
                f"too few frozen pool users ({len(pool_users)}) for {folds} folds"
            )
        if len(pool_user_ids) < folds:
            raise ValueError(
                f"selection fold has only {len(pool_user_ids)} users with "
                f"scenario records for {folds} folds; every fold needs "
                "validation records and a non-empty training set"
            )
        split_seed = manifest["split"]["seed"]
        if isinstance(split_seed, bool) or not isinstance(split_seed, int):
            raise ValueError("syndrome manifest has an invalid split seed")
        fold_of_user = {}
        for stratum in sorted(pool_by_stratum):
            users = sorted(pool_by_stratum[stratum])
            rng = random.Random(
                f"{split_seed}|{SELECTION_FOLD_VERSION}|{repeat}|{stratum}"
            )
            rng.shuffle(users)
            offset = rng.randrange(folds)
            for index, user_id in enumerate(users):
                fold_of_user[user_id] = (offset + index) % folds
        observed_folds = {
            fold_of_user[subject_users[subject_id]] for subject_id in pool_ids
        }
        empty_folds = sorted(set(range(folds)) - observed_folds)
        if empty_folds:
            raise ValueError(
                "frozen selection fold assignment has no observed users in "
                f"folds {empty_folds}; scenario filtering cannot rebalance users"
            )

        train_ids = [
            sid for sid in pool_ids
            if fold_of_user[subject_users[sid]] != fold
        ]
        dev_ids = [
            sid for sid in pool_ids
            if fold_of_user[subject_users[sid]] == fold
        ]
        return train_ids, dev_ids

    raise ValueError("only syndrome selection folds are supported")


def load_selection(manifest_path: Path, dataset_path: Path):
    """Read and validate a selection-only dataset before using any records."""
    manifest = _decode_selection_manifest(
        _read_regular_file_no_follow(manifest_path, "selection manifest"), "syndrome"
    )
    if (manifest.get("label_tier") != "PUBLISHED"
          or manifest.get("n_classes") != 148
          or manifest.get("label_taxonomy_version") != "tcm-sd-syndrome-vocab-v1"
          or manifest.get("modalities", {}).get("text", {}).get("kind") != "hashed_text"
          or manifest.get("manifest_sha256") != SELECTION_MANIFEST_SHA256
          or manifest.get("dataset_sha256") != SELECTION_DATASET_SHA256):
        raise ValueError("selection data do not match the frozen TCM-SD protocol")
    raw = read_verified_dataset_bytes(dataset_path, manifest["dataset_sha256"])
    records = json.loads(raw)
    if not isinstance(records, list):
        raise ValueError("selection records must be a list")
    parsed = []
    for item in records:
        if not isinstance(item, dict):
            raise ValueError("invalid selection record")
        row = item.get("subject_index")
        user = item.get("user_id")
        text = item.get("input_text")
        label = item.get("label")
        if (isinstance(row, bool) or not isinstance(row, int) or row < 1
                or not isinstance(user, str) or not user or not isinstance(text, str)):
            raise ValueError("invalid selection record identity or text")
        if label is not None and (isinstance(label, bool) or not isinstance(label, int)
                                  or not 0 <= label < manifest["n_classes"]):
            raise ValueError("selection label outside the frozen taxonomy")
        parsed.append(SimpleNamespace(
            subject_index=row, user_id=user, input_text=text, label=label,
        ))
    _validate_selection_records("syndrome", manifest, parsed)
    return manifest, {record.subject_index: record for record in parsed}


def synthetic_selection():
    """Construct a fixed, entirely artificial selection pool for smoke checks."""
    subjects = [f"s{i:02}" for i in range(24)]
    users = {subject: f"u{i // 2:02}" for i, subject in enumerate(subjects)}
    partitions = {user: "train" for user in users.values()}
    rows = {subject: i + 1 for i, subject in enumerate(subjects)}
    raw_records = [
        {"subject_index": row, "user_id": users[subject],
         "input_text": ("头痛" if row % 2 else "腹痛") + str(row),
         "label": row % 2, "provenance": {"source": "synthetic-autoresearch"}}
        for subject, row in rows.items()
    ]
    manifest = {
        "task": "syndrome", "task_type": "syndrome_classification",
        "label_tier": "SMOKE_ONLY", "label_taxonomy_version": "synthetic-v1",
        "n_classes": 2, "modalities": {"text": {"dim": 16, "kind": "hashed_text"}},
        "split": {"seed": 123, "train": subjects, "dev": [], "holdout": []},
        "subject_row_by_id": rows, "subject_user_by_id": users,
        "user_partition_by_id": partitions,
        "user_stratum_by_id": {user: str(i % 3) for i, user in enumerate(sorted(partitions))},
        "selection_parent_manifest_sha256": "0" * 64,
        "selection_parent_dataset_sha256": "1" * 64,
        "dataset_sha256": hashlib.sha256(
            json.dumps(raw_records, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    _validate_selection_manifest("syndrome", manifest)
    records = [SimpleNamespace(**record) for record in raw_records]
    _validate_selection_records("syndrome", manifest, records)
    return manifest, {record.subject_index: record for record in records}


def fold_inputs(manifest: dict, records: dict, fold: int, folds: int, repeat: int):
    """Use the frozen grouped fold and one-way exact-text dev deduplication."""
    train_ids, dev_ids = selection_fold_ids(manifest, fold, folds, repeat)
    rows = manifest["subject_row_by_id"]
    train = [records[rows[sid]] for sid in train_ids]
    train_texts = {record.input_text for record in train}
    dev = [records[rows[sid]] for sid in dev_ids
           if records[rows[sid]].input_text not in train_texts]
    train = [record for record in train if record.label is not None]
    dev = [record for record in dev if record.label is not None]
    if not train or not dev:
        raise ValueError("selection fold has no labelled training or validation records")
    return train, dev


def evaluate(manifest: dict, records: dict, candidate: bytes, *,
             budget_seconds: int, seed: int, folds: int, repeats: int,
             repo_root: Path | None = None, progress=None) -> dict:
    """Evaluate one candidate on fixed selection folds, never on holdout."""
    if manifest["split"]["holdout"]:
        raise ValueError("holdout must be empty for candidate evaluation")
    if (folds < 2 or folds > 10 or repeats < 1 or repeats > 10
            or folds * repeats > 32 or budget_seconds * folds * repeats > 14400):
        raise ValueError("fold count, repeats or total compute budget exceed the run limit")
    code_hashes = fixed_code_hashes()
    device = local_device()
    scores = []
    for repeat in range(repeats):
        for fold in range(folds):
            if progress:
                progress({"state": "TRAINING", "fold": fold + 1, "folds": folds,
                          "repeat": repeat + 1, "repeats": repeats,
                          "deadline": time.time() + budget_seconds + 45})
            train, dev = fold_inputs(manifest, records, fold, folds, repeat)
            training = [{"text": row.input_text, "label": row.label} for row in train]
            inputs = [row.input_text for row in dev]
            started = time.monotonic()
            predictions = run_candidate(candidate, training, inputs,
                                        budget_seconds=budget_seconds, seed=seed,
                                        repo_root=repo_root)
            elapsed = round(time.monotonic() - started, 3)
            score = score_predictions([row.label for row in dev], predictions,
                                      manifest["n_classes"])
            scores.append({"fold": fold, "repeat": repeat, "macro_f1": score,
                           "n_validation": len(dev), "train_seconds": elapsed})
    if fixed_code_hashes() != code_hashes:
        raise RuntimeError("fixed evaluator code changed during the run")
    values = [row["macro_f1"] for row in scores]
    smoke = manifest["label_tier"] == "SMOKE_ONLY"
    return {
        "metric": "syndrome/macro_f1",
        "direction": "maximize",
        "metric_version": "tcm-sd-eval-v3",
        "code_sha256": code_hashes,
        "promotion": "strict",
        "macro_f1": statistics.mean(values),
        "fold_std": statistics.pstdev(values),
        "fold_count": folds,
        "repeats": repeats,
        "folds": scores,
        "candidate_sha256": hashlib.sha256(candidate).hexdigest(),
        "dataset_sha256": manifest["dataset_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "seed": seed,
        "budget_seconds": budget_seconds,
        "device": device["device"],
        "gpu": device["gpu"],
        "peak_vram_mb": "NOT_AVAILABLE",
        "evidence": EVIDENCE_SMOKE if smoke else EVIDENCE_LOCAL,
    }


def read_regular(path: Path, limit: int) -> bytes:
    """Read a bounded regular file without following a final symlink."""
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError("input is not a bounded regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            content = stream.read(limit + 1)
        if len(content) > limit:
            raise ValueError("input exceeds the size limit")
        return content
    finally:
        os.close(descriptor)


def open_regular(path: Path):
    """Open a private output file without following symbolic or extra hard links."""
    flags = os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("output must be a regular file without extra hard links")
        return os.fdopen(descriptor, "r+", encoding="utf-8", newline="")
    except BaseException:
        os.close(descriptor)
        raise


def committed_source(repo: Path, commit: str, editable: str) -> bytes:
    """Read one bounded committed file, without interpreting a revision argument."""
    if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit):
        raise ValueError("commit must be a full hexadecimal Git commit ID")
    path = PurePosixPath(editable)
    if path.is_absolute() or ".." in path.parts or not path.parts or str(path) != editable:
        raise ValueError("editable must be a relative Git path")
    obj = f"{commit}:{editable}"
    size = subprocess.run(["git", "cat-file", "-s", obj], cwd=repo,
                          stdin=subprocess.DEVNULL, capture_output=True,
                          text=True, check=True, timeout=15).stdout.strip()
    if not size.isdigit() or not 0 < int(size) <= 2 * 1024 * 1024:
        raise ValueError("commit does not contain a bounded training file")
    return subprocess.run(["git", "show", obj], cwd=repo, stdin=subprocess.DEVNULL,
                          capture_output=True, check=True, timeout=15).stdout


def write_new_atomic(path: Path, payload: bytes) -> None:
    """Create a complete file atomically without overwriting an existing result."""
    temporary = path.parent / (".result-" + uuid.uuid4().hex)
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
    finally:
        temporary.unlink(missing_ok=True)


def history_evidence(path: Path) -> str | None:
    """Evidence level already used by this run (a crash has no result file)."""
    try:
        contents = read_regular(path, 2 * 1024 * 1024).decode("utf-8")
    except FileNotFoundError:
        return None
    rows = list(csv.DictReader(io.StringIO(contents), delimiter="\t", strict=True))
    levels = {row.get("evidence") for row in rows if row.get("evidence")}
    if len(levels) > 1:
        raise ValueError("results file mixes evidence levels")
    return next(iter(levels), None)


def resolved_evidence(results: Path, result: dict | None, explicit: str | None) -> str:
    """A failed LOCAL_EXPERIMENTAL attempt keeps its original evidence level."""
    if explicit:
        return explicit
    if isinstance(result, dict) and result.get("evidence") in (EVIDENCE_LOCAL, EVIDENCE_SMOKE):
        return str(result["evidence"])
    return str(history_evidence(results) or EVIDENCE_LOCAL)


def event_id_for(*, results: Path, candidate_id: str, attempt: int, commit: str,
                 decision: str, description: str, value: float | None, evidence: str,
                 repair_of: str | None) -> str:
    """Stable idempotency key: the same record command produces the same event."""
    identity = json.dumps(
        [str(results), candidate_id, attempt, commit, decision, description,
         value, evidence, repair_of],
        ensure_ascii=False, separators=(",", ":"), sort_keys=True, allow_nan=False)
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def append_event(path: Path, event: dict) -> bool:
    """Append one immutable audit event; an identical event_id is a no-op."""
    if not isinstance(event, dict) or any(key not in event for key in EVENT_KEYS):
        raise ValueError("audit event is missing required fields")
    repair_of = event["repair_of"]
    if (event["schema_version"] != EVENT_SCHEMA
            or not re.fullmatch(r"[0-9a-f]{64}", str(event["event_id"]))
            or not str(event["candidate_id"])
            or isinstance(event["attempt"], bool) or not isinstance(event["attempt"], int)
            or event["attempt"] < 1
            or event["decision"] not in EVENT_DECISIONS
            or event["evidence"] not in (EVIDENCE_LOCAL, EVIDENCE_SMOKE)
            or (repair_of is not None
                and not re.fullmatch(r"[0-9a-f]{64}", str(repair_of)))
            or not str(event["description"]).strip() or len(str(event["description"])) > 400):
        raise ValueError("invalid audit event")
    value = event["value"]
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                              or not math.isfinite(value)):
        raise ValueError("audit event has an invalid score")
    canonical = {key: event[key] for key in EVENT_KEYS}
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.parent / ("." + path.name + ".lock")
    with open_regular(lock_path) as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            contents = read_regular(path, MAX_EVENT_LOG_BYTES).decode("utf-8")
        except FileNotFoundError:
            contents = ""
        if contents and not contents.endswith("\n"):
            raise ValueError("event log has an incomplete final event")
        existing = []
        for line in contents.splitlines():
            if not line.strip():
                continue
            try:
                prior = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("event log contains an invalid JSON line") from exc
            if not isinstance(prior, dict):
                raise ValueError("event log contains a non-object event")
            existing.append(prior)
        for prior in existing:
            if prior.get("event_id") != canonical["event_id"]:
                continue
            if any(prior.get(key) != canonical[key] for key in EVENT_IDENTITY_KEYS):
                raise ValueError("event_id is already used by a different event")
            return False
        payload = (contents + json.dumps(canonical, ensure_ascii=False, sort_keys=True,
                                         allow_nan=False) + "\n").encode("utf-8")
        if len(payload) > MAX_EVENT_LOG_BYTES:
            raise ValueError("event log exceeds the size limit")
        temporary = path.parent / (".event-" + uuid.uuid4().hex)
        try:
            with temporary.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return True


def add_event_arguments(parser: argparse.ArgumentParser) -> None:
    """Shared CLI for the append-only crash / repair audit log."""
    parser.add_argument("--candidate-id", default=None,
                        help="round id shared by a crash and its repair retries")
    parser.add_argument("--attempt", type=int, default=1,
                        help="1 for the first attempt, 2+ for a repair retry")
    parser.add_argument("--repair-of", default=None,
                        help="event_id of the crash this attempt repairs")
    parser.add_argument("--event-id", default=None,
                        help="explicit idempotency key (default: hash of the event identity)")
    parser.add_argument("--event-log", type=Path, default=None,
                        help="append-only JSONL log (default: events.jsonl beside results.tsv)")
    parser.add_argument("--log", type=Path, default=None,
                        help="optional round log path recorded with the event")
    parser.add_argument("--evidence", choices=(EVIDENCE_LOCAL, EVIDENCE_SMOKE), default=None,
                        help="evidence level for a crash without a result file")


def record_event(args: argparse.Namespace, results: Path, *, metric: str,
                 value: float | None, protocol: str, evidence: str, decision: str,
                 commit: str, description: str) -> bool:
    """Record KEEP / REVERT / CRASH as one event, linked to its repair chain."""
    candidate_id = args.candidate_id or commit
    event_id = args.event_id or event_id_for(
        results=results, candidate_id=candidate_id, attempt=args.attempt, commit=commit,
        decision=decision, description=description, value=value, evidence=evidence,
        repair_of=args.repair_of)
    event = {
        "schema_version": EVENT_SCHEMA,
        "event_id": event_id,
        "candidate_id": candidate_id,
        "attempt": args.attempt,
        "commit": commit,
        "decision": decision,
        "evidence": evidence,
        "metric": metric,
        "value": value,
        "protocol_sha256": protocol,
        "description": description,
        "repair_of": args.repair_of,
        "result_path": str(args.result) if getattr(args, "result", None) else None,
        "log_path": str(args.log) if getattr(args, "log", None) else None,
        "results_path": str(results),
    }
    event_log = Path(args.event_log) if args.event_log else results.parent / EVENT_LOG_NAME
    return append_event(event_log, event)


def append_history(path: Path, *, commit: str, metric: str, value: float | None,
                   direction: str, decision: str, description: str,
                   protocol: str, evidence: str, peak_vram: str = "NOT_AVAILABLE",
                   promotion: str = "strict") -> None:
    """Append one decision; reject incomparable protocols and false improvements."""
    if (not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit)
            or direction not in ("minimize", "maximize")
            or decision not in ("KEEP", "REVERT", "CRASH")
            or promotion not in ("strict", "external")):
        raise ValueError("invalid direction or decision")
    if not metric or not evidence or not description.strip() or len(description) > 400:
        raise ValueError("metric, evidence and a short description are required")
    if decision == "CRASH":
        if value is not None:
            raise ValueError("a crash cannot carry a numeric score")
    elif (isinstance(value, bool) or not isinstance(value, (int, float))
          or not math.isfinite(value) or not re.fullmatch(r"[0-9a-f]{64}", protocol)):
        raise ValueError("a scored result needs a finite value and protocol hash")
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.parent / ("." + path.name + ".lock")
    with open_regular(lock_path) as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            contents = read_regular(path, 2 * 1024 * 1024).decode("utf-8")
        except FileNotFoundError:
            contents = ""
        if contents and not contents.endswith("\n"):
            raise ValueError("results file has an incomplete final row")
        reader = csv.DictReader(io.StringIO(contents), delimiter="\t", strict=True)
        existing = list(reader)
        if reader.fieldnames and reader.fieldnames != list(FIELDS):
            raise ValueError("results file has an incompatible schema")
        if any(None in row or any(row.get(field) is None for field in FIELDS) for row in existing):
            raise ValueError("results file contains an incomplete row")
        for row in existing:
            if row["decision"] not in ("KEEP", "REVERT", "CRASH"):
                raise ValueError("results file contains an invalid decision")
            if row["decision"] == "CRASH":
                if row["value"]:
                    raise ValueError("crash row contains a score")
            else:
                try:
                    prior = float(row["value"])
                except ValueError as exc:
                    raise ValueError("results file contains an invalid score") from exc
                if (not math.isfinite(prior)
                        or not re.fullmatch(r"[0-9a-f]{64}", row["protocol_sha256"])):
                    raise ValueError("results file contains an invalid score or protocol")
        if protocol and any(row.get("protocol_sha256") not in ("", protocol) for row in existing):
            raise ValueError("result protocol differs from this run's earlier results")
        if any((row.get("metric"), row.get("direction"), row.get("evidence"), row.get("promotion"))
               != (metric, direction, evidence, promotion) for row in existing):
            raise ValueError("result metadata differs from this run's earlier results")
        new_row = dict(zip(FIELDS, (commit, metric, "" if value is None else str(value),
                                   direction, decision, description, peak_vram, protocol,
                                   evidence, promotion), strict=True))
        # Repeating an identical record command must not create a second event.
        if any(row == new_row for row in existing):
            return
        same_commit = [row for row in existing if row.get("commit") == commit]
        if same_commit and any(row.get("decision") != "CRASH" for row in same_commit):
            raise ValueError("candidate commit is already recorded with a scored decision")
        # Earlier rows for this commit are unscored crashes: the repair retry is a
        # distinct append-only event and must never overwrite them.
        if decision == "KEEP" and promotion == "strict":
            previous = [float(row["value"]) for row in existing
                        if row.get("decision") == "KEEP" and row.get("value")]
            if previous and ((direction == "maximize" and value <= max(previous))
                             or (direction == "minimize" and value >= min(previous))):
                raise ValueError("KEEP requires an improvement over the previous champion")
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(existing)
        writer.writerow(new_row)
        payload = output.getvalue().encode("utf-8")
        if len(payload) > 2 * 1024 * 1024:
            raise ValueError("results file exceeds the size limit")
        temporary = path.parent / (".history-" + uuid.uuid4().hex)
        try:
            with temporary.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def record_main(argv: list[str]) -> int:
    """Append one decision to results.tsv and one immutable event to events.jsonl."""
    parser = argparse.ArgumentParser(description=record_main.__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--decision", required=True, choices=EVENT_DECISIONS)
    parser.add_argument("--description", required=True)
    parser.add_argument("--result", type=Path)
    add_event_arguments(parser)
    args = parser.parse_args(argv)
    candidate = committed_source(args.repo, args.commit, "src/research/train.py")
    score: float | None = None
    peak_vram = "NOT_AVAILABLE"
    protocol = ""
    result_payload = None
    if args.result is not None:
        try:
            result_payload = json.loads(read_regular(args.result, 1024 * 1024))
        except (OSError, ValueError):
            # A crash often leaves no result file; its audit event is still recorded.
            result_payload = None
    evidence = resolved_evidence(args.results, result_payload, args.evidence)
    if args.decision != "CRASH":
        if not isinstance(result_payload, dict):
            parser.error("a scored decision requires a readable --result")
        result = result_payload
        if result.get("candidate_sha256") != hashlib.sha256(candidate).hexdigest():
            raise ValueError("result does not match the committed training file")
        score = result["macro_f1"]
        peak_vram = result.get("peak_vram_mb", "NOT_AVAILABLE")
        if isinstance(score, bool) or not isinstance(score, (float, int)) or not 0 <= score <= 1:
            raise ValueError("result has an invalid macro-F1")
        if result.get("evidence") not in (EVIDENCE_SMOKE, EVIDENCE_LOCAL):
            raise ValueError(f"result evidence must be {EVIDENCE_SMOKE} or {EVIDENCE_LOCAL}")
        if args.evidence is not None and args.evidence != result["evidence"]:
            raise ValueError("--evidence conflicts with the result evidence")
        # A scored round's own result file is authoritative for the evidence level.
        evidence = result["evidence"]
        if (result.get("metric") != "syndrome/macro_f1"
                or result.get("metric_version") != "tcm-sd-eval-v3"
                or result.get("direction") != "maximize"):
            raise ValueError("result does not match the frozen metric contract")
        # A scored run must name the exact frozen inputs it used. Real runs must
        # carry the pinned protocol-v2 hashes; smoke runs the built-in fixture.
        expected = (synthetic_selection()[0] if evidence == EVIDENCE_SMOKE else None)
        if expected is None:
            if (result.get("dataset_sha256") != SELECTION_DATASET_SHA256
                    or result.get("manifest_sha256") != SELECTION_MANIFEST_SHA256):
                raise ValueError(
                    "result does not match the frozen protocol-v2 selection inputs"
                )
        elif (result.get("dataset_sha256") != expected["dataset_sha256"]
                or result.get("manifest_sha256") != expected["manifest_sha256"]):
            raise ValueError("result does not match the built-in synthetic protocol")
        keys = ("metric", "direction", "metric_version", "dataset_sha256", "manifest_sha256",
                "seed", "budget_seconds", "fold_count", "repeats", "evidence",
                "code_sha256", "promotion")
        if any(key not in result for key in keys):
            raise ValueError("result is missing a frozen protocol field")
        if result["code_sha256"] != fixed_code_hashes() or result["promotion"] != "strict":
            raise ValueError("fixed evaluator code or promotion rule has changed")
        folds = result["fold_count"]
        repeats = result["repeats"]
        budget = result["budget_seconds"]
        if (any(isinstance(v, bool) or not isinstance(v, int)
                for v in (folds, repeats, budget, result["seed"]))
                or not 2 <= folds <= 10 or not 1 <= repeats <= 10
                or folds * repeats > 32 or not 1 <= budget <= 3600
                or folds * repeats * budget > 14400):
            raise ValueError("result has invalid protocol parameters")
        details = result.get("folds")
        if not isinstance(details, list) or len(details) != folds * repeats:
            raise ValueError("result does not contain all fold scores")
        observed = set()
        fold_scores = []
        for item in details:
            if not isinstance(item, dict):
                raise ValueError("invalid fold score")
            fold, repeat, value = item.get("fold"), item.get("repeat"), item.get("macro_f1")
            n_validation = item.get("n_validation")
            if (isinstance(fold, bool) or not isinstance(fold, int) or not 0 <= fold < folds
                    or isinstance(repeat, bool) or not isinstance(repeat, int)
                    or not 0 <= repeat < repeats or (fold, repeat) in observed
                    or isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 <= value <= 1
                    or isinstance(n_validation, bool) or not isinstance(n_validation, int)
                    or n_validation < 1):
                raise ValueError("invalid fold score")
            observed.add((fold, repeat))
            fold_scores.append(value)
        if not math.isclose(statistics.mean(fold_scores), score, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("reported metric does not match fold scores")
        if peak_vram != "NOT_AVAILABLE":
            raise ValueError("GPU memory was not measured by this evaluator")
        protocol = hashlib.sha256(json.dumps(
            {key: result[key] for key in keys}, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
    append_history(
        args.results, commit=args.commit, metric="syndrome/macro_f1",
        value=score, direction="maximize", decision=args.decision,
        description=args.description, protocol=protocol, evidence=evidence,
        peak_vram=peak_vram,
    )
    record_event(args, args.results, metric="syndrome/macro_f1", value=score,
                 protocol=protocol, evidence=evidence, decision=args.decision,
                 commit=args.commit, description=args.description)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "record":
        return record_main(argv[1:])
    parser = argparse.ArgumentParser(description="Fixed TCM-SD selection evaluator")
    parser.add_argument("--manifest", type=Path,
                        help="frozen protocol-v2 selection_manifest.json")
    parser.add_argument("--dataset", type=Path,
                        help="frozen protocol-v2 selection_records.json")
    parser.add_argument("--candidate", type=Path, required=True,
                        help="the one editable training file (src/research/train.py)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=None,
                        help="worktree the candidate was committed in (recorded for provenance)")
    parser.add_argument("--budget", type=int, default=60,
                        help="fixed wall-clock seconds per fold")
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--synthetic", action="store_true",
                        help="use the built-in smoke fixture instead of real data")
    args = parser.parse_args(argv)
    if args.synthetic:
        if args.manifest or args.dataset:
            parser.error("synthetic mode uses the built-in fixture and accepts no data paths")
    elif not (args.manifest and args.dataset):
        parser.error("real evaluation needs --manifest and --dataset "
                     "(or --synthetic for the smoke fixture)")
    if args.out.exists():
        parser.error("output already exists; use a new result file")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    real = not args.synthetic
    task_name = "tcm-sd-syndrome" if real else "synthetic-syndrome"
    evidence = EVIDENCE_LOCAL if real else EVIDENCE_SMOKE

    def status(value: dict) -> None:
        value = {**value, "task": task_name, "updated_at": time.time(),
                 "evidence": evidence}
        target = args.out.parent / "status.json"
        temporary = args.out.parent / (".status-" + uuid.uuid4().hex)
        try:
            with temporary.open("x", encoding="utf-8") as stream:
                json.dump(value, stream, ensure_ascii=False)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    with open_regular(args.out.parent / ".evaluation.lock") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another evaluation is active in this run") from exc
        try:
            status({"state": "PREPARING", "deadline": time.time() + 60})
            if real:
                manifest, records = load_selection(args.manifest, args.dataset)
            else:
                manifest, records = synthetic_selection()
            candidate = read_regular(args.candidate, 2 * 1024 * 1024)
            result = evaluate(manifest, records, candidate,
                              budget_seconds=args.budget, seed=args.seed,
                              folds=args.folds, repeats=args.repeats,
                              repo_root=args.repo_root, progress=status)
            payload = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            write_new_atomic(args.out, payload.encode("utf-8"))
            status({"state": "COMPLETE", "macro_f1": result["macro_f1"]})
        except Exception as exc:
            status({"state": "FAILED", "reason": type(exc).__name__})
            raise
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
