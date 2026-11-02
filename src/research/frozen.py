"""THE FROZEN BOUNDARY of the source-level AutoResearch loop.

Mirrors the role of `prepare.py` in karpathy/autoresearch: everything the search
must not be able to touch lives here or beside it — the data, the frozen split
and its scaler, the label contract, the evaluator (`research/evaluator.py`,
which runs in its own process on a rebuilt fold), the metric, and the training
time budget. A candidate imports from this module and nothing else.

The agent edits exactly one file: `research/candidate.py`. The loop rejects any
patch that touches another path, and rejects any patch whose text mentions a
forbidden symbol (see `FORBIDDEN_SYMBOLS`). That separation is checkable rather
than promised, which is the difference between this loop and the v1-v8 one that
only let an agent choose six hyperparameter values.

Why a fixed time budget: upstream fixes 5 minutes per experiment so model size,
batch size and schedule can all be explored without "train longer" being an
available win. Here that was a live defect — `trainer.epochs` sat in the
editable scope, so a candidate could raise its score by training longer, which
is not a modelling result.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

#: Fixed training wall clock per fold, in seconds. Not editable, and not a
#: scope key: every candidate gets the same budget, so a better score can only
#: come from a better use of it.
#:
#: 20s, not upstream's 300s: this model has ~6k parameters and 174 training rows
#: per fold, and the whole loop must stay runnable interactively. The choice is
#: declared here rather than tuned — raising it changes what "better use of the
#: budget" means for every archived round at once, so it is part of the frozen
#: contract, not a knob.
TIME_BUDGET_SECONDS = 20.0
#: Hard-kill margin over TIME_BUDGET_SECONDS. A candidate that overruns is a
#: crash (upstream: "if a run exceeds 10 minutes, kill it and treat it as a
#: failure"), never a slow winner.
TIME_BUDGET_GRACE_SECONDS = 20.0

#: What a candidate may import from this module.
FROZEN_PUBLIC_API = (
    "TIME_BUDGET_SECONDS",
    "FoldData",
    "build_fold",
)

#: Names a candidate patch may never mention. A candidate that can reach the
#: evaluator, the split, the labels or the scaler can report a number of its own
#: choosing. The loop always reads the score from the frozen evaluator and never
#: from the candidate; these symbol checks are what make that number mean
#: something. (They catch the careless and the accidental — a determined
#: candidate could obfuscate the name — so they are one layer, not the wall;
#: the wall is that scoring happens in a separate `research.evaluator` process
#: that rebuilds the fold from the frozen manifest and never sees the objects
#: the candidate trained against.)
FORBIDDEN_SYMBOLS = (
    "TCMTrustedEvaluator",
    "EvalManifest",
    "real_eval_manifest",
    "manifest_sha256",
    "holdout",
    "split_seed",
    "fold_manifest",
    "derive_classification_targets",
    "_fit_scaler",
    "input_spec_version",
    "exec(",
    "compile(",
    "__import__",
    # R1 F4: the syndrome task's frozen scoring/split machinery is equally
    # out of reach for a candidate patch (candidate_sd.py's editable scope).
    "SyndromeTrustedEvaluator",
    "build_syndrome_manifest",
    "task_sd",
    "subject_partitions",
)
#: `eval(` is deliberately NOT a text symbol: the substring matches the
#: legitimate `model.eval()` training idiom. Dynamic-eval calls are rejected
#: precisely, as AST Call nodes with Name in the zero-tolerance set — see
#: `guard.validate_source_ast`.

#: Deliberately NOT forbidden: `category_labels`, `target_mask`, `score_mask`.
#: Those are the keys the frozen provider HANDS a candidate — it cannot train
#: without reading them. Listing them made the guard reject a legitimate patch
#: whose diff merely quoted the baseline's own training line as context (round
#: 2 of the first run), which is a false positive that teaches nothing. What is
#: forbidden is the SCORING side: the evaluator, the split, the scaler, the
#: holdout and the label-derivation function.


@dataclass(frozen=True)
class FoldData:
    """One fold's data, scaled by that fold's own statistic.

    `train` and `dev` are frozen providers; `input_spec` is the contract the
    candidate's model must declare. Everything else about the fold — which
    subjects, which labels, which rows are held out — is deliberately absent.
    """

    train: object
    dev: object
    input_spec: object
    dims: dict
    n_train_subjects: int
    n_dev_subjects: int
    seed: int

    def summary(self) -> dict:
        return {
            "input_spec_version": self.input_spec.version,
            "dims": dict(self.dims),
            "n_train_subjects": self.n_train_subjects,
            "n_dev_subjects": self.n_dev_subjects,
            "time_budget_seconds": TIME_BUDGET_SECONDS,
        }


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
        from research.task_sd import SELECTION_FOLD_VERSION

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

    pool = sorted(list(manifest["split"]["train"]) + list(manifest["split"]["dev"]))
    rng = random.Random(manifest["split"]["seed"] + repeat)
    rng.shuffle(pool)
    fold_of = {s: i % folds for i, s in enumerate(pool)}
    train_ids = [s for s in pool if fold_of[s] != fold]
    dev_ids = [s for s in pool if fold_of[s] == fold]
    return train_ids, dev_ids


def build_fold(
    config: dict,
    manifest: dict,
    fold: int,
    folds: int,
    repeat: int,
    records: list,
) -> tuple[FoldData, dict]:
    """Draw one fold from a validated selection-only manifest and records."""
    from data.real_split import fold_manifest
    from research.tasks import validate_manifest_contract
    from runtime.real_data import RealBatchProvider, real_input_spec

    task = config.get("task", "constitution")
    validate_manifest_contract(task, manifest, selection_pool=True)
    if manifest["split"]["holdout"] != []:
        raise ValueError("selection manifest must have an empty holdout partition")
    if records is None:
        raise ValueError("fold records must come from validated selection inputs")
    expected_rows = set(manifest["subject_row_by_id"].values())
    record_rows = [record.subject_index for record in records]
    if len(record_rows) != len(set(record_rows)) or set(record_rows) != expected_rows:
        raise ValueError("fold records do not match the selection manifest rows")

    train_ids, dev_ids = selection_fold_ids(manifest, fold, folds, repeat)
    data = fold_manifest(
        manifest, records, train_ids, dev_ids,
        notes=f"source-level selection fold {fold} repeat {repeat}",
    ).data
    fold_rows = set(data["subject_row_by_id"].values())
    fold_records = [record for record in records if record.subject_index in fold_rows]
    batch_size = int(config.get("train_provider", {}).get("batch_size", 32))
    fold_data = FoldData(
        train=RealBatchProvider(data, fold_records, "train", batch_size=batch_size),
        dev=RealBatchProvider(data, fold_records, "dev", batch_size=64),
        input_spec=real_input_spec(data),
        dims={
            "modalities": {name: p["dim"] for name, p in data["modalities"].items()},
            "targets": 9,
        },
        n_train_subjects=len(train_ids),
        n_dev_subjects=len(dev_ids),
        seed=int(config.get("trainer", {}).get("seed", 0)),
    )
    return fold_data, data


def training_fold_data(fit_manifest: dict, records, config: dict) -> FoldData:
    """The FoldData a candidate trains against: train rows only, no dev.

    The training subprocess's inputs are the fit manifest (holdout emptied,
    dev partition dropped, subject_row_by_id pruned to the training rows) and
    records pruned to those same rows — the candidate-visible world contains
    no dev labels to fit against (F-06) and no full-set rows to diff (F-08).
    """
    from runtime.real_data import RealBatchProvider, real_input_spec

    return FoldData(
        train=RealBatchProvider(
            fit_manifest, records, "train",
            batch_size=int(config.get("train_batch_size", 32)),
        ),
        dev=None,
        input_spec=real_input_spec(fit_manifest),
        dims={
            "modalities": {
                name: params["dim"] for name, params in fit_manifest["modalities"].items()
            },
            "targets": 9,
        },
        n_train_subjects=len(fit_manifest["split"]["train"]),
        n_dev_subjects=0,
        seed=int(config.get("seed", 0)),
    )


def assert_api_is_closed() -> None:
    """Every declared public name must exist in this module.

    A name in FROZEN_PUBLIC_API that does not exist would let a candidate import
    a typo silently, and a public name NOT declared would widen the boundary
    without an edit to this list — so both directions are checked.
    """
    missing = [name for name in FROZEN_PUBLIC_API if name not in globals()]
    if missing:
        raise ValueError(f"FROZEN_PUBLIC_API names that do not exist: {missing}")


assert_api_is_closed()
