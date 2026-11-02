"""Build #56 protocol-v2 development inputs for the #53 syndrome task.

This builder loads records only through
``load_tcm_sd_development_records``. That loader verifies and opens exactly
``train.json`` and ``dev.json``; it never hashes or opens the consumed final
``test.json`` split. The exact sd-v1 unblinded user set is loaded from a
static ``--exposed-users-file`` (produced once from the unlock delivery
artifacts) and excluded from the pool; the remaining users get a fresh
internal train/dev split over the clean population. The manifest's
``holdout`` partition is that internal development partition. This builder
does not score or unlock it. Issue #56 results use development data and
user-grouped cross-validation only.

Default output directory is runs/sd-v2/<scenario>. The builder emits the
development bundle plus selection_records.json and selection_manifest.json.
The selection pair contains all non-empty scenario rows for the frozen
train+dev users, has an empty holdout split, and records the parent manifest
SHA-256. Evaluation-side deduplication is applied after each split or fold is
frozen. Choose short for chief complaints or clinical for the full text fields.

Replay: ``PYTHONPATH=src python -m scripts.build_sd_campaign_inputs`` with
optional ``--raw-dir data/local/tcm-sd``, ``--out <directory>``, and
``--scenario short|clinical`` arguments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from data.tcm_sd import (
    LABEL_TAXONOMY_ID,
    SPLIT_SEED,
    load_syndrome_vocab,
    load_tcm_sd_development_records,
    subject_partitions,
    user_majority_strata,
)
from research.task_sd import (
    SyndromeRecord,
    build_syndrome_manifest,
    build_syndrome_selection_artifacts,
    records_to_json,
)


def load_exposed_users(path: Path) -> tuple[frozenset[str], dict]:
    """Load the static sd-v1 exposed-user set from the unlock artifacts.

    The file is produced once from the sd-v1 unlocked delivery artifacts
    (``runs/sd-campaign/manifest.json`` ``split.holdout`` row ids mapped
    through ``records.json`` user ids); this builder never opens the final
    test split. Fail closed on schema drift so a stale or hand-edited file
    cannot silently narrow the exclusion.
    """
    payload = path.read_bytes()
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: exposed-users file must be a JSON object")
    users = data.get("users")
    if (
        not isinstance(users, list)
        or not all(isinstance(u, str) and u for u in users)
        or users != sorted(set(users))
    ):
        raise ValueError(f"{path}: users must be a sorted list of unique ids")
    if data.get("n_users") != len(users):
        raise ValueError(f"{path}: n_users does not match the users list")
    for key in ("source", "n_rows"):
        if not isinstance(data.get(key), (str, int)) or not data.get(key):
            raise ValueError(f"{path}: missing {key!r} provenance field")
    return frozenset(users), {
        "source": data["source"],
        "n_users": len(users),
        "n_rows": data["n_rows"],
        "file_sha256": hashlib.sha256(payload).hexdigest(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/local/tcm-sd")
    parser.add_argument("--out", default=None)
    parser.add_argument("--scenario", choices=("short", "clinical"), default="short")
    parser.add_argument(
        "--exposed-users-file",
        required=True,
        help=(
            "JSON with the exact sd-v1 unblinded user set (from the unlock "
            "delivery artifacts); required, fail-closed"
        ),
    )
    args = parser.parse_args(argv)
    out = Path(args.out) if args.out else Path("runs/sd-v2") / args.scenario
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output directory: {out}")

    if args.scenario == "short":
        from data.tcm_sd import build_short_text as text_fn
    else:
        from data.tcm_sd import build_clinical_text as text_fn

    raw_dir = Path(args.raw_dir)
    records = load_tcm_sd_development_records(raw_dir)
    vocab = load_syndrome_vocab(raw_dir)
    label_index = {name: i for i, name in enumerate(vocab)}
    unknown = sorted({r.norm_syndrome for r in records} - set(label_index))
    if unknown:
        raise ValueError(
            f"{len(unknown)} norm_syndrome values outside the frozen "
            f"taxonomy {LABEL_TAXONOMY_ID}: {unknown[:5]}"
        )

    # Freeze strata over all loaded users before scenario text filtering.
    # Keep duplicate text rows here: each dev/CV evaluator must deduplicate
    # only its frozen evaluation side after the fold is assigned.
    user_strata = user_majority_strata(records)

    # The sd-v1 delivery unblinded an exact set of users (its scored holdout
    # rows, recorded in the unlock delivery artifacts). Those users are
    # excluded from this protocol's pool by loading the static exposed-user
    # file — NOT by a seed replay, which would permute differently over this
    # reduced train+dev population and leave exposed users in the pool.
    # The remaining users get a fresh internal train/dev split over the clean
    # population; no holdout or test file is read.
    exposed_users, exclusion_provenance = load_exposed_users(
        Path(args.exposed_users_file)
    )
    loaded_users = {r.user_id for r in records}
    exposed_in_population = exposed_users & loaded_users
    clean_records = [r for r in records if r.user_id not in exposed_in_population]
    internal = subject_partitions(clean_records, seed=SPLIT_SEED)
    partitions = {
        # the internal replay's own test slice folds back into train: it is a
        # development-split device, not a holdout — the only sealed users here
        # are the exact sd-v1 exposed set.
        "train": frozenset(internal["train"] | internal["test"]),
        "dev": frozenset(internal["dev"]),
        "test": exposed_in_population,
    }
    exposed_rows = sum(1 for record in records if record.user_id in exposed_in_population)

    # Empty scenario text (e.g. 19 cases with an empty chief complaint) has no
    # input for this scenario and the hashed_text contract is fail-closed;
    # drop them here and record the count — production inference on such
    # input is an ABSTAIN, never a guess. Duplicate text rows remain in the
    # pool; the evaluator handles only the active dev side of each fold.
    texted = [r for r in records if text_fn(r).strip()]
    empty_text_dropped = len(records) - len(texted)

    rows = [
        SyndromeRecord(
            subject_index=i,
            user_id=r.user_id,
            features={},
            label=label_index[r.norm_syndrome],
            norm_syndrome=r.norm_syndrome,
            input_text=text_fn(r),
            provenance={"source_file": r.source_file, "lcd_id": r.lcd_id},
        )
        for i, r in enumerate(texted, start=1)
    ]
    payload = records_to_json(rows)
    dataset_sha256 = hashlib.sha256(payload.encode()).hexdigest()
    manifest = build_syndrome_manifest(
        rows,
        dataset_sha256=dataset_sha256,
        n_classes=len(vocab),
        label_taxonomy_version=LABEL_TAXONOMY_ID,
        label_tier="PUBLISHED",
        modality_kind="hashed_text",
        notes=(
            f"scenario={args.scenario}; text basis={text_fn.__name__}; "
            "preserve train+dev rows before fold assignment; evaluator applies "
            "one-way exact-text dedup to each frozen dev side; "
            f"hashed features frozen in-manifest (HashingVectorizer "
            f"char_wb 1-3, 65536, l2); no fitted statistics"
        ),
        frozen_user_partitions=partitions,
        frozen_user_strata=user_strata,
    )
    selection_rows, selection_manifest = build_syndrome_selection_artifacts(
        manifest, rows
    )
    selection_payload = records_to_json(selection_rows)

    out.mkdir(parents=True, exist_ok=False)
    (out / "records.json").write_text(payload, encoding="utf-8")
    manifest.save(out / "manifest.json")
    (out / "selection_records.json").write_text(
        selection_payload, encoding="utf-8"
    )
    selection_manifest.save(out / "selection_manifest.json")
    (out / "build-summary.json").write_text(
        json.dumps(
            {
                "scenario": args.scenario,
                "text_basis": text_fn.__name__,
                "n_records": len(rows),
                "evaluation_dedup_policy": (
                    "preserve selection rows; exact scenario text dedup occurs "
                    "on each frozen evaluation side"
                ),
                "empty_text_dropped": empty_text_dropped,
                "dataset_sha256": dataset_sha256,
                "manifest_sha256": manifest.sha256,
                "selection_n_records": len(selection_rows),
                "selection_dataset_sha256": selection_manifest.data["dataset_sha256"],
                "selection_manifest_sha256": selection_manifest.sha256,
                "user_partition_sha256": manifest.data["user_partition_sha256"],
                "sd_v1_exposed_user_exclusion": {
                    "policy": (
                        "exact sd-v1 unblinded user set loaded from the unlock "
                        "delivery artifacts file; excluded from the selection "
                        "pool; not a seed replay"
                    ),
                    **exclusion_provenance,
                    "n_users_in_population": len(exposed_in_population),
                    "n_rows": exposed_rows,
                },
                "split_seed": SPLIT_SEED,
                "taxonomy": LABEL_TAXONOMY_ID,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "n_records": len(rows),
                "evaluation_dedup": "deferred until frozen dev/CV fold",
                "manifest_sha256": manifest.sha256[:16],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
