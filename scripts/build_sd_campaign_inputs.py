"""Build #56 protocol-v2 development inputs for the #53 syndrome task.

This builder loads records only through
``load_tcm_sd_development_records``. That loader verifies and opens exactly
``train.json`` and ``dev.json``; it never hashes or opens the consumed final
``test.json`` split. The builder freezes user-level partitions across the
complete loaded train+dev population and preserves all non-empty scenario
rows in the hashed-text selection pool. The evaluator removes exact-text
duplicates from the evaluation side of each frozen dev split or CV fold.

The manifest's ``holdout`` partition is an internal development partition
derived only from train+dev. This builder does not score or unlock it. Issue
#56 results use development data and user-grouped cross-validation only.

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/local/tcm-sd")
    parser.add_argument("--out", default=None)
    parser.add_argument("--scenario", choices=("short", "clinical"), default="short")
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

    # Freeze strata and partitions over all loaded users before scenario text
    # filtering. Keep duplicate text rows here: each dev/CV evaluator must
    # deduplicate only its frozen evaluation side after the fold is assigned.
    user_strata = user_majority_strata(records)
    partitions = subject_partitions(records, seed=SPLIT_SEED)

    # The sd-v1 delivery already unblinded a test assignment over this
    # population, replayed deterministically here from the frozen seed. Those
    # users form this builder's internal holdout: they are excluded from the
    # selection pool and never scored in #56. No holdout or test file is read.
    exposed_users = partitions["test"]
    exposed_rows = sum(1 for record in records if record.user_id in exposed_users)

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
                        "sd-v1 unblinded test users, replayed deterministically "
                        "from the frozen split seed over the loaded train+dev "
                        "population; excluded from the selection pool"
                    ),
                    "n_users": len(exposed_users),
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
