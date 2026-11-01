"""Build the AutoResearch campaign inputs for the #53 syndrome task (#53 stage 1).

Generates, on the dev host (acam-lab, real data stays in data/local/):
  runs/sd-campaign/records.json    — one row per case: user_id, input_text
                                     (short scenario = chief complaint),
                                     label index, provenance; NO inline
                                     features (hashed_text modality).
  runs/sd-campaign/manifest.json   — hashed_text manifest from
                                     research.task_sd.build_syndrome_manifest
                                     (subject-level split via
                                     data.tcm_sd.subject_partitions, seed
                                     20261101, single source of truth).

Evaluation-side one-way dedup (mainline arbitration 2026-11-02) is applied on
the SCENARIO's own text basis BEFORE the manifest is frozen, so the campaign's
dev scores measure the same dev set the frozen baseline reports:
short basis drops 2,065 dev cases (0.3835 → 0.3449 for the TF-IDF+LR
reference). Train is kept in full; holdout rows enter the manifest redacted by
the engine's holdout clearing and are scored only at the paid unlock.

Replay: PYTHONPATH=src python -m scripts.build_sd_campaign_inputs \
    [--raw-dir data/local/tcm-sd] [--out runs/sd-campaign] [--scenario short]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from data.tcm_sd import (
    LABEL_TAXONOMY_ID,
    SPLIT_SEED,
    deduplicate_evaluation_records,
    load_syndrome_vocab,
    load_tcm_sd_records,
    subject_partitions,
)
from data.tcm_sd import build_short_text as build_text
from research.task_sd import (
    SyndromeRecord,
    build_syndrome_manifest,
    records_to_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/local/tcm-sd")
    parser.add_argument("--out", default="runs/sd-campaign")
    parser.add_argument("--scenario", choices=("short", "clinical"), default="short")
    args = parser.parse_args(argv)

    if args.scenario == "short":
        from data.tcm_sd import build_short_text as text_fn
    else:
        from data.tcm_sd import build_clinical_text as text_fn

    raw_dir = Path(args.raw_dir)
    records = load_tcm_sd_records(raw_dir)
    vocab = load_syndrome_vocab(raw_dir)
    label_index = {name: i for i, name in enumerate(vocab)}

    partitions = subject_partitions(records, seed=SPLIT_SEED)
    kept, dedup_stats = deduplicate_evaluation_records(
        records, partitions, text_basis=text_fn
    )

    # Empty scenario text (e.g. 19 cases with an empty chief complaint) has no
    # input for this scenario and the hashed_text contract is fail-closed;
    # drop them here and record the count — production inference on such
    # input is an ABSTAIN, never a guess.
    texted = [r for r in kept if text_fn(r).strip()]
    empty_text_dropped = len(kept) - len(texted)

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
        for i, r in enumerate(texted)
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
            f"one-way evaluation dedup on the scenario basis: "
            f"{json.dumps(dedup_stats['dropped'])} dropped; "
            f"hashed features frozen in-manifest (HashingVectorizer "
            f"char_wb 1-3, 65536, l2); no fitted statistics"
        ),
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "records.json").write_text(payload, encoding="utf-8")
    manifest.save(out / "manifest.json")
    (out / "build-summary.json").write_text(
        json.dumps(
            {
                "scenario": args.scenario,
                "text_basis": text_fn.__name__,
                "n_records": len(rows),
                "dedup": dedup_stats,
                "empty_text_dropped": empty_text_dropped,
                "dataset_sha256": dataset_sha256,
                "manifest_sha256": manifest.sha256,
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
                "dedup_dropped": dedup_stats["dropped"],
                "manifest_sha256": manifest.sha256[:16],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
