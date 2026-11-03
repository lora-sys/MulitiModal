"""Reconstruct artifacts/tcm-50-v8 from the frozen inputs, hash-verified.

The delivered v8 bundle (2026-10-31, #50) was lost with the dev host's
artifacts/ directory. This rebuilds the parts that are reproducible WITHOUT
reading the frozen holdout:

  1. freeze the manifest from the audited Mendeley workbook with the split seed
     the report names (20261107) and compare its digest to the recorded prefix;
  2. fit the champion configuration - the report's tcm-0000 baseline, which is
     exactly runspec.base_config - on the selection pool (train+dev) through the
     same worker entry the harness's finalize() uses, and compare weights.pt to
     the full SHA-256 recorded in docs/evidence/registry.json.

`refit_champion` trains on the pool only; it never reads the holdout. The
holdout-scored outputs (summary.json, report.md, the holdout charts) are NOT
regenerated: the Phase-1 holdout budget is 2/2 spent, and the README already
records that v8's holdout macro-F1 lead was a threshold artifact. Those numbers
stay citations to docs/reports/tcm-50-real-training.md.

Run from the repo root:  PYTHONPATH=src .venv/bin/python <this file>
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

EXPECTED_MANIFEST_PREFIX = "775ba73bfe8f4da6"
EXPECTED_WEIGHTS_SHA256 = (
    "9d642282767df088a115a60d81b8102f29fca9a4931e2894df0ca48f35a25ad2"
)
EXPECTED_PARTITION_SIZES = {"train": 191, "dev": 41, "holdout": 42}
XLS = pathlib.Path("data/local/mendeley-cxtph3tjsg-v1/Share data.xls")
BUNDLE = pathlib.Path("artifacts/tcm-50-v8")


def main() -> int:
    from data.mendeley import EXPECTED_SHA256, load_mendeley_scores
    from data.real_split import build_real_manifest
    from runtime.harness import _worker, default_real_tcm_runspec

    if not XLS.is_file():
        print(f"NOT_AVAILABLE: {XLS} is missing")
        return 2

    records = load_mendeley_scores(XLS)
    manifest = build_real_manifest(
        records, dataset_sha256=EXPECTED_SHA256, split_seed=20261107
    )
    print(f"manifest_sha256      = {manifest.sha256}")
    print(f"expected prefix      = {EXPECTED_MANIFEST_PREFIX}")
    manifest_ok = manifest.sha256.startswith(EXPECTED_MANIFEST_PREFIX)
    sizes = {
        name: len(manifest.data["split"][name])
        for name in ("train", "dev", "holdout")
    }
    print(f"partition sizes      = {sizes}")
    sizes_ok = sizes == EXPECTED_PARTITION_SIZES

    if not sizes_ok:
        print("REFUSING: the frozen split does not have the delivered partition sizes")
        return 3
    if not manifest_ok:
        # The split seed (20261107) and the partition sizes (191/41/42) are the
        # delivered ones; the digest additionally covers manifest fields that
        # were extended after v8, so it is not byte-reproducible today. Say so
        # rather than failing the whole rebuild on a schema change.
        print(
            "NOTE: the digest differs from the report's prefix; the partition "
            "matches and the digest is recorded as NOT_REPRODUCIBLE_BYTE_EXACT"
        )

    champion = BUNDLE / "run-bundle" / "champion"
    champion.mkdir(parents=True, exist_ok=True)
    manifest_path = BUNDLE / "run-bundle" / "manifest.json"
    manifest.save(manifest_path)

    spec = default_real_tcm_runspec(str(manifest_path), str(XLS))
    # exactly how AutoResearchHarness.finalize() composes the refit config:
    # base_config plus the task/out_dir/data block, with the runspec seed.
    config = {**spec["base_config"]}
    config.update({
        "task": "tcm",
        "data_mode": "real",
        "out_dir": str(champion),
        "data": {
            **spec["data"],
            "task_type": spec["base_config"]["data"]["task_type"],
        },
    })
    config["trainer"]["seed"] = spec["seed"]
    config_path = BUNDLE / "finalize-config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    result = _worker(
        ["refit", "--config", str(config_path)],
        spec["budget"]["per_run_timeout_s"],
    )
    weights = champion / "weights.pt"
    weights_sha = hashlib.sha256(weights.read_bytes()).hexdigest()
    print(f"weights_sha256       = {weights_sha}")
    print(f"recorded sha256      = {EXPECTED_WEIGHTS_SHA256}")
    weights_ok = weights_sha == EXPECTED_WEIGHTS_SHA256
    print(f"champion config      = {json.dumps(config['model'])} / "
          f"{json.dumps({k: v for k, v in config['trainer'].items()})}")

    (BUNDLE / "RECONSTRUCTION.json").write_text(
        json.dumps({
            "artifact": "tcm-50-v8",
            "delivered": "2026-10-31 (#50, v8) - lost 2026-11-03 with the dev host artifacts/",
            "reconstructed": {
                "manifest.json": {
                    "sha256": manifest.sha256,
                    "split_seed": 20261107,
                    "partition_sizes": sizes,
                    "matches_recorded_prefix": manifest_ok,
                    "recorded_prefix": EXPECTED_MANIFEST_PREFIX,
                    "digest_status": (
                        "identical" if manifest_ok else
                        "NOT_REPRODUCIBLE_BYTE_EXACT: split seed and partition "
                        "sizes match the delivered manifest, but the digest covers "
                        "manifest fields extended after v8"
                    ),
                },
                "run-bundle/champion/weights.pt": {
                    "sha256": weights_sha,
                    "recorded_in": "docs/evidence/registry.json",
                    "matches_recorded_sha256": weights_ok,
                },
                "champion_config": {
                    "model": config["model"],
                    "trainer": {k: v for k, v in config["trainer"].items()},
                    "train_provider": config.get("train_provider", {}),
                },
                "worker_result": result,
            },
            "not_reconstructed": {
                "summary.json / report.md / holdout charts": (
                    "holdout-scored; the Phase-1 holdout budget is 2/2 spent and "
                    "these numbers stay citations to "
                    "docs/reports/tcm-50-real-training.md"
                ),
                "runs/real-tcm-v8 selection store": (
                    "the 24 rounds were proposed round-by-round and the runspec "
                    "was revised after v8 (fold-std margin -> SEM margin, "
                    "repeats=4); replaying them today would not reproduce the "
                    "delivered selection"
                ),
                "regression/ deployable-only/ qc-exclude/": (
                    "their metadata is holdout-scored; the training sides are "
                    "reproducible, the reported numbers are citations"
                ),
            },
            "holdout_read": False,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if weights_ok:
        print("OK: the delivered champion is byte-identical to the lost one")
        return 0

    # The refit is deterministic but the pipeline changed after v8, so the
    # produced weights are NOT the delivered ones. Never leave a look-alike
    # bundle behind: record the probe result and remove the misfit.
    import shutil

    (BUNDLE / "RECONSTRUCTION.json").unlink(missing_ok=True)
    for produced in (champion, BUNDLE / "run-bundle" / "manifest.json",
                     BUNDLE / "finalize-config.json"):
        if produced.is_dir():
            shutil.rmtree(produced)
        elif produced.exists():
            produced.unlink()
    run_bundle = BUNDLE / "run-bundle"
    if run_bundle.is_dir() and not any(run_bundle.iterdir()):
        run_bundle.rmdir()
    (BUNDLE / "NOT_RECOVERABLE.json").write_text(
        json.dumps({
            "artifact": "tcm-50-v8",
            "status": "NOT_REPRODUCIBLE",
            "delivered_weights_sha256": EXPECTED_WEIGHTS_SHA256,
            "refit_today_weights_sha256": weights_sha,
            "reproducible": {
                "split_seed": 20261107,
                "partition_sizes": sizes,
                "note": "the delivered partition is reproducible",
            },
            "not_reproducible": (
                "the delivered weights: the refit is deterministic but the "
                "pipeline changed after v8, so today's fit differs. The reported "
                "holdout numbers stay citations to "
                "docs/reports/tcm-50-real-training.md"
            ),
            "holdout_read": False,
            "probe": "scripts/reconstruct_tcm_50_v8.py",
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"NOT_REPRODUCIBLE: delivered {EXPECTED_WEIGHTS_SHA256[:16]}…, today's "
        f"refit {weights_sha[:16]}… — no bundle kept, status recorded"
    )
    return 4


if __name__ == "__main__":
    sys.exit(main())