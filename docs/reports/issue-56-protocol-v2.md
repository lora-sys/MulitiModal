# Issue #56 protocol-v2 development evaluation

**Status: DEV_AND_CV_COMPLETE; SYNTHETIC_TESTS_PASS_WITH_SKIPS; HOST_DOCKER_WORKER_NOT_RUN**

The protocol-v2 builder created short and clinical selection-only inputs from the frozen train/dev users. Both CV runs completed successfully. This report uses only the selection records and manifests. The final test set, ledger, old campaign artifacts, and real-data finalization were not accessed.

## Scope and evidence boundary

The evaluator consumes only runs/sd-v2/<scenario>/selection_records.json
and selection_manifest.json. The selection manifest retains frozen train/dev
user assignments and strata, including users with no remaining scenario row;
it has an empty holdout partition and no holdout user mapping. The evaluator
requires no-follow regular-file descriptors, verifies the selection manifest
and exact dataset-byte SHA-256 before parsing records, then trains and scores
only CV train/dev rows.

The report covers short and clinical scenarios. Holdout records, test files,
ledger events, finalization, candidate patches, and prior sd-campaign
artifacts are outside this evaluation.

## Replay commands

Run the builder once for each fresh scenario directory. The evaluator output
directory must also be fresh and separate from the scenario input directory.

    PYTHONPATH=src python -m scripts.build_sd_campaign_inputs --scenario short --out runs/sd-v2/short
    PYTHONPATH=src python -m scripts.evaluate_sd_v2_cv --scenario short --out runs/sd-v2/cv-short-<fresh-id>

    PYTHONPATH=src python -m scripts.build_sd_campaign_inputs --scenario clinical --out runs/sd-v2/clinical
    PYTHONPATH=src python -m scripts.evaluate_sd_v2_cv --scenario clinical --out runs/sd-v2/cv-clinical-<fresh-id>

The evaluator prints a concise run summary and writes cv-report.json under
each requested output directory. Preserve those reports with the review
evidence; they are not checked into Git.

For the frozen train-to-dev benchmark, use a new output directory:

    PYTHONPATH=src python -m scripts.evaluate_sd_v2_cv --mode dev --scenario short --out runs/sd-v2/dev-short-<fresh-id>
    PYTHONPATH=src python -m scripts.evaluate_sd_v2_cv --mode dev --scenario clinical --out runs/sd-v2/dev-clinical-<fresh-id>

The recorded dev runs used these arguments inside the offline CPU image.
The image mounted src/ and scripts/ read-only, mounted only each scenario's
selection_manifest.json and selection_records.json read-only, and wrote the
report to a bounded temporary output volume before copying the report into
runs/sd-v2/.

## Frozen protocol

| Field | Protocol |
| --- | --- |
| Selection pool | Frozen train + dev rows from the selection-only builder artifact |
| Fold assignment | selection_fold_ids; user-grouped, stratified by frozen user majority-label stratum |
| Fold schedule | 4 folds × 2 repeats |
| Repeat seed source | Frozen split seed + fold version + repeat + frozen user stratum |
| Within-fold text deduplication | research.task_sd.build_fold; exact input_text, one-way CV dev against CV train |
| Baseline | TfidfNgramBaseline |
| Text analyzer and n-grams | Character 1–3 grams |
| min_df | 2 |
| Sublinear term frequency | true |
| C | 4.0 |
| Class weighting | balanced |
| Maximum iterations | 2000 |
| Solver | ovr-liblinear |
| Baseline seed | 20261101 |
| Metric version | tcm-sd-eval-v3 |
| Primary metric | syndrome/macro_f1 |
| Fixed label count and macro denominator | 148 |
| Frozen user→partition assignment | user_partition_sha256 in the manifest: canonical SHA-256 over the complete user→partition map (data.tcm_sd.subject_partition_sha256) |
| Per-class support | Count of true labels in post-dedup CV dev rows |
| Undefined per-class values | zero_division=0; all 148 labels remain in the macro denominator; support is 0 when absent from CV dev |
| Taxonomy version | TCM-SD v1 (LABEL_TAXONOMY_ID) |

Every fold report records its per-class support and scores, training
configuration, environment, and the selection artifact hashes.

## Exposed-user exclusion

The sd-v1 delivery unblinded a test assignment over the TCM-SD population,
replayed deterministically by the builder from the frozen split seed
20261101 (`data.tcm_sd.subject_partitions` over the loaded train+dev
records). Every user in that replayed test partition is excluded from the
protocol-v2 selection pool: their rows form the builder's internal holdout
partition and are never scored in #56. No holdout or test file was read;
the exclusion list is derived deterministically from the frozen seed, and
the builder records the excluded user and row counts in build-summary.json
under `sd_v1_exposed_user_exclusion`.

## Replay evidence

Values below come from the two saved cv-report.json files. Both scenarios completed with status OK and eight fold runs (4 folds × 2 repeats). Results use the uncommitted worktree based on 4a73a199f7c2cfb97012507ee06f7ed24c0b3cd6.

| Evidence | short | clinical |
| --- | --- | --- |
| Status | OK | OK |
| Source revision | Uncommitted worktree based on 4a73a199f7c2cfb97012507ee06f7ed24c0b3cd6 | Same uncommitted worktree based on 4a73a199f7c2cfb97012507ee06f7ed24c0b3cd6 |
| Selection manifest SHA-256 | 286fdf4412111a352974cf01184ed2b0d79f5f908366573a1e128c83a83e4ea9 | 5a749511e9b36051f7c679ebc270cc93ac5798f19ee0ef48c02436b1f34efe97 |
| Selection dataset SHA-256 | a0ddb09c1f06c6eb7dae621c1a43ba5d0c8b1a2fb8af6d9317002898bd060206 | 1e795296ec0ee9dd0af9a4a8e53de6a82ac975fc5cd3aac3071731c7b84434bc |
| Parent manifest SHA-256 | 897c2ce1f7ba6d0871b81f0ed0b2e72ceecf29b22f777e83dc8ff539293850c3 | 1701f3fe58915ec95b94f836af58fa128e062946f9b1c1697d9c6808f8ffc633 |
| Parent dataset SHA-256 | 8769c150aff5e5e26f13cf112890cd466cb5ae8a571dfa55da8d68986a67ea96 | 3b4d9f9b7ea9d6b2bcaa2693d1e578efdc82bf29c1ab7f4600183de449793216 |
| Selection records | 42,982 | 44,258 |
| Selection users with rows / frozen pool users | 41,573 / 43,249 | 42,785 / 43,249 |
| Fold runs | 8 | 8 |
| Fold macro-F1 values by repeat | 0: 0.303738, 0.315914, 0.343637, 0.341493; 1: 0.318011, 0.339925, 0.303033, 0.307265 | 0: 0.543068, 0.576595, 0.545957, 0.559918; 1: 0.549773, 0.558867, 0.550877, 0.550120 |
| Mean fold macro-F1 | 0.321627 | 0.554397 |
| Population std. dev. across folds | 0.016323 | 0.009968 |
| Evaluator report path | runs/sd-v2/cv-short-issue56-round2/cv-report.json | runs/sd-v2/cv-clinical-issue56-round2/cv-report.json |
| Python / scikit-learn / NumPy versions | 3.12.16 / 1.5.2 / 1.26.4 | 3.12.16 / 1.5.2 / 1.26.4 |

## Development-set benchmark

| Evidence | short | clinical |
| --- | --- | --- |
| Dev macro-F1 | 0.314839 | 0.554424 |
| Training rows | 39,877 | 39,892 |
| Dev rows scored | 3,105 | 4,366 |
| Exact text duplicates dropped | 0 | 0 |
| Logical manifest SHA-256 | 286fdf4412111a352974cf01184ed2b0d79f5f908366573a1e128c83a83e4ea9 | 5a749511e9b36051f7c679ebc270cc93ac5798f19ee0ef48c02436b1f34efe97 |
| Dataset SHA-256 | a0ddb09c1f06c6eb7dae621c1a43ba5d0c8b1a2fb8af6d9317002898bd060206 | 1e795296ec0ee9dd0af9a4a8e53de6a82ac975fc5cd3aac3071731c7b84434bc |
| Raw manifest file SHA-256 | 7eca44ece57686427b279db60cc47ee397346826a4b83ca1f1c4d528d8ea3240 | 8b8278620499d38358b8839aaf2ad32e8cd0a1bdf9feb29986992bec3bead733 |
| Report path | runs/sd-v2/dev-short-20261102-issue56/dev-report.json | runs/sd-v2/dev-clinical-20261102-issue56/dev-report.json |
| Environment | CPU scikit-learn; Python 3.12.16 / scikit-learn 1.5.2 / NumPy 1.26.4; seed 20261101 | CPU scikit-learn; Python 3.12.16 / scikit-learn 1.5.2 / NumPy 1.26.4; seed 20261101 |

Both runs used selection-only files mounted read-only in a networkless
container. No mixed records/manifest, test.json, ledger, old campaign artifacts,
or Docker socket was mounted.

These development and CV scores are from the CPU TF-IDF n-gram baseline. Issue
#56 validates the split and evaluation protocol; it did not train a neural
candidate. Neural model comparison and GPU experiments are scoped to Issue #57.

## Post-evaluation hardening

The final hardening edits do not change the split, feature, or scoring
protocol. The latest synthetic regression suite reports 304 passed and 17
skipped: 5 Docker integration tests (no daemon in the offline test container),
4 TCM-SD real-data checks, 3 checks requiring local BERT weights, and 5
syndrome real-data checks. Ruff 0.17.3 and git diff --check pass. The test
container mounted only source, scripts, tests, and pyproject.toml read-only; it
had no data, runs, ledger, holdout, or Docker socket. The full host Docker
worker chain remains NOT_RUN.

## Interpretation

These results describe development-set model selection under protocol v2.
They do not establish clinical validity, research-claim readiness, or holdout
performance. Only the selection records and manifests were used; no holdout
evaluation was performed. The earlier sd-v1 scores are marked
NOT_COMPARABLE_TO_PROTOCOL_V2 in research-sd-v1.md.
