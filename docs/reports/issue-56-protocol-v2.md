# Issue #56 protocol-v2 development evaluation

**Status: DEV_AND_CV_COMPLETE; SYNTHETIC_TESTS_PASS_WITH_SKIPS; HOST_DOCKER_WORKER_CHAIN_RUN (acam-lab); POST-REVIEW-EVIDENCE-REGENERATED**

The protocol-v2 builder created short and clinical selection-only inputs from
the frozen train/dev users, excluding the exact sd-v1 unblinded user set. Both
CV runs completed successfully. This report uses only the selection records
and manifests, plus the static exposed-user id file derived from the unlock
delivery artifacts. The final test split file, ledger, and real-data
finalization were not accessed.

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

    PYTHONPATH=src python -m scripts.build_sd_campaign_inputs --scenario short --out runs/sd-v2/short --exposed-users-file runs/sd-campaign/sd-v1-exposed-users.json
    PYTHONPATH=src python -m scripts.evaluate_sd_v2_cv --scenario short --out runs/sd-v2/cv-short-<fresh-id>

    PYTHONPATH=src python -m scripts.build_sd_campaign_inputs --scenario clinical --out runs/sd-v2/clinical --exposed-users-file runs/sd-campaign/sd-v1-exposed-users.json
    PYTHONPATH=src python -m scripts.evaluate_sd_v2_cv --scenario clinical --out runs/sd-v2/cv-clinical-<fresh-id>

The evaluator prints a concise run summary and writes cv-report.json under
each requested output directory. Preserve those reports with the review
evidence; they are not checked into Git.

For the frozen train-to-dev benchmark, use a new output directory:

    PYTHONPATH=src python -m scripts.evaluate_sd_v2_cv --mode dev --scenario short --out runs/sd-v2/dev-short-<fresh-id>
    PYTHONPATH=src python -m scripts.evaluate_sd_v2_cv --mode dev --scenario clinical --out runs/sd-v2/dev-clinical-<fresh-id>

The regenerated evidence below was executed directly on the dev host
(acam-lab) with the committed tree; the dev runs used these same arguments.

## Frozen protocol

| Field | Protocol |
| --- | --- |
| Selection pool | Frozen train + dev rows from the selection-only builder artifact, minus the exact sd-v1 unblinded user set (static `--exposed-users-file`, fail-closed) |
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

The sd-v1 delivery unblinded the labels of 4,049 holdout rows (3,905 unique
users). Those users are excluded from the protocol-v2 selection pool by an
**exact static user set**, not a seed replay: a seed replay over this
builder's reduced train+dev population would permute differently than the
original 52,295-user assignment and would leave most exposed users in the
pool. The set is published once as
`runs/sd-campaign/sd-v1-exposed-users.json` (acam-lab), derived by mapping
the unlocked delivery's scored holdout row ids
(`runs/sd-campaign/manifest.json` `split.holdout`) through
`runs/sd-campaign/records.json` user ids; no test file was read. The builder
requires `--exposed-users-file`, validates the file's schema and integrity
(`file_sha256` in build-summary.json), and fails closed on a missing or
tampered file. Exposed users who do not appear in train+dev are recorded as
`n_users_in_population` separately from the file's total `n_users`. The
remaining users receive a fresh internal train/dev split over the clean
population (`data.tcm_sd.subject_partitions`, seed 20261101) — that split is
a development-only assignment and is not the sd-v1 holdout.

## Replay evidence

**Supersession note (2026-11-03 review round):** the previously recorded
evidence (CV short 0.3216 / clinical 0.5544, dev 0.3148 / 0.5544) was produced
by an intermediate builder whose exposed-user exclusion used a seed replay
over the reduced train+dev population; that mechanism is wrong (a per-stratum
reshuffle makes the replayed "test" set almost disjoint from the actually
unblinded sd-v1 users) and its numbers are **NOT_REPRODUCIBLE** from this
tree. All evidence below is regenerated with the committed builder and the
exact static exposed-user file, on the dev host (acam-lab, CPU ovr-liblinear,
Python 3.12.3 / scikit-learn 1.9.1 / NumPy 2.5.4, seed 20261101). Both
scenarios completed with status OK and eight fold runs (4 folds × 2 repeats).

| Evidence | short | clinical |
| --- | --- | --- |
| Status | OK | OK |
| Source revision | fix/issue-56-tcm-sd-eval (post-review fix) | same |
| Selection manifest SHA-256 | f6ac971a9224542551163d692d40ec095b78465dcdd4ea0a67e0bd168b98f04e | 0f57c2ea4c0d273e0e2434d3ce2ee774dd728f1f6676cb848b8aa80f532c49e8 |
| Selection dataset SHA-256 | e63682908e6bed7488ada3cd68cf50b4a1817a4bfbfa28c0a00b74fbc0af4d55 | be4f2f73142e139711fe45d421b720281672d862b93d9e7b98dc8338e9314710 |
| Parent manifest SHA-256 | 093e7336ab4ae9e501ed049cfd169424d522f03f2b7393e25fa3c9e5569dbf30 | 74e47ac658569cede4c5d687cd221eecefb43cb974442e2dbcf6687936798a31 |
| Parent dataset SHA-256 | 7257d9a02f8e7b0bea2dcba32d18836c2cc1b434a76ef9f3b5d88009d3b4853a | eb7e5cab1afca62b21ddd5c0d5e6e006fa44ab7bc6ba7276ab1e3c6417e744de |
| Selection records | 45,026 | 45,043 |
| Selection users with rows / frozen pool users | 43,504 / 43,520 | 43,520 / 43,520 |
| Excluded sd-v1 exposed users (file total / in population) | 3,905 / 3,489 | 3,905 / 3,489 |
| Fold runs | 8 | 8 |
| Fold macro-F1 values | 0.3108, 0.3259, 0.3396, 0.3209, 0.3142, 0.3195, 0.3221, 0.3062 | 0.5406, 0.5614, 0.5578, 0.5568, 0.5446, 0.5464, 0.5503, 0.5457 |
| Mean fold macro-F1 | 0.319921 | 0.550453 |
| Population std. dev. across folds | 0.009593 | 0.006934 |
| Evaluator report path | runs/sd-v2/cv-short-issue56-fix/cv-report.json | runs/sd-v2/cv-clinical-issue56-fix/cv-report.json |

## Development-set benchmark

| Evidence | short | clinical |
| --- | --- | --- |
| Dev macro-F1 | 0.332107 | 0.543784 |
| Dev rows scored | 2,844 | 4,043 |
| Report path | runs/sd-v2/dev-short-issue56-fix/dev-report.json | runs/sd-v2/dev-clinical-issue56-fix/dev-report.json |
| Environment | acam-lab CPU; Python 3.12.3 / scikit-learn 1.9.1 / NumPy 2.5.4; seed 20261101 | same |

The reruns executed directly on the dev host with the committed tree; no
test.json, ledger, or old campaign artifact was read (the exposed-user file
is the only sd-v1-derived input, and it contains user ids only).

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
