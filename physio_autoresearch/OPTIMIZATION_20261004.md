# Code optimization and experiments — 2026-10-04

Baseline: PR20, `d6abaf1065d6e2359c37a80b96d5902e5baae4b8`.
Branch: `optimize/subject-scoring-20261004`. This second-stage change depends on
[PR20](https://github.com/lora-sys/MulitiModal/pull/20) and targets its
`review/physio-v0-20261004` branch while that PR remains open. The original
`E:\zzzz\MulitModel` working tree is preserved.
No dependencies, model architecture, fixed-search parameters or metrics were changed.

## Retained changes

1. Separate dependency versions from full environment provenance. `load_round`
   still hashes current source/data/contracts and reads current dependency/lock
   versions on every call, but no longer starts Git or repeats the source scan.
   Provenance remains complete at ingestion and reviewed upgrades. No global cache.
2. Reject prediction arrays with a wrong sample/target axis instead of reshaping
   them into apparently valid results. Ordinary sklearn 1-D single-target outputs
   and exact 2-D outputs remain supported.
3. Reject NUL-containing CSVs: the pandas C parser can silently truncate IDs and
   numeric cells; two distinct subjects could previously collapse into one.
4. Reject contracts whose input names collide with generated prediction names.
   IDs or measured targets could otherwise be overwritten or receive merge suffixes.
5. Apply the existing maximum-depth caps when a tree candidate omits depth.
   Previously RF defaulted to unbounded depth and XGBoost to depth 6, bypassing
   V0's caps of 6 and 3. Explicit fixed-search configurations remain unchanged.
   This is bounds enforcement, not a claim that those defaults improve accuracy.

Production delta: four source files, 18 added / 4 removed lines. Tests and
reproduction scripts are separate from production code.

## Single-factor measurements and rejected candidates

| Factor | Result | Decision |
|---|---|---|
| Round integrity check | Final warm median 76.17 → 26.01 ms (2.93×); identical return values; one source scan, zero Git calls instead of two scans and one Git call | Retain |
| Score all targets together | Six-subject, single-target case 1.37 → 2.35 ms; 1000-subject/16-target case faster but Python allocation peak grew about 52% | Reject for current workload |
| Single-output prediction dispatch | Three seeds: baseline 1.05–1.12 ms → 0.86–0.90 ms; about 0.20 ms saved, equal predictions | Defer; new wrapper/serialization dependency is not justified here |
| Omit intermediate float mask cast | Paired timing gains vary from negative to positive; float32 behavior needs an explicit float64 cast elsewhere | Reject; no stable measured gain |

Round-check Python allocation peaks: 1,153,253 → 1,153,315 bytes, effectively
unchanged. These are `tracemalloc` peaks, not process RSS. Timings exclude Python
startup; they are not an end-to-end training or hardware latency claim. An earlier
round-check run measured 3.07×; both raw runs are retained.
The benchmark compares baseline loader/provenance functions with the optimized
loader using the same current helpers and snapshot, isolating that factor. It is
not a timing comparison of two complete historical checkouts.

Scoring and inference candidates use seeds 42/43/44, fixed inputs and paired or
randomized timing order. Round validation reuses one frozen six-subject snapshot
with two sessions each and seed 42. These factors were measured separately.

## Validation and prediction equivalence

- **69 Python tests passed** (14 more than PR20), **4 offline Pi tests passed**.
- Independent GPT-6.1 Sol Ultra read-only review found no blocker in the final diff.
- Replayed all **12** fixed six-subject candidates, across all four model families,
  with identical data/contract/split hashes and seed 42. Every held-out prediction,
  evaluation and KEEP/REVERT decision matched exactly; maximum prediction delta 0.
- Champion MAE: **2.112936226052805 → 2.112936226052805 bpm**. All 12 model sizes
  matched; champion remained 2,027 bytes. Replay added zero training attempts.
- `git diff --check` passed. No model API calls, company data, new installations,
  historical Torch matrix or checkpoint migration were used.

## Reproduction and evidence

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/physio_autoresearch --basetemp runs/optimization/final-tests -q --junitxml=runs/optimization/final-python.xml
npm run test:pi
& .\.venv\Scripts\python.exe -m scripts.benchmark_round_validation
& .\.venv\Scripts\python.exe -m scripts.compare_physio_pipeline
```

If a previously frozen benchmark round was created before these source changes,
inspect the diff first and invoke `benchmark_round_validation --review-code-upgrade`
explicitly; the normal command refuses an unreviewed source change. Pipeline
comparison uses the preserved first-stage synthetic snapshot below `runs/physio-v0`.
It verifies that the recorded baseline source hashes match PR20's source blobs,
and that baseline/current dependency versions match. A fresh checkout must first
generate this synthetic baseline using `scripts.physio_experiments` at PR20, then
preserve its ignored `runs/physio-v0` artifacts for the comparison. The recorded
baseline environment predates the commit; matching source hashes authenticate it.

Raw evidence below ignored `runs/optimization/`:

- `round-validation/comparison.json` and `comparison-before-correctness-guards.json`;
- `pipeline-comparison/comparison.json`, complete candidate models/predictions/journal;
- `final-python.xml`;
- `rejected-vectorization/worker.py`, benchmark script and 12-case result;
- `profile-agent/benchmark_inference.py` and `inference-results.json`.
- `generalization-audit/negative_controls.py` and `results.json`.

## Generalization audit before publication

No ordinary fold leakage was found in the current path. `data.py:70-89` groups
all sessions sharing a supplied subject ID into one LOSO validation fold and
checks complete, disjoint splits. `trusted/worker.py:33-37` creates and fits a
fresh pipeline on training rows; `models.py:37-38` includes the imputer/scaler.
`trusted/worker.py:98-108` verifies evaluator-owned folds and predicts held-out
subjects with fold models. The separately fitted full-data model is for inference.
`contracts.py:66-70,128-138` limits candidate inputs to declared pre features.

**Absence of overfitting is not established.** `harness.py:267-278,245-250` reuses
the same development LOSO scores for 12 candidates and KEEP/Champion selection.
There is no independent final test or nested outer assessment. Candidate scores,
including the historically recorded 13th proposal's lower score, remain selection
scores and can be optimistic. An evaluator subprocess is not an independent sample.
The synthetic target is deliberately `0.8 * pre_hr + 8 + 0.03 * age + noise`.
At the same session count, seed-42 six-subject generation includes the earlier
five-subject rows; those demonstrations are not independent cohorts or final tests.

Record-ID uniqueness does not detect copied windows under new IDs or a person
assigned multiple subject IDs. The future-feature name/phase rules do not verify
actual measurement timestamps or lineage. These require an upstream real-data
audit. The six-subject synthetic fixture had no duplicate feature/target rows;
intentional mixed-subject folds, subject ID features and post-target features were
rejected in the negative controls.

A fixed ElasticNet (`alpha=0.1`, `l1_ratio=0.2`, model seed 42), without search,
was evaluated on a separate six-subject/two-session synthetic fixture. Its mean
training subject MAE was 0.8097 and LOSO MAE 2.2892 bpm; Naive LOSO MAE was 9.5874.
Whole-subject label permutations at seeds 42/43/44 produced ElasticNet LOSO MAE
7.5521/17.2349/8.1232, versus Naive 9.5874. All three results are retained, including
permutations that still beat Naive by chance. Train-only imputer/scaler statistics
were checked for each fold. This tiny diagnostic is not a significance test or a
real generalization proof; the training/validation gap and selection risk remain.
Run the preserved local diagnostic with:

```powershell
& .\.venv\Scripts\python.exe runs/optimization/generalization-audit/negative_controls.py
```

## Remaining accuracy work

These changes preserve valid synthetic predictions and improve validation
reliability; they do not prove real physiological accuracy. Real evaluation needs
an explicit data path and approved contract (provenance, measured post outcomes,
pre-only inputs, units/scales and primary endpoint). Keep all sessions of a person
together, select configurations on development subjects, then evaluate once on
separate future/held-out subjects. Do not tune on that final test set.

Equivalent numeric parameter representations such as `alpha=1` and `alpha=1.0`
still have different candidate hashes. Broader default/identity normalization is
deferred; no round budget, historical record or existing asset was rewritten.
