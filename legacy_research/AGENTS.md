# Historical research rules

This is the historical research asset area, not the industrial development path. It preserves the minimum credible evidence needed to explain old models, weights, experiments, and decisions. Industrial modules must not import implementations from this directory.

## Default read-only

Original checkpoints, metrics, configs, logs, and code snapshots are immutable. Record later discoveries in `audits/`; never edit history to make an old experiment appear reproducible or correct.

## Evidence levels

- **L1 reproduced** — code, data version, config, checkpoint, scaler, commit, and repeatable result.
- **L2 complete assets, not rerun** — identity and loading contract are complete, but the result has not been reproduced.
- **L3 result evidence, incomplete assets** — logs/JSON/figures exist but checkpoint, config, or data contract is missing.
- **L4 documentation claim** — only prose, comments, or paper claims support it.

Every cited historical conclusion states its evidence level.

## Data provenance

Classify every dataset as exactly one of: `company_collected`, `public`, `simulated`, `mixed`, or `unknown`. Never use unqualified `real` or `realonly`. Describe mixed assets per modality.

## Checkpoints

Every checkpoint manifest records filename, SHA-256, size, source path and machine, backup, commit, model class, architecture, input schema, task, training data, evidence level, trained/frozen parameters, parent checkpoint, known limitations, and intended reuse.

Original and derived assets remain separate. Never fine-tune in place or overwrite an authenticated checkpoint.

## Allowed reuse

`representation` may consume authenticated checkpoints, read-only model snapshots, scalers, schemas, parameter keys/shapes, architecture evidence, failure evidence, and minimal machine-readable metrics. It must not import WESAD loaders, giant experiment entrypoints, label-leaking generators, hard-coded paths, unknown scalers, old task heads, or compatibility fallbacks.

## Reproduction

Run historical reproductions only when explicitly requested and only into a new isolated run. Preserve original and reproduced results together, record environment and commit, and never tune hidden choices merely to match a paper number.

## Local failure modes

- Editing history and presenting it as the original fact.
- Guessing data or checkpoint identity from filenames.
- Letting industrial code depend on paper code.
- Archiving every cache and duplicate in the name of completeness.

