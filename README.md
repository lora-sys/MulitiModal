# MulitiModal

Industrial multimodal representation and constrained massage-decision model.

The first offline physiological research pipeline is now implemented in
[`physio_autoresearch/`](physio_autoresearch/README.md), following the owner's
shared V0 plan. It provides data QC, subject-level evaluation, four tabular
baselines, bounded candidate rounds, independent scoring, KEEP/REVERT, local
experiment memory, and a restricted Pi SDK bridge. Current acceptance uses
synthetic data only; it does not establish physiological validity or recursive
self-improvement. See the module guide for Windows commands and evidence limits.

The repository is being separated from its historical paper experiments before core product implementation begins:

- [`legacy_research/`](legacy_research/) preserves authenticated pretrained assets, historical code, paper evidence, and known failures.
- [`representation/`](representation/) will own multimodal representation, pretrained migration, missing-modality handling, Gate A/B, and TCM Late Reinjection.
- [`massage_decision/`](massage_decision/) will own program/intensity heads, HOLD, action catalogs, and deterministic decision policy.
- [`data_contracts/`](data_contracts/) will hold versioned schemas without company data.
- [`configs/`](configs/) will hold explicit reproducible configuration.

See [`PROJECT_STATUS.md`](PROJECT_STATUS.md) for the current phase and [`DECISIONS.md`](DECISIONS.md) for accepted long-term decisions. Agent rules begin in [`AGENTS.md`](AGENTS.md).

The representation and massage-decision product models remain a separate later
milestone. The new offline research pipeline does not import historical
experiment implementations or change their pretrained assets.

