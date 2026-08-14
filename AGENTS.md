# MulitiModal

MulitiModal is an industrial, real-time multimodal representation and constrained massage-decision model. It starts from the historical OPLRI pretrained model and will be fine-tuned on company-collected data to recommend a massage program and one of three intensity levels.

This project is not a multimodal LLM. The current scope is model training, evaluation, and inference. Hardware control, user interfaces, and the complete product system are outside the current scope.

## What we never compromise on

1. **Reuse evidence, not assumptions.** Prefer the historical architecture and pretrained weights, but let company data decide the industrial model. Historical paper metrics do not prove product performance.
2. **Reach a working industrial prototype quickly.** Prefer the smallest end-to-end system that is trainable, testable, and fast. Do not restore a paper-sized experiment matrix or add impressive machinery without a demonstrated need.
3. **Treat missing modalities as a first-class case.** Missing is not zero and low quality is not missing. Use explicit availability masks; never disguise missing sensors with normal defaults or zero waveforms.
4. **Keep data and model assets traceable.** Distinguish company-collected, public, simulated, mixed, and unknown data. Every checkpoint needs provenance, schema, task, commit, SHA-256, and known limitations.
5. **Keep the industrial path separate from research history.** Historical assets live under `legacy_research`; product code must not import historical experiment implementations. Remove superseded paths after replacement instead of maintaining compatibility shells.

## A note from the project owner

We prefer rapid, exploratory progress, but every step should move toward a working product. The project has moved from paper experiments to an industrial model. Reuse the architecture and pretrained weights that were already earned, make only the necessary changes, and optimize for simple code, fast training, low coupling, and high cohesion. Product definitions, labels, modality scope, and long-term rules are decisions we make together; the agent must stop and ask before choosing a material direction alone.

## Shared language

- **I** means the project owner and final product decision-maker.
- **You** means the agent reading this file and doing the current work.
- **We** means the project owner and the currently working agent.
- **Company data** means controlled company-collected data that never enters Git.
- **Historical experiments** means paper and exploratory code, data, results, and models.
- **Pretrained model** means the authenticated OPLRI checkpoint and its semantically transferable parameters.
- **Representation model** converts multimodal human inputs into a unified human-state representation.
- **Decision model** recommends massage program and intensity from the representation and current massage context.
- **Profile context** is low-frequency background such as age, sex, height, weight, and BMI.
- **Current state** is the changing state expressed by vital, diagnostic, neuro, and future pressure modalities.
- **Diagnostic state** means tongue, facial, or diagnostic-device results. Do not use the ambiguous historical name `static_scores` for new contracts.
- **TCM prior** is the nine-dimensional constitution prior produced by the TCM model.
- **Gate A** conditionally modulates current state with the TCM prior.
- **Gate B** corrects current state using state and modality quality.
- **Late Reinjection** reintroduces the effective TCM prior immediately before product decision heads.
- **HOLD** means keep the current program or intensity.
- **Industrial prototype** is the smallest model that trains on company data, supports missing modalities, and performs real-time inference.
- **Best model** is meaningful only with a named dataset version, task, split, metrics, and deployment constraints.

## The six easiest ways to hurt this project again

1. Calling public, simulated, company, or mixed data simply "real data."
2. Hiding missing modalities behind normal defaults, means, or zero waveforms.
3. Using `strict=False` and claiming pretrained loading succeeded without an explicit parameter report.
4. Splitting windows or sessions from one person across train and test.
5. Letting paper loaders, hard-coded paths, giant experiment scripts, or compatibility fallbacks leak into the industrial path.
6. Mixing code, company data, checkpoints, logs, figures, and generated paper artifacts in one commit.

## Check every affected surface

For each model change, state which of these apply:

- modality semantics, order, units, shape, update cadence, availability, and quality;
- exact, partial, semantic-mismatch, reinitialized, disabled, and unused pretrained parameters;
- identical preprocessing across training, validation, and inference;
- fair M0/M1/M2 variation, changing only the planned gates;
- complete, field-missing, modality-missing, and low-quality inputs;
- program, three-level intensity, HOLD, confidence, and legal action combinations;
- latency, size, stability, NaN handling, and degradation behavior;
- config, data version, parent checkpoint hash, load report, and metrics provenance.

## Data and training

- Company data never enters Git. Supply it through one explicit configuration or environment variable; do not guess paths through fallback chains.
- Fit preprocessing and scalers on the training split only.
- Keep every person's sessions in one data split.
- Synthetic tensors prove structure only, never product performance.
- Record the parent checkpoint hash and every frozen or unfrozen parameter group.
- The product fine-tuning entry may coordinate `representation` and `massage_decision` end to end, while their module responsibilities remain separate.

## Verification

Use the smallest proof that demonstrates the change. Schema changes need contract validation; adapters need shape, mask, missing-field, and gradient tests; checkpoint work needs a parameter mapping report and deterministic forward pass; gate work needs controlled M0/M1/M2 comparisons; head work needs program, intensity, HOLD, and legality tests. Bug fixes require a focused regression test.

Do not describe synthetic forward success, training-set fit, or a historical metric as product validation. Do not run the full historical matrix unless explicitly requested.

## Git and commits

- Do not commit, push, or open a pull request unless explicitly requested.
- Preserve existing user changes. Never clean the working tree implicitly.
- One purpose per commit. Use clear Conventional Commit titles.
- Important commit bodies explain: problem, changes, verification, data impact, model impact, and checkpoint impact.
- Never use messages such as `update`, `fix`, `!`, `..`, `finally fix`, or `last train`.
- Keep checkpoint import, historical archival, governance, and product refactoring in separate commits.
- Company data, `runs/`, large logs, caches, duplicate figures, and unregistered checkpoints do not belong in ordinary Git history.

## How it works

```text
company multimodal inputs
  -> feature schemas and modality adapters
  -> current-state representation
  -> Gate A (TCM conditioning)
  -> Gate B (state and quality correction)
  -> profile-context fusion
  -> patient representation
  -> TCM late reinjection
  -> decision representation
  -> program and three-level intensity heads
  -> constrained massage action
  -> future hardware adapter
```

M0, M1, and M2 share one pretrained origin and differ only in Gate A/B usage. They are selection variants, not three permanent product models. The model proposes a constrained action; it never directly controls motors.

## Where things live

- `legacy_research/` — immutable historical evidence, experiments, and authenticated pretrained assets.
- `representation/` — modality adapters, missing-modality handling, pretrained migration, Gate A/B, late reinjection, and human-state representation.
- `massage_decision/` — program and intensity heads, HOLD, confidence, catalog constraints, and decision policy.
- `data_contracts/` — schemas, units, missing rules, labels, and anonymous examples; never company data.
- `configs/` — explicit reproducible training and inference configuration.
- `runs/` — generated training outputs and load reports; ignored by Git.
- `PROJECT_STATUS.md` — current phase, blockers, pending decisions, and next step.
- `DECISIONS.md` — accepted long-term product and architecture decisions.

## Taste

- Reuse validated architecture; do not preserve historical mistakes.
- Concentrate complexity at modality adapters and checkpoint mapping.
- The representation module does not know chair protocols; the decision module does not parse raw sensor files.
- One scalar is not one modality. Group by source, semantics, and update cadence.
- Matching shape does not imply matching meaning.
- A checkpoint existing does not prove its parameters participate in forward or training.
- Historical best does not mean industrial best.
- Add modalities through explicit adapters without breaking other contracts.
- Do not add large Transformers, MoE, generative imputation, or online RL because they look advanced.
- Training speed, inference latency, and adjustment cost are architecture metrics.
- Verify existing dependencies before adding or rebuilding common capabilities.
- If a rule conflicts with the actual product need, stop and get owner approval.

