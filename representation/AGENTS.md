# Representation model rules

This area converts multimodal human inputs into a stable 128-dimensional decision representation using authenticated historical pretrained assets. It does not define chair programs, hardware protocols, or company-data storage.

## Stable interface

Inputs are versioned groups: `profile_context`, `vital_state`, `diagnostic_state`, `neuro_state`, and `pressure_dynamic`, each with values, availability, and quality. Outputs include the 128D decision embedding, availability/quality summaries, nine-dimensional TCM prior, TCM confidence, and active M0/M1/M2 variant.

## Modality responsibilities

- `profile_context` is low-frequency background and joins after Gate A/B.
- `vital_state` contains changing vital results and joins current state.
- `diagnostic_state` is an independent tongue/facial/diagnostic-result modality; never reuse the ambiguous new name `static_scores`.
- `neuro_state` contains derived neuro results. Never feed them to the historical ECG/EDA or pressure encoder.
- `pressure_dynamic` keeps a stable interface but is unavailable in v1. Never feed a zero waveform to simulate presence.

## Missingness and quality

Hard availability masks apply before learnable gates. Quality describes present-but-unreliable data. A learnable gate cannot restore masked information. Do not invent precise quality values without a device or business rule.

## TCM migration

TCM is an auxiliary prior, never a controller. Transfer tokenizer rows only where field semantics match. CLS, Transformer layers, normalization, and classifier may initialize compatible models. New fields require new tokenizer parameters and missing embeddings. Shape equality alone never authorizes loading.

## Representation path

Current-state adapters and mask-aware fusion feed Gate A, then Gate B. Profile context joins through concat plus projection. The effective nine-dimensional TCM prior is late-reinjected through the historical 128+9 -> 128 contract.

- Gate A uses the TCM prior to condition current state, not profile context.
- Gate B corrects current state using state and a small quality bias; it does not determine availability.
- Late Reinjection reintroduces only the availability/confidence-adjusted TCM prior, not every raw feature.

## Pretrained loading

Never claim success from `strict=False`. Report `loaded_exact`, `loaded_partial`, `shape_mismatch`, `semantic_mismatch`, `reinitialized`, `disabled`, `unused_from_checkpoint`, and `missing_in_checkpoint`, plus loaded, active, frozen, trainable, disabled, and newly initialized parameter totals.

## Fine-tuning stages

0. Verify hashes, mapping, masks, disabled modules, and variant differences without training.
1. Freeze historical parameters; train only new adapters, fusion, missing/quality mechanisms, and task heads.
2. Unfreeze Gate A/B and Late Reinjection with a lower learning rate.
3. Only with data and subject-wise evidence, unfreeze TCM projection or final Transformer layers. Do not default to full unfreezing.

## Variants

- M0: no Gate A/B; Late Reinjection enabled.
- M1: Gate A enabled; Gate B disabled; Late Reinjection enabled.
- M2: Gate A/B and Late Reinjection enabled.

All variants share checkpoint, data, split, adapters, heads, budget, freezing, seed, and stopping rules. `M2-from-scratch` is a low-cost proof of pretraining value, not a fourth product model.

## Local failure modes

- Feeding derived neuro scalars into a time-series encoder.
- Pretending pressure exists with zeros.
- Loading parameters by shape despite semantic mismatch.
- Letting TCM directly choose intensity.
- Changing gates, encoders, heads, and budget in one ablation.
- Retaining failed variants behind compatibility layers.
- Calling a synthetic forward pass product validation.

