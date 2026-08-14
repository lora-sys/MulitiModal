# Decisions

All decisions below are accepted unless marked otherwise. Superseded decisions remain in this file with their replacement ID.

## D-001: Move from paper experiments to an industrial model

- Status: accepted
- Date: 2026-08-12
- Decision: Current scope is massage-model training, evaluation, and inference. Paper completeness is no longer the product goal.
- Rationale: The project now targets a working company prototype.

## D-002: Build a representation and constrained decision model

- Status: accepted
- Date: 2026-08-12
- Decision: The product is not a multimodal LLM and does not generate low-level actuator sequences.

## D-003: Use the authenticated OPLRI checkpoint as the first pretrained origin

- Status: accepted
- Date: 2026-08-12
- Decision: Transfer only parameters whose shape and semantics match; report every mapping explicitly.
- Evidence: SHA-256 `89f75e66d5fa6a65e7158fa3e39e4f886ff13fa7ae37f0f892a3555e6e2f65ba`, source commit `f743da3ab3d78df8cb3206047d8b507872ba87f0`.

## D-004: Preserve the later OPLRI ideas as migration candidates

- Status: accepted
- Date: 2026-08-12
- Decision: Gate A, Gate B, and TCM Late Reinjection remain candidates rather than being discarded for a from-scratch simple model.

## D-005: Select one model through M0/M1/M2

- Status: accepted
- Date: 2026-08-12
- Decision: M0 disables both gates, M1 enables Gate A, and M2 enables Gate A/B; all retain Late Reinjection and share one pretrained origin. Only one industrial model remains after controlled evaluation.

## D-006: Use explicit industrial modality groups

- Status: accepted
- Date: 2026-08-12
- Decision: Use `profile_context`, `vital_state`, `diagnostic_state`, `neuro_state`, and `pressure_dynamic`, with `constitution_tcm` as an auxiliary prior branch.

## D-007: Keep pressure unavailable in v1

- Status: accepted
- Date: 2026-08-12
- Decision: Preserve the pressure interface with availability zero. Do not substitute zero waveforms or normal defaults.

## D-008: Keep diagnostic and neuro results independent

- Status: accepted
- Date: 2026-08-12
- Decision: Company tongue/facial results enter `diagnostic_state`; derived neuro results enter `neuro_state`. Neither is forced into the old time-series encoder.

## D-009: Define Gate A, Gate B, profile fusion, and Late Reinjection roles

- Status: accepted
- Date: 2026-08-12
- Decision: Gate A conditions current state with TCM; Gate B performs quality-aware state correction; profile joins after the gates through concat plus projection; only the effective 9D TCM prior is late-reinjected through 128+9 -> 128.

## D-010: Use hierarchical program and intensity heads

- Status: accepted
- Date: 2026-08-12
- Decision: One shared decision representation feeds a program/HOLD head and a program-conditioned intensity/HOLD head. Intensity semantics are gentle, comfortable, and strong. Program classes wait for a versioned catalog.

## D-011: Separate three work areas

- Status: accepted
- Date: 2026-08-12
- Decision: Use `legacy_research`, `representation`, and `massage_decision` with one-way dependency from authenticated history toward product modules.

## D-012: Keep company data outside Git

- Status: accepted
- Date: 2026-08-12
- Decision: Git stores contracts, anonymous examples, versions, and hashes only.

## D-013: Layer project governance

- Status: accepted
- Date: 2026-08-12
- Decision: Root `CLAUDE.md` references root `AGENTS.md`; each work area has local rules; `PROJECT_STATUS.md` stores dynamic state; this file stores long-term decisions.

## D-014: Remove unrelated WishLive material after backup

- Status: accepted
- Date: 2026-08-12
- Decision: WishLive under `paper/hackthon` does not enter the historical massage archive.

## D-015: Restructure safely and avoid compatibility shells

- Status: accepted
- Date: 2026-08-12
- Decision: Freeze baseline and assets, establish the new path, archive history, then remove superseded material. Do not maintain long-term compatibility fallbacks.

## D-016: Keep module ownership separate but fine-tune through one composition entry

- Status: accepted
- Date: 2026-08-12
- Decision: `representation` owns the backbone and `massage_decision` owns product heads; a single product fine-tuning composition may train them end to end without adding a fourth top-level work area.

## D-017: First implementation milestone is migration skeleton verification

- Status: accepted
- Date: 2026-08-12
- Decision: The first implementation milestone will authenticate and map pretrained assets and verify M0/M1/M2 structural forward paths without company data or training. Core implementation is not part of the current restructure-preparation phase.

