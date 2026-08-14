# Project Status

## Current Phase

Restructure preparation. Backup, authenticated asset import, governance documents, and historical archival are allowed. Core model implementation, algorithm changes, data-pipeline implementation, and training remain out of scope.

## Current Objective

Prepare the historical OPLRI/TCM pretrained assets and a clean three-area repository structure for later company-data fine-tuning into a real-time multimodal massage representation and decision model.

## Confirmed

- The project has moved from paper experiments to an industrial model.
- The current product is a multimodal representation and constrained decision model, not a multimodal LLM.
- The authenticated server OPLRI checkpoint is the first pretrained origin.
- Gate A, Gate B, and TCM Late Reinjection are migration candidates evaluated as M0/M1/M2.
- Pressure remains in the interface but is unavailable in v1.
- Diagnostic and neuro results are independent modalities.
- TCM is an auxiliary prior and never directly controls intensity.
- Program/intensity use hierarchical heads; intensity semantics are gentle, comfortable, and strong.
- Company data remains outside Git. Detailed data contracts wait for actual device/data fields.
- Work areas are `legacy_research`, `representation`, and `massage_decision`.

## In Progress

- Verifying the pre-industrial backup, authenticated asset import, governance documents, and historical archive layout.

## External Inputs Still Unknown

- Company data fields, units, cadence, and missing patterns.
- Tongue/facial diagnostic output fields.
- Neuro-device output fields.
- Massage program catalog.
- Pressure hardware and hardware-state feedback.

## Available Assets

- Server OPLRI checkpoint: SHA-256 `89f75e66d5fa6a65e7158fa3e39e4f886ff13fa7ae37f0f892a3555e6e2f65ba`.
- Server TCM checkpoint: SHA-256 `b5c92665226a127d0a683af47fd782b71e74bafd14ac0d067e55ca4e4c9422f0`.
- Pre-restructure local TCM checkpoint: SHA-256 `c4faa0b48042b2b38a13ca65ff2f8c5db2d999b15280a491225e20b7e72ddfa9`.
- Shared 8D scaler: SHA-256 `41b5af434c410e7b9e98086faa04ece3bbfaa9c50291b59f91e1d5bcb9a81b5b`.

## Known Risks

- OPLRI's historical ECG/EDA input semantics differ from the new task.
- Existing checkpoints did not pretrain the new diagnostic or neuro fields.
- No historical 8D TCM checkpoint exists; all authenticated TCM checkpoints are 4D.
- Gate A/B value on company modalities is not yet proven.
- Historical experiments contain naming conflicts, incomplete checkpoints, and label leakage.

## Storage Notes

- The working-tree copy of the public WESAD dataset was removed after backup verification, reducing the active project from roughly 17 GB to 1 GB.
- The pre-industrial snapshot retains a recoverable WESAD copy; it is not part of the industrial training path.

## Next Step

Review the mechanical restructure and decide whether the two remaining root cleanup candidates (`.DS_Store` and the empty `。` file) should be removed. Core model implementation remains a later milestone.

## Last Updated

2026-08-12
