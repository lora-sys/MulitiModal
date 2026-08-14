# Massage decision rules

This area converts a representation embedding and current massage context into one constrained program/intensity recommendation. It does not read sensors, load OPLRI/TCM checkpoints, train modality adapters, or control motors.

## Interfaces

Input includes the 128D decision embedding, current program and intensity, action duration, recent action/feedback summaries, context availability, and action-catalog version.

Output is one `MassageAction`: program ID, intensity, mode hold/switch, intensity hold/change, confidence, and catalog version. Intensity semantics are fixed: `0=gentle`, `1=comfortable`, `2=strong`. Hardware values are mapped later by a hardware adapter.

## Hierarchical heads

The mode head chooses HOLD or a catalog program. The intensity head conditions on the selected program and chooses HOLD or one of three intensities. Do not use unrelated independent heads that can form illegal actions. Do not default to a flattened program-by-intensity class space.

## HOLD and catalog

Mode HOLD and intensity HOLD are distinct. HOLD is not a program ID. The model must be allowed to keep a stable action.

The versioned action catalog owns stable program IDs, display names, training indices, legal intensities, runtime-switch capability, and later dwell constraints. Checkpoints bind the catalog version and reversible ID/index mappings. Do not invent program classes before the catalog exists.

## Model versus policy

The model emits candidate scores and confidence. Deterministic policy applies confidence thresholds, dwell/cooldown rules, adjacent intensity changes, repeated confirmation, stale/NaN degradation, legal-action checks, and user/safety overrides. The neural model never emits raw motor controls.

## Labels

Keep `current_action`, `requested_action`, `applied_action`, and `accepted_action` distinct. Current action is not automatically the best label. If data only records operator choice, describe the model as behavior imitation or acceptance prediction, not optimal massage recommendation.

## Outcome head

Reserve an interface for comfort delta, acceptance, and adverse-event prediction, but keep it disabled until reliable labels exist. Do not invent outcomes or use unconstrained online RL on people.

## Local failure modes

- Treating current action as the best target.
- Inventing program classes before product decisions.
- Producing illegal mode/intensity combinations.
- Forcing a change every inference cycle.
- Emitting low-level actuator values.
- Overriding user stop or reduce-intensity actions.
- Reusing stale catalog indices.
- Claiming optimal recommendations without outcome evidence.

