---
name: autoresearch
description: Run or monitor a bounded autonomous research loop using a researcher-supplied task, frozen evaluation and one editable experiment file. Use for iterative model or algorithm research; not for one-off training.
---

# AutoResearch

Use this skill to improve a researcher-defined artifact through sequential, evidence-backed experiments. Read `program.md` and the repository's applicable instructions. The skill defines the method; task data, metrics, commands and limits belong in a separate research brief, not in this skill.

Before a campaign, identify the research question, data and permitted partitions, baseline, primary metric and direction, fixed evaluator, editable file, execution environment, per-run and total budgets, constraints, and stop/delivery criteria. Reuse an existing brief when present. Obtain missing information only when it changes the experiment or authorization; do not guess a metric, data provenance or safety boundary.

Establish an unchanged baseline. For each round, read the current artifact and prior results, form a falsifiable hypothesis, modify only the designated file, record its commit, run the fixed evaluation, and keep or revert according to the frozen decision rule. Continue without asking between rounds until the user stops the campaign, the agreed budget is exhausted, or a genuine safety or technical boundary blocks execution. Record failures as failures, not numeric scores. Do not use locked test data to select candidates.

The researcher or trusted environment owns the evaluator, labels, data partitions, metric, budget and promotion criteria. Untrusted code must run in an OS/container boundary; the evaluator and reference labels must be outside its writable environment. If independence from the proposing agent has not been established, do not claim trusted scores or use the system for real-data research. A separate process or a reviewed diff alone does not establish isolation.

Keep a durable experiment history with commit, hypothesis, protocol identity, result, decision, resources and evidence status. Preserve actionable observations with their supporting run and limits; treat them as observations until independently reproduced. A monitor reads those records and reports missing, stale and failed states honestly. It must not control the experiment.
