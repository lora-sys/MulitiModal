# Research loop contract

The research brief is the authority for one campaign. It must name:

- **Goal and output:** the research question, intended artifact and what success would mean.
- **Data:** source and permission, input/output schema, training and selection partitions, protected test data and any grouping or preprocessing rules.
- **Measurement:** baseline, fixed evaluator and version, primary metric and whether higher or lower is better, comparison and promotion rules, and the evidence needed for a claim.
- **Change boundary:** the single editable file, frozen inputs and evaluator, dependency and architecture constraints, and how the trusted evaluator is kept outside the agent's writable environment.
- **Execution:** setup and exact baseline/evaluation/record/monitor commands, runtime and resource limits, seed policy, total budget, logs and artifact locations.
- **Stopping and handoff:** when to stop, what needs human judgment, and which code, evidence and unresolved questions to retain.

Do not start a real campaign if an essential field is unknown. Make unknowns explicit. The researcher sets these boundaries; the agent can choose experiments within them.

## Procedure

1. Create a dedicated clean experiment branch/worktree and an ignored run directory. Record the task brief and the version or hash of each frozen input. Confirm the evaluator's independence and execution boundary for real data.
2. Evaluate and record the unchanged baseline. A synthetic smoke run proves only that the mechanics work.
3. Read the latest champion, prior outcomes and current editable file. Choose one testable hypothesis and the smallest change that can distinguish it. Commit that file on the experiment branch.
4. Run the fixed evaluation with the same protocol and resource budget. Capture the actual result or failure and the candidate commit. Do not derive the score from candidate-reported values when the evaluator is the authority.
5. Append the result and decision to the run history. Keep the candidate only when it satisfies the brief's promotion rule; otherwise reset the dedicated clean worktree to the prior champion. A crash, timeout or invalid protocol is not a model score.
6. Write down a useful observation with its evidence and scope, then use the result to select the next hypothesis. Continue until the specified stop condition; do not repeatedly ask the researcher whether to proceed.

The read-only monitor consumes the run's status and results; it must clearly show the task, metric, direction, evidence state, failures and missing values. Human review decides whether an observation becomes reusable practice and whether an artifact is ready for downstream use.
