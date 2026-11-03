# AutoResearch Skill

Guide a Coding Agent through an AutoResearch campaign against **this repository's
module** (`src/research/`). The Skill only tells an agent *how to call* the
module; all research logic, training, diagnosis and storage live in the module
itself. Installing or invoking this Skill does not modify lora-mode or any other
project.

## When to use it

Use it when the user asks to "run research", "optimize the model", "hill-climb
the metric", "keep/roll back experiments", or wants a live view of a campaign.
Do not use it for one-off manual training runs — that is `runtime.cli smoke`.

## What the agent must not do

* Do not edit data, labels, splits, the evaluator, the metric, the budget or the
  holdout. They are frozen, and a patch that reaches them is refused.
* Do not produce a batch of pre-planned experiments and call it autonomous
  research. One round = one hypothesis, proposed **after** reading the previous
  round's result.
* Do not trust a score the candidate reports. The trusted evaluator scores, in a
  separate process.
* Do not treat `INVALID_PROTOCOL`, `UNSUPPORTED_ARCHITECTURE` or `CRASH` as
  evidence that the model is worse.

## Procedure

1. **Pick the task.** `toy` is the TCM-free example task and works anywhere
   (synthetic data, CPU, no download). Real tasks (`constitution`, `syndrome`)
   need their own frozen manifest and dataset on the dev host.

2. **Prepare a run** (creates a dedicated git worktree on
   `autoresearch/<name>`, writes the frozen spec, materializes the toy inputs,
   and runs the Doctor's preflight):

   ```bash
   python -m research.cli run --task toy --out runs/toy-1 --repo .
   ```
   It exits `1` with `"blocked": true` if the Doctor refuses the protocol. Fix
   what it names; do not work around it.

3. **Run the campaign.** With a real Coding Agent CLI available:

   ```bash
   python -m research.cli run --task toy --out runs/toy-1 --repo . \
     --agent-command "claude -p" --max-rounds 8 --budget 20 --folds 4 --repeats 2 \
     --dashboard --port 8765
   ```
   The agent command reads the prompt on stdin and prints one JSON object
   (see `program.md`). If no agent CLI exists, drive it one round at a time —
   `run --block 0` prepares, then per round:
   read `status`, write the proposal JSON, then `step`.

   ```bash
   python -m research.cli status --run runs/toy-1
   python -m research.cli step  --run runs/toy-1 --proposal-file p.json \
     --proposer "<your-agent-id>"
   ```

4. **Watch it.** `python -m research.cli dashboard --run runs/toy-1 --port 8765`
   serves a read-only page on `http://127.0.0.1:8765/`. Tell the user that URL.
   It polls every 3 s, shows the current stage/fold, the hypothesis, the trusted
   metric curve, KEEP/REVERT history and the Doctor's findings, and marks a run
   `STALE` when the heartbeat stops. Closing the tab does not affect training.

5. **Stop / resume.** `stop --run <dir>` asks the run to stop and ends the live
   training child. `resume --out <dir> ...` continues from the manifest, never
   replaying a finished round. `report --run <dir>` writes
   `report.json` + `report-<run_id>.md` from persisted state only.

## What to look at when a round is not kept

Read the Doctor's findings before proposing anything new:

* `BLOCKED` — the protocol is not trustworthy; the campaign already stopped.
  Fix the named check, then start a **new** campaign.
* `WARNING` + `measurement_resolution` — the difference is smaller than one
  standard error. Change the information available (features, labels, data) or
  raise the resolution (more folds/repeats); do not lower the margin.
* `WARNING` + `per_class_errors` / `class_support` — a macro average is being
  dragged by a tail class. That is a task property, not a model property.
* `UNVERIFIED` — the check could not be performed. Never read it as a pass.

## Limits to state honestly

* A run driven by `--allow-stub` is **not** autonomous research; the manifest and
  the report both say `autonomous_research: false`.
* The dashboard is read-only and loopback-only. Remote access, authentication
  and sharing are explicitly out of scope for this module.
* The toy task's labels are synthetic: a rising metric there says the plumbing
  works, nothing more.
* Holdout evaluation is a separate, budgeted step (`finalize`); the search never
  reads the locked test.