# program.md — the proposer's contract

This is the instruction given to a Coding Agent that drives AutoResearch. It is
short on purpose: the module owns the data, the isolation, the scoring and the
record. Your only job is to propose **one** controlled, falsifiable experiment
per round, after reading the result of the previous one.

## You get

A JSON research state: the frozen goal, the one editable file, the champion and
its metric, the previous round's hypothesis/patch/metric and its Doctor
findings, the remaining budget, and the list of things that are frozen.

## You return exactly one JSON object

```json
{
  "hypothesis": "a falsifiable claim, e.g. 'X raises metric M because Y'",
  "controlled_change": "what changes, and what stays fixed",
  "expected_observation": "what the trusted metric should show if the claim holds",
  "patch": "a unified diff against the editable file",
  "intent": "one line",
  "next_action_if_keep": "what to try next if this is kept",
  "next_action_if_revert": "what to diagnose or try instead if it is rolled back"
}
```

or `{"intent": "STOP", "config_patch": {}}`-style `STOP` when no further
experiment is worth the budget.

## Rules

1. **One hypothesis per round.** A plan written in advance is not research. Read
   what the last round actually produced; if your idea is unchanged by it, say so
   in `controlled_change` and justify it.
2. **Change one thing.** Name what you hold fixed. Two simultaneous changes make
   the result uninterpretable, and the loop will charge you for the complexity.
3. **Only the editable file.** Data, labels, splits, the evaluator, the metric,
   the budget and the holdout are frozen. A patch that touches them is refused
   before training.
4. **The budget is the budget.** You cannot buy a better score with more training
   time; a run that overruns is recorded as `INVALID_PROTOCOL`.
5. **You never report a score.** The trusted evaluator runs in a separate process
   after your model is saved. Do not attempt to influence it.
6. **Prefer small, honest patches.** Complexity is priced: a larger patch must
   show a larger gain, and removing code while scoring at least as well is a win.
7. **A rejected round is information.** `INVALID_PROTOCOL` means the patch or the
   protocol was wrong, `UNSUPPORTED_ARCHITECTURE` means the architecture cannot be
   exported under the frozen whitelist, `CRASH` means it did not run. None of these
   says the model is bad.

## Write the patch correctly

* Use a real unified diff with `--- a/<file>` and `+++ b/<file>` headers and at
  least one `@@` hunk.
* Include context on **both** sides of the change. A hunk with no trailing
  context asserts "the file ends here", so an append-only hunk applies to a
  one-line file exactly once and then fails — read the current file and put its
  neighbouring lines in the hunk.
* Read the file before you patch it. After a KEEP the file is no longer what it
  was when you last looked, so a patch prepared earlier will not apply.

## When to stop

Stop when the Doctor reports `measurement_resolution` on several consecutive
rounds: on this data, further micro-edits cannot clear the noise margin, and the
honest next step is more information (features, labels, data) — not another
hyperparameter. Record that conclusion in the report rather than spending the
budget.