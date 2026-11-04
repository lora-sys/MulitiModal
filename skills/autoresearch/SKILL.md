---
name: autoresearch
description: Run, watch, stop and report an AutoResearch campaign on this repository (bounded source-level experiments with a trusted evaluator and KEEP/REVERT). Use when the user says 跑研究 / 自动研究 / AutoResearch / RSI / 爬山 / 调模型 / 加指标 / 做实验并保留最好的 / keep or revert / watch the training live, or asks to optimize a model over many rounds, diagnose a run, or open the live dashboard. The module drives real training and scoring; this skill only tells you which CLI commands to run and in what order.
---

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

* **Editable surface v2 (2026-11-04):** you may edit ANY `src/**/*.py` model,
  training or feature-engineering source file, and may add new Python modules.
  `editable_file` in the state is the anchor candidate, NOT a limit. See the
  state's `frozen_paths` denylist: data, labels, splits, the evaluator, the
  metric, the budget, the holdout ledger, governance docs, tests and the trusted
  engine are frozen, and a patch that reaches them is refused. See
  `docs/reports/2026-11-04-editable-surface-v2.md` for the authorisation.
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
   and runs the Doctor's preflight). `prepare` needs NO Coding Agent:

   ```bash
   export PYTHONPATH=src
   python -m research.cli prepare --task toy --out runs/toy-1 --repo .
   ```
   It exits `1` with `"blocked": true` if the Doctor refuses the protocol. Fix
   what it names; do not work around it. Call this ONCE per run — a second
   `prepare` on the same `--out` is refused.

3. **Run the campaign — the agent that loaded this Skill is the proposer.**
   This is the intended flow, and it needs no external agent CLI. Once per
   round, in this order:

   ```bash
   python -m research.cli state --run runs/toy-1          # FULL proposer state:
   #   frozen_goal, editable_scope, frozen_paths, champion, last_round
   #   (metric/decision/reason/diagnostics/doctor), remaining_budget
   # decide ONE hypothesis that reacts to that state, write it to p.json
   # (see program.md), then:
   python -m research.cli step --run runs/toy-1 --proposal-file p.json \
     --proposer "<your-agent-id>"
   ```
   `state` is the machine-readable `research_state()`; `status` is only the
   coordinator's live summary and does NOT contain `last_round` or the Doctor
   findings. `step` enforces the frozen round/wall-clock budget, honours a
   `stop` request, supports a `{"intent":"STOP"}` proposal, applies exactly
   one hypothesis, trains, scores with the frozen evaluator, records the
   decision and prints the NEW state. So the NEXT hypothesis is always written
   after the previous round's result exists. That is what `batch_supplied: false`
   plus the recorded `context_sha256` / `proposal_sha256` certify.

   An optional `--agent-command <cmd>` on `run` exists for a fully automated
   campaign (it reads the prompt on stdin and prints one JSON object). It is
   **not used in the Skill-driven flow**: the agent driving the session is the
   proposer. `run` without `--agent-command` fails closed by design.

4. **Watch it.** `python -m research.cli dashboard --run runs/toy-1 --port 8765`
   serves a read-only page on `http://127.0.0.1:8765/`. Tell the user that URL.
   It polls every 3 s, shows the current stage/fold, the hypothesis, the trusted
   metric curve, KEEP/REVERT history and the Doctor's findings, and marks a run
   `STALE` when the heartbeat stops. Closing the tab does not affect training.

5. **Prove the page really paints.** The acceptance step is a picture, not an
   assertion:

   ```bash
   python scripts/autoresearch_screenshot.py \
     --url http://127.0.0.1:8765/static --out shots/<run_id>.png --size 1280x1750
   ```
   It uses a headless browser the host already has (chromium / chrome / snap) and
   writes a PNG; exit `2` with `NOT_AVAILABLE` means no browser exists, which is a
   stated gap rather than a silent pass. Point it at `/static`: the live page
   repaints every three seconds, so an automated capture of it may never settle.

6. **Stop / resume.** `stop --run <dir>` asks the run to stop and ends the live
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

## Upstream visual skill: lieflat-charts (charts)

The dashboard's charts are not styled by taste. They follow **lieflat-charts**, the
template-driven data-visualisation skill:

```bash
git clone --depth 1 https://github.com/larashero3-dotcom/lieflat-charts.git
```

What we adopt, and where it came from:

* **The Mono grammar**, with tokens taken from that skill's `mono-tokens.js` rather
  than approximated: paper `#F0EFEB`, ink `#1C1C1A`, muted `#8F8E88`, faint
  `#C6C5BF`, grid `#DEDDD6`, and the 7-step grey ladder.
* **Structural rules**: 24px borderless cards separated by whitespace (no borders,
  no shadows), a four-part card (conclusion-style title · subtitle that explains the
  series · chart · uppercase source line), a title that states the conclusion
  instead of naming the chart type, and a minimum SVG label size of 5.5–6.5px.
* **Brightness encodes importance**: the primary metric is ink, the best-so-far
  reference is muted and dashed, a rolled-back round is a hollow faint dot, a kept
  round is a filled ink dot. No gradients, no glows, no decorative fills, no
  broken-axis tricks; value and visual stay proportional.
* **Data is deterministic**: every number comes from the run manifest, so a reload
  paints the same picture.

**Why the Glance family and not Lupi.** lieflat-charts requires that choice to be
justified in writing: Lupi Editorial and Lupi Basics are static editorial charts,
while this page is a live monitoring strip read at a glance, repainting every three
seconds — the case its own rules assign to Glance. The dashboard is therefore a
Glance-family page, not a re-invention.

**Two deliberate deviations**, both because a loopback monitoring page must work
with no network: the remote Inter webfont is replaced by a system-font stack that
starts with Inter, and there is no entrance-animation library (a CSS reveal that
honours `prefers-reduced-motion`).

**Attribution.** lieflat-charts asks that publicly distributed output credit its
developer ("本 skill 由‘躺在废墟里’开发"). That is not stamped into the dashboard UI;
it belongs in documentation, which is this line.

## Limits to state honestly

* A run driven by `--allow-stub` is **not** autonomous research; the manifest and
  the report both say `autonomous_research: false`.
* The dashboard is read-only and loopback-only. Remote access, authentication
  and sharing are explicitly out of scope for this module.
* The toy task's labels are synthetic: a rising metric there says the plumbing
  works, nothing more.
* Holdout evaluation is a separate, budgeted step (`finalize`); the search never
  reads the locked test.