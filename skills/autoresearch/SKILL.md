---
name: autoresearch
description: Run or monitor sequential single-file TCM-SD AutoResearch experiments with fixed evaluation and Git keep or revert.
---

# AutoResearch

Read `AGENTS.md` and `program.md`. Use this skill when the user asks for an AutoResearch campaign or its monitor. The agent drives each experiment; no separate proposer is required.

The only editable research file is `src/research/train.py`. The fixed evaluator, dataset, split, metric and budget are outside the experiment worktree and must not be changed by the candidate. Candidate execution requires OS/container isolation; a separate process is insufficient. Holdout is never used for selection. The current replacement evaluator permits only explicitly synthetic inputs until the independent evaluator isolation is verified; do not work around this restriction.

Use a dedicated `autoresearch/<tag>` Git worktree, a new run directory, and an explicit fixed budget. Evaluate the unchanged file first. Then repeat: read the current history and training file; formulate one hypothesis; change only `train.py`; commit; evaluate the commit; record its result; keep an improvement or reset that dedicated worktree to the previous champion. Follow `program.md` for commands and failure handling. Continue until the user stops the campaign or a stated budget or a real safety/technical boundary is reached; do not ask between rounds.

Serve the read-only monitor with `./solve.sh dashboard --run runs/<tag>`. It shows the current fold and the same `results.tsv` written by the loop. The default bind is localhost and has no authentication. Synthetic results are engineering evidence only; local result files are not independently authenticated.
