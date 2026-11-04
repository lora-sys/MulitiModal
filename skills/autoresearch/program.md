# Experiment loop

Use a dedicated clean `autoresearch/<tag>` Git worktree and a separate ignored `runs/<tag>` directory. Never reset another branch or a dirty worktree. Run the fixed evaluator from the trusted checkout; pass the worktree's `src/research/train.py` as `--candidate`. The candidate container cannot write the evaluator, data, split, metric, or budget. Independent isolation from the proposing agent remains unverified. Follow `AGENTS.md` for isolation and holdout rules.

For a new tag, use `git worktree add -b autoresearch/<tag> /absolute/path/to/worktree HEAD` from a committed trusted checkout. Create the ignored run directory separately. Check that the branch does not already exist and that the worktree is clean before each reset. Do not operate on the user's main checkout.

1. Establish the baseline using the unchanged committed training file. On this version the evaluator accepts `--synthetic` only and uses its built-in `SMOKE_ONLY` fixture. Do not run real-data evaluation while `AGENT_ISOLATION_UNVERIFIED`.
2. Read `results.tsv` and the current training file. Form one hypothesis, edit only that file, then commit it in the dedicated worktree.
3. Evaluate under the same fixture, image, folds, seed and wall-clock budget. Example from the trusted checkout:

   ```bash
   PYTHONPATH=src python -m research.prepare --synthetic \
     --candidate /absolute/path/worktree/src/research/train.py \
     --out /absolute/path/runs/tag/round-001.json --budget 60 --folds 2 --repeats 1
   ```

4. Read the evaluator's JSON. Record the candidate commit and decision against that exact JSON:

   ```bash
   PYTHONPATH=src python -m research.prepare record --repo /absolute/path/worktree \
     --results /absolute/path/runs/tag/results.tsv --commit <candidate-sha> \
     --result /absolute/path/runs/tag/round-001.json --decision KEEP --description '<hypothesis>'
   ```

   Use `REVERT` when the score does not improve, or `CRASH` without `--result` when execution failed. Never record a crash as a numeric score. Once recorded, keep the candidate commit if improved; otherwise reset only the dedicated clean worktree to the previous champion commit. The monitor reads this TSV and the evaluator's status file.
5. Read the new history before choosing the next hypothesis. Continue without asking between rounds; stop on user request, a declared budget, or an unresolved safety/technical failure. Preserve the result files and report missing evidence honestly.
