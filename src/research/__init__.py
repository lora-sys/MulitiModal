"""Source-level AutoResearch (karpathy/autoresearch shape, adapted).

    frozen.py    the boundary the search may not touch (data, split, labels,
                 evaluator, metric, time budget)
    candidate.py THE ONLY FILE THE AGENT EDITS (model, loss, optimiser, loop)
    runner.py    subprocess: fit each fold under the fixed budget, then score
                 with the frozen evaluator
    guard.py     the whitelist, enforced (one file, no frozen symbols)
    loop.py      rounds: apply -> train -> measure -> commit or reset

Why this exists next to `runtime/harness.py`: the harness lets an agent choose
six hyperparameter VALUES inside a frozen model, which measured 23/23 REVERTs
and never moved the champion — an empty search space cannot produce research.
`loop.py` lets it rewrite the model and the training loop, and advances a git
branch on measured improvement, so the artifact of the search is code.

Nothing is imported here: the runner imports `research.candidate` inside the
subprocess, and a candidate must be loadable without dragging the loop's
runtime dependencies in.
"""
