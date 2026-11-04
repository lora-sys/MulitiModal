# Issue #56 candidate sandbox resource boundary

Candidate source validation and execution run inside the networkless Docker
worker. The host reads at most 1 MiB of candidate source through a
regular-file-only, no-follow, nonblocking descriptor to bind its SHA-256.
Syntax, AST, imports, and candidate API checks run after the container gate
and before the finalizer charges an unlock event.

The container uses a read-only root, 4 GiB memory with equal memory-swap,
2 CPUs, and a 32-process limit. A trusted root supervisor retains SETUID, SETGID, KILL, and
DAC_READ_SEARCH. DAC_READ_SEARCH lets the supervisor read staged inputs
that are private to the execution account. The supervisor drops the
candidate child to UID/GID 65532 before it runs candidate code; the child
does not retain DAC_READ_SEARCH.
Docker daemon logging is disabled. The supervisor bounds worker runtime to
120 seconds, kills remaining worker-UID processes, publishes a bounded
result record, then waits at most 30 seconds for the host artifact handoff.
The container is auto-removed on exit; the host also kills and removes it in
a finally cleanup path.

Candidate output is written to a 128 MiB tmpfs limited to 256 inodes.
The temporary directory is a separate 64 MiB tmpfs, and the host-release
control directory is a 1 MiB root-only tmpfs. The host exports only the
result record, weights, and architecture file. Each exported file is limited
to 64 MiB and the aggregate export, including the result record, is limited
to 96 MiB. Export streams are size-checked before host writes; symlinks,
FIFOs, other non-regular files, unexpected paths, and excess sizes fail
closed.

## Remaining host boundary

The code enforces a per-run cap on candidate output and exported artifacts.
It does not impose a quota across all completed run directories on the host.
Repeated successful runs can accumulate up to their per-run export limits;
a project or filesystem quota still requires host-side administration.

## Verification

The targeted synthetic regression suite passed 304 tests; 17 were skipped:
5 Docker integration tests (no daemon in the offline test container), 4
TCM-SD real-data checks, 3 checks requiring local BERT weights, and 5 syndrome
real-data checks. It ran in an offline container with read-only source,
scripts, test, and project configuration mounts. The container had no data,
runs, ledger, holdout, or Docker socket mounted. Ruff 0.17.3 and
git diff --check passed. The full host Docker worker chain was not run, so this
evidence does not establish end-to-end candidate isolation.
