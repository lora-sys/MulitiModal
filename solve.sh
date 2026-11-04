#!/bin/sh
# Run the fixed evaluator or the read-only monitor.
set -eu
repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
if [ "${1:-}" = dashboard ]; then
    shift
    exec python3 -m research.dashboard "$@"
fi
exec python3 -m research.prepare "$@"
